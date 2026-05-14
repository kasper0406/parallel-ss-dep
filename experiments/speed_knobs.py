"""Shared training-precision helpers.

`apply_speed_knobs(model, bf16=True, tf32=True)` is the canonical "make it
fast on this hardware" call. Used by `train_lm.py`, `sft_code.py`,
`train_distill.py` (and any future trainer). Keeps the bf16-autocast
wrapping logic in one place so all trainers get the same numerics.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def apply_speed_knobs(model: nn.Module, bf16: bool = True, tf32: bool = True,
                       compile_model: bool = False,
                       verbose: bool = True) -> nn.Module:
    """Apply bf16 autocast + TF32 (+ optional torch.compile) to a built
    model. Returns the model.

    Must be called AFTER model construction but BEFORE the optimizer
    references `model.parameters()` (parameters are unaffected; the
    forward wrap doesn't change `nn.Parameter` identity).
    """
    if tf32:
        torch.set_float32_matmul_precision("high")
        if verbose:
            print("TF32 enabled for fp32 matmul (high precision mode)")
    if bf16:
        # Wrap model.forward in bf16 autocast. Master weights stay fp32
        # (Muon + AdamW expect that); only forward+intermediates run in
        # bf16. Backward runs through the same autocast graph (PyTorch
        # handles the casts). No GradScaler needed (bf16 has fp32 exp range).
        _orig_forward = model.forward

        def _bf16_forward(*args, **kwargs):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return _orig_forward(*args, **kwargs)

        model.forward = _bf16_forward
        if verbose:
            print("bf16 autocast wrapping model.forward")
    if compile_model:
        # Compile the (possibly bf16-wrapped) forward. The FLA Triton
        # kernels are opaque to Dynamo, so each is a graph break — compile
        # still fuses the PyTorch glue between them (RMSNorm, GLU MLP,
        # FiLM projections, gate/memory heads). fullgraph=False is
        # required (the graph breaks are expected, not errors).
        # Control-flow changes (e.g. the K-self-feed curriculum flipping
        # _film_bypass) trigger a one-time recompile — acceptable.
        model.forward = torch.compile(model.forward, fullgraph=False)
        if verbose:
            print("torch.compile applied to model.forward "
                  "(fullgraph=False; FLA kernels are graph breaks)")
    return model
