"""Shared training-precision helpers.

`apply_speed_knobs(model, bf16=True, tf32=True)` is the canonical "make it
fast on this hardware" call. Used by `train_lm.py`, `sft_code.py`,
`train_distill.py` (and any future trainer). Keeps the bf16-autocast
wrapping logic in one place so all trainers get the same numerics.
"""
from __future__ import annotations

import torch
import torch.nn as nn


_FLA_DYNAMO_PATCHED = False


def _disable_dynamo_on_fla_helpers(verbose: bool = True) -> None:
    """Mark FLA's data-dependent helpers as `torch._dynamo.disable`.

    `prepare_chunk_indices` builds an index tensor from `cu_seqlens.tolist()`
    — a CPU sync over a data-dependent length list. Under `torch.compile`,
    Dynamo cannot trace this and recompiles every batch (the doc-count
    varies). Disabling tracing for the helper lets it run in eager,
    silences the recompile loop, and is a prerequisite for ever enabling
    `mode="reduce-overhead"` (CUDA Graphs need stable shapes).
    """
    global _FLA_DYNAMO_PATCHED
    if _FLA_DYNAMO_PATCHED:
        return
    try:
        import fla.ops.utils.index as _fla_index
        disabled = torch._dynamo.disable(_fla_index.prepare_chunk_indices)
        _fla_index.prepare_chunk_indices = disabled
        # The kernel callers did `from fla.ops.utils.index import
        # prepare_chunk_indices` at import time, so each holds a local
        # binding to the ORIGINAL function — patching the source module
        # alone leaves the recompile loop in place. Patch every chunk
        # kernel module that imported the helper.
        import importlib
        patched_modules = []
        for mod_name in (
            "fla.ops.delta_rule.chunk",
            "fla.ops.gated_delta_rule.chunk",
            "fla.ops.gated_delta_product.chunk",
            "fla.ops.delta_product.chunk",
        ):
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            if hasattr(mod, "prepare_chunk_indices"):
                mod.prepare_chunk_indices = disabled
                patched_modules.append(mod_name)
        _FLA_DYNAMO_PATCHED = True
        if verbose:
            print(f"torch._dynamo.disable applied to "
                  f"prepare_chunk_indices in "
                  f"fla.ops.utils.index + {len(patched_modules)} caller "
                  f"modules (prevents cu_seqlens-driven recompile loop)")
    except ImportError:
        pass  # FLA not installed → nothing to patch.


def apply_speed_knobs(model: nn.Module, bf16: bool = True, tf32: bool = True,
                       compile_model: bool = False,
                       compile_mode: str = "default",
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
        # Prevent the cu_seqlens-driven recompile loop in FLA's
        # `prepare_chunk_indices` (does a CPU sync over a data-dependent
        # length list) BEFORE wrapping forward in torch.compile.
        _disable_dynamo_on_fla_helpers(verbose=verbose)
        # STRICT COMPILE: turn off dynamo's silent eager fallback. The
        # default `suppress_errors=True` will install torch.compile, then
        # — on the first compile failure — silently revert that frame to
        # eager and print a warning that's easy to miss. Past runs have
        # shipped at production speed because of this exact footgun.
        # When the caller asks for compile, treat any compile error as a
        # hard failure so we notice immediately.
        torch._dynamo.config.suppress_errors = False
        # WORKAROUND (2026-05-21) for a torch.compile / AOTAutograd bug:
        # reconstructing aliased graph outputs by replaying ViewMeta
        # sequences corrupts SymInt view-metadata under dynamic shapes
        # (matches pytorch/pytorch issue #124382), causing a
        # non-deterministic backward-pass segfault / garbage-shape
        # RuntimeError. In this repo it is triggered by the trunk gist
        # loss (cumsum + slice over a dynamic sequence dim) feeding
        # autograd.Function (FLA) nodes under activation checkpointing.
        # Forcing as_strided reconstruction instead of ViewMeta replay
        # dodges it — no correctness impact, negligible cost. Verified
        # 3x against the real pretrain trainer.
        try:
            import torch._functorch.config as _ffc
            _ffc.view_replay_for_aliased_outputs = False
            if verbose:
                print("torch._functorch view_replay_for_aliased_outputs"
                      "=False (ViewMeta-replay segfault workaround)")
        except Exception as _e:  # pragma: no cover
            if verbose:
                print(f"WARNING: view_replay workaround not applied: {_e}")
        # Compile the (possibly bf16-wrapped) forward. The FLA Triton
        # kernels are opaque to Dynamo, so each is a graph break — compile
        # still fuses the PyTorch glue between them (RMSNorm, GLU MLP,
        # FiLM projections, gate/memory heads). fullgraph=False is
        # required (the graph breaks are expected, not errors).
        # Control-flow changes (e.g. the K-self-feed curriculum flipping
        # _film_bypass) trigger a one-time recompile — acceptable.
        # Stash the bf16-wrapped (but NOT yet compiled) forward as
        # `model._eager_forward` so callers that need an out-of-shape
        # extra forward — e.g. `compute_process_reward_loss` /
        # `compute_gate_calibration_loss` (their `after_ids` is
        # `(N, L_after)` ≠ the main `(B, T)` shape) — can opt out of
        # compile per call without triggering a recompile / Inductor
        # symbolic-shape assertion. The main hot path still runs through
        # the compiled forward installed below.
        model._eager_forward = model.forward
        compiled_fwd = torch.compile(model.forward, fullgraph=False,
                                      mode=compile_mode)
        # Verify dynamo can install + run the wrapper at all. If the model
        # has constructs Dynamo refuses to trace (e.g. exotic ModuleDict
        # dispatch in the cross-layer xattn path), torch.compile-install
        # itself succeeds but the FIRST CALL throws. Probing here means
        # the trainer crashes BEFORE step 1, not silently runs eager
        # forever — the exact failure mode that motivated this guard.
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        model.forward = compiled_fwd
        if verbose:
            print(f"torch.compile applied to model.forward "
                  f"(fullgraph=False, mode={compile_mode!r}, "
                  "strict-errors=ON; FLA kernels are graph breaks)")
    return model
