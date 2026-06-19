"""bf16-STATE variant of the DeltaNet per-head matrix optimizer + a per-(arm,
regime) optimizer builder for the dense-LM bf16-regime probe.

The production training regime (see `speed_knobs.apply_speed_knobs` +
`bf16_optim.BF16StateAdamW/BF16StateMuon`) is:
    torch.autocast(bf16) fwd/bwd  +  fp32 MASTER weights  +  bf16 OPTIMIZER STATE.

So relative to the fp32 control, the bf16 regime changes exactly two things:
  (a) gradients are noisier (bf16 fwd/bwd rounding); the .grad on the fp32
      master params is still fp32, so Newton-Schulz still dualizes an fp32 grad.
  (b) the momentum buffer is STORED in bf16 (lifted to fp32 for the math, cast
      back for storage) -- identical treatment to `BF16StateMuon`.

`DeltaNetProjMuonBF16` mirrors `DeltaNetProjMuon` EXACTLY except the momentum
buffer is stored bf16. It overrides only `_upd` (the single place momentum is
touched); the per-head / qk-coupled `step()` orthogonalization is inherited
unchanged. `_ns_batched` already casts to bf16 internally in BOTH regimes
(matching torch Muon's NS), so the NS polynomial precision is regime-invariant.
"""
from __future__ import annotations

import torch

from experiments.exp_deltanet_precond_optim import (
    DeltaNetProjMuon, build_units_from_model,
)
from experiments.bf16_optim import BF16StateAdamW, BF16StateMuon


class DeltaNetProjMuonBF16(DeltaNetProjMuon):
    """DeltaNetProjMuon with bf16 momentum-buffer storage (math in fp32)."""

    def _upd(self, p, mom, nesterov):
        g = p.grad
        st = self.state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(g, dtype=torch.bfloat16)
        buf = st["momentum_buffer"].to(torch.float32)          # lift to fp32
        g32 = g if g.dtype == torch.float32 else g.to(torch.float32)
        buf.lerp_(g32, 1 - mom)                                # buf = mom*buf + (1-mom)*g
        update = g32.lerp(buf, mom) if nesterov else buf       # nesterov look-ahead
        st["momentum_buffer"].copy_(buf)                       # round-to-nearest-even -> bf16
        return update


_EMBED_LIKE = {"embed.weight", "pos_embed.weight", "lm_head.weight"}


def build_dense_opts(model, *, arm: str, regime: str, lr_mat: float,
                     lr_adamw: float, wd: float, momentum: float):
    """Build the optimizer set for one (arm, regime).

    arm    : 'muon' (whole-matrix Muon on all 2D hidden mats) |
             'perhead' (per-head NS on q/k/v/b + Muon on o_proj+MLP).
    regime : 'fp32'  -> torch.optim.{AdamW,Muon} + DeltaNetProjMuon (fp32 state)
             'bf16'  -> BF16State{AdamW,Muon} + DeltaNetProjMuonBF16 (bf16 state)

    Returns (opts, matrix_idx). opts[0] is always the AdamW group (embeddings /
    lm_head / 1D norms); matrix_idx are the indices of the matrix optimizer(s)
    whose LR the cosine schedule drives with lr_mat.
    """
    assert arm in ("muon", "perhead")
    assert regime in ("fp32", "bf16")
    bf16 = regime == "bf16"

    # ---- AdamW group: embedding-like + every non-2D param (1D norms, conv) ----
    adamw_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in _EMBED_LIKE or p.ndim != 2:
            adamw_params.append(p)
    if bf16:
        # compile_step=False: the per-param compile is a pure speed knob and is
        # irrelevant to the bf16-STATE precision question under test; turning it
        # off keeps the 72-run sweep fast and free of per-shape compile latency.
        adamw = BF16StateAdamW(adamw_params, lr=lr_adamw, betas=(0.9, 0.95),
                               weight_decay=0.0, compile_step=False)
    else:
        adamw = torch.optim.AdamW(adamw_params, lr=lr_adamw, betas=(0.9, 0.95),
                                  weight_decay=0.0)
    opts = [adamw]
    matrix_idx = []

    def _muon(params):
        if bf16:
            return BF16StateMuon(params, lr=lr_mat, momentum=momentum,
                                 weight_decay=wd, nesterov=True)
        return torch.optim.Muon(params, lr=lr_mat, momentum=momentum,
                                weight_decay=wd, nesterov=True)

    if arm == "muon":
        muon_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad or p.ndim != 2 or n in _EMBED_LIKE:
                continue
            muon_params.append(p)
        opts.append(_muon(muon_params))
        matrix_idx.append(1)
    else:
        _tp, units, other_2d = build_units_from_model(model, mode="perhead")
        assert units, ("no DeltaNet q/k/v/b units discovered — perhead arm would "
                       "silently collapse into the muon arm")
        opts.append(_muon(other_2d))
        cls = DeltaNetProjMuonBF16 if bf16 else DeltaNetProjMuon
        opts.append(cls(units, lr=lr_mat, momentum=momentum,
                        weight_decay=wd, nesterov=True))
        matrix_idx.extend([1, 2])
    return opts, matrix_idx
