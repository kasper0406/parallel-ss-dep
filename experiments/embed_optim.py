"""Embedding-layer optimizers (the modular-norm 'dualizer' for tables).

`RowNormEmbed` is the embedding/lm_head analogue of what Muon does for the
hidden 2D matrices: maintain a (Nesterov) momentum buffer, then *dualize* the
update before applying it. For a hidden matrix Muon dualizes via Newton-Schulz
(spectral / RMS→RMS operator norm). For an embedding TABLE the natural unit is
a single ROW (one token's vector): the modular norm of the table is the worst
row's RMS norm, so steepest descent under that norm normalizes EACH ROW's
update to unit RMS and scales by lr. Every token vector then takes a step of
fixed RMS magnitude `lr`, independent of its raw per-row gradient scale (rare
tokens get the same effective step as frequent ones — the μP / modular-norm
prescription for embedding layers).

Decoupled weight decay + momentum mirror Muon so a Muon-matrix + RowNorm-embed
optimizer is a clean "dualize everything" configuration. State is fp32 (one
momentum buffer per param; ~same persistent cost as bf16-AdamW's two buffers).
"""
from __future__ import annotations

import torch


class RowNormEmbed(torch.optim.Optimizer):
    """Per-row RMS-normalized update for embedding / lm_head tables.

    update_row = momentum_buffer(row)            (Nesterov-blended)
    update_row /= RMS(update_row)                 (-> unit RMS per row)
    param_row -= lr * update_row                  (+ decoupled weight decay)

    Args mirror Muon's defaults (momentum=0.95, nesterov, decoupled wd).
    Params are expected to be >=2D tables; rows are dim 0, everything else is
    flattened into the per-row vector for the RMS.
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 eps: float = 1e-8):
        if lr < 0.0:
            raise ValueError(f"invalid lr {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum {momentum}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("RowNormEmbed does not support sparse grads")
                g = g.float()
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom) if nesterov else buf
                # Per-row RMS normalization. Row = dim 0; flatten the rest.
                u2 = u.reshape(u.shape[0], -1)
                rms = u2.pow(2).mean(dim=1, keepdim=True).sqrt()
                u2 = u2 / (rms + eps)
                u = u2.reshape(u.shape)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u.to(p.dtype), alpha=-lr)
        return loss
