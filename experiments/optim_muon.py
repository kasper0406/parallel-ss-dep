"""Muon optimizer — Keller Jordan, github.com/KellerJordan/Muon.

Newton-Schulz orthogonalisation of the momentum buffer, applied to 2D
weight matrices only. For 1D parameters (norms, biases) and embeddings
fall back to AdamW. The recipe that's consistently 1.5-2× more sample-
efficient than AdamW on small transformer-like models.

This is a self-contained reference port — no external dependency. Uses
fp32 for the Newton-Schulz iteration (MPS-friendly; the original is bf16).
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5,
                                eps: float = 1e-7) -> torch.Tensor:
    """Compute G's "zeroth power" U V^T via 5-step quintic Newton-Schulz.

    Approximates the orthogonal factor of G's polar decomposition.
    The polynomial coefficients (a, b, c) = (3.4445, -4.7750, 2.0315)
    are tuned for fast quintic convergence — Keller Jordan's choice.
    """
    assert G.ndim == 2, f"Newton-Schulz expects 2D, got {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon — orthogonalised SGD for 2D parameters.

    Apply to the "hidden" 2D weight matrices in your model. Use a
    separate AdamW (or similar) for the rest (embeddings, output head,
    biases, normalizations).
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise RuntimeError(
                        f"Muon expects 2D params; got {p.shape}. "
                        f"Use a separate optimizer (AdamW) for 1D params."
                    )
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # Scale by max(1, sqrt(out/in)) — Keller's calibration so the
                # update spectrum is comparable across parameter shapes.
                update = update * max(1.0, (update.size(0) / update.size(1)) ** 0.5)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(update, alpha=-lr)
        return loss


def muon_param_groups(model: torch.nn.Module) -> tuple[list, list]:
    """Split model parameters into (muon_2d, adamw_other) lists.

    Keep embeddings and lm_head with AdamW (per Keller's recipe — input
    and output token embeddings benefit more from per-parameter scaling).
    Everything else 2D goes to Muon.
    """
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = ("embed" in name) or ("lm_head" in name)
        if p.ndim == 2 and not is_embed:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params
