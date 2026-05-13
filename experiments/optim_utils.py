"""Shared optimizer + LR-scheduler construction.

Two optimizer modes:
- "adamw": single AdamW with α scalars in their own param group
  (weight_decay=alpha_wd, default 0.0 per CLAUDE.md mandate).
- "muon": Muon for ≥2D hidden matrices + AdamW for embeddings, lm_head,
  1D params, and ≥3D tensors. FiLM α scalars again get their own AdamW
  group.

Both modes return parallel `(opts, scheds)` lists. The caller calls
`s.step()` on every entry of `scheds` once per training step (and
`o.step()` on every entry of `opts`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


_EMBED_OR_HEAD_NAMES = {"embed.weight", "pos_embed.weight", "lm_head.weight"}


def is_film_alpha(name: str) -> bool:
    """Match the learnable α inside any feedback container
    (sparse_feedback, xattn_feedback, feedback)."""
    if not name.endswith(".alpha"):
        return False
    return any(name.startswith(prefix) or f".{prefix}" in name
               for prefix in ("sparse_feedback.", "xattn_feedback.", "feedback."))


def build_optimizer(model: nn.Module, *, optimizer: str, lr: float,
                    lr_muon: float, alpha_wd: float, steps: int,
                    wd: float = 0.1,
                    verbose: bool = True
                    ) -> tuple[list[torch.optim.Optimizer], list]:
    """Build optimizer(s) + cosine schedulers. See module docstring."""
    if optimizer == "adamw":
        regular, alphas = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (alphas if is_film_alpha(name) else regular).append(p)
        groups = [{"params": regular, "weight_decay": wd}]
        if alphas:
            groups.append({"params": alphas, "weight_decay": alpha_wd})
            if verbose:
                print(f"  α-WD split: {len(alphas)} FiLM α params get "
                      f"weight_decay={alpha_wd}")
        opts = [torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))]
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(
            opts[0], T_max=steps, eta_min=lr * 0.1)]
        return opts, scheds

    if optimizer != "muon":
        raise ValueError(f"unknown optimizer {optimizer!r}")

    # Muon: 2D hidden matrices only. Embeddings, lm_head, 1D, 3D+ → AdamW.
    muon_params, adamw_regular, adamw_alpha = [], [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if name in _EMBED_OR_HEAD_NAMES or p.ndim != 2:
            (adamw_alpha if is_film_alpha(name) else adamw_regular).append(p)
        else:
            muon_params.append(p)
    if verbose:
        print(f"  optimizer split: {len(muon_params)} Muon params "
              f"({sum(p.numel() for p in muon_params)/1e6:.1f}M), "
              f"{len(adamw_regular) + len(adamw_alpha)} AdamW params "
              f"({sum(p.numel() for p in adamw_regular + adamw_alpha)/1e6:.1f}M)")
    adamw_groups = [{"params": adamw_regular, "weight_decay": wd}]
    if adamw_alpha:
        adamw_groups.append({"params": adamw_alpha, "weight_decay": alpha_wd})
        if verbose:
            print(f"  α-WD split: {len(adamw_alpha)} FiLM α params get "
                  f"weight_decay={alpha_wd}")
    opts = [
        torch.optim.Muon(muon_params, lr=lr_muon, momentum=0.95, weight_decay=wd),
        torch.optim.AdamW(adamw_groups, lr=lr, betas=(0.9, 0.95)),
    ]
    if verbose:
        print(f"  Muon weight_decay={wd}; AdamW(regular) weight_decay={wd}")
    scheds = [
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opts[0], T_max=steps, eta_min=lr_muon * 0.1),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opts[1], T_max=steps, eta_min=lr * 0.1),
    ]
    return opts, scheds
