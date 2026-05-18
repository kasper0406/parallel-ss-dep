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

import math

import torch
import torch.nn as nn


_EMBED_OR_HEAD_NAMES = {"embed.weight", "pos_embed.weight", "lm_head.weight"}


def _is_embedding_like(name: str) -> bool:
    """Names of params Muon should NOT touch (they're table-shaped, not
    hidden matrices): embeddings, lm_head, and PKM value tables (each
    PKM value head is an nn.Embedding named pkm_layer.values.{i}.weight).
    Newton-Schulz orthogonalisation on a 65 k×144 lookup table is wasted
    compute and conceptually wrong."""
    if name in _EMBED_OR_HEAD_NAMES:
        return True
    if name.startswith("pkm_layer.values.") and name.endswith(".weight"):
        return True
    return False


def _is_pkm_value(name: str) -> bool:
    """The 38-65 M-param PKM value tables. Eligible for the v7.1 value-LR
    multiplier (--pkm_value_lr_mult), which compensates for the α·w_k·∂loss
    multiplicative dampening of per-row gradient."""
    return name.startswith("pkm_layer.values.") and name.endswith(".weight")


def _wsd_lambda(total_steps: int, warmup_steps: int, decay_frac: float):
    """Warmup-Stable-Decay LR multiplier in [0, 1].

    - warmup: linear 0 -> 1 over `warmup_steps`
    - stable: constant 1.0 for the bulk of training
    - decay:  cosine 1.0 -> 0.0 over the last `decay_frac` of steps

    The stable phase means no wasted low-LR tail; the decay can be run
    from any stopping point to "cash out" a checkpoint, which is why WSD
    suits open-ended-horizon runs better than cosine.
    """
    decay_steps = max(1, int(decay_frac * total_steps))
    stable_end = max(warmup_steps, total_steps - decay_steps)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step < stable_end:
            return 1.0
        progress = min(1.0, (step - stable_end) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def _make_scheduler(opt, *, base_lr: float, schedule: str, steps: int,
                    warmup_steps: int, decay_frac: float):
    """Cosine (legacy, byte-identical to before) or WSD."""
    if schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=base_lr * 0.1)
    if schedule == "wsd":
        return torch.optim.lr_scheduler.LambdaLR(
            opt, _wsd_lambda(steps, warmup_steps, decay_frac))
    raise ValueError(f"unknown lr_schedule {schedule!r}")


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
                    lr_schedule: str = "cosine",
                    warmup_steps: int = 0,
                    decay_frac: float = 0.15,
                    bf16_optim_state: bool = False,
                    pkm_value_lr_mult: float = 1.0,
                    verbose: bool = True
                    ) -> tuple[list[torch.optim.Optimizer], list]:
    """Build optimizer(s) + LR schedulers. See module docstring.

    `lr_schedule`: "cosine" (legacy) or "wsd" (warmup-stable-decay).
    `bf16_optim_state`: if True, AdamW exp_avg/exp_avg_sq and Muon
        momentum_buffer are stored as bf16 (saves ~550 MB persistent
        on the 218 M v4 model). See `experiments/bf16_optim.py`.
    """
    if bf16_optim_state:
        from experiments.bf16_optim import BF16StateAdamW, BF16StateMuon
        AdamWCls = BF16StateAdamW
        MuonCls = BF16StateMuon
    else:
        AdamWCls = torch.optim.AdamW
        MuonCls = torch.optim.Muon

    if optimizer == "adamw":
        regular, alphas, pkm_values = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_pkm_value(name) and pkm_value_lr_mult != 1.0:
                pkm_values.append(p)
            elif is_film_alpha(name):
                alphas.append(p)
            else:
                regular.append(p)
        groups = [{"params": regular, "weight_decay": wd}]
        if alphas:
            groups.append({"params": alphas, "weight_decay": alpha_wd})
            if verbose:
                print(f"  α-WD split: {len(alphas)} FiLM α params get "
                      f"weight_decay={alpha_wd}")
        if pkm_values:
            groups.append({"params": pkm_values, "weight_decay": wd,
                           "lr": lr * pkm_value_lr_mult})
            if verbose:
                print(f"  PKM-value LR boost: {len(pkm_values)} value tables "
                      f"get lr={lr * pkm_value_lr_mult:.2e} "
                      f"({pkm_value_lr_mult}× base lr={lr:.2e})")
        opts = [AdamWCls(groups, lr=lr, betas=(0.9, 0.95))]
        scheds = [_make_scheduler(opts[0], base_lr=lr, schedule=lr_schedule,
                                  steps=steps, warmup_steps=warmup_steps,
                                  decay_frac=decay_frac)]
        if verbose:
            print(f"  lr_schedule={lr_schedule}"
                  + (f" (warmup={warmup_steps}, decay_frac={decay_frac})"
                     if lr_schedule == "wsd" else ""))
        return opts, scheds

    if optimizer != "muon":
        raise ValueError(f"unknown optimizer {optimizer!r}")

    # Muon: 2D hidden matrices only. Embeddings, lm_head, 1D, 3D+ → AdamW.
    muon_params, adamw_regular, adamw_alpha, adamw_pkm_values = [], [], [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if _is_embedding_like(name) or p.ndim != 2:
            if _is_pkm_value(name) and pkm_value_lr_mult != 1.0:
                adamw_pkm_values.append(p)
            elif is_film_alpha(name):
                adamw_alpha.append(p)
            else:
                adamw_regular.append(p)
        else:
            muon_params.append(p)
    if verbose:
        print(f"  optimizer split: {len(muon_params)} Muon params "
              f"({sum(p.numel() for p in muon_params)/1e6:.1f}M), "
              f"{len(adamw_regular) + len(adamw_alpha) + len(adamw_pkm_values)} AdamW params "
              f"({sum(p.numel() for p in adamw_regular + adamw_alpha + adamw_pkm_values)/1e6:.1f}M)")
    adamw_groups = [{"params": adamw_regular, "weight_decay": wd}]
    if adamw_alpha:
        adamw_groups.append({"params": adamw_alpha, "weight_decay": alpha_wd})
        if verbose:
            print(f"  α-WD split: {len(adamw_alpha)} FiLM α params get "
                  f"weight_decay={alpha_wd}")
    if adamw_pkm_values:
        adamw_groups.append({"params": adamw_pkm_values, "weight_decay": wd,
                              "lr": lr * pkm_value_lr_mult})
        if verbose:
            print(f"  PKM-value LR boost: {len(adamw_pkm_values)} value "
                  f"tables get lr={lr * pkm_value_lr_mult:.2e} "
                  f"({pkm_value_lr_mult}× base lr={lr:.2e})")
    opts = [
        MuonCls(muon_params, lr=lr_muon, momentum=0.95, weight_decay=wd),
        AdamWCls(adamw_groups, lr=lr, betas=(0.9, 0.95)),
    ]
    if verbose:
        print(f"  Muon weight_decay={wd}; AdamW(regular) weight_decay={wd}"
              + ("  [bf16 optimizer state]" if bf16_optim_state else ""))
    scheds = [
        _make_scheduler(opts[0], base_lr=lr_muon, schedule=lr_schedule,
                        steps=steps, warmup_steps=warmup_steps,
                        decay_frac=decay_frac),
        _make_scheduler(opts[1], base_lr=lr, schedule=lr_schedule,
                        steps=steps, warmup_steps=warmup_steps,
                        decay_frac=decay_frac),
    ]
    if verbose:
        print(f"  lr_schedule={lr_schedule}"
              + (f" (warmup={warmup_steps}, decay_frac={decay_frac})"
                 if lr_schedule == "wsd" else ""))
    return opts, scheds
