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


def _is_think_adapter(name: str) -> bool:
    """Phase B ThinkAdapter params (2026-05-26). The adapter lives at
    `blocks.{L}.think_adapter.{fc1,fc2}.{weight,bias}` plus the learnable
    scalar `blocks.{L}.think_adapter.alpha`. ALL go to AdamW (NOT Muon —
    the matrices are small and the dedicated thinking-time function makes
    Newton-Schulz orthogonalisation conceptually wrong; α is a scalar
    that Muon would handle as 1D-AdamW anyway).
    """
    return ".think_adapter." in name or name.endswith(".think_adapter.alpha")


def _is_latent_feedback_adapter(name: str) -> bool:
    """LatentFeedbackAdapter params (2026-06-01). Lives at
    `latent_feedback_adapter.{norm.weight,proj.{weight,bias},alpha}`. ALL go to
    AdamW: the `proj` matrix is zero-init with an identity residual + learnable
    α (the FiLM-α opt-in-via-gradient curriculum), so Muon's Newton-Schulz
    orthogonalisation is the wrong inductive bias (it would immediately push the
    zero matrix toward an orthogonal one).
    """
    return name.startswith("latent_feedback_adapter.")


def _is_refinement_head(name: str) -> bool:
    """Phase D RefinementHead params (2026-05-27). Lives at
    `refinement_head.{W_q,W_k,W_v,W_o,W_up,W_down,attn_norm,mlp_norm}.*`
    plus the scalar `refinement_head.alpha`. ALL go to AdamW: attention
    matrices need adaptive moments; Muon's Newton-Schulz orthogonalisation
    is the wrong inductive bias for short local-window attention.
    """
    return name.startswith("refinement_head.") or name == "refinement_head.alpha"


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
    """Match learnable α scalars that must get NO weight decay (CLAUDE.md
    mandate): FiLM α inside any feedback container, plus the top-level
    cooperation/retrieval scalars `mem_alpha` and `retrieval_input_alpha`
    (init 0.1; WD on these fights the FiLM-α curriculum that grows them only
    if useful)."""
    if name.endswith("mem_alpha") or name.endswith("retrieval_input_alpha"):
        return True
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
                    soap_precond_freq: int = 10,
                    soap_normalize_grads: bool = False,
                    matrix_optimizer: str = "muon",
                    embed_lr_mult: float = 1.0,
                    embed_optimizer: str = "adam",
                    verbose: bool = True
                    ) -> tuple[list[torch.optim.Optimizer], list]:
    """Build optimizer(s) + LR schedulers. See module docstring.

    `lr_schedule`: "cosine" (legacy) or "wsd" (warmup-stable-decay).
    `bf16_optim_state`: if True, AdamW exp_avg/exp_avg_sq and Muon
        momentum_buffer are stored as bf16 (saves ~550 MB persistent
        on the 218 M v4 model). See `experiments/bf16_optim.py`.

    `optimizer == "soap"`: SOAP (Shampoo-in-Adam-eigenbasis, Vyas et al.
        2024, vendored in `experiments/soap.py`) replaces Muon on EXACTLY
        the same 2D hidden-matrix params Muon would handle; embeddings /
        lm_head / PKM value tables / 1D / α scalars still route to AdamW,
        identical to the muon split — so a muon-vs-soap A/B isolates the
        matrix optimizer. The SOAP matrix LR is taken from `lr_muon` (the
        "matrix-optimizer LR" slot); SOAP's per-tensor preconditioner is
        refreshed every `soap_precond_freq` steps. `bf16_optim_state` is
        ignored for the SOAP group (the vendored impl is fp32-state only);
        the AdamW group still honours it.

    `matrix_optimizer` (only consulted when `optimizer == "muon"`): which
        orthogonalizing matrix optimizer to use on the EXACT SAME 2D
        hidden-matrix set Muon would handle.
        - "muon" (default) → `torch.optim.Muon` / `BF16StateMuon`, i.e.
          BYTE-IDENTICAL to the legacy path (every existing launcher /
          checkpoint is unaffected — the flag is purely additive).
        - "fused_deltanet_ns" → `FusedDeltaNetMuon`: per-head Newton–Schulz
          on the DeltaNet q/k/v/b projections (head-structured modular-norm
          dualizer) + whole-matrix Muon on o_proj / MLP / every other 2D
          matrix. The matrix-optimizer PARAM SET is identical to the muon
          arm by construction (units ∪ other_2d == muon_params), and ALL
          other param groups (AdamW for embeds/norms/1D, the FiLM-α group,
          the PKM-value LR group) are untouched — so a muon-vs-fused A/B
          isolates ONLY the q/k/v/b orthogonalization boundary. See
          `experiments/exp_deltanet_precond_fused.py` and
          DELTANET_PRECONDITIONER{,_PERF}.md.

    `embed_lr_mult` / `embed_optimizer` (only consulted when
        `optimizer == "muon"`): how to treat the embedding/lm_head AdamW group
        (the params matched by `_is_embedding_like`, excluding PKM value tables
        which keep their own `--pkm_value_lr_mult` path). Both default to the
        legacy behaviour (mult 1.0, "adam") which is BYTE-IDENTICAL — the
        embed-like params then stay in the shared AdamW group at the base lr.
        - `embed_lr_mult X` (X != 1.0): split the embed-like params into their
          own AdamW group at `lr * X` (the μP-flavoured higher embedding LR).
        - `embed_optimizer "rownorm"`: route the embed-like params to a SEPARATE
          `RowNormEmbed` optimizer (per-row / per-token RMS-normalized update —
          the modular-norm dualizer for embedding tables) at `lr * embed_lr_mult`,
          momentum/wd mirroring Muon. See `experiments/embed_optim.py`.
        Non-default embed treatment requires `--optimizer muon` (it splits the
        Muon-mode AdamW group); it is rejected otherwise rather than silently
        ignored, mirroring the matrix_optimizer guard.
    """
    if bf16_optim_state:
        from experiments.bf16_optim import BF16StateAdamW, BF16StateMuon
        AdamWCls = BF16StateAdamW
        MuonCls = BF16StateMuon
    else:
        AdamWCls = torch.optim.AdamW
        MuonCls = torch.optim.Muon

    # Validate matrix_optimizer up front (covers the adamw branch too — it would
    # otherwise silently ignore the flag and return before the muon-branch check).
    if matrix_optimizer not in ("muon", "fused_deltanet_ns"):
        raise ValueError(f"unknown matrix_optimizer {matrix_optimizer!r}")
    if matrix_optimizer == "fused_deltanet_ns" and optimizer != "muon":
        raise ValueError(
            "matrix_optimizer='fused_deltanet_ns' requires --optimizer muon "
            f"(got {optimizer!r}); it replaces the Muon matrix optimizer only.")

    if embed_optimizer not in ("adam", "rownorm"):
        raise ValueError(f"unknown embed_optimizer {embed_optimizer!r}")
    # Non-default embedding treatment is implemented only for the muon split
    # (it carves the embed-like params out of the Muon-mode AdamW group). Reject
    # rather than silently ignore on other base optimizers (mirrors the
    # matrix_optimizer guard — the adamw branch would otherwise return early).
    embed_special = (embed_lr_mult != 1.0) or (embed_optimizer == "rownorm")
    if embed_special and optimizer != "muon":
        raise ValueError(
            f"non-default embedding treatment (embed_lr_mult={embed_lr_mult}, "
            f"embed_optimizer={embed_optimizer!r}) requires --optimizer muon "
            f"(got {optimizer!r}).")

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

    if optimizer not in ("muon", "soap"):
        raise ValueError(f"unknown optimizer {optimizer!r}")

    # Muon / SOAP: 2D hidden matrices only. Embeddings, lm_head, 1D, 3D+ → AdamW.
    # ThinkAdapter (Phase B, 2026-05-26): both the 2D fc weights AND the
    # 1D α scalar route to AdamW — the adapter is a small specialized
    # head, not a general d×d hidden matrix Newton-Schulz should normalize.
    muon_params, adamw_regular, adamw_alpha, adamw_pkm_values = [], [], [], []
    adamw_embed = []  # embed/lm_head split out only when embed_special
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if _is_think_adapter(name):
            adamw_regular.append(p)
            continue
        if _is_refinement_head(name):
            adamw_regular.append(p)
            continue
        if _is_latent_feedback_adapter(name):
            adamw_regular.append(p)
            continue
        if _is_embedding_like(name) or p.ndim != 2:
            if _is_pkm_value(name) and pkm_value_lr_mult != 1.0:
                adamw_pkm_values.append(p)
            elif is_film_alpha(name):
                adamw_alpha.append(p)
            elif embed_special and _is_embedding_like(name) and not _is_pkm_value(name):
                adamw_embed.append(p)
            else:
                adamw_regular.append(p)
        else:
            muon_params.append(p)
    if verbose:
        _all_adamw = adamw_regular + adamw_alpha + adamw_pkm_values + adamw_embed
        print(f"  optimizer split: {len(muon_params)} Muon params "
              f"({sum(p.numel() for p in muon_params)/1e6:.1f}M), "
              f"{len(_all_adamw)} AdamW params "
              f"({sum(p.numel() for p in _all_adamw)/1e6:.1f}M)")
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
    # Embedding/lm_head with a higher AdamW LR (embed_optimizer == "adam").
    # When embed_optimizer == "rownorm", adamw_embed is instead handed to a
    # separate RowNormEmbed optimizer below (NOT added to the AdamW groups).
    if adamw_embed and embed_optimizer == "adam":
        adamw_groups.append({"params": adamw_embed, "weight_decay": wd,
                             "lr": lr * embed_lr_mult})
        if verbose:
            print(f"  embed LR mult: {len(adamw_embed)} embed/lm_head params "
                  f"get AdamW lr={lr * embed_lr_mult:.2e} "
                  f"({embed_lr_mult}× base lr={lr:.2e})")
    if optimizer == "muon":
        if matrix_optimizer == "fused_deltanet_ns":
            # Per-head Newton-Schulz on DeltaNet q/k/v/b + whole-matrix Muon
            # on everything else. CRUCIAL fairness property: the matrix
            # optimizer must touch EXACTLY `muon_params` (the same set the
            # plain-muon arm uses), so we derive the per-head units from the
            # model but route every muon_param NOT captured by a q/k/v/b unit
            # to whole-matrix Muon. union(units, other_2d) == muon_params.
            from experiments.exp_deltanet_precond_optim import (
                build_units_from_model)
            from experiments.exp_deltanet_precond_fused import FusedDeltaNetMuon
            _, _all_units, _ = build_units_from_model(model, mode="perhead")
            muon_ids = {id(p) for p in muon_params}
            kept_units, unit_ids = [], set()
            for u in _all_units:
                if all(id(p) in muon_ids for p in u["params"]):
                    kept_units.append(u)
                    unit_ids.update(id(p) for p in u["params"])
            other_2d = [p for p in muon_params if id(p) not in unit_ids]
            if not kept_units:
                raise ValueError(
                    "matrix_optimizer='fused_deltanet_ns' found no DeltaNet "
                    "q/k/v/b projections — is --arch deltanet set? The fused "
                    "per-head optimizer only applies to DeltaNet blocks.")
            matrix_opt = FusedDeltaNetMuon(
                kept_units, other_2d, lr=lr_muon, momentum=0.95,
                weight_decay=wd, nesterov=True, ns_steps=5,
                bf16_state=bf16_optim_state, batch_across_layers=False)
            if verbose:
                _n_unit = sum(p.numel() for u in kept_units for p in u["params"])
                _n_other = sum(p.numel() for p in other_2d)
                print(f"  matrix_optimizer=fused_deltanet_ns: "
                      f"{len(kept_units)} per-head q/k/v/b units "
                      f"({_n_unit/1e6:.1f}M) + {len(other_2d)} whole-matrix "
                      f"Muon params ({_n_other/1e6:.1f}M); "
                      f"matrix set == muon arm ({len(muon_params)} params); "
                      f"weight_decay={wd}"
                      + ("  [bf16 optimizer state]" if bf16_optim_state else ""))
        else:
            matrix_opt = MuonCls(muon_params, lr=lr_muon, momentum=0.95,
                                 weight_decay=wd)
            if verbose:
                print(f"  Muon weight_decay={wd}; AdamW(regular) "
                      f"weight_decay={wd}"
                      + ("  [bf16 optimizer state]" if bf16_optim_state else ""))
    else:  # soap — Shampoo-in-Adam-eigenbasis on the same 2D matrix params.
        from experiments.soap import SOAP
        matrix_opt = SOAP(muon_params, lr=lr_muon, betas=(0.95, 0.95),
                          weight_decay=wd, precondition_frequency=soap_precond_freq,
                          # vocab-sized dims excluded by max_precond_dim; our 2D
                          # hidden matrices are all <= max_precond_dim so both
                          # sides get a full preconditioner. 1D excluded here
                          # anyway (1D params route to AdamW above).
                          precondition_1d=False,
                          # SOAP-recommended companion to large precond_freq
                          # (~100): per-layer grad normalization (helps at high
                          # freq, hurts at low freq per the paper).
                          normalize_grads=soap_normalize_grads)
        if verbose:
            print(f"  SOAP(matrix) lr={lr_muon:.2e} betas=(0.95,0.95) "
                  f"wd={wd} precond_freq={soap_precond_freq}; "
                  f"AdamW(regular) weight_decay={wd}"
                  + ("  [AdamW bf16 state; SOAP state fp32]"
                     if bf16_optim_state else ""))
    opts = [
        matrix_opt,
        AdamWCls(adamw_groups, lr=lr, betas=(0.9, 0.95)),
    ]
    scheds = [
        _make_scheduler(opts[0], base_lr=lr_muon, schedule=lr_schedule,
                        steps=steps, warmup_steps=warmup_steps,
                        decay_frac=decay_frac),
        _make_scheduler(opts[1], base_lr=lr, schedule=lr_schedule,
                        steps=steps, warmup_steps=warmup_steps,
                        decay_frac=decay_frac),
    ]
    # RowNorm dualizer for embed/lm_head: a SEPARATE optimizer (+ its own
    # WSD/cosine scheduler) on the carved-out embed group. Added only when
    # requested, so the default path is byte-identical (no extra optimizer).
    if adamw_embed and embed_optimizer == "rownorm":
        from experiments.embed_optim import RowNormEmbed
        embed_lr = lr * embed_lr_mult
        rownorm_opt = RowNormEmbed(adamw_embed, lr=embed_lr, momentum=0.95,
                                   nesterov=True, weight_decay=wd)
        opts.append(rownorm_opt)
        scheds.append(_make_scheduler(rownorm_opt, base_lr=embed_lr,
                                      schedule=lr_schedule, steps=steps,
                                      warmup_steps=warmup_steps,
                                      decay_frac=decay_frac))
        if verbose:
            print(f"  embed_optimizer=rownorm: {len(adamw_embed)} embed/lm_head "
                  f"params get per-row RMS-normalized (modular-norm dualizer) "
                  f"updates at lr={embed_lr:.2e} "
                  f"({embed_lr_mult}× base lr={lr:.2e}), momentum=0.95, wd={wd}")
    if verbose:
        print(f"  lr_schedule={lr_schedule}"
              + (f" (warmup={warmup_steps}, decay_frac={decay_frac})"
                 if lr_schedule == "wsd" else ""))
    return opts, scheds
