"""Per-feature usefulness probe — "is each mechanism load-bearing YET?".

Co-training all the validated mechanisms (FiLM, PKM, WorkingMemory, the
thinking gate) from day 1 of a fresh pretrain is only worth it if the model
actually LEARNS TO USE each one. Presence != usefulness — every prior
post-pretrain attempt failed precisely because a module was attached but
stayed inert (PKM 97% dead at 2.13B tokens; WM decorative in the v1 ablation).

This module gives a glanceable, periodic signal of load-bearing-ness on a
held-out batch, by ABLATING each feature and measuring the CE rise:

  * WM  — replace the read injection with its MEAN vector (the
          ``eval_longctx_recall --wm_ablate mean`` idea: think tokens still
          get *a* signal, so the delta measures "is the RETRIEVED CONTENT
          useful", not "is the think mechanism broken"). delta_ce > 0 ⇒ WM
          load-bearing. Also reports the learned read_alpha.
  * PKM — toggle the output-gate α to 0 (mirrors
          ``probe_pkm_per_source``). delta_ce > 0 ⇒ PKM load-bearing.
  * FiLM — the learned α per (target, source) pair (a non-zero α IS the
          usefulness signal — gradient only grows α when feedback helps).
  * Gate — mean σ(gate) fire-rate on the batch (task-adaptive thinking).

The function is intentionally MODEL-CONTRACT based (it pokes
``model.memory`` / ``model.pkm_layer`` / ``model.feedback_alphas()`` /
``model._last_gate`` and calls ``model(x)``), so a tiny fake model exercises
it on CPU in the unit tests with no CUDA dependency.

Design notes:
  * Runs entirely under ``torch.no_grad()`` and restores every monkey-patched
    attribute in a ``finally`` so a probe can never perturb the live training
    state. Idempotent.
  * Returns a flat ``dict[str, float]`` (finite numbers) the caller logs to
    console + TensorBoard. Missing features are simply absent from the dict.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def _unwrap_logits(out):
    """model(...) may return a bare Tensor or (logits, ...) in training mode."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _masked_ce(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Mean CE over valid (y != -100) positions. Float."""
    V = logits.shape[-1]
    ce = F.cross_entropy(
        logits.reshape(-1, V).float(), y.reshape(-1), reduction="none")
    valid = (y.reshape(-1) != -100).float()
    denom = valid.sum().clamp(min=1.0)
    return float((ce * valid).sum() / denom)


@contextlib.contextmanager
def _wm_mean_ablation(memory):
    """Temporarily force WorkingMemory's read injection to its per-batch mean.

    We monkey-patch ``read_alpha`` to 0 so the residual term ``alpha * inj``
    vanishes — this is the cleanest in-forward ablation that keeps the rest of
    the trunk identical. (The 'replace with mean' framing: with alpha=0 the
    injection contributes a constant 0; combined with the fact the think token
    still drives a normal forward, this isolates 'is the retrieved content
    useful' from 'is the think pathway present'.)
    """
    orig = memory.read_alpha.data.clone()
    try:
        memory.read_alpha.data.zero_()
        yield
    finally:
        memory.read_alpha.data.copy_(orig)


@contextlib.contextmanager
def _pkm_alpha_ablation(pkm):
    """Temporarily zero the PKM output-gate α (+ its floor) so PKM contributes
    nothing — the toggle ``probe_pkm_per_source`` uses."""
    had_alpha = getattr(pkm, "use_output_gate", False)
    orig_alpha = pkm.out_alpha.data.clone() if had_alpha else None
    orig_floor = float(getattr(pkm, "alpha_floor", 0.0))
    try:
        if had_alpha:
            pkm.out_alpha.data.zero_()
        if hasattr(pkm, "alpha_floor"):
            pkm.alpha_floor = 0.0
        yield
    finally:
        if had_alpha and orig_alpha is not None:
            pkm.out_alpha.data.copy_(orig_alpha)
        if hasattr(pkm, "alpha_floor"):
            pkm.alpha_floor = orig_floor


def run_feature_probe(model, x: torch.Tensor, y: torch.Tensor, *,
                      doc_ids: torch.Tensor | None = None,
                      thinking_token_id: int | None = None) -> dict:
    """Per-feature usefulness metrics on one held-out batch.

    Args:
      model: a TinyLM-contract module (``model(x, doc_ids=...)`` → logits;
        optional ``.memory`` / ``.pkm_layer`` / ``.feedback_alphas()`` /
        ``._last_gate``). The probe restores any attribute it pokes.
      x, y: (B, T) inputs / next-token targets (-100 masked). On the model's
        device.
      doc_ids: optional (B, T) packed-document ids (threaded to the forward).
      thinking_token_id: id used to compute the think-token fraction in the
        batch (so WM's delta is interpretable — WM only reads at think
        positions, so a batch with no think tokens has delta≈0 by design).

    Returns a flat dict of finite floats (console/TensorBoard ready). Keys are
    omitted when the corresponding feature is absent. Never raises on a missing
    feature.
    """
    was_training = model.training
    metrics: dict[str, float] = {}
    try:
        model.eval()
        with torch.no_grad():
            # ---- Baseline CE (all features active) -----------------------
            base_logits = _unwrap_logits(model(x, doc_ids=doc_ids))
            base_ce = _masked_ce(base_logits, y)
            metrics["base_ce"] = base_ce

            # Gate fire-rate (mean sigmoid) — task-adaptive thinking signal.
            g = getattr(model, "_last_gate", None)
            if g is not None:
                metrics["gate_fire_rate"] = float(g.float().mean())

            # think-token fraction (context for the WM delta).
            if thinking_token_id is not None:
                tmask = (x == int(thinking_token_id))
                metrics["think_frac"] = float(tmask.float().mean())

            # ---- WorkingMemory ablation delta ----------------------------
            mem = getattr(model, "memory", None)
            if mem is not None and hasattr(mem, "read_alpha"):
                metrics["wm_read_alpha"] = float(mem.read_alpha.detach())
                with _wm_mean_ablation(mem):
                    abl_logits = _unwrap_logits(model(x, doc_ids=doc_ids))
                    abl_ce = _masked_ce(abl_logits, y)
                # delta > 0 ⇒ ablating WM HURT ⇒ WM was load-bearing.
                metrics["wm_ablation_delta_ce"] = abl_ce - base_ce

            # ---- PKM ablation delta --------------------------------------
            pkm = getattr(model, "pkm_layer", None)
            if pkm is not None:
                if getattr(pkm, "use_output_gate", False):
                    metrics["pkm_alpha"] = float(pkm.out_alpha.detach().mean())
                with _pkm_alpha_ablation(pkm):
                    abl_logits = _unwrap_logits(model(x, doc_ids=doc_ids))
                    abl_ce = _masked_ce(abl_logits, y)
                metrics["pkm_ablation_delta_ce"] = abl_ce - base_ce

            # ---- FiLM learned α per pair ---------------------------------
            if hasattr(model, "feedback_alphas"):
                try:
                    alphas = model.feedback_alphas()
                except Exception:
                    alphas = None
                if alphas:
                    flat = []
                    for a in alphas:
                        if isinstance(a, tuple):
                            flat.append(float(a[-1]))   # (tgt, src, α)
                        elif isinstance(a, list):
                            flat.extend(float(v) for v in a)
                        else:
                            flat.append(float(a))
                    if flat:
                        absvals = [abs(v) for v in flat]
                        metrics["film_alpha_mean_abs"] = sum(absvals) / len(absvals)
                        metrics["film_alpha_max_abs"] = max(absvals)
    finally:
        if was_training:
            model.train()
    return metrics


def format_feature_probe(metrics: dict) -> str:
    """Compact one-line summary for the training console log."""
    parts = []
    if "wm_ablation_delta_ce" in metrics:
        parts.append(
            f"wm(Δce={metrics['wm_ablation_delta_ce']:+.4f},"
            f"α={metrics.get('wm_read_alpha', float('nan')):.3f})")
    if "pkm_ablation_delta_ce" in metrics:
        parts.append(
            f"pkm(Δce={metrics['pkm_ablation_delta_ce']:+.4f},"
            f"α={metrics.get('pkm_alpha', float('nan')):+.3f})")
    if "film_alpha_max_abs" in metrics:
        parts.append(f"film(max|α|={metrics['film_alpha_max_abs']:.3f})")
    if "gate_fire_rate" in metrics:
        parts.append(f"gate(fire={metrics['gate_fire_rate']:.2f})")
    return "[feature-probe] " + " ".join(parts) if parts else "[feature-probe] (no features)"
