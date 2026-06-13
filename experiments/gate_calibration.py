"""Gate-calibration auxiliary loss for SFT (Layer 1 of the "think only where
helpful" plan; see THINKING_GATE_SELECTIVITY_2026_05_30.md).

The output gate is supervised in pretrain only by an entropy proxy
(``exp(-H/T)`` — "think where uncertain"). But *uncertainty != thinking-helps*:
much uncertainty is irreducible. This module gives the gate a DENSE,
PER-POSITION teacher of where a think actually improves the next-token
prediction.

Mechanism (the training-time twin of ``probe_gate_calibration``):

  1. From the MAIN forward we already have the gate logit ``g_t`` (snapshotted
     from ``model._last_gate_logits`` BEFORE any extra forward clobbers it —
     this exact bug bit the 2026-05-27 wiring).
  2. Sample a fraction of clean positions. For each, measure whether LATENT
     thinking helps: run ``R`` state-readonly LATENT think steps on
     ``prefix_0..t`` (the validated Coconut-style hidden-feedback mechanism,
     ``thinking.latent_think_logp``) and compare
     ``Delta_t = logp_after_R_latent_thinks(true_{t+1}) - logp_no_think(true_{t+1})``.
  3. target ``y_t = 1{Delta_t > margin}``.
  4. Loss ``+= BCE_with_logits(g_t, y_t)`` — SYMMETRIC: pushes sigma UP where
     thinking helps, DOWN where it doesn't.

IMPORTANT — mechanism. The "does thinking help" measurement uses LATENT
thinking (feed the trunk's own hidden state back as the think-slot input
embedding), NOT discrete ``[THINK]*K`` token appends. Discrete-token thinking
is the validated ``mode="token"`` baseline that does NOT help (0.09 vs 1.00 on
the latent-think validation task); calibrating the gate against it taught the
gate to suppress thinking. The single shared primitive lives in
``thinking.latent_think_logp`` so this loss and the probe can never diverge on
mechanism again.

Cost is ~R forwards on the sampled fraction only (bounded by
``sample_frac x max_positions``). Default weight 0.0 = OFF (byte-identical to
no-aux SFT).

torch.compile note: the variable-length extra forward crashed Inductor in
prior work (2026-05-27). ``thinking.latent_think_logp`` is wrapped in
``torch._dynamo.disable`` so a compiled SFT run does not crash; ``--no-compile``
is the documented fallback if the disable boundary is insufficient.

GPU-free testable: every function takes a model exposing the TinyLM contract
(``model(input_ids, return_hidden=...)`` + ``model._last_gate_logits`` +
``model.embed`` + ``inputs_embeds=``) and runs on whatever device the inputs
live on, so the CPU tests pass a tiny fake model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.thinking import latent_think_logp


# pad id used to build the batched extra forward. MUST differ from the
# thinking token id (pad-as-think silently triggers state_readonly_at_think on
# padding positions, corrupting the after-forward's recurrent state). See
# CLAUDE.md "Pad-id MUST differ" + probe_gate_calibration.
PAD_ID = 0


@dataclass
class GateCalibrationResult:
    """Outputs of one gate-calibration step (None when no positions sampled)."""
    loss: torch.Tensor          # scalar, differentiable through the gate logits
    n_positions: int            # M positions actually scored
    target_frac_pos: float      # fraction with y==1 (thinking helped)
    mean_sigma: float           # mean sigma(gate) at scored positions
    mean_delta: float           # mean Delta logp at scored positions


def _unwrap_logits(out):
    """forward(...) may return a bare Tensor, or (logits, ...) in training
    mode (TinyLM appends a gist-loss scalar / hidden). Take the first item."""
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _clean_position_mask(input_ids: torch.Tensor, targets: torch.Tensor,
                         *, thinking_token_id: int,
                         eos_id: int | None) -> torch.Tensor:
    """(B, T) bool: positions t that are valid calibration positions.

    Valid iff the next-token target_t is a real emit token (not -100 / think /
    pad / eos) AND input_t is itself real (so h_t / the gate logit at t is a
    normal hidden, not a think/pad position)."""
    valid = targets != -100
    valid = valid & (targets != int(thinking_token_id))
    valid = valid & (targets != PAD_ID)
    if eos_id is not None:
        valid = valid & (targets != int(eos_id))
    valid = valid & (input_ids != int(thinking_token_id))
    valid = valid & (input_ids != PAD_ID)
    return valid


def compute_gate_calibration_loss(
    model,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    gate_logits_snapshot: torch.Tensor,
    *,
    thinking_token_id: int,
    latent_R: int = 4,
    margin: float = 0.0,
    sample_frac: float = 0.1,
    max_positions: int = 256,
    sigma_low: float = 0.0,
    sigma_high: float = 1.0,
    eos_id: int | None = None,
    think_batch: int = 64,
    pos_weight: float | None = None,
    generator: torch.Generator | None = None,
) -> GateCalibrationResult | None:
    """Differentiable gate-calibration BCE loss for one (B, T) batch.

    Args:
      model: object exposing the TinyLM contract. Must already have been run
        for the MAIN forward (so the LM logits exist) — we re-derive lp0 here
        from a fresh no_grad forward to keep the helper self-contained and the
        gradient confined to the gate.
      input_ids, targets: (B, T) the SAME tensors the main LM loss used.
        ``targets`` is next-token labels (-100 where masked).
      gate_logits_snapshot: (B, T) the gate logits captured from the MAIN
        forward BEFORE any extra forward clobbers ``model._last_gate_logits``.
        This is the tensor the BCE gradient flows into; it MUST be the
        grad-carrying gate logits, not a detached copy.
      thinking_token_id: the [THINKING] token id. Asserted != PAD_ID.
      latent_R: number of state-readonly LATENT think steps in the extra
        forward (the validated hidden-feedback mechanism; see
        ``thinking.latent_think_logp``).
      margin: ``y = 1{Delta_logp > margin}``.
      sample_frac, max_positions: bound the number of scored positions per
        batch to ``min(sample_frac * n_valid, max_positions)``.
      sigma_low/high: optionally restrict to positions whose current
        sigma(gate) is in (low, high) to focus compute where the gate is
        undecided. Default (0.0, 1.0) keeps all sampled positions.
      eos_id: end-of-sequence id excluded from valid positions (optional).
      think_batch: positions per extra-forward chunk (memory bound).
      generator: optional torch.Generator for reproducible sampling.

    Returns a GateCalibrationResult, or None if no positions qualified
    (caller should treat None as "add nothing this step").
    """
    assert int(thinking_token_id) != PAD_ID, (
        f"thinking_token_id ({thinking_token_id}) must differ from PAD_ID "
        f"({PAD_ID}) — pad-as-think corrupts the state-readonly mask")
    device = input_ids.device
    B, T = input_ids.shape
    if gate_logits_snapshot.shape != (B, T):
        raise ValueError(
            f"gate_logits_snapshot shape {tuple(gate_logits_snapshot.shape)} "
            f"must match input_ids (B, T) = {(B, T)}")

    valid = _clean_position_mask(input_ids, targets,
                                 thinking_token_id=thinking_token_id,
                                 eos_id=eos_id)
    # Need a t+1 target -> drop the last column.
    valid[:, T - 1] = False

    # Optional sigma band restriction (focus on undecided positions).
    if sigma_low > 0.0 or sigma_high < 1.0:
        sigma = torch.sigmoid(gate_logits_snapshot.detach())
        valid = valid & (sigma > sigma_low) & (sigma < sigma_high)

    bsel, tsel = torch.nonzero(valid, as_tuple=True)
    P = bsel.numel()
    if P == 0:
        return None

    n_keep = min(int(sample_frac * P) if sample_frac < 1.0 else P, max_positions)
    n_keep = max(1, min(n_keep, P))
    if n_keep < P:
        # randperm on the generator's device (CPU generator is the common
        # caller), then move the index to the selection device. Avoids the
        # "generator device != tensor device" error when a CPU rng is reused.
        gen_device = generator.device if generator is not None else bsel.device
        perm = torch.randperm(P, generator=generator,
                              device=gen_device)[:n_keep].to(bsel.device)
        bsel, tsel = bsel[perm], tsel[perm]
        P = bsel.numel()

    true_next = targets[bsel, tsel]                           # (P,)

    # Baseline lp0 AND post-think lpR are both computed on the SAME left-padded
    # prefix (fixed 2026-06-13): previously lp0 used the clean full sequence while
    # lpR used a left-padded prefix, so Δ=lpR-lp0 conflated "thinking helped" with
    # "left-pad corrupted lpR's recurrent state". Computing the no-think baseline
    # on the identical padded prefix (last real position = index -1) makes Δ
    # isolate the latent-think effect.
    Lmax = int(tsel.max().item()) + 1
    lpR = torch.empty(P, device=device)
    lp0 = torch.empty(P, device=device)
    for s in range(0, P, think_batch):
        e = min(s + think_batch, P)
        rows = []
        for j in range(s, e):
            b = int(bsel[j].item())
            t = int(tsel[j].item())
            pref = input_ids[b, : t + 1]
            padlen = Lmax - pref.numel()
            if padlen > 0:
                pad = torch.full((padlen,), PAD_ID, dtype=pref.dtype,
                                 device=device)
                pref = torch.cat([pad, pref], dim=0)          # LEFT pad
            rows.append(pref)
        chunk_pref = torch.stack(rows, dim=0)
        # no-think baseline on the SAME padded prefix: logp(true_next) at the
        # last (real) position.
        with torch.no_grad():
            cl = _unwrap_logits(model(chunk_pref)).float()[:, -1, :]
            lp0[s:e] = F.log_softmax(cl, dim=-1).gather(
                1, true_next[s:e].view(-1, 1)).squeeze(1)
        lpR[s:e] = latent_think_logp(
            model, chunk_pref, true_next[s:e], R=latent_R,
            thinking_token_id=thinking_token_id, pad_id=PAD_ID)

    delta = (lpR - lp0).detach()
    y = (delta > margin).float()       # 1 where a latent think HELPS the truth

    # POLARITY (fixed 2026-06-13): the gate convention everywhere — the
    # gate-terms LM loss (`g·CE + (1-g)·λ`), eval (`g=P(emit), emit iff
    # g≥threshold`), and the entropy-aux (confident→σ→1→emit) — is
    # σ = sigmoid(gate_logit) = P(EMIT). We want the gate to THINK where
    # thinking helps, i.e. σ→0 (P(think)=1-σ→1) at y=1 positions. So train
    # P(think) = σ(-gate_logit) toward y by negating the logit in the BCE.
    # (The previous code trained σ(+logit)→y, which drove EMIT exactly where
    # thinking helps — backwards.) pos_weight up-weights the helpful "think"
    # minority (~10% of code positions) so BCE doesn't collapse to all-emit.
    gate_logit_sel = gate_logits_snapshot[bsel, tsel]
    pw = (torch.tensor(float(pos_weight), device=device)
          if pos_weight is not None else None)
    loss = F.binary_cross_entropy_with_logits(-gate_logit_sel, y, pos_weight=pw)

    with torch.no_grad():
        sigma_sel = torch.sigmoid(gate_logit_sel.detach())
    return GateCalibrationResult(
        loss=loss,
        n_positions=int(P),
        target_frac_pos=float(y.mean().item()),
        mean_sigma=float(sigma_sel.mean().item()),
        mean_delta=float(delta.mean().item()),
    )
