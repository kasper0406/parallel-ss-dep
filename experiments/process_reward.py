"""Phase A — process-reward auxiliary loss.

Trains the trunk + WM to make thinks *productive*: at sampled positions
where the gate already wants to fire, run a second forward with K think
tokens inserted before position t and check whether the resulting
next-token prediction has more probability mass on the actual next
token. The loss:

    L_process = mean_t [ -log p_after(y_{t+1}) + log p_before(y_{t+1}) ]

becomes negative as `p_after` puts more mass on the truth than
`p_before`. Minimising it pushes the optimizer to make thinks reduce
next-token error.

Default `process_reward_weight=0.0` → off → byte-identical to existing
training.

Design notes
------------
* "Before": uses the ORIGINAL main-forward logits already computed for
  the LM loss. No extra compute.
* "After": one extra forward pass over a small batch of synthesised
  sequences. For each sampled position `(b, t)` we build
  `[x[b, 0..t], K * THINK_ID]` of length `t + 1 + K`, then read the
  logits at the last position. The model's prediction at that final
  step is what would be sampled to choose `y[t+1]` if the model had
  thought K times. To keep compute bounded we cap the number of
  sampled positions per batch via `sample_frac` and a hard upper bound.
* Retrieval-as-input mode (additive α-gated WM injection) is honoured
  when the caller passes `retrieval_as_input=True`; otherwise the
  embedding-table path is used (still gives think positions a single
  shared embedding, but the loss can still measure the consequence).

See `THINKING_PLAN.md` Phase A for full motivation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ProcessRewardStats:
    """Diagnostics from one call to compute_process_reward_loss."""
    n_candidates: int        # positions where gate>min_sigma & target valid
    n_sampled: int           # positions actually used in the after-forward
    mean_log_ratio: float    # mean(log p_after - log p_before); positive = good
    frac_positive: float     # fraction of sampled positions with after > before


@dataclass
class GateCalibrationStats:
    """Diagnostics from one call to compute_gate_calibration_loss."""
    n_candidates: int        # positions where σ in [min_sigma, max_sigma]
    n_sampled: int           # positions actually evaluated
    target_frac_one: float   # fraction of sampled positions where target=1
                             # (i.e. thinking helped)
    mean_sigma: float        # mean σ over sampled positions (pre-clamp)
    mean_log_ratio: float    # mean(log p_think - log p_no_think); diagnostic


def _select_candidate_positions(
    gate: torch.Tensor,             # (B, T) sigmoid gate
    target_shifted: torch.Tensor,   # (B, T-1) labels for y[t+1], -100 = mask
    apply_min_sigma: float,
    sample_frac: float,
    rng: torch.Generator,
    max_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (b_idx, t_idx) of sampled candidate positions.

    A candidate is a position `t` in `[0, T-2]` such that:
      - `gate[b, t] > apply_min_sigma` — the gate already wants to think
      - `target_shifted[b, t] != -100` — there's a real next-token target
    Out of all candidates we keep a uniform-random fraction
    `sample_frac`, hard-capped at `max_positions` to bound compute.
    """
    B, T = gate.shape
    # Restrict to t in [0, T-2] so y[t+1] is in range.
    gate_left = gate[:, :T - 1]
    valid_target = target_shifted.ne(-100)
    cand_mask = (gate_left > apply_min_sigma) & valid_target  # (B, T-1)
    cand_flat = cand_mask.flatten().nonzero(as_tuple=True)[0]
    n_cand = int(cand_flat.numel())
    if n_cand == 0:
        empty = torch.empty(0, dtype=torch.long, device=gate.device)
        return empty, empty
    target_n = min(max(1, int(round(n_cand * sample_frac))), max_positions)
    if target_n >= n_cand:
        pick = cand_flat
    else:
        perm = torch.randperm(n_cand, generator=rng, device="cpu")[:target_n]
        pick = cand_flat[perm.to(gate.device)]
    Tm1 = T - 1
    b_idx = (pick // Tm1).to(torch.long)
    t_idx = (pick % Tm1).to(torch.long)
    return b_idx, t_idx


def _build_after_sequences(
    x: torch.Tensor,                # (B, T) input_ids
    b_idx: torch.Tensor,            # (N,)
    t_idx: torch.Tensor,            # (N,)
    K: int,
    thinking_token_id: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each sampled (b, t), build [x[b, 0..t], K * THINK_ID] left-
    padded to a uniform length. Return:
      after_ids   (N, L_max)         — left-padded with `pad_token_id`
      last_pos    (N,)               — index of the last real position
                                       (where logits should be read).
    `L_max` is `t_idx.max() + 1 + K` capped at the original T (which is
    guaranteed to be enough room because t <= T-2 → t+1+K <= T-1+K, so
    we use T+K). We use LEFT padding so the last token's logits are the
    "predict y[t+1] after thinking" we want.
    """
    N = b_idx.numel()
    T = x.shape[1]
    L_after = int(t_idx.max().item()) + 1 + K  # tokens of real content
    # Build (N, L_after), left-pad with pad_token_id.
    out = torch.full((N, L_after), pad_token_id, dtype=x.dtype,
                     device=x.device)
    last_pos = torch.empty(N, dtype=torch.long, device=x.device)
    # We can't avoid a small Python loop here unless we vectorise via
    # gather + roll. Sampled-N is small (default ~10% of valid positions,
    # capped) so this is fine.
    for i in range(N):
        b = int(b_idx[i].item())
        t = int(t_idx[i].item())
        prefix_len = t + 1
        pad_n = L_after - prefix_len - K
        # Real content sits in the rightmost (prefix_len + K) slots.
        out[i, pad_n:pad_n + prefix_len] = x[b, :prefix_len]
        out[i, pad_n + prefix_len:pad_n + prefix_len + K] = thinking_token_id
        last_pos[i] = L_after - 1
    return out, last_pos


def _call_model_eager(model, *args, **kwargs):
    """Invoke the model forward, skipping torch.compile when possible.

    `speed_knobs.apply_speed_knobs(..., compile_model=True)` stashes the
    pre-compile (bf16-wrapped) forward as `model._eager_forward`. The
    aux-loss extra forwards run on `(N, L_after)`-shaped inputs that do
    NOT match the main `(B, T)` graph, so routing them through the
    compiled forward either triggers expensive recompiles or — when
    Inductor specialises on the main shape — a symbolic-shape assert.
    Using the eager forward sidesteps both. When compile is OFF the
    attribute is absent and we fall back to the normal `model(...)`.
    """
    eager = getattr(model, "_eager_forward", None)
    if eager is not None:
        return eager(*args, **kwargs)
    return model(*args, **kwargs)


def _retrieval_input_embeds(
    model,
    ids: torch.Tensor,
    thinking_token_id: int,
) -> torch.Tensor:
    """Build inputs_embeds for the after-forward in v7 additive
    retrieval-as-input mode.

    Mirrors the logic in sft_code.py's main forward: an extra no_grad
    forward to populate `memory._last_injection`, then add the lagged
    injection at think positions, scaled by `model.retrieval_input_alpha`.
    """
    with torch.no_grad():
        _ = _call_model_eager(model, ids)
    inj = model.memory._last_injection                  # (N, L, d), detached
    base_emb = model.embed(ids)
    is_think = (ids == int(thinking_token_id)).unsqueeze(-1)
    shifted_inj = torch.cat(
        [torch.zeros_like(inj[:, :1]), inj[:, :-1]], dim=1)
    alpha = model.retrieval_input_alpha
    return (
        base_emb
        + is_think.to(base_emb.dtype) * alpha * shifted_inj.to(base_emb.dtype)
    )


def compute_process_reward_loss(
    model,
    x: torch.Tensor,                 # (B, T) input_ids of the main batch
    y: torch.Tensor,                 # (B, T) labels; -100 = mask
    gate: torch.Tensor,              # (B, T) σ(gate) from main forward
    main_logits: torch.Tensor,       # (B, T, V_real) main-forward logits
                                     #   (already truncated to base_vocab)
    *,
    thinking_token_id: int,
    K: int,
    apply_min_sigma: float,
    sample_frac: float,
    rng: torch.Generator,
    pad_token_id: int,
    retrieval_as_input: bool,
    base_vocab_for_loss: int | None,
    max_positions: int = 256,
) -> tuple[torch.Tensor, ProcessRewardStats]:
    """Compute the process-reward auxiliary loss.

    Returns `(loss, stats)`. When there are no candidate positions the
    loss is a zero scalar with `requires_grad=False`; callers should
    multiply by the configured weight and add to the total loss.
    """
    if pad_token_id == thinking_token_id:
        raise ValueError(
            "pad_token_id must NOT equal thinking_token_id. Pad-as-think "
            "silently triggers state_readonly_at_think / "
            "mem_write_only_at_think on padding positions, corrupting the "
            "after-forward's recurrent state. Pass a different pad id."
        )
    B, T = x.shape
    device = x.device
    # Shifted labels: y[t+1] is the target for position t.
    target_shifted = y[:, 1:]                                # (B, T-1)

    # Pretrain note: y may contain `thinking_token_id` at think-burst
    # positions. The loss path slices logits to base_vocab_for_loss
    # (which excludes the think token), so a target equal to
    # `thinking_token_id` is out-of-range for the gather and triggers
    # a CUDA device-side assert. Treat such positions as masked.
    target_shifted_masked = target_shifted.masked_fill(
        target_shifted == int(thinking_token_id), -100)

    b_idx, t_idx = _select_candidate_positions(
        gate.detach(), target_shifted_masked,
        apply_min_sigma=apply_min_sigma,
        sample_frac=sample_frac,
        rng=rng,
        max_positions=max_positions,
    )
    n_cand_mask = (gate.detach()[:, :T - 1] > apply_min_sigma) & (
        target_shifted_masked != -100)
    n_candidates = int(n_cand_mask.sum().item())
    n_sampled = int(b_idx.numel())
    if n_sampled == 0:
        zero = torch.zeros((), device=device, requires_grad=False)
        return zero, ProcessRewardStats(n_candidates=n_candidates,
                                        n_sampled=0,
                                        mean_log_ratio=0.0,
                                        frac_positive=0.0)

    # --- "Before": log p of y[t+1] under the main forward at position t.
    # main_logits has shape (B, T, V_real). We need logits at position t.
    # Already truncated to base_vocab so log_softmax is over real tokens.
    main_logits_at = main_logits[b_idx, t_idx, :]                # (N, V)
    targets = target_shifted_masked[b_idx, t_idx]                # (N,)
    log_p_before = F.log_softmax(main_logits_at.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)).squeeze(-1)                   # (N,)
    log_p_before = log_p_before.detach()  # not part of the optimisation

    # --- "After": forward over the synthesised K-think sequences.
    after_ids, last_pos = _build_after_sequences(
        x, b_idx, t_idx, K=K,
        thinking_token_id=thinking_token_id,
        pad_token_id=pad_token_id,
    )
    if retrieval_as_input:
        inputs_embeds = _retrieval_input_embeds(
            model, after_ids, thinking_token_id=thinking_token_id)
        after_out = _call_model_eager(
            model, after_ids, inputs_embeds=inputs_embeds)
    else:
        after_out = _call_model_eager(model, after_ids)
    # TinyLM.forward returns a TUPLE in training mode when gist_loss is
    # enabled (`(logits, gist_loss_scalar)`). Our extra forward should
    # only care about the logits — discard the gist contribution (it's
    # computed off a throwaway forward and we don't want to backprop it).
    after_logits = after_out[0] if isinstance(after_out, tuple) else after_out
    # Some callers truncate logits to base_vocab; mirror that so the
    # softmax denominator matches "before".
    if base_vocab_for_loss is not None:
        after_logits = after_logits[..., :base_vocab_for_loss]
    # Gather logits at the last (real) position per row.
    after_logits_at = after_logits[
        torch.arange(after_ids.shape[0], device=device), last_pos, :]
    log_p_after = F.log_softmax(after_logits_at.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)).squeeze(-1)                   # (N,)

    # Loss: want log_p_after > log_p_before → minimise (log_p_before - log_p_after).
    loss = (log_p_before - log_p_after).mean()
    diff = (log_p_after - log_p_before).detach()
    stats = ProcessRewardStats(
        n_candidates=n_candidates,
        n_sampled=n_sampled,
        mean_log_ratio=float(diff.mean().item()),
        frac_positive=float((diff > 0).float().mean().item()),
    )
    return loss, stats


# ---------------------------------------------------------------------------
# Gate-calibration auxiliary loss
# ---------------------------------------------------------------------------
#
# Sibling of process_reward. Same extra-forward mechanism, different loss
# direction:
#   * process_reward trains the TRUNK to make K-think predictions more
#     accurate than no-think predictions (pushes log p_after up).
#   * gate_calibration trains the GATE to PREDICT whether thinking will
#     help at this position. The target is a binary label derived from
#     the same extra forward — target_t = 1 if log p_think >
#     log p_no_think else 0 — and the loss is BCE on σ_t against that
#     label. Symmetric: penalises both "gate fires but thinking didn't
#     help" AND "gate didn't fire but thinking would have helped".
#
# Default off (`weight=0`). When on, the loss reuses the main-forward
# "before" logits (already computed by the caller's LM forward), runs
# one extra forward over [prefix, K * THINK] for sampled positions, and
# returns a single scalar BCE loss.

def _select_candidate_positions_window(
    gate: torch.Tensor,              # (B, T) sigmoid gate
    target_shifted: torch.Tensor,    # (B, T-1) labels for y[t+1]; -100 mask
    apply_min_sigma: float,
    apply_max_sigma: float,
    sample_frac: float,
    rng: torch.Generator,
    max_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Like `_select_candidate_positions` but candidacy is a
    `[min, max]` σ window — gate-calibration deliberately wants to
    train *uncertain* gate positions, so positions where the gate is
    already saturated (σ ~ 0 or σ ~ 1) are skipped.
    """
    B, T = gate.shape
    gate_left = gate[:, :T - 1]
    valid_target = target_shifted.ne(-100)
    cand_mask = (
        (gate_left >= apply_min_sigma)
        & (gate_left <= apply_max_sigma)
        & valid_target
    )
    cand_flat = cand_mask.flatten().nonzero(as_tuple=True)[0]
    n_cand = int(cand_flat.numel())
    if n_cand == 0:
        empty = torch.empty(0, dtype=torch.long, device=gate.device)
        return empty, empty
    target_n = min(max(1, int(round(n_cand * sample_frac))), max_positions)
    if target_n >= n_cand:
        pick = cand_flat
    else:
        perm = torch.randperm(n_cand, generator=rng, device="cpu")[:target_n]
        pick = cand_flat[perm.to(gate.device)]
    Tm1 = T - 1
    b_idx = (pick // Tm1).to(torch.long)
    t_idx = (pick % Tm1).to(torch.long)
    return b_idx, t_idx


def compute_gate_calibration_loss(
    model,
    x: torch.Tensor,                # (B, T) input_ids of the main batch
    y: torch.Tensor,                # (B, T) labels; -100 = mask
    gate: torch.Tensor,             # (B, T) σ(gate) from main forward
    main_logits: torch.Tensor,      # (B, T, V_real) main-forward logits
                                    #   (already truncated to base_vocab)
    *,
    thinking_token_id: int,
    K: int,
    apply_min_sigma: float,
    apply_max_sigma: float,
    sample_frac: float,
    rng: torch.Generator,
    pad_token_id: int,
    retrieval_as_input: bool,
    base_vocab_for_loss: int | None,
    max_positions: int = 32,
    gate_logits: torch.Tensor | None = None,  # (B, T) pre-sigmoid gate
                                              # — preferred for numerical
                                              # stability (BCE-with-logits)
    smooth_target_scale: float = 0.0,  # > 0 → use σ((logp_think -
                                       # logp_no_think) * scale) instead
                                       # of hard {0,1} target.
) -> tuple[torch.Tensor, GateCalibrationStats]:
    """Compute the gate-calibration auxiliary loss.

    For each sampled position t:
      - "no-think" log-prob from `main_logits[b, t, y[b, t+1]]`
      - "think" log-prob from an extra forward over [prefix, K * THINK]
      - target = 1 if log p_think > log p_no_think else 0
      - loss term = BCE(σ_t, target)

    Returns `(loss, stats)`. When there are no candidate positions the
    loss is a zero scalar with `requires_grad=False`.
    """
    if pad_token_id == thinking_token_id:
        raise ValueError(
            "pad_token_id must NOT equal thinking_token_id. Pad-as-think "
            "silently triggers state_readonly_at_think / "
            "mem_write_only_at_think on padding positions, corrupting the "
            "after-forward's recurrent state. Pass a different pad id."
        )
    if apply_max_sigma < apply_min_sigma:
        raise ValueError(
            f"apply_max_sigma ({apply_max_sigma}) must be >= "
            f"apply_min_sigma ({apply_min_sigma})."
        )
    B, T = x.shape
    device = x.device
    target_shifted = y[:, 1:]                                # (B, T-1)

    # Pretrain note (mirrors compute_process_reward_loss above): mask
    # positions where the TARGET is the thinking token — those would
    # gather-out-of-range when logits are sliced to base_vocab_for_loss.
    target_shifted_masked = target_shifted.masked_fill(
        target_shifted == int(thinking_token_id), -100)

    gate_detached = gate.detach()
    b_idx, t_idx = _select_candidate_positions_window(
        gate_detached, target_shifted_masked,
        apply_min_sigma=apply_min_sigma,
        apply_max_sigma=apply_max_sigma,
        sample_frac=sample_frac,
        rng=rng,
        max_positions=max_positions,
    )
    n_cand_mask = (
        (gate_detached[:, :T - 1] >= apply_min_sigma)
        & (gate_detached[:, :T - 1] <= apply_max_sigma)
        & (target_shifted_masked != -100)
    )
    n_candidates = int(n_cand_mask.sum().item())
    n_sampled = int(b_idx.numel())
    if n_sampled == 0:
        zero = torch.zeros((), device=device, requires_grad=False)
        return zero, GateCalibrationStats(
            n_candidates=n_candidates, n_sampled=0,
            target_frac_one=0.0, mean_sigma=0.0, mean_log_ratio=0.0)

    # --- "No-think": log p of y[t+1] under main forward at position t.
    main_logits_at = main_logits[b_idx, t_idx, :]                # (N, V)
    targets = target_shifted_masked[b_idx, t_idx]                # (N,)
    log_p_no_think = F.log_softmax(main_logits_at.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)).squeeze(-1)                   # (N,)
    log_p_no_think = log_p_no_think.detach()  # label, not optimised

    # --- "Think": extra forward over [prefix, K * THINK_ID] sequences.
    after_ids, last_pos = _build_after_sequences(
        x, b_idx, t_idx, K=K,
        thinking_token_id=thinking_token_id,
        pad_token_id=pad_token_id,
    )
    with torch.no_grad():
        # The think-branch output is a LABEL for the gate, not a
        # gradient path — no need to track grad through the second
        # forward. Saves activation memory vs process_reward.
        if retrieval_as_input:
            inputs_embeds = _retrieval_input_embeds(
                model, after_ids, thinking_token_id=thinking_token_id)
            after_out = _call_model_eager(
                model, after_ids, inputs_embeds=inputs_embeds)
        else:
            after_out = _call_model_eager(model, after_ids)
        # Tuple in training mode with gist_loss; discard gist contribution.
        after_logits = (after_out[0] if isinstance(after_out, tuple)
                        else after_out)
        if base_vocab_for_loss is not None:
            after_logits = after_logits[..., :base_vocab_for_loss]
        after_logits_at = after_logits[
            torch.arange(after_ids.shape[0], device=device), last_pos, :]
        log_p_think = F.log_softmax(
            after_logits_at.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)).squeeze(-1)            # (N,)

    diff = (log_p_think - log_p_no_think).detach()                # (N,)

    # --- Build target.
    if smooth_target_scale > 0.0:
        target = torch.sigmoid(diff * float(smooth_target_scale))
    else:
        target = (diff > 0).float()
    target = target.detach()

    # --- BCE on the gate. Prefer logits-form when available (numerical
    # stability); fall back to direct σ BCE for callers that don't
    # stash the pre-sigmoid logit.
    if gate_logits is not None:
        gl = gate_logits[b_idx, t_idx]                             # (N,)
        loss = F.binary_cross_entropy_with_logits(gl.float(), target)
    else:
        sig = gate[b_idx, t_idx].clamp(1e-6, 1.0 - 1e-6).float()   # (N,)
        loss = F.binary_cross_entropy(sig, target)

    stats = GateCalibrationStats(
        n_candidates=n_candidates,
        n_sampled=n_sampled,
        target_frac_one=float(target.mean().item()),
        mean_sigma=float(gate_detached[b_idx, t_idx].mean().item()),
        mean_log_ratio=float(diff.mean().item()),
    )
    return loss, stats
