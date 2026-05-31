"""Tests for the stochastic-gate-as-policy machinery in train_rl_grader.py.

Pivot (2026-05-26): instead of teaching the gate to fire via CoT imitation,
let RL DISCOVER the optimal thinking pattern by Bernoulli-sampling
emit/think and rewarding the gate's decisions with the same group-relative
advantage that rewards the emit-token PPO.

These tests cover:
- The Bernoulli math (log p, log 1-p, entropy at p=0.5).
- The rollout records gate decisions with the right log-prob.
- A unchanged-policy PPO ratio is exactly 1.0.
- Variation in decisions across rollouts when gate is near 0.5.
- force_emit positions are NOT recorded (model had no choice).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.train_rl_grader import (
    Rollout,
    compute_entropy_bonus,
    policy_loss_for_rollouts_batched,
    rollout_group_batched,
)


# ---------------------------------------------------------------------------
# Stub model that mimics TinyLM's _last_gate stash + a fixed-logits forward.
# ---------------------------------------------------------------------------

class StubLM(nn.Module):
    """Minimal stand-in: fixed gate value at every position, fixed logits
    so emit-sampling is deterministic given temperature=0. No actual params
    are needed for the rollout test, but we add one tiny linear so
    next(model.parameters()) works in the PPO loss path."""

    def __init__(self, vocab_size: int, *, gate_value: float):
        super().__init__()
        self.vocab_size = vocab_size
        self._gate_value = gate_value
        self.d_model = 4
        # lm_head whose product with the constant ones-hidden reproduces the
        # old fixed logits (token 5 = 5.0, rest 0). The policy loss now applies
        # lm_head only at gathered emit positions (skip_lm_head path), so the
        # stub must expose a real hidden + lm_head rather than canned logits.
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[5, :] = 5.0 / self.d_model
        self._last_gate: torch.Tensor | None = None
        self._last_gate_logits: torch.Tensor | None = None

    def forward(self, ids: torch.Tensor, skip_lm_head: bool = False):
        B, T = ids.shape
        device = ids.device
        # Constant hidden -> lm_head(hidden) gives token-5-wins logits.
        hidden = torch.ones(B, T, self.d_model, device=device,
                            dtype=torch.float32)
        # Stash gate as a (B, T) tensor at the requested value.
        g = torch.full((B, T), float(self._gate_value), device=device,
                       dtype=torch.float32)
        self._last_gate = g
        self._last_gate_logits = torch.logit(g.clamp(1e-6, 1 - 1e-6))
        if skip_lm_head:
            return hidden
        return self.lm_head(hidden)


class StubTokenizer:
    eos_token_id = 1

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


# ---------------------------------------------------------------------------
# Pure-math sanity tests (no model needed).
# ---------------------------------------------------------------------------

def test_bernoulli_log_prob_emit_branch():
    p = torch.tensor([0.2, 0.5, 0.8])
    # log p when emit=True
    lp = torch.log(p)
    assert torch.allclose(lp,
                          torch.tensor([math.log(0.2),
                                         math.log(0.5),
                                         math.log(0.8)]))


def test_bernoulli_log_prob_think_branch():
    p = torch.tensor([0.2, 0.5, 0.8])
    # log(1-p) when emit=False (i.e. chose think)
    lp = torch.log(1.0 - p)
    assert torch.allclose(lp,
                          torch.tensor([math.log(0.8),
                                         math.log(0.5),
                                         math.log(0.2)]))


def test_bernoulli_entropy_max_at_half():
    """H(p) = -(p log p + (1-p) log(1-p)). Max ≈ 0.693 at p=0.5,
    monotone toward 0 at p=0 or p=1."""
    p_half = torch.tensor(0.5)
    H_half = -(p_half * torch.log(p_half)
               + (1 - p_half) * torch.log(1 - p_half))
    assert abs(float(H_half.item()) - math.log(2.0)) < 1e-6

    # Near 0: very small entropy.
    p_near_zero = torch.tensor(1e-4)
    H_near_zero = -(p_near_zero * torch.log(p_near_zero)
                    + (1 - p_near_zero) * torch.log(1 - p_near_zero))
    assert float(H_near_zero.item()) < 0.01

    # Near 1: also very small entropy.
    p_near_one = torch.tensor(1.0 - 1e-4)
    H_near_one = -(p_near_one * torch.log(p_near_one)
                   + (1 - p_near_one) * torch.log(1 - p_near_one))
    assert float(H_near_one.item()) < 0.01


# ---------------------------------------------------------------------------
# Rollout-recording tests.
# ---------------------------------------------------------------------------

def test_rollout_records_gate_decisions_when_stochastic_on():
    """With stochastic_gate=True, each rollout's gate_* fields are populated
    and contain at least one decision (since p_emit is finite and there are
    multiple iterations)."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.5)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3, 4]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=4,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=4,
        max_think_per_step=2,
        total_think_budget=8,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,  # deterministic emit (token 5 always)
        min_emit_before_eos=0,
        stochastic_gate=True,
    )
    assert len(rollouts) == 4
    for r in rollouts:
        # Lists are populated (even if length zero on a corner case).
        assert r.gate_decisions is not None
        assert r.gate_log_probs is not None
        assert r.gate_positions is not None
        assert len(r.gate_decisions) == len(r.gate_log_probs)
        assert len(r.gate_decisions) == len(r.gate_positions)
        # Log-probs are -log(2) ≈ -0.693 at p=0.5 (both branches identical).
        for lp, dec in zip(r.gate_log_probs, r.gate_decisions):
            expected = math.log(0.5)
            assert abs(lp - expected) < 1e-5, (lp, expected, dec)


def test_rollout_default_stochastic_off_leaves_gate_fields_none():
    """Backwards-compat: default stochastic_gate=False → gate fields are None."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.9)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=2,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=3,
        max_think_per_step=2,
        total_think_budget=4,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
    )
    for r in rollouts:
        assert r.gate_decisions is None
        assert r.gate_log_probs is None
        assert r.gate_positions is None


def test_rollout_variation_across_rollouts_at_p_half():
    """At p=0.5, Bernoulli sampling should produce variation across rollouts.
    With enough rollouts and decisions, we expect both emit and think to appear."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.5)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=16,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=4,
        max_think_per_step=2,
        total_think_budget=8,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
        stochastic_gate=True,
    )
    all_decisions = [d for r in rollouts for d in (r.gate_decisions or [])]
    assert any(all_decisions), "expected at least one emit decision"
    assert any(not d for d in all_decisions), "expected at least one think decision"


def test_rollout_force_emit_excluded_from_gate_record():
    """When `total_think_budget=0`, every position is force_emit and so NO
    gate decisions should be recorded (the model had no choice)."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.01)  # would WANT to think if free
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=4,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=3,
        max_think_per_step=2,
        total_think_budget=0,  # ZERO budget → force_emit every iter
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
        stochastic_gate=True,
    )
    for r in rollouts:
        assert r.gate_decisions == [], (
            "force_emit positions should not be recorded as policy choices")


# ---------------------------------------------------------------------------
# PPO policy-loss tests.
# ---------------------------------------------------------------------------

def test_ppo_gate_ratio_is_one_when_policy_unchanged():
    """Sanity: same model, same gate value → new_lp == old_lp → ratio == 1."""
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.5)
    # Build a synthetic rollout: 2 emit tokens, 2 gate decisions (one emit,
    # one think) at known positions with log_p = log(0.5) each.
    full_ids = [2, 3, 5, 9, 5]  # prompt=[2,3], emit=5 at pos2, think=9 at pos3, emit=5 at pos4
    rollout = Rollout(
        prompt_len=2,
        emit_token_ids=[5, 5],
        emit_log_probs=[float(F.log_softmax(
            torch.tensor([0.0] * 5 + [5.0] + [0.0] * (vocab - 6)).float(),
            dim=-1)[5].item())] * 2,
        emit_positions=[2, 4],
        full_ids=full_ids,
        depth=1,
        text="5 5",
        gate_decisions=[True, False],
        gate_log_probs=[math.log(0.5), math.log(0.5)],
        gate_positions=[2, 3],
    )
    loss, ratio, kl, gate_stats = policy_loss_for_rollouts_batched(
        [rollout], [1.0], model,
        clip_eps=0.1,
        thinking_token_id=thinking_id,
        temperature=1.0,
        pad_id=0,
        stochastic_gate=True,
        gate_entropy_bonus=0.0,
    )
    # Emit ratio ≈ 1.0 (model logits unchanged; bf16 autocast introduces
    # ~0.01 numerical drift between rollout-time fp32 and policy-time bf16).
    assert abs(float(ratio.item()) - 1.0) < 0.02, ratio
    # Gate ratio should be 1.0 (gate value unchanged at 0.5; gate is set in
    # the stub OUTSIDE autocast and round-trips through fp32 exp/log).
    assert abs(gate_stats["gate_ratio"] - 1.0) < 1e-4, gate_stats
    # Entropy at p=0.5 is log(2).
    assert abs(gate_stats["gate_entropy"] - math.log(2.0)) < 1e-4, gate_stats
    # Fire rate = 1/2 (one emit decision out of two).
    assert abs(gate_stats["gate_fire_rate"] - 0.5) < 1e-4, gate_stats


def test_entropy_bonus_subtracts_from_loss():
    """Larger entropy_bonus → strictly smaller loss (since entropy > 0)."""
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.5)
    full_ids = [2, 3, 5, 5]
    emit_lp = float(F.log_softmax(
        torch.tensor([0.0] * 5 + [5.0] + [0.0] * (vocab - 6)).float(),
        dim=-1)[5].item())
    rollout = Rollout(
        prompt_len=2, emit_token_ids=[5, 5], emit_log_probs=[emit_lp, emit_lp],
        emit_positions=[2, 3], full_ids=full_ids, depth=0, text="",
        gate_decisions=[True, True],
        gate_log_probs=[math.log(0.5), math.log(0.5)],
        gate_positions=[2, 3],
    )
    loss_no_bonus, *_ = policy_loss_for_rollouts_batched(
        [rollout], [1.0], model,
        clip_eps=0.1, thinking_token_id=thinking_id, temperature=1.0,
        pad_id=0, stochastic_gate=True, gate_entropy_bonus=0.0)
    loss_with_bonus, *_ = policy_loss_for_rollouts_batched(
        [rollout], [1.0], model,
        clip_eps=0.1, thinking_token_id=thinking_id, temperature=1.0,
        pad_id=0, stochastic_gate=True, gate_entropy_bonus=1.0)
    # entropy bonus is subtracted → loss_with_bonus < loss_no_bonus.
    assert float(loss_with_bonus.item()) < float(loss_no_bonus.item()), (
        loss_no_bonus.item(), loss_with_bonus.item())
    # Specifically, loss decreased by exactly bonus * entropy = 1.0 * log(2).
    delta = float(loss_no_bonus.item()) - float(loss_with_bonus.item())
    assert abs(delta - math.log(2.0)) < 1e-4, (delta, math.log(2.0))


# ---------------------------------------------------------------------------
# Selective stochastic gate tests (Phase A, 2026-05-26).
# Sample only when σ(gate) is in (low, high); decisive positions use the
# deterministic threshold and are NOT recorded as policy choices.
# ---------------------------------------------------------------------------

def test_selective_sampling_excludes_decisive():
    """At σ ≈ 0.95 with sample range (0.1, 0.9), every position is decisive
    → threshold-emit AND no gate decisions recorded."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.95)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=4,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=4,
        max_think_per_step=2,
        total_think_budget=8,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
        stochastic_gate=True,
        gate_sample_range_low=0.1,
        gate_sample_range_high=0.9,
    )
    for r in rollouts:
        # Decisive (σ=0.95 > 0.9) → no Bernoulli sample recorded.
        assert r.gate_decisions == [], (
            "decisive σ should be excluded from gate_decisions")
        assert r.gate_positions == []
        assert r.gate_log_probs == []
        assert r.gate_n_sampled == 0
        # Should have AT LEAST one decisive count (every non-force-emit
        # step contributes one).
        assert r.gate_n_decisive >= 1, r.gate_n_decisive
        # The >=0.9 bucket should be the one populated.
        assert r.gate_sigma_bucket_counts is not None
        assert r.gate_sigma_bucket_counts[3] >= 1


def test_selective_sampling_includes_uncertain():
    """At σ = 0.5 with sample range (0.1, 0.9), every position is uncertain
    → Bernoulli-sampled AND recorded as a gate decision; across many
    rollouts we see both emit and think."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.5)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=16,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=4,
        max_think_per_step=2,
        total_think_budget=8,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
        stochastic_gate=True,
        gate_sample_range_low=0.1,
        gate_sample_range_high=0.9,
    )
    all_decisions = [d for r in rollouts for d in (r.gate_decisions or [])]
    # Sampled at uncertain positions → both branches must appear.
    assert any(all_decisions), "expected at least one emit decision"
    assert any(not d for d in all_decisions), "expected at least one think"
    # Every recorded decision is from a sampled position; sum should match.
    for r in rollouts:
        assert len(r.gate_decisions) == r.gate_n_sampled, (
            len(r.gate_decisions), r.gate_n_sampled)
        # σ=0.5 → falls into [0.5, 0.9) bucket.
        assert r.gate_sigma_bucket_counts[2] >= 1


def test_range_default_samples_everywhere():
    """Default range [0.0, 1.0] matches legacy sample-everywhere behavior.
    Regression test for backwards-compat."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    # σ=0.95 — would be decisive under (0.1, 0.9) but inside default (0.0, 1.0).
    model = StubLM(vocab, gate_value=0.95)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=4,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=4,
        max_think_per_step=2,
        total_think_budget=8,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
        stochastic_gate=True,
        # Defaults — every position is in (0.0, 1.0) → sampled.
        gate_sample_range_low=0.0,
        gate_sample_range_high=1.0,
    )
    for r in rollouts:
        # Every non-force-emit position sampled → at least one decision
        # (and zero decisive positions).
        assert r.gate_n_decisive == 0
        assert r.gate_n_sampled >= 1
        assert len(r.gate_decisions) == r.gate_n_sampled


def test_range_boundary_handling():
    """Convention: σ STRICTLY inside (low, high) is sampled. σ == low or
    σ == high is OUT (treated decisive)."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    # σ = 0.1 EXACTLY — should be treated decisive when low=0.1.
    model = StubLM(vocab, gate_value=0.1)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=2,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=3,
        max_think_per_step=2,
        total_think_budget=6,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
        stochastic_gate=True,
        gate_sample_range_low=0.1,
        gate_sample_range_high=0.9,
    )
    for r in rollouts:
        # σ == low → out of range → decisive, no recording.
        assert r.gate_decisions == [], r.gate_decisions
        assert r.gate_n_sampled == 0
        assert r.gate_n_decisive >= 1


def test_diagnostics_fields_present_in_rollout():
    """Phase A diagnostic fields are populated on the Rollout output:
    gate_sigma_bucket_counts (4 buckets), gate_n_sampled, gate_n_decisive."""
    torch.manual_seed(0)
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.5)  # uncertain → sampled under (0.1, 0.9)
    tok = StubTokenizer()
    prompt = torch.tensor([[2, 3]])

    rollouts = rollout_group_batched(
        model, tok, prompt,
        n_rollouts=2,
        thinking_token_id=thinking_id,
        eos_token_id=tok.eos_token_id,
        max_gen=3,
        max_think_per_step=2,
        total_think_budget=6,
        emit_threshold=0.5,
        gate_floor=0.0,
        temperature=0.0,
        min_emit_before_eos=0,
        stochastic_gate=True,
        gate_sample_range_low=0.1,
        gate_sample_range_high=0.9,
    )
    for r in rollouts:
        assert r.gate_sigma_bucket_counts is not None
        assert len(r.gate_sigma_bucket_counts) == 4
        # Every recorded position is accounted for in exactly one of the
        # two partitions (sampled or decisive); buckets sum to total.
        bucket_sum = sum(r.gate_sigma_bucket_counts)
        assert bucket_sum == r.gate_n_sampled + r.gate_n_decisive, (
            bucket_sum, r.gate_n_sampled, r.gate_n_decisive)


def test_stochastic_off_yields_no_gate_stats():
    """Backwards-compat path: stochastic_gate=False → gate_stats are NaN."""
    vocab = 16
    thinking_id = 9
    model = StubLM(vocab, gate_value=0.5)
    full_ids = [2, 3, 5]
    emit_lp = float(F.log_softmax(
        torch.tensor([0.0] * 5 + [5.0] + [0.0] * (vocab - 6)).float(),
        dim=-1)[5].item())
    rollout = Rollout(
        prompt_len=2, emit_token_ids=[5], emit_log_probs=[emit_lp],
        emit_positions=[2], full_ids=full_ids, depth=0, text="",
        # gate_* default to None (the legacy / non-stochastic case).
    )
    loss, ratio, kl, gate_stats = policy_loss_for_rollouts_batched(
        [rollout], [1.0], model,
        clip_eps=0.1, thinking_token_id=thinking_id, temperature=1.0,
        pad_id=0, stochastic_gate=False)
    assert math.isnan(gate_stats["gate_ratio"])
    assert math.isnan(gate_stats["gate_entropy"])
    assert math.isnan(gate_stats["gate_fire_rate"])
    # Loss / ratio behave as the prior emit-only path (bf16 autocast ~1 %).
    assert abs(float(ratio.item()) - 1.0) < 0.02


# ---------------------------------------------------------------------------
# Phase B: entropy-bonus curriculum tests.
# ---------------------------------------------------------------------------

def test_entropy_curriculum_linear_schedule():
    start, end, total = 0.05, 0.001, 200
    assert abs(compute_entropy_bonus(
        0, static=0.01, start=start, end=end, total=total) - start) < 1e-9
    assert abs(compute_entropy_bonus(
        total, static=0.01, start=start, end=end, total=total) - end) < 1e-9
    mid = compute_entropy_bonus(
        total // 2, static=0.01, start=start, end=end, total=total)
    assert abs(mid - (start + end) / 2) < 1e-9, mid


def test_entropy_curriculum_clamps_after_total():
    start, end, total = 0.05, 0.001, 100
    val = compute_entropy_bonus(
        total * 2, static=0.01, start=start, end=end, total=total)
    assert abs(val - end) < 1e-9, val


def test_entropy_curriculum_off_falls_back_to_static():
    static = 0.013
    for step in (0, 10, 100, 10_000):
        val = compute_entropy_bonus(
            step, static=static, start=0.0, end=0.5, total=200)
        assert val == static, (step, val)
