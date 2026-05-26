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
        # Trivial param so .parameters() / .device-lookup work.
        self.dummy = nn.Linear(1, 1)
        self._last_gate: torch.Tensor | None = None
        self._last_gate_logits: torch.Tensor | None = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        device = ids.device
        # Deterministic logits: token 5 always wins (so emit sampling is
        # predictable). Vocab > 5 + thinking_token_id assumed by callers.
        logits = torch.zeros(B, T, self.vocab_size, device=device,
                             dtype=torch.float32)
        logits[..., 5] = 5.0
        # Stash gate as a (B, T) tensor at the requested value.
        g = torch.full((B, T), float(self._gate_value), device=device,
                       dtype=torch.float32)
        self._last_gate = g
        self._last_gate_logits = torch.logit(g.clamp(1e-6, 1 - 1e-6))
        return logits


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
