"""CPU-only tests for within-group think-budget diversity (FIX 2,
THINKING_GATE_SELECTIVITY_2026_05_30.md).

Tests the pure budget-assignment helper and the per-row force-emit wiring in
rollout_group_batched. No GPU, no real model — the helper is pure arithmetic;
the rollout wiring is exercised with a tiny CPU TinyLM.

Run ONLY this file (the full suite has CUDA tests that would OOM a co-resident
training run):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \\
        experiments/test_think_budget_diversity.py -v
"""
from __future__ import annotations

import pytest
import torch

from experiments.train_rl_grader import compute_think_budget_spread


# ---------------------------------------------------------------------------
# Pure helper: spread correctness.
# ---------------------------------------------------------------------------

def test_diversity_zero_is_uniform():
    # diversity=0 -> every rollout gets the full budget (byte-identical).
    for n in (1, 2, 4, 8):
        assert compute_think_budget_spread(120, n, 0.0) == [120] * n


def test_single_rollout_always_full_budget():
    # n==1 -> the spread degenerates to the base budget regardless of d.
    assert compute_think_budget_spread(120, 1, 0.5) == [120]
    assert compute_think_budget_spread(120, 1, 1.0) == [120]


def test_spread_ascending_and_endpoints():
    out = compute_think_budget_spread(120, 4, 0.5)
    assert len(out) == 4
    # Ascending.
    assert out == sorted(out)
    # Top rollout gets the full budget; bottom gets ~budget*(1-d)=60.
    assert out[-1] == 120
    assert out[0] == 60


def test_spread_within_bounds():
    # Every budget in [1, budget] for any (budget, n, d).
    for budget in (1, 5, 40, 120):
        for n in (1, 2, 3, 5, 8):
            for d in (0.0, 0.25, 0.5, 0.9, 1.0):
                out = compute_think_budget_spread(budget, n, d)
                assert len(out) == n
                for b in out:
                    assert 1 <= b <= budget


def test_diversity_one_floor_is_one():
    # d=1 -> lowest rollout gets budget 1, highest gets the full budget.
    out = compute_think_budget_spread(100, 5, 1.0)
    assert out[0] == 1
    assert out[-1] == 100
    assert out == sorted(out)


def test_diversity_clamped():
    # d outside [0,1] is clamped.
    assert compute_think_budget_spread(120, 4, -0.5) == [120] * 4
    assert compute_think_budget_spread(120, 4, 5.0) == \
        compute_think_budget_spread(120, 4, 1.0)


def test_empty_group():
    assert compute_think_budget_spread(120, 0, 0.5) == []


# ---------------------------------------------------------------------------
# Wiring: rollout_group_batched honors a per-row budget.
# ---------------------------------------------------------------------------

import torch.nn as nn


class _StubLM(nn.Module):
    """Minimal stand-in (mirrors test_stochastic_gate.StubLM): fixed gate value
    + deterministic logits. Crucially it has NO forward_step/prefill, so the
    rollout uses the full-forward path (no DeltaNet/Triton needed on CPU)."""

    def __init__(self, vocab_size: int, *, gate_value: float):
        super().__init__()
        self.vocab_size = vocab_size
        self._gate_value = gate_value
        self.dummy = nn.Linear(1, 1)
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        device = ids.device
        logits = torch.zeros(B, T, self.vocab_size, device=device,
                             dtype=torch.float32)
        logits[..., 5] = 5.0   # token 5 always wins (THINK_ID is 6, no clash)
        g = torch.full((B, T), float(self._gate_value), device=device,
                       dtype=torch.float32)
        self._last_gate = g
        self._last_gate_logits = torch.logit(g.clamp(1e-6, 1 - 1e-6))
        return logits


class _StubTokenizer:
    eos_token_id = 1

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


THINK_ID = 6  # != winning emit token (5), != EOS (1), != PAD (0)


def test_rollout_accepts_scalar_and_list_budget():
    """A scalar budget and an equivalent per-row list produce the SAME rollouts
    (the per-row path with all-equal budgets is byte-identical to the scalar
    path) — and a per-row list of the WRONG length raises."""
    from experiments.train_rl_grader import rollout_group_batched
    m = _StubLM(64, gate_value=0.9)  # σ=0.9 >= 0.5 -> emit-mostly
    tok = _StubTokenizer()
    prompt = torch.randint(7, 50, (1, 6))

    common = dict(
        thinking_token_id=THINK_ID, eos_token_id=1, max_gen=8,
        max_think_per_step=4, emit_threshold=0.5, gate_floor=0.0,
        temperature=0.0, min_emit_before_eos=4)

    g_scalar = rollout_group_batched(
        m, tok, prompt, n_rollouts=3, total_think_budget=10, **common)
    g_list = rollout_group_batched(
        m, tok, prompt, n_rollouts=3, total_think_budget=[10, 10, 10], **common)
    assert [r.emit_token_ids for r in g_scalar] == \
        [r.emit_token_ids for r in g_list]
    assert [r.depth for r in g_scalar] == [r.depth for r in g_list]

    with pytest.raises(ValueError):
        rollout_group_batched(
            m, tok, prompt, n_rollouts=3, total_think_budget=[10, 10],
            **common)


def test_per_row_budget_caps_think_totals():
    """With distinct per-row budgets and a gate forced to always think, each
    rollout's total think count must equal exactly its assigned budget (it
    thinks until the per-row budget forces emit)."""
    from experiments.train_rl_grader import rollout_group_batched
    m = _StubLM(64, gate_value=0.01)  # σ=0.01 < emit_threshold -> always think
    tok = _StubTokenizer()
    prompt = torch.randint(7, 50, (1, 6))

    budgets = [1, 3, 7]
    rollouts = rollout_group_batched(
        m, tok, prompt, n_rollouts=3,
        thinking_token_id=THINK_ID, eos_token_id=1, max_gen=16,
        max_think_per_step=100,  # don't let per-step cap interfere
        total_think_budget=budgets,
        emit_threshold=0.5,      # σ=0.01 never reaches -> always think...
        gate_floor=0.0,
        temperature=0.0, min_emit_before_eos=4)
    # ...until the per-row total budget forces emit.
    for r, b in zip(rollouts, budgets):
        assert r.depth == b, (r.depth, b)
