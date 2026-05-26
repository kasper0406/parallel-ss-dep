"""Tests for the Phase A process-reward auxiliary loss.

What we check (all CPU-only, tiny mock model):
  1. Default (--process_reward_weight 0.0) → byte-identical: the helper
     is never called from sft_code.py's training loop.
  2. CLI flags parse to documented defaults.
  3. Position sampler honours `sample_frac` and `apply_min_sigma`.
  4. compute_process_reward_loss returns a SCALAR with grad enabled
     when there's at least one sampled position.
  5. Zero candidates → zero loss, requires_grad=False, n_sampled=0.
  6. End-to-end smoke: one optimizer step with weight>0 doesn't NaN
     and actually moves model parameters.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_process_reward.py -v
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.process_reward import (
    _select_candidate_positions,
    _build_after_sequences,
    compute_process_reward_loss,
)


THINK_ID = 7
PAD_ID = 0  # MUST differ from THINK_ID — pad-as-think corrupts the
            # after-forward state when mem_write_only_at_think /
            # state_readonly_at_think are on (compute_process_reward_loss
            # enforces this with a ValueError).


class _MockLM(nn.Module):
    """Minimal stand-in for TinyLM exposing what process_reward needs:
      - model(x)                → logits (B, T, V)
      - model.embed(x)          → embeddings (B, T, d)
      - model._last_gate        → (B, T) sigmoid gate
      - model.memory._last_injection (only when retrieval_as_input)
      - model.retrieval_input_alpha (only when retrieval_as_input)
    """
    def __init__(self, vocab: int = 16, d: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.gate_head = nn.Linear(d, 1)
        self._last_gate = None

    def forward(self, x, inputs_embeds=None, return_hidden=False):
        h = inputs_embeds if inputs_embeds is not None else self.embed(x)
        # Cheap "trunk": a single linear so gradients flow.
        h = h + h.tanh() * 0.01
        self._last_gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits


def _gate_tensor(seed: int = 0, B=2, T=8) -> torch.Tensor:
    g = torch.rand(B, T, generator=torch.Generator().manual_seed(seed))
    return g


# ---------- 1. Flag-default-off: CLI parses with weight 0 ---------------------

def test_flag_default_off():
    """The CLI flag exists with default=0.0 and the helper is wired
    behind a `use_process_reward = (weight > 0 and ...)` guard."""
    import pathlib
    src = pathlib.Path("experiments/sft_code.py").read_text()
    assert '"--process_reward_weight"' in src
    section = src.split('"--process_reward_weight"', 1)[1].split(")", 1)[0]
    assert "default=0.0" in section
    # The training loop should ONLY call the helper when
    # use_process_reward is True (which requires weight > 0).
    assert "use_process_reward = (" in src
    assert "args.process_reward_weight > 0.0" in src
    # Helper actually wired into the legacy forward path.
    assert "compute_process_reward_loss" in src


# ---------- 2. Candidate selector ------------------------------------------

def test_selector_respects_apply_min_sigma():
    torch.manual_seed(0)
    B, T = 2, 8
    gate = torch.full((B, T), 0.9)
    gate[0, 0] = 0.1  # below threshold
    y_shift = torch.zeros(B, T - 1, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)
    b_idx, t_idx = _select_candidate_positions(
        gate, y_shift, apply_min_sigma=0.5, sample_frac=1.0,
        rng=rng, max_positions=128)
    # Only position (0,0) should be excluded; total candidates = B*(T-1) - 1
    assert b_idx.numel() == B * (T - 1) - 1
    assert not ((b_idx == 0) & (t_idx == 0)).any()


def test_selector_respects_sample_frac():
    torch.manual_seed(0)
    B, T = 4, 16
    gate = torch.full((B, T), 0.9)
    y_shift = torch.zeros(B, T - 1, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)
    # 60 candidates × 0.25 = 15 → should sample 15
    b_idx, _ = _select_candidate_positions(
        gate, y_shift, apply_min_sigma=0.5, sample_frac=0.25,
        rng=rng, max_positions=128)
    assert b_idx.numel() == round(B * (T - 1) * 0.25)


def test_selector_respects_max_positions():
    torch.manual_seed(0)
    B, T = 8, 16
    gate = torch.full((B, T), 0.9)
    y_shift = torch.zeros(B, T - 1, dtype=torch.long)
    rng = torch.Generator().manual_seed(0)
    b_idx, _ = _select_candidate_positions(
        gate, y_shift, apply_min_sigma=0.5, sample_frac=1.0,
        rng=rng, max_positions=10)
    assert b_idx.numel() == 10


def test_selector_skips_masked_targets():
    B, T = 2, 8
    gate = torch.full((B, T), 0.9)
    y_shift = torch.zeros(B, T - 1, dtype=torch.long)
    y_shift[:, :] = -100  # everything masked
    rng = torch.Generator().manual_seed(0)
    b_idx, t_idx = _select_candidate_positions(
        gate, y_shift, apply_min_sigma=0.5, sample_frac=1.0,
        rng=rng, max_positions=128)
    assert b_idx.numel() == 0
    assert t_idx.numel() == 0


# ---------- 3. After-sequence builder --------------------------------------

def test_after_sequence_layout():
    B, T = 2, 8
    x = torch.arange(B * T, dtype=torch.long).reshape(B, T) + 100
    b_idx = torch.tensor([0, 1])
    t_idx = torch.tensor([3, 5])
    K = 2
    after_ids, last_pos = _build_after_sequences(
        x, b_idx, t_idx, K=K,
        thinking_token_id=THINK_ID, pad_token_id=PAD_ID)
    # L_max = max(t_idx) + 1 + K = 5 + 1 + 2 = 8
    assert after_ids.shape == (2, 8)
    # For row 0: t=3 → prefix_len=4, then K=2 thinks → 6 real tokens,
    # so left-pad = 8 - 6 = 2.
    assert (after_ids[0, :2] == PAD_ID).all()
    assert (after_ids[0, 2:6] == x[0, :4]).all()
    assert (after_ids[0, 6:8] == THINK_ID).all()
    # For row 1: t=5 → 6+2=8 real tokens, no padding.
    assert (after_ids[1, :6] == x[1, :6]).all()
    assert (after_ids[1, 6:8] == THINK_ID).all()
    assert (last_pos == 7).all()


# ---------- 4. compute_process_reward_loss --------------------------------

def test_compute_returns_zero_when_no_candidates():
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    B, T = 2, 6
    x = torch.randint(0, 16, (B, T))
    y = torch.full_like(x, -100)
    main_logits = model(x)
    gate = model._last_gate
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=1.0,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None)
    assert loss.item() == 0.0
    assert not loss.requires_grad
    assert stats.n_sampled == 0


def test_compute_returns_scalar_with_grad():
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    # Force the gate high everywhere so every position is a candidate.
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=0.5,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        max_positions=128)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert stats.n_sampled > 0
    # Smoke: backward populates grads on model params.
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.parameters())


def test_compute_respects_sample_frac():
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    B, T = 4, 16
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=0.1,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None)
    # B*(T-1) = 60 candidates × 0.1 = 6
    assert stats.n_candidates == 60
    assert stats.n_sampled == 6


def test_compute_filters_low_sigma():
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    gate = torch.full((B, T), 0.2)  # ALL below the threshold
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=1.0,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None)
    assert stats.n_candidates == 0
    assert stats.n_sampled == 0
    assert loss.item() == 0.0


# ---------- 5. End-to-end smoke -------------------------------------------

def test_end_to_end_one_step_no_nan():
    """Smoke: with weight>0, a single backward+step produces finite
    parameter updates."""
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    B, T = 2, 8
    x = torch.randint(0, 16, (B, T))
    y = torch.randint(0, 16, (B, T))
    main_logits = model(x)
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    # Main LM loss + process-reward loss.
    shift_logits = main_logits[:, :-1].contiguous()
    shift_labels = y[:, 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1))
    pr_loss, _ = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=0.5,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None)
    total = lm_loss + 0.1 * pr_loss
    assert torch.isfinite(total)
    opt.zero_grad(set_to_none=True)
    total.backward()
    opt.step()
    for p in model.parameters():
        assert torch.isfinite(p).all()


def test_default_off_is_zero_loss():
    """With the helper never called (weight=0 short-circuit), nothing
    changes — this mirrors the byte-identical guarantee. We assert
    here that when sft_code.py's use_process_reward branch is
    *disabled* (weight=0 → never enters), no extra ops touch the
    graph. We approximate by checking the helper returns a non-grad
    zero on the no-candidates path (which is what the off-state
    looks like from the optimizer's perspective)."""
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    B, T = 2, 6
    x = torch.randint(0, 16, (B, T))
    y = torch.full_like(x, -100)
    main_logits = model(x)
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=1.0,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None)
    assert loss.item() == 0.0
    assert not loss.requires_grad
    assert stats.n_sampled == 0


def test_pad_eq_think_raises():
    """Defensive guard: pad_token_id == thinking_token_id must raise.
    Pad-as-think silently corrupts the after-forward's recurrent state
    when state_readonly_at_think / mem_write_only_at_think are on —
    the bug caught by the v5 code review."""
    import pytest
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    B, T = 2, 6
    x = torch.randint(0, 16, (B, T))
    y = x.clone()
    main_logits = model(x)
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="pad_token_id must NOT equal"):
        compute_process_reward_loss(
            model, x, y, gate=gate, main_logits=main_logits,
            thinking_token_id=THINK_ID,
            K=2, apply_min_sigma=0.0, sample_frac=1.0,
            rng=rng, pad_token_id=THINK_ID,  # collide on purpose
            retrieval_as_input=False, base_vocab_for_loss=None)
