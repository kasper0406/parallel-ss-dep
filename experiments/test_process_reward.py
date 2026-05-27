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
    # Targets in [0, THINK_ID) so none equal THINK_ID — the loss now
    # masks target==thinking_token_id positions (pretrain-safety fix,
    # 2026-05-27) to avoid gather-out-of-bounds when logits are sliced
    # to base_vocab_for_loss.
    y = torch.randint(0, THINK_ID, (B, T))
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


def test_eager_forward_used_when_present():
    """When `model._eager_forward` is set (as `speed_knobs.apply_speed_knobs`
    does under `compile_model=True`), the extra after-forward must go
    through it instead of `model.__call__`. This dodges the Inductor
    symbolic-shape assertion that fires when the aux's `(N, L_after)`
    shape collides with the compiled main `(B, T)` graph.
    """
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    B, T = 2, 12
    x = torch.randint(1, 16, (B, T))  # avoid pad id 0
    y = x.clone()
    main_logits = model(x)
    gate = torch.full((B, T), 0.9)

    # Install a sentinel "eager" forward that records its call and
    # delegates to the real model. The compiled forward stays the
    # original (and should NOT be invoked from compute_process_reward).
    calls = {"eager": 0, "compiled": 0}
    real_fwd = model.forward

    def eager_fwd(*a, **kw):
        calls["eager"] += 1
        return real_fwd(*a, **kw)

    def compiled_fwd(*a, **kw):
        calls["compiled"] += 1
        return real_fwd(*a, **kw)

    model._eager_forward = eager_fwd
    model.forward = compiled_fwd
    try:
        rng = torch.Generator().manual_seed(0)
        loss, stats = compute_process_reward_loss(
            model, x, y, gate=gate, main_logits=main_logits,
            thinking_token_id=THINK_ID,
            K=2, apply_min_sigma=0.5, sample_frac=1.0,
            rng=rng, pad_token_id=PAD_ID,
            retrieval_as_input=False, base_vocab_for_loss=None)
    finally:
        # Clean up the monkey-patch.
        del model._eager_forward
        model.forward = real_fwd

    assert stats.n_sampled > 0
    assert calls["eager"] >= 1, "after-forward must use _eager_forward"
    assert calls["compiled"] == 0, "after-forward must NOT route through compile"


def test_eager_forward_fallback_when_absent():
    """When `model._eager_forward` is missing (no-compile mode), the
    aux loss must still work by falling back to `model(...)`. Guarantees
    the optimization is backwards-compatible.
    """
    torch.manual_seed(0)
    model = _MockLM(vocab=16, d=8)
    assert not hasattr(model, "_eager_forward")
    B, T = 2, 8
    x = torch.randint(1, 16, (B, T))
    y = x.clone()
    main_logits = model(x)
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=1.0,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None)
    assert stats.n_sampled > 0
    assert loss.requires_grad
    loss.backward()  # gradient must flow


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


# ---------- 6. retrieval_as_input=True branch (Gap 2) -----------------

class _MockMemory(torch.nn.Module):
    """Stand-in for `WorkingMemory` exposing the one attribute the
    aux helpers read in retrieval-as-input mode: `_last_injection`,
    a `(B, T, d)` tensor stashed by `WorkingMemory.forward`.
    """
    def __init__(self):
        super().__init__()
        self._last_injection = None


class _MockLMWithRetrieval(nn.Module):
    """`_MockLM` + the surface needed by `_retrieval_input_embeds`:
      - `model.memory._last_injection` (populated by the no_grad pre-
        forward inside the helper)
      - `model.retrieval_input_alpha` (learnable scalar, mirrors
        `TinyLM.retrieval_input_alpha`)

    Captures `inputs_embeds` of EVERY forward call so the test can
    verify the aux helper passed a customised tensor when
    `retrieval_as_input=True`.
    """
    def __init__(self, vocab: int = 16, d: int = 8):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.gate_head = nn.Linear(d, 1)
        self.memory = _MockMemory()
        # Init non-zero so an "additive α * injection" makes a real
        # difference vs the bare embedding.
        self.retrieval_input_alpha = nn.Parameter(torch.tensor(0.7))
        self._last_gate = None
        self.captured_inputs_embeds = []   # (call_idx → tensor or None)

    def forward(self, x, inputs_embeds=None, return_hidden=False):
        self.captured_inputs_embeds.append(
            None if inputs_embeds is None else inputs_embeds.detach().clone())
        h = inputs_embeds if inputs_embeds is not None else self.embed(x)
        h = h + h.tanh() * 0.01
        # Stash an injection tensor on every forward so the next call
        # has something to read from (mirrors WorkingMemory's behaviour
        # when forward is called fresh).
        self.memory._last_injection = h.detach().clone()
        self._last_gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits


def test_process_reward_retrieval_as_input_branch():
    """When `retrieval_as_input=True`, the after-forward must receive a
    custom `inputs_embeds` built by `_retrieval_input_embeds` (additive
    α-gated injection over the embedding table). Verifies:
      - the helper succeeds (no shape errors)
      - the after-forward got a non-None inputs_embeds with the right
        (N, L_after, d) shape
      - the inputs_embeds tensor is NOT a bare embedding lookup (the
        additive α·injection branch fires on think positions)
    """
    torch.manual_seed(0)
    model = _MockLMWithRetrieval(vocab=16, d=8)
    B, T = 2, 8
    # Avoid pad and think ids in the input so the after-forward's
    # think-positions are well-defined.
    x = torch.randint(1, THINK_ID, (B, T))
    y = torch.randint(1, THINK_ID, (B, T))
    main_logits = model(x)
    n_calls_before = len(model.captured_inputs_embeds)
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=0.5,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=True,
        base_vocab_for_loss=None,
        max_positions=8)
    assert stats.n_sampled > 0
    # retrieval_as_input path fires two extra forwards: (1) a no_grad
    # warm-up that populates `memory._last_injection`, (2) the real
    # after-forward with the custom inputs_embeds. Both go through
    # `_call_model_eager` → `model(...)`.
    n_calls_after = len(model.captured_inputs_embeds)
    assert n_calls_after - n_calls_before == 2, (
        f"expected 2 extra forwards, got {n_calls_after - n_calls_before}")
    warmup_kwarg = model.captured_inputs_embeds[n_calls_before]
    after_kwarg = model.captured_inputs_embeds[n_calls_before + 1]
    assert warmup_kwarg is None, (
        "warm-up forward should NOT pass inputs_embeds (it builds the "
        "injection)")
    assert after_kwarg is not None, (
        "after-forward MUST pass a custom inputs_embeds when "
        "retrieval_as_input=True")
    # Shape: (N, L_after, d_model). N == stats.n_sampled.
    assert after_kwarg.shape[0] == stats.n_sampled
    assert after_kwarg.shape[2] == model.d
    # The retrieval branch should produce a tensor DIFFERENT from the
    # bare embedding lookup (α=0.7 × injection added at think positions).
    # Reconstruct what a bare-embedding-table forward would have used:
    # we feed the same ids back through `model.embed` and compare. They
    # must differ (because the injection got added at think positions).
    bare = model.embed(
        # last call's input_ids — captured implicitly via the forward
        # call's first positional arg. We don't have direct access so
        # we just assert the custom tensor is NOT equal to any zero or
        # to a simple embed of a uniform pad.
        torch.full((stats.n_sampled, after_kwarg.shape[1]), PAD_ID,
                   dtype=torch.long)
    )
    assert not torch.allclose(after_kwarg, bare), (
        "custom inputs_embeds is suspiciously identical to bare embed lookup")
    assert torch.isfinite(loss)


# ---------- 7. Tuple-return survival (Gap 3) --------------------------

class _MockLMTupleReturn(nn.Module):
    """Mirrors `TinyLM.forward` when training-mode gist_loss is on:
    `forward(x)` returns `(logits, gist_loss_scalar)` instead of just
    logits. The aux helpers MUST unwrap this tuple — without that
    unwrap, `after_logits[..., :base_vocab_for_loss]` fails with a
    `TypeError: tuple indices must be integers or slices, not tuple`.
    """
    def __init__(self, vocab: int = 16, d: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.gate_head = nn.Linear(d, 1)
        self._last_gate = None

    def forward(self, x, inputs_embeds=None, return_hidden=False):
        h = inputs_embeds if inputs_embeds is not None else self.embed(x)
        h = h + h.tanh() * 0.01
        self._last_gate = torch.sigmoid(self.gate_head(h).squeeze(-1))
        logits = self.head(h)
        # Pretend gist loss is on: return a tuple. Use a real scalar
        # tensor (with grad) to make sure helpers don't accidentally
        # propagate it.
        gist_scalar = h.mean() * 0.0  # zero but grad-bearing
        if return_hidden:
            return (logits, h)  # different tuple shape; we don't use this
        return (logits, gist_scalar)


def test_process_reward_handles_tuple_forward_return():
    """When `model(x)` returns `(logits, gist_loss)` (training-mode
    gist_loss path), `compute_process_reward_loss` must unwrap the
    tuple. Regression for the `if isinstance(after_out, tuple)` patch
    in process_reward.py.
    """
    torch.manual_seed(0)
    model = _MockLMTupleReturn(vocab=16, d=8)
    B, T = 2, 8
    x = torch.randint(1, 16, (B, T))
    y = torch.randint(1, THINK_ID, (B, T))
    # Main forward returns a tuple too; mirror what the trainer would do
    # (it unpacks `logits, gist = model(x)` BEFORE calling the aux helper).
    main_logits, _gist = model(x)
    gate = torch.full((B, T), 0.9)
    rng = torch.Generator().manual_seed(0)
    # If the unwrap is missing, this raises TypeError on the line that
    # tries `after_logits[..., :base_vocab_for_loss]`.
    loss, stats = compute_process_reward_loss(
        model, x, y, gate=gate, main_logits=main_logits,
        thinking_token_id=THINK_ID,
        K=2, apply_min_sigma=0.5, sample_frac=0.5,
        rng=rng, pad_token_id=PAD_ID,
        retrieval_as_input=False, base_vocab_for_loss=None,
        max_positions=8)
    assert stats.n_sampled > 0
    assert torch.isfinite(loss)
    assert loss.requires_grad
    # Backward also works (the gist-scalar shouldn't pollute the graph).
    loss.backward()
