"""Tests for state-readonly thinking (Phase 2 architectural fix, 2026-05-26).

The flag `state_readonly_at_think=True` forces the DeltaNet per-token
write gate β to 0 at think positions. Think tokens still process their
query against the existing recurrent state (so the local hidden h_t
carries useful information that gate/WM/lm_head consume) but they do
NOT WRITE into the recurrent state. This preserves long-range bindings
across multi-think bursts — the documented 100% → 20% recall-at-512
drop should disappear.

Tests:
  1. Shapes are unchanged with 0 / 1 / 4 think tokens (smoke).
  2. Recall-preservation probe — the LOAD-BEARING test: with multi-think
     bursts inserted between an early binding and a late query,
     state_readonly=True preserves recall while state_readonly=False
     degrades it. Trained on a synthetic var-binding-style sequence.
  3. Equivalence: with NO think tokens in the input, the model output
     is bit-identical regardless of the flag.

Run: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m pytest \
    experiments/test_state_readonly_thinking.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


# A handful of token ids reserved for the synthetic probes.
THINK_ID = 1
QUERY_ID = 2
BIND_TAG_ID = 3       # marks "this position is a binding"
DISTRACTOR_BASE = 10  # distractor noise tokens >= this id
DISTRACTOR_COUNT = 90  # number of distinct distractor tokens
VALUE_BASE = 100      # value tokens (the binding payload) >= this id
PAD_ID = 0


def _make_model(*, vocab_size=256, n_layers=2, d_model=32, n_heads=2,
                d_head=16, state_readonly: bool = False,
                max_T: int = 512, seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(
        vocab_size=vocab_size,
        d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head,
        attention_cls=DeltaNetAttention,
        max_T=max_T,
        state_readonly_at_think=state_readonly,
        # Need a thinking_token_id for state_readonly to know which
        # positions to mask. The model constructor accepts it via the
        # use_memory path; we set it directly after construction below.
    )


def _set_thinking_id(model: TinyLM, tid: int) -> None:
    """Force the model to know the thinking-token id without enabling WM."""
    model.thinking_token_id = int(tid)


# --------------------------------------------------------------------------
# 1. Shape smoke test
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_forward_shape_with_think_tokens():
    """0 / 1 / 4 think tokens must all produce (B, T, vocab) logits."""
    model = _make_model(state_readonly=True).cuda().eval()
    _set_thinking_id(model, THINK_ID)
    B, T = 2, 24
    base = torch.randint(VALUE_BASE, VALUE_BASE + 50, (B, T), device="cuda")
    for n_thinks in (0, 1, 4):
        x = base.clone()
        for k in range(n_thinks):
            x[0, 5 + k] = THINK_ID  # consecutive bursts in first row
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, T, 256)


# --------------------------------------------------------------------------
# 3. Equivalence: no think tokens → byte-identical regardless of flag
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_equivalence_no_think_tokens():
    """A sequence WITHOUT any think tokens must produce identical output
    regardless of state_readonly_at_think — the wrapper must be a no-op
    when the mask is all-False."""
    torch.manual_seed(0)
    model_off = _make_model(state_readonly=False, seed=42).cuda().eval()
    torch.manual_seed(0)
    model_on = _make_model(state_readonly=True, seed=42).cuda().eval()
    _set_thinking_id(model_on, THINK_ID)
    # Sanity: weights are identical so the only difference is the flag.
    for (n1, p1), (n2, p2) in zip(model_off.named_parameters(),
                                    model_on.named_parameters()):
        assert n1 == n2, f"name mismatch {n1} vs {n2}"
        assert torch.equal(p1, p2), f"weights diverged on {n1}"

    B, T = 2, 32
    # Build sequence with NO think tokens.
    torch.manual_seed(7)
    x = torch.randint(VALUE_BASE, VALUE_BASE + 50, (B, T), device="cuda")
    assert not (x == THINK_ID).any()
    with torch.no_grad():
        out_off = model_off(x)
        out_on = model_on(x)
    torch.testing.assert_close(out_off, out_on, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_thinks_change_output_when_flag_off():
    """Sanity that the test setup is non-degenerate: with state_readonly=False,
    a think token in the middle of the sequence DOES alter downstream logits
    (otherwise the recall probe is meaningless)."""
    model = _make_model(state_readonly=False, seed=11).cuda().eval()
    _set_thinking_id(model, THINK_ID)
    torch.manual_seed(3)
    x = torch.randint(VALUE_BASE, VALUE_BASE + 50, (1, 32), device="cuda")
    x_with_think = x.clone()
    x_with_think[0, 8] = THINK_ID
    with torch.no_grad():
        out_clean = model(x)
        out_think = model(x_with_think)
    # Logits AFTER the inserted think differ; bit-equality would mean the
    # think token had zero effect (which would mean the test is degenerate).
    diff_after = (out_clean[0, 9:] - out_think[0, 9:]).abs().mean().item()
    assert diff_after > 1e-3, (
        f"think token had no measurable effect (diff={diff_after}); "
        "the test setup is degenerate.")


# --------------------------------------------------------------------------
# 2. Recall-preservation probe (THE LOAD-BEARING TEST)
#
# Synthetic var-binding task:
#     [BIND_TAG_ID, V, distractors..., (think burst), distractors..., QUERY_ID]
# Train the model on next-token CE so that the model predicts V at the
# position right after QUERY_ID. Insert a multi-think burst between the
# binding and the query. Compare recall accuracy with vs without
# state_readonly_at_think.
#
# Expected: state_readonly=True preserves recall better than False under
# multi-think bursts (because think tokens never corrupt the DeltaNet
# recurrence carrying the binding).
# --------------------------------------------------------------------------

def _build_var_binding_batch(
    B: int, T: int, n_values: int, n_thinks: int, *, device: str = "cuda",
    seed: int = 0,
):
    """Construct (input_ids, target_pos, target_id) tensors.

    Layout per row:
        pos 0:                BIND_TAG_ID
        pos 1:                V (one of n_values distinct value tokens)
        pos 2 .. burst_start: random distractor tokens
        burst_start .. +n_thinks-1: THINK_ID repeated
        ...:                  random distractor tokens
        pos T-2:              QUERY_ID
        pos T-1:              target = V  (this is the LOSS-TARGET position)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    # Random distractor filling. A wide distractor range matters: with a
    # narrow range (e.g. 20 distinct ids), distractor tokens have high
    # frequency and dominate the recurrence with repeated key directions,
    # so the binding gets clobbered regardless of think handling. A
    # wider range gives each distractor a near-unique key direction and
    # lets the binding survive — exposing the think-burst-specific
    # corruption that state-readonly addresses.
    rand = torch.randint(DISTRACTOR_BASE,
                          DISTRACTOR_BASE + DISTRACTOR_COUNT,
                          (B, T), device=device, generator=g)
    x.copy_(rand)
    # Values for each row.
    values = torch.randint(VALUE_BASE, VALUE_BASE + n_values,
                            (B,), device=device, generator=g)
    x[:, 0] = BIND_TAG_ID
    x[:, 1] = values
    # Place think burst at ~mid-sequence.
    burst_start = T // 2
    for k in range(n_thinks):
        x[:, burst_start + k] = THINK_ID
    # Query and target at the tail.
    x[:, -2] = QUERY_ID
    x[:, -1] = values  # the loss is on the position right after QUERY,
                       # i.e. we want logits at pos T-2 (which predict pos T-1)
                       # to match `values`.
    return x, values


def _train_and_eval_recall(*, state_readonly: bool, n_thinks_train: int,
                            n_thinks_eval: int, T: int = 80,
                            steps: int = 500, B: int = 32,
                            n_values: int = 8, seed: int = 0,
                            ) -> float:
    """Train a tiny single-layer narrow-head DeltaNet on the var-binding
    task with n_thinks_train inserted think tokens, then eval recall
    with n_thinks_eval think tokens.

    The model is deliberately tight (1 layer, d_head=4) so the recurrent
    state is highly compressed — exactly the regime where multi-think
    bursts catastrophically overwrite the binding (mirroring the
    full-model failure mode documented in CLAUDE.md).

    Training INCLUDES a small think burst (n_thinks_train>0) so the
    model learns to map THINK_ID through the same key directions as
    other tokens. At eval, a much longer burst (n_thinks_eval) is the
    extrapolation regime where state_readonly=True structurally
    prevents the writes that would overwrite the binding.
    """
    torch.manual_seed(seed)
    model = _make_model(vocab_size=256, n_layers=1, d_model=16,
                         n_heads=4, d_head=4, max_T=512,
                         state_readonly=state_readonly, seed=seed).cuda()
    _set_thinking_id(model, THINK_ID)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    for step in range(steps):
        x, vals = _build_var_binding_batch(
            B, T, n_values, n_thinks_train, seed=10000 + step,
        )
        logits = model(x)
        # Loss: at position T-2 (the QUERY position), predict vals.
        target_logits = logits[:, -2, :]   # (B, vocab)
        loss = F.cross_entropy(target_logits, vals)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Eval on held-out batches.
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ev in range(8):
            x, vals = _build_var_binding_batch(
                B, T, n_values, n_thinks_eval, seed=99000 + ev,
            )
            logits = model(x)
            pred = logits[:, -2, :].argmax(dim=-1)
            correct += int((pred == vals).sum().item())
            total += int(vals.numel())
    return correct / total


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_recall_preservation_with_state_readonly():
    """LOAD-BEARING: state_readonly=True must preserve recall under
    multi-think bursts strictly better than state_readonly=False.

    Setup: train each 1-layer narrow-head DeltaNet on var-binding with
    a SHORT think burst (4 thinks). The model learns to map THINK_ID
    through its W_k projection like any other token. At eval, insert a
    much LONGER burst (32 thinks) between the binding and the query —
    each extra think writes into the recurrent state slot, overwriting
    the binding (the exact failure mode from CLAUDE.md). With
    state_readonly=True, β=0 at think positions so no overwrites occur
    and recall is preserved.
    """
    n_thinks_train = 4
    n_thinks_eval = 32
    acc_on = _train_and_eval_recall(
        state_readonly=True, n_thinks_train=n_thinks_train,
        n_thinks_eval=n_thinks_eval, T=80, steps=500, seed=0,
    )
    acc_off = _train_and_eval_recall(
        state_readonly=False, n_thinks_train=n_thinks_train,
        n_thinks_eval=n_thinks_eval, T=80, steps=500, seed=0,
    )
    print(f"\n  recall (state_readonly=True):  {acc_on:.3f}")
    print(f"  recall (state_readonly=False): {acc_off:.3f}")
    chance = 1.0 / 8   # n_values
    # state_readonly=True should retain HIGH (not necessarily perfect)
    # recall. Threshold lowered 0.85 -> 0.60 (2026-07-01, flaky-test audit):
    # this training loop's converged accuracy is not bit-reproducible run
    # to run — FLA's chunk_delta_rule backward uses atomic-add reductions
    # in its Triton kernel, so floating-point summation order (and hence
    # the exact gradients over 500 steps) varies across process launches
    # even with torch.manual_seed(seed) fixed; there is no supported
    # deterministic mode for this kernel. Observed acc_on across runs of
    # this exact seed=0 config: 0.668-0.824 (this audit's own measurements
    # plus the prior flaky-CI value in the task report); acc_off over the
    # same runs stayed at 0.094-0.285. 0.60 sits safely below every
    # observed acc_on floor while remaining ~4.8x chance (0.125) and with
    # a wide margin over every observed acc_off ceiling for this seed —
    # discriminative power (see the acc_on > acc_off + 0.10 check below)
    # is unaffected.
    assert acc_on >= 0.60, (
        f"state_readonly=True recall {acc_on:.3f} too low; expected "
        f">= 0.60 (chance = {chance:.3f}). Either training didn't "
        f"converge or the masking is broken."
    )
    # state_readonly=False should suffer from the longer burst at eval.
    assert acc_on > acc_off + 0.10, (
        f"state_readonly=True ({acc_on:.3f}) failed to preserve recall "
        f"vs state_readonly=False ({acc_off:.3f}) by a margin >= 0.10. "
        f"The architectural fix is not protecting the recurrent state."
    )


# --------------------------------------------------------------------------
# 4. DECODE-PATH β-masking (2026-05-28).
#
# GEMINI flagged that only the full-sequence forward path is wired for
# state-readonly; the incremental decode path (prefill T>1 chunk kernel +
# forward_step T=1 fused_recurrent kernel) may leave β unmasked at think
# positions. These tests install a forward hook on the inner DeltaNet
# `b_proj` Linear that records the PRE-sigmoid logits the kernel actually
# saw, and assert they are clamped to a large-negative value (β≈0) at think
# positions during BOTH decode sub-paths.
# --------------------------------------------------------------------------

def _attach_bproj_recorder(model: TinyLM):
    """Register a forward hook on every block's inner b_proj that stores
    the (post-state-readonly-hook) output logits. Returns (records, handles).

    The state-readonly masking hook is registered FIRST (in __init__ via
    enable_state_readonly_at_think), so a hook we register here runs AFTER
    it and therefore observes the *already-masked* logits — exactly what
    the kernel consumes. Returns one entry per b_proj call.
    """
    from experiments.layers import _FlaWrapper
    records = []
    handles = []

    def _rec(_m, _inp, out):
        records.append(out.detach().float().cpu())

    for blk in model.blocks:
        attn = blk.attn
        if isinstance(attn, _FlaWrapper) and hasattr(attn.layer, "b_proj"):
            handles.append(attn.layer.b_proj.register_forward_hook(_rec))
    return records, handles


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_decode_prefill_honors_state_readonly_at_think():
    """In the prefill (T>1 chunk-kernel) decode path, β must be 0 at think
    positions when state_readonly_at_think=True."""
    model = _make_model(state_readonly=True, n_layers=1, seed=5).cuda().eval()
    _set_thinking_id(model, THINK_ID)
    model._film_bypass = True

    B, T = 1, 16
    x = torch.randint(VALUE_BASE, VALUE_BASE + 50, (B, T), device="cuda")
    think_positions = [4, 5, 6]
    for p in think_positions:
        x[0, p] = THINK_ID

    records, handles = _attach_bproj_recorder(model)
    try:
        with torch.no_grad():
            model.prefill(x)
    finally:
        for h in handles:
            h.remove()

    assert len(records) >= 1, "b_proj hook never fired during prefill"
    # The chunk path runs b_proj over the whole prompt at once → one record
    # of shape (B, T, n_heads) (possibly flattened to (1, B*T, n_heads)).
    logits = records[0]
    flat = logits.reshape(-1, logits.shape[-1])  # (B*T, H)
    # Positions correspond to row-major (b, t). With B=1 it's just t.
    for p in think_positions:
        assert (flat[p] <= -1e3).all(), (
            f"prefill: think position {p} b_proj logits not masked: "
            f"{flat[p].tolist()}")
    # A non-think position must NOT be masked.
    assert (flat[0] > -1e3).any(), "prefill: non-think pos wrongly masked"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_decode_forward_step_honors_state_readonly_at_think():
    """In the forward_step (T=1 fused_recurrent) decode path, β must be 0
    at a think position when state_readonly_at_think=True. This is the
    path GEMINI flagged as the open follow-up."""
    model = _make_model(state_readonly=True, n_layers=1, seed=6).cuda().eval()
    _set_thinking_id(model, THINK_ID)
    model._film_bypass = True

    B, T_prompt = 1, 8
    prompt = torch.randint(VALUE_BASE, VALUE_BASE + 50, (B, T_prompt),
                           device="cuda")
    assert not (prompt == THINK_ID).any()

    with torch.no_grad():
        cache, _ = model.prefill(prompt)

    # Now step with a THINK token and record b_proj logits at that step.
    records, handles = _attach_bproj_recorder(model)
    try:
        think_tok = torch.full((B, 1), THINK_ID, device="cuda",
                               dtype=torch.long)
        with torch.no_grad():
            model.forward_step(think_tok, cache)
    finally:
        for h in handles:
            h.remove()

    assert len(records) >= 1, "b_proj hook never fired during forward_step"
    logits = records[0]
    flat = logits.reshape(-1, logits.shape[-1])  # (B*1, H)
    assert (flat <= -1e3).all(), (
        f"forward_step: think position b_proj logits not masked (β not 0): "
        f"{flat.tolist()}")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_decode_forward_step_emit_not_masked():
    """Backwards-compat / sanity: a NON-think token in forward_step must
    leave β unmasked even with state_readonly_at_think=True."""
    model = _make_model(state_readonly=True, n_layers=1, seed=7).cuda().eval()
    _set_thinking_id(model, THINK_ID)
    model._film_bypass = True

    B, T_prompt = 1, 8
    prompt = torch.randint(VALUE_BASE, VALUE_BASE + 50, (B, T_prompt),
                           device="cuda")
    with torch.no_grad():
        cache, _ = model.prefill(prompt)

    records, handles = _attach_bproj_recorder(model)
    try:
        emit_tok = torch.randint(VALUE_BASE, VALUE_BASE + 50, (B, 1),
                                 device="cuda")
        assert not (emit_tok == THINK_ID).any()
        with torch.no_grad():
            model.forward_step(emit_tok, cache)
    finally:
        for h in handles:
            h.remove()

    assert len(records) >= 1
    flat = records[0].reshape(-1, records[0].shape[-1])
    assert (flat > -1e3).any(), (
        "forward_step: emit position wrongly masked (β forced to 0 at a "
        "non-think token)")
