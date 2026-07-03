"""Tests for the two generation-path wiring gaps fixed 2026-07-03
(pilot post-mortem: the deployed incremental stack never applied FiLM,
and the WM copy head could never fire at inference).

GAP 1 — FiLM at decode. `TinyLM.prefill` / `forward_step` had an
`use_film_at_decode` branch that was never exercised in practice (every
existing generator sets `model._film_bypass = True`, and
`test_incremental_decode.py` follows that convention too, so the branch
had zero test coverage). It also recomputed the FiLM lag-seed via a
wasteful *second* full forward that silently dropped `think_mask`
(divergent from the real cache-populating pass whenever
`state_readonly_at_think` / `use_think_adapter` are active). Fixed by
capturing the lag seed inline from the real pass, and generalized the
single-slot cache to a length-`feedback_lag` ring buffer (the old code
only worked by accident for the default `feedback_lag=1`).

GAP 2 — WM copy head at decode. `_apply_copy_head` was a hard no-op
whenever `mem_read_mask is None`, and no generator threads a mask at
inference (there's no oracle "recall answer span" at generation time) —
so the trained copy/pointer readout, despite having its own
input-driven trigger (`me`, the match-existence gate behind
`mem_copy_require_match`), silently never fired outside of training.
Fixed: when the memory is `always_read` (already injects the WM read at
every position when no mask is supplied), the copy head now mixes at
every position too, gated by `g_eff = g * me` — `me == 0` (no
causally-valid match) is a strict no-op, so this can only ever recover
lost behaviour, never introduce a new failure mode not already reached
via `always_read`'s own read.

Run:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -m pytest \
        experiments/test_film_decode_wiring.py -v -s
"""
from __future__ import annotations

import pytest
import torch

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeltaNet Triton kernels require CUDA")


# ===========================================================================
# Helpers
# ===========================================================================

def _lean_model(seed: int = 0) -> TinyLM:
    """No feedback at all -- the config the `else:` (plain block loop /
    bypass) branch has always served, untouched by this session's edit."""
    torch.manual_seed(seed)
    return TinyLM(
        vocab_size=64, d_model=32, n_layers=4, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=64,
    )


def _film_model(seed: int = 0, feedback_self_k: int = 0,
                feedback_lag: int = 1, alpha_init: float = 0.15) -> TinyLM:
    """Tiny sparse-FiLM model. `feedback_self_k` only changes the
    TRAINING forward's branch (`model.forward`, multi-pass self-feed) --
    `prefill` / `forward_step` never read `self.feedback_self_k` at all,
    so it has no bearing on the decode-path tests below. `alpha_init` is
    kept modest (roughly the 0.02-0.1 magnitude real trained FiLM alphas
    land at, e.g. `checkpoints/feature_pilot_B.pt`'s 0.02) rather than
    large -- the K=1-lag deploy scheme is a fixed-point APPROXIMATION of
    K-self-feed training (see model.py's `lagged_sources` docstring), and
    that approximation is only expected to be tight in the small-alpha
    regime the architecture was actually designed and trained in."""
    torch.manual_seed(seed)
    return TinyLM(
        vocab_size=64, d_model=32, n_layers=4, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=64,
        feedback_mode="film", feedback_pairs=((0, 2),),
        feedback_self_k=feedback_self_k, feedback_lag=feedback_lag,
        feedback_alpha_init=alpha_init,
    )


def _prefix_reference(model: TinyLM, full_ids: torch.Tensor, prompt_len: int,
                      n_gen: int) -> torch.Tensor:
    """Path A from test_incremental_decode.py: at each step i, a FRESH
    full forward over the growing prefix, last-position logits only.
    This is what `forward_step` is logically equivalent to."""
    outs = []
    for i in range(n_gen):
        prefix_ids = full_ids[:, : prompt_len + i]
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            li = model(prefix_ids).float()
        outs.append(li[:, -1:, :].clone())
    return torch.cat(outs, dim=1)


def _incremental_decode(model: TinyLM, full_ids: torch.Tensor, prompt_len: int,
                        n_gen: int) -> torch.Tensor:
    """Path B: prefill(prompt) + forward_step on each subsequent token."""
    prompt_ids = full_ids[:, :prompt_len]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cache, prefill_last_logits = model.prefill(prompt_ids)
        outs = [prefill_last_logits[:, -1:, :].float().clone()]
        for i in range(n_gen - 1):
            next_tok = full_ids[:, prompt_len + i: prompt_len + i + 1]
            step_logits, cache = model.forward_step(next_tok, cache)
            outs.append(step_logits.float().clone())
    return torch.cat(outs, dim=1)


def _argmax_match(a: torch.Tensor, b: torch.Tensor) -> int:
    a_amax = a.argmax(dim=-1).flatten().tolist()
    b_amax = b.argmax(dim=-1).flatten().tolist()
    return sum(int(x == y) for x, y in zip(a_amax, b_amax))


# ===========================================================================
# TEST 1 -- lean (feedback=none) ckpt: unaffected by this session's edit.
# ===========================================================================

@needs_cuda
def test_lean_model_prefill_step_matches_reference_and_is_bypass_invariant():
    """(a) sanity: prefill+forward_step on a feedback-free model tracks
    the reference full-forward-on-growing-prefix path (same contract the
    pre-existing `test_incremental_decode.py` validates).
    (b) the actual regression guard for 'byte-identical before/after':
    this session's diff only touches the `if use_film_at_decode:`
    branches, which for `feedback_pairs=()` are unreachable in BOTH the
    old and new code (`use_film_at_decode` is `self.feedback_pairs and
    ...`, and `()` is falsy) -- so toggling `_film_bypass` on a lean
    model must be a complete no-op, and IS the closest same-process
    proxy for 'this code path never changed'."""
    torch.manual_seed(0)
    device = "cuda"
    model = _lean_model().to(device).eval()
    vocab = 64
    prompt_len, n_gen = 12, 8
    full_ids = torch.randint(0, vocab, (1, prompt_len + n_gen), device=device)

    ref = _prefix_reference(model, full_ids, prompt_len, n_gen)

    model._film_bypass = False
    dec_default = _incremental_decode(model, full_ids, prompt_len, n_gen)
    model._film_bypass = True
    dec_bypass = _incremental_decode(model, full_ids, prompt_len, n_gen)

    # (b) bypass-invariance: bit-identical regardless of the toggle.
    torch.testing.assert_close(dec_default, dec_bypass, atol=0.0, rtol=0.0)

    # (a) tracks the true reference within the established bf16 kernel
    # noise floor (chunk-with-cache vs fresh chunk-without-cache).
    diff = (ref - dec_default).abs().max().item()
    matches = _argmax_match(ref, dec_default)
    print(f"\n[lean] max|delta|={diff:.4f} argmax={matches}/{n_gen}")
    assert diff < 1.0
    assert matches >= n_gen - 2


# ===========================================================================
# TEST 2 -- FiLM model: incremental decode vs the full K=1-lagged forward.
# ===========================================================================

@needs_cuda
def test_film_incremental_decode_matches_k1_lagged_full_forward():
    """The equivalence test the repo never had. Reference =
    `model.forward()` with `feedback_self_k=0` (`_film_bypass=False`) --
    that's the 'Standard 2-pass sparse FiLM' branch in `forward()`:
    pass 1 vanilla, pass 2 modulated by `_shift_right_by_k(pass1, lag)`.
    Re-run on a growing prefix it is CAUSAL/prefix-length-invariant
    (pass 1 never looks ahead, and pass 2's FiLM input only ever reaches
    back into pass 1), so repeated full forwards on growing prefixes are
    a legitimate step-by-step reference -- exactly mirroring
    `test_incremental_decode.py`'s existing "Path A" pattern.

    `prefill` / `forward_step` never read `feedback_self_k` (only
    `feedback_pairs` / `_film_bypass` / `feedback_xattn_pairs`), so this
    reference is valid for a checkpoint trained at ANY `feedback_self_k`
    -- what changes between K values is which forward the WEIGHTS were
    optimized against, not which forward `prefill`/`forward_step` run.
    """
    torch.manual_seed(3)
    device = "cuda"
    ref_model = _film_model(seed=3, feedback_self_k=0).to(device).eval()
    dec_model = _film_model(seed=3, feedback_self_k=3).to(device).eval()
    # Same seed already gives identical block weights in practice, but
    # extra-param allocation order can perturb the RNG stream between
    # differently-shaped constructions -- force it explicitly so the
    # comparison isolates the DECODE-PATH wiring, not incidental weight
    # drift. K only changes forward()'s training branch, not the module
    # set, so state_dicts are key-for-key identical.
    assert set(ref_model.state_dict()) == set(dec_model.state_dict())
    dec_model.load_state_dict(ref_model.state_dict())

    vocab = 64
    prompt_len, n_gen = 20, 10
    full_ids = torch.randint(0, vocab, (1, prompt_len + n_gen), device=device)

    ref = _prefix_reference(ref_model, full_ids, prompt_len, n_gen)
    dec = _incremental_decode(dec_model, full_ids, prompt_len, n_gen)

    diff = (ref - dec).abs()
    per_step_max = diff.amax(dim=-1).flatten().tolist()
    overall_max = float(diff.max().item())
    matches = _argmax_match(ref, dec)
    print(f"\n[FiLM K=1-lag equivalence] per-step max|delta|="
         f"{[f'{x:.4f}' for x in per_step_max]}")
    print(f"  overall max|delta|={overall_max:.4f} argmax={matches}/{n_gen}")

    # Generous bf16-kernel-noise-plus-approximation tolerance, calibrated
    # against test_incremental_decode.py's established LOGIT_TOL=0.5 /
    # ARGMAX_MIN=14/16 floor for the SAME kernel-disagreement effect on a
    # feedback-free model; FiLM adds a genuine (small-alpha) source of
    # extra divergence within the prompt region alone (bf16 through an
    # extra module), so we allow a bit more headroom here.
    assert overall_max < 1.5, per_step_max
    assert matches >= n_gen - 3, f"{matches}/{n_gen}"

    # The prompt-boundary claim from the wiring analysis: prefill's own
    # last-prompt-position logits (no forward_step yet) should be TIGHT
    # -- prefill runs pass1+pass2 over the WHOLE prompt exactly like the
    # reference's first prefix call, just with a cache attached.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prompt_ref = ref_model(full_ids[:, :prompt_len]).float()[:, -1, :]
        cache, prompt_dec = dec_model.prefill(full_ids[:, :prompt_len])
    prompt_diff = (prompt_ref - prompt_dec.float()[:, -1, :]).abs().max().item()
    print(f"  prompt-boundary max|delta|={prompt_diff:.4f}")
    assert prompt_diff < 1.0


@needs_cuda
def test_film_lag_gt_1_ring_buffer_shape_and_causality():
    """Regression guard for the `feedback_lag > 1` generalization: the
    single-slot cache the old code implicitly assumed `lag==1` for is
    now a length-`lag` ring buffer. Doesn't need to match a reference --
    just needs to run, keep a constant (B, lag, d) shape across many
    steps, and differ from the lag=1 config (proving the lag value is
    actually being honored, not silently truncated to 1)."""
    torch.manual_seed(4)
    device = "cuda"
    m_lag1 = _film_model(seed=4, feedback_self_k=0, feedback_lag=1).to(device).eval()
    m_lag3 = _film_model(seed=4, feedback_self_k=0, feedback_lag=3).to(device).eval()
    m_lag3.load_state_dict(
        {k: v for k, v in m_lag1.state_dict().items()}, strict=True)

    vocab = 64
    prompt_len, n_gen = 10, 6
    full_ids = torch.randint(0, vocab, (1, prompt_len + n_gen), device=device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        cache3, _ = m_lag3.prefill(full_ids[:, :prompt_len])
        src = m_lag3.sparse_target_to_source[0]
        assert cache3["lagged_sources"][src].shape == (1, 3, 32)
        for i in range(n_gen - 1):
            tok = full_ids[:, prompt_len + i: prompt_len + i + 1]
            _, cache3 = m_lag3.forward_step(tok, cache3)
            assert cache3["lagged_sources"][src].shape[1] == 3

    dec_lag1 = _incremental_decode(m_lag1, full_ids, prompt_len, n_gen)
    dec_lag3 = _incremental_decode(m_lag3, full_ids, prompt_len, n_gen)
    assert not torch.allclose(dec_lag1, dec_lag3, atol=1e-3), (
        "lag=1 and lag=3 decode produced identical logits -- the ring "
        "buffer generalization is not actually taking effect")


# ===========================================================================
# TEST 3 -- `_film_bypass=True` is still the explicit escape hatch.
# ===========================================================================

@needs_cuda
def test_film_bypass_still_forces_plain_block_loop():
    """`model._film_bypass = True` must reproduce the SAME output as a
    sibling model with `feedback_mode="none"` (identical block/embed/
    lm_head weights, just literally no sparse_feedback modules at all)
    -- i.e. bypass is a real ablation escape hatch, not merely 'FiLM
    with alpha near zero'."""
    torch.manual_seed(5)
    device = "cuda"
    none_model = _lean_model(seed=5).to(device).eval()
    film_model = _film_model(seed=5, feedback_self_k=3, alpha_init=0.7).to(device).eval()
    missing, unexpected = film_model.load_state_dict(
        none_model.state_dict(), strict=False)
    # film_model is a strict superset of none_model's keys (feedback_pairs
    # adds sparse_feedback.* params) -- those are legitimately `missing`
    # from the lean donor's state dict (left at film_model's own random
    # init, untouched) and every key present in BOTH must load with no
    # `unexpected` leftovers.
    assert set(missing) == {
        "sparse_feedback.0.alpha", "sparse_feedback.0.W_scale.weight",
        "sparse_feedback.0.W_shift.weight"}, missing
    assert unexpected == [], f"expected no unexpected keys: {unexpected}"

    vocab = 64
    prompt_len, n_gen = 14, 8
    full_ids = torch.randint(0, vocab, (1, prompt_len + n_gen), device=device)

    ref = _incremental_decode(none_model, full_ids, prompt_len, n_gen)
    film_model._film_bypass = True
    bypassed = _incremental_decode(film_model, full_ids, prompt_len, n_gen)
    torch.testing.assert_close(ref, bypassed, atol=0.0, rtol=0.0)

    # And the un-bypassed FiLM model (large alpha) must differ -- proves
    # the bypass toggle is actually doing something, not that FiLM was
    # a no-op to begin with.
    film_model._film_bypass = False
    unbypassed = _incremental_decode(film_model, full_ids, prompt_len, n_gen)
    assert not torch.allclose(ref, unbypassed, atol=1e-3), (
        "un-bypassed FiLM decode == bypassed decode -- FiLM isn't "
        "actually being applied at decode")


# ===========================================================================
# TEST 4 -- WM copy head fires at inference (mem_read_mask=None).
# ===========================================================================

def _copy_head_model() -> TinyLM:
    torch.manual_seed(0)
    return TinyLM(
        vocab_size=10, d_model=8, n_layers=1, n_heads=2, d_head=4,
        attention_cls=DeltaNetAttention, max_T=32, output_gate=True,
        use_memory=True, mem_size=32, thinking_token_id=9, pad_token_id=0,
        mem_decoupled_kv=True, mem_discrete_key=True, use_copy_head=True,
        mem_always_read=True,
    )


def _copy_head_scene():
    """Hand-built (h, input_ids, lm_logits) plus hand-stashed WM read
    state -- mirrors test_wm_discrete_key.py's
    test_match_existence_gating_suppresses_copy_on_no_match, generalized
    to exercise EVERY position (not just two masked ones), since the
    fix under test is specifically about the `mem_read_mask is None`
    (every-position) path."""
    model = _copy_head_model()
    mem = model.memory
    with torch.no_grad():
        model.copy_head.gate.bias.fill_(10.0)   # g ~= 1: copy clearly active
        model.copy_head.gate.weight.zero_()
    B, T, K, V, d = 1, 6, 2, 10, 8
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    h = torch.randn(B, T, d)
    lm_logits = torch.randn(B, T, V)
    mem._last_top_idx_buf = torch.tensor([[0, 1]])
    attn = torch.zeros(B, T, K)
    attn[0, 3, 0] = 1.0
    attn[0, 4, 0] = 1.0
    mem._last_read_attn_grad = attn
    # Only position 3 has a causally-valid match; every other position
    # (including 4, which has non-zero attention/would-be copy content)
    # has none.
    mem._last_match_exists = torch.tensor(
        [[False, False, False, True, False, False]])
    mem.copy_require_match = True
    return model, mem, h, ids, lm_logits


def test_copy_head_silent_by_default_without_always_read():
    """Byte-compat: a copy-head model WITHOUT always_read must still be
    a hard no-op when no mask is given (the pre-fix behaviour for every
    non-`always_read` config)."""
    model, mem, h, ids, lm_logits = _copy_head_scene()
    mem.always_read = False
    out = model._apply_copy_head(lm_logits.clone(), h, ids, None)
    assert torch.equal(out, lm_logits)


def test_copy_head_fires_everywhere_at_inference_gated_by_match():
    """GAP 2 fix: with always_read on and NO mask supplied (the actual
    generation-time calling convention), the copy head must mix at
    match positions and be an exact no-op at non-match positions."""
    model, mem, h, ids, lm_logits = _copy_head_scene()
    assert mem.always_read is True
    plain = torch.softmax(lm_logits, dim=-1)

    out = model._apply_copy_head(lm_logits.clone(), h, ids, None)
    out_dist = torch.softmax(out, dim=-1)

    # match position (3): copy head actually shifted the distribution.
    assert (out_dist[0, 3] - plain[0, 3]).abs().sum() > 0.5
    # every non-match position, INCLUDING pos 4 which has real attention
    # mass and would copy garbage if `me` weren't gating it: unchanged.
    for t in (0, 1, 2, 4, 5):
        torch.testing.assert_close(out_dist[0, t], plain[0, t], atol=1e-5, rtol=1e-4)

    # Explicit-mask path (train-time / any caller that threads a mask)
    # is untouched by this fix -- same call, but now WITH an explicit
    # mask covering only the two positions the old test already covers.
    rm = torch.zeros(1, 6, dtype=torch.bool)
    rm[0, 3] = True
    rm[0, 4] = True
    out_masked = model._apply_copy_head(lm_logits.clone(), h, ids, rm)
    out_masked_dist = torch.softmax(out_masked, dim=-1)
    torch.testing.assert_close(out_masked_dist[0, 3], out_dist[0, 3])
    torch.testing.assert_close(out_masked_dist[0, 4], plain[0, 4])


def test_copy_inference_off_restores_old_behaviour():
    """`model._copy_inference_off = True` is the explicit escape hatch:
    with it set, generation-time forwards (mem_read_mask=None) are a
    hard no-op again, byte-identical to the pre-fix code, regardless of
    always_read / match state."""
    model, mem, h, ids, lm_logits = _copy_head_scene()
    assert mem.always_read is True
    model._copy_inference_off = True
    out = model._apply_copy_head(lm_logits.clone(), h, ids, None)
    assert torch.equal(out, lm_logits)

    # Sanity: the toggle only affects the no-mask path -- an explicit
    # mask still fires normally even with the escape hatch set (it only
    # disables the NEW always-on inference behaviour, not the original
    # trained/supervised mask-driven path).
    rm = torch.zeros(1, 6, dtype=torch.bool)
    rm[0, 3] = True
    out_masked = model._apply_copy_head(lm_logits.clone(), h, ids, rm)
    plain = torch.softmax(lm_logits, dim=-1)
    out_masked_dist = torch.softmax(out_masked, dim=-1)
    assert (out_masked_dist[0, 3] - plain[0, 3]).abs().sum() > 0.5


# ===========================================================================
# TEST 5 -- training-time self-calibrating copy gate (extension,
# owner-approved 2026-07-03): the maskless branch also fires in TRAINING
# forwards, so plain-CE batches supply negatives at me-positions while
# masked recall batches keep supplying supervised positives.
# ===========================================================================

def _training_copy_model(device="cuda"):
    """Tiny always_read + copy-head model for real training forwards.

    Uses legacy dot-product WM addressing (no tokenizer-dependent
    discrete/namekey machinery), which never populates
    `_last_match_exists` -- so tests that need an `me` signal install a
    wrapper around `_apply_memory` that stamps a fixed `me` after the
    real WM read, standing in for the discrete/ctx-namekey addressing
    the production configs use."""
    torch.manual_seed(7)
    model = TinyLM(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=64,
        use_memory=True, mem_size=64, thinking_token_id=63, pad_token_id=0,
        mem_always_read=True, use_copy_head=True,
    ).to(device)
    with torch.no_grad():
        # Lift the copy gate off its very-negative cold-start init so the
        # mix has measurable influence (g ~= 0.88).
        model.copy_head.gate.bias.fill_(2.0)
    return model


def _install_me_stamp(model, me: torch.Tensor):
    """Wrap model._apply_memory so `mem._last_match_exists` is `me` after
    every WM read (the legacy addressing leaves it None)."""
    orig = model._apply_memory

    def patched(h, input_ids, read_mask=None, doc_ids=None):
        out = orig(h, input_ids, read_mask=read_mask, doc_ids=doc_ids)
        model.memory._last_match_exists = me.to(h.device)
        return out

    model._apply_memory = patched


@needs_cuda
def test_copy_head_training_maskless_fires_at_me_and_grads_copy_gate():
    """Training-mode forward WITHOUT mem_read_mask: (a) logits differ
    from the _copy_inference_off control exactly at me=1 positions and
    are bit-identical at me=0 positions; (b) plain CE backward delivers
    gradient to the copy gate (the 'negatives from plain CE' half of the
    self-calibrating design)."""
    device = "cuda"
    model = _training_copy_model(device)
    B, T = 1, 12
    me = torch.zeros(B, T, dtype=torch.bool)
    me[0, 5] = True
    me[0, 9] = True
    _install_me_stamp(model, me)

    ids = torch.randint(1, 62, (B, T), device=device)
    model.train()

    logits_on = model(ids)
    if isinstance(logits_on, tuple):
        logits_on = logits_on[0]

    model._copy_inference_off = True
    logits_off = model(ids)
    if isinstance(logits_off, tuple):
        logits_off = logits_off[0]
    model._copy_inference_off = False

    p_on = torch.softmax(logits_on.detach(), dim=-1)
    p_off = torch.softmax(logits_off.detach(), dim=-1)
    for t in range(T):
        delta = (p_on[0, t] - p_off[0, t]).abs().sum().item()
        if me[0, t]:
            assert delta > 1e-4, f"me=1 position {t} shows no copy influence"
        else:
            # index_put never touched these rows -- exact equality.
            assert torch.equal(logits_on[0, t].detach(),
                              logits_off[0, t].detach()), \
                f"me=0 position {t} was modified"

    # (b) gradient reaches the copy gate from plain CE alone.
    tgt = torch.randint(1, 62, (B, T), device=device)
    loss = torch.nn.functional.cross_entropy(
        logits_on.reshape(-1, logits_on.shape[-1]), tgt.reshape(-1))
    model.zero_grad(set_to_none=True)
    loss.backward()
    gw = model.copy_head.gate.weight.grad
    gb = model.copy_head.gate.bias.grad
    assert gw is not None and gb is not None
    assert (gw.abs().sum() + gb.abs().sum()).item() > 0, \
        "plain CE delivered zero gradient to the copy gate"


@needs_cuda
def test_copy_head_training_explicit_mask_immune_to_toggle():
    """When a data mask IS passed (the supervised-positive recall
    batches), behavior must be exactly as before: _copy_inference_off
    must have zero effect on the explicit-mask path."""
    device = "cuda"
    model = _training_copy_model(device)
    B, T = 1, 12
    me = torch.ones(B, T, dtype=torch.bool)     # me everywhere; mask decides
    _install_me_stamp(model, me)
    ids = torch.randint(1, 62, (B, T), device=device)
    rm = torch.zeros(B, T, dtype=torch.bool, device=device)
    rm[0, 4] = True
    model.train()

    out_a = model(ids, mem_read_mask=rm)
    if isinstance(out_a, tuple):
        out_a = out_a[0]
    model._copy_inference_off = True
    out_b = model(ids, mem_read_mask=rm)
    if isinstance(out_b, tuple):
        out_b = out_b[0]
    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)


@needs_cuda
def test_copy_head_training_maskless_noop_without_match_signal():
    """Legacy addressing never populates `_last_match_exists` -> the
    maskless path must be a hard no-op ('apply where me fires' -- no me,
    no application), NOT an ungated all-positions blanket (which would
    also corrupt the contiguous-run answer offset)."""
    device = "cuda"
    model = _training_copy_model(device)      # no me-stamp installed
    ids = torch.randint(1, 62, (1, 12), device=device)
    model.train()

    out_a = model(ids)
    if isinstance(out_a, tuple):
        out_a = out_a[0]
    model._copy_inference_off = True
    out_b = model(ids)
    if isinstance(out_b, tuple):
        out_b = out_b[0]
    assert model.memory._last_match_exists is None
    torch.testing.assert_close(out_a, out_b, atol=0.0, rtol=0.0)


# ===========================================================================
# TEST 6 -- `_wm_read_one` respects `always_read` (third wiring gap):
# WM injection live at incremental decode for always_read models.
# ===========================================================================

def _wm_model(always_read: bool, device="cuda"):
    torch.manual_seed(21)
    return TinyLM(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=64,
        use_memory=True, mem_size=64, thinking_token_id=63, pad_token_id=0,
        mem_always_read=always_read,
    ).to(device).eval()


@needs_cuda
def test_wm_always_read_incremental_matches_full_forward():
    """always_read model, NO think tokens in the stream: incremental
    decode must (a) actually apply the WM injection at every step
    (nonzero -- proven by divergence from a read_alpha=0 control) and
    (b) stay consistent with the full forward's per-position injection
    (per-prefix reference, bf16 tolerance)."""
    device = "cuda"
    model = _wm_model(always_read=True, device=device)
    prompt_len, n_gen = 12, 8
    # Tokens in [1, 62]: no pad (0), no think (63).
    full_ids = torch.randint(1, 62, (1, prompt_len + n_gen), device=device)

    ref = _prefix_reference(model, full_ids, prompt_len, n_gen)
    dec = _incremental_decode(model, full_ids, prompt_len, n_gen)

    diff = (ref - dec).abs().max().item()
    matches = _argmax_match(ref, dec)
    print(f"\n[always_read WM incremental] max|delta|={diff:.4f} "
         f"argmax={matches}/{n_gen}")
    assert diff < 1.0
    assert matches >= n_gen - 2

    # Positive control: with the read injection disabled (alpha=0) the
    # step outputs must CHANGE -- proving the injection was genuinely
    # live during the run above (pre-fix, inj_mask was all-zero at every
    # step, so alpha had no effect on forward_step outputs at all).
    with torch.no_grad():
        saved = model.memory.read_alpha.clone()
        model.memory.read_alpha.zero_()
    dec_noinj = _incremental_decode(model, full_ids, prompt_len, n_gen)
    with torch.no_grad():
        model.memory.read_alpha.copy_(saved)
    # Skip index 0: that logit row comes from prefill (full forward),
    # which respected always_read both before and after the fix.
    assert not torch.allclose(dec[:, 1:], dec_noinj[:, 1:], atol=1e-4), (
        "zeroing read_alpha did not change forward_step outputs -- WM "
        "injection is still dead during incremental decode")


@needs_cuda
def test_wm_read_one_non_always_read_semantics_unchanged():
    """Unit-level byte-compat: with always_read=False, `_wm_read_one` at
    a NON-think position returns its input unchanged (injection masked
    to zero), and at a think position it injects -- the exact pre-fix
    semantics."""
    device = "cuda"
    model = _wm_model(always_read=False, device=device)
    ids = torch.randint(1, 62, (1, 4), device=device)
    with torch.no_grad():
        h_normed = model.out_norm(model.embed(ids))
        buf = model._wm_init_buffer_from_prompt(h_normed, ids)

        h_new = torch.randn(1, 1, 32, device=device)
        emit_tok = torch.tensor([[5]], device=device)
        out_emit = model._wm_read_one(buf, h_new, None, emit_tok)
        torch.testing.assert_close(out_emit, h_new, atol=0.0, rtol=0.0)

        think_tok = torch.tensor([[63]], device=device)
        out_think = model._wm_read_one(buf, h_new, None, think_tok)
        assert not torch.allclose(out_think, h_new, atol=1e-6), \
            "think-position read produced no injection"

    # And the always_read model DOES inject at the emit token.
    model_ar = _wm_model(always_read=True, device=device)
    with torch.no_grad():
        h_normed = model_ar.out_norm(model_ar.embed(ids))
        buf = model_ar._wm_init_buffer_from_prompt(h_normed, ids)
        out_ar = model_ar._wm_read_one(buf, h_new, None, emit_tok)
        assert not torch.allclose(out_ar, h_new, atol=1e-6), \
            "always_read read produced no injection at an emit token"


@needs_cuda
def test_wm_read_one_always_read_guarded_for_unfaithful_addressing():
    """`_wm_read_one` only implements the LEGACY dot-product scoring; for
    decoupled_kv / discrete / namekey addressing an always-on step-read
    would inject a wrong-scoring read at every position of a trained
    ckpt. The guard keeps those configs on the old (think-only, i.e.
    zero here) behaviour."""
    device = "cuda"
    torch.manual_seed(23)
    model = TinyLM(
        vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=64,
        use_memory=True, mem_size=64, thinking_token_id=63, pad_token_id=0,
        mem_always_read=True, mem_decoupled_kv=True,
    ).to(device).eval()
    ids = torch.randint(1, 62, (1, 4), device=device)
    with torch.no_grad():
        h_normed = model.out_norm(model.embed(ids))
        buf = model._wm_init_buffer_from_prompt(h_normed, ids)
        h_new = torch.randn(1, 1, 32, device=device)
        emit_tok = torch.tensor([[5]], device=device)
        out = model._wm_read_one(buf, h_new, None, emit_tok)
        torch.testing.assert_close(out, h_new, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
