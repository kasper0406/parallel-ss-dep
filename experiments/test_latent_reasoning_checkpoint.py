"""Tests for the latent-reasoning-cotrain OOM fix (2026-07-02).

ROOT CAUSE: `latent_reasoning_cotrain.py`'s `_answer_span_latent_loss` runs an
R-step latent unroll (up to R=8) x n_examples (=4 in the phase-1 launcher)
FULL forwards through the whole trunk, all summed into ONE graph before the
caller's single combined `.backward()`. `clean_latent_thread(...,
no_activation_ckpt=True)` deliberately keeps the model's PER-BLOCK checkpoint
OFF for this path (it hits a Blackwell "unspecified launch failure" recomputing
FLA kernels at the latent thread's short/odd lengths) — so every one of those
forwards previously retained its FULL ~32-layer activation trace, ~16 GB by
itself on the 32L x 960d config, and is what killed both Arm-B attempts at
step 620 (see project_phase1_ab_features_nettax memory note).

The fix wraps each latent-thread `model(...)` call in an OUTER (whole-model,
not per-block) `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`
boundary — a coarser, different checkpoint granularity than the per-block one
implicated in the Blackwell bug, while still discarding the internal
activation trace between calls. These tests pin: (1) the checkpointed loss
VALUE is unchanged (checkpointing is exact recomputation, not an
approximation); (2) gradients reach both the latent adapter AND the trunk,
unchanged by checkpointing; (3) the deepest rung (R=8) runs cleanly; (4) the
3-tuple return path (gist loss active) is handled correctly under
checkpointing, matching the pre-fix `out[1]`-indexing contract.

CPU-only, no CUDA / FLA / HF required — mirrors test_latent_feedback_adapter.py's
tiny-model pattern (SoftmaxAttention swapped in for the CUDA-only DeltaNet).
"""
from __future__ import annotations

import json

import torch

from experiments.gist_loss import build_gist_heads, parse_horizons
from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.latent_reasoning_cotrain import (
    LatentReasoningCotrain,
    _answer_span_latent_loss,
)

THINK_ID = 5  # != PAD_ID (0)
PAD_ID = 0
EOS_ID = 1
VOCAB = 64
D_MODEL = 32


def _tiny_model(*, seed: int = 0, with_gist: bool = False) -> TinyLM:
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=2, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=256,
        output_gate=True, state_readonly_at_think=True,
        use_latent_feedback_adapter=True,
    )
    m.thinking_token_id = THINK_ID
    if with_gist:
        # Mirrors train_lm.py's gist wiring exactly (model_builder does not
        # set this up itself) — exercises the training-mode 3-tuple return
        # (logits, hidden, gist_loss) that _answer_span_latent_loss's
        # docstring specifically calls out as the reason it INDEXES
        # `out[1]` instead of tuple-unpacking.
        horizons = parse_horizons("2,4")
        m.gist_heads = build_gist_heads(D_MODEL, horizons)
        m._gist_horizons = horizons
        m._gist_loss_enabled = True
    m.train()
    return m


def _example(seed: int = 0, plen: int = 10, slen: int = 3):
    g = torch.Generator().manual_seed(seed)
    comment_ids = torch.randint(2, VOCAB - 1, (plen,), generator=g).tolist()
    sol_ids = torch.randint(2, VOCAB - 1, (slen,), generator=g).tolist()
    return comment_ids, sol_ids


# ---------------------------------------------------------------------------
# 1. Checkpointed loss value == unchecked loss value.
# ---------------------------------------------------------------------------

def test_checkpoint_matches_unchecked_loss_value():
    m = _tiny_model(seed=0)
    comment_ids, sol_ids = _example()
    loss_ckpt = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=3, thinking_id=THINK_ID,
        device="cpu", checkpoint_latent=True)
    loss_plain = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=3, thinking_id=THINK_ID,
        device="cpu", checkpoint_latent=False)
    assert torch.allclose(loss_ckpt, loss_plain, atol=1e-5), (
        float(loss_ckpt), float(loss_plain))


def test_checkpoint_matches_unchecked_loss_with_gate_weight():
    """gate_weight>0 adds a second loss term reading model._last_gate_logits
    — the exact attribute checkpoint's first (grad-tracked) call sets, so this
    pins that the gate-calibration term also survives checkpointing intact."""
    m = _tiny_model(seed=1)
    comment_ids, sol_ids = _example(seed=1)
    loss_ckpt = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=4, thinking_id=THINK_ID,
        device="cpu", gate_weight=0.05, checkpoint_latent=True)
    loss_plain = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=4, thinking_id=THINK_ID,
        device="cpu", gate_weight=0.05, checkpoint_latent=False)
    assert torch.allclose(loss_ckpt, loss_plain, atol=1e-5), (
        float(loss_ckpt), float(loss_plain))


def test_checkpoint_matches_unchecked_with_gist_active():
    """Training-mode model(..., return_hidden=True) returns a 3-tuple
    (logits, hidden, gist) when the trunk gist loss is enabled — pin that
    checkpoint()'s tuple-output handling doesn't disturb the hidden-at-
    index-1 convention _answer_span_latent_loss relies on."""
    m = _tiny_model(seed=2, with_gist=True)
    comment_ids, sol_ids = _example(seed=2)
    loss_ckpt = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=3, thinking_id=THINK_ID,
        device="cpu", checkpoint_latent=True)
    loss_plain = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=3, thinking_id=THINK_ID,
        device="cpu", checkpoint_latent=False)
    assert torch.isfinite(loss_ckpt) and torch.isfinite(loss_plain)
    assert torch.allclose(loss_ckpt, loss_plain, atol=1e-5), (
        float(loss_ckpt), float(loss_plain))


# ---------------------------------------------------------------------------
# 2. Gradients: checkpointed == unchecked, and reach BOTH the adapter and the
#    trunk (not just the adapter).
# ---------------------------------------------------------------------------

def _grads_for(m: TinyLM, comment_ids, sol_ids, R, checkpoint_latent):
    for p in m.parameters():
        p.grad = None
    loss = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=R, thinking_id=THINK_ID,
        device="cpu", checkpoint_latent=checkpoint_latent)
    loss.backward()
    grads = {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in m.named_parameters()}
    for p in m.parameters():
        p.grad = None
    return grads


def test_gradients_flow_to_adapter_and_trunk():
    m = _tiny_model(seed=0)
    comment_ids, sol_ids = _example()
    grads = _grads_for(m, comment_ids, sol_ids, R=3, checkpoint_latent=True)

    adapter_grad = grads["latent_feedback_adapter.proj.weight"]
    assert adapter_grad is not None and adapter_grad.abs().sum() > 0, \
        "adapter must receive nonzero gradient"

    embed_grad = grads["embed.weight"]
    assert embed_grad is not None and embed_grad.abs().sum() > 0, \
        "trunk embedding must receive nonzero gradient"

    # At least one block-level trunk parameter must also see gradient (not
    # just the embedding table / adapter).
    block_grad_keys = [n for n in grads if n.startswith("blocks.0.")
                       and grads[n] is not None]
    assert block_grad_keys, "no blocks.0.* parameter received a gradient"
    assert any(grads[n].abs().sum() > 0 for n in block_grad_keys), \
        "block 0 gradients are all exactly zero"


def test_checkpoint_matches_unchecked_gradients():
    m = _tiny_model(seed=0)
    comment_ids, sol_ids = _example()
    g_ckpt = _grads_for(m, comment_ids, sol_ids, R=3, checkpoint_latent=True)
    g_plain = _grads_for(m, comment_ids, sol_ids, R=3, checkpoint_latent=False)
    assert set(g_ckpt.keys()) == set(g_plain.keys())
    compared = 0
    for name in g_ckpt:
        a, b = g_ckpt[name], g_plain[name]
        assert (a is None) == (b is None), name
        if a is None:
            continue
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (
            name, float((a - b).abs().max()))
        compared += 1
    assert compared > 0


# ---------------------------------------------------------------------------
# 3. Deepest rung (R=8) runs cleanly under checkpointing.
# ---------------------------------------------------------------------------

def test_rung_8_runs_under_checkpoint():
    m = _tiny_model(seed=3)
    comment_ids, sol_ids = _example(seed=3, plen=8, slen=2)
    loss = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=8, thinking_id=THINK_ID,
        device="cpu", gate_weight=0.05, checkpoint_latent=True)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert m.latent_feedback_adapter.proj.weight.grad is not None
    assert m.latent_feedback_adapter.proj.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# 4. clean_latent_thread's restore contract is untouched (model.
#    activation_checkpointing must still come back False -> whatever it was,
#    checkpoint_latent is an independent, outer mechanism).
# ---------------------------------------------------------------------------

def test_checkpointing_does_not_touch_model_activation_checkpointing_flag():
    m = _tiny_model(seed=0)
    m.activation_checkpointing = True  # arbitrary pre-existing value
    comment_ids, sol_ids = _example()
    _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R=2, thinking_id=THINK_ID,
        device="cpu", checkpoint_latent=True)
    assert m.activation_checkpointing is True, (
        "_answer_span_latent_loss must not mutate model.activation_checkpointing "
        "(that flag is clean_latent_thread's, a DIFFERENT, per-block mechanism "
        "left off for the documented Blackwell-recompute reason)")


# ---------------------------------------------------------------------------
# 5. End-to-end through LatentReasoningCotrain.step() — checkpoint_latent
#    default True, threaded from the class down into the loss helper.
# ---------------------------------------------------------------------------

class _FakeTok:
    """Deterministic, dependency-free stand-in for a HF tokenizer — avoids
    needing network / cached SmolLM2 tokenizer files for this unit test."""

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (VOCAB - 4)) for c in text]


def _write_rung_file(path, n, n_records=3):
    with open(path, "w") as f:
        for i in range(n_records):
            f.write(json.dumps({
                "prompt": f"x{i} = {i}\nprint(x{i})",
                "answer": i,
            }) + "\n")


def test_latentreasoningcotrain_checkpoint_default_on(tmp_path):
    prefix = str(tmp_path / "toy_ptrchase_train")
    for n in (2, 3):
        _write_rung_file(f"{prefix}_n{n}.jsonl", n)

    m = _tiny_model(seed=0)
    reasoner = LatentReasoningCotrain(
        train_prefix=prefix, rungs=[2, 3], tok=_FakeTok(),
        thinking_id=THINK_ID, eos_id=EOS_ID, device="cpu", max_len=64,
        no_ramp=True, seed=0)
    assert reasoner.checkpoint_latent is True, "checkpointing must default ON"

    loss, rung = reasoner.step(m, step=0, total_steps=100, n_examples=2)
    assert torch.isfinite(loss)
    assert rung in (2, 3)
    loss.backward()
    assert m.latent_feedback_adapter.proj.weight.grad is not None


def test_latentreasoningcotrain_checkpoint_escape_hatch(tmp_path):
    prefix = str(tmp_path / "toy_ptrchase_train2")
    _write_rung_file(f"{prefix}_n2.jsonl", 2)

    m = _tiny_model(seed=0)
    reasoner = LatentReasoningCotrain(
        train_prefix=prefix, rungs=[2], tok=_FakeTok(),
        thinking_id=THINK_ID, eos_id=EOS_ID, device="cpu", max_len=64,
        no_ramp=True, seed=0, checkpoint_latent=False)
    assert reasoner.checkpoint_latent is False

    loss, rung = reasoner.step(m, step=0, total_steps=100, n_examples=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.latent_feedback_adapter.proj.weight.grad is not None


# ---------------------------------------------------------------------------
# 6. Regression pin for the "ambient config != clean-latent-thread state,
#    caller backwards LATE" bug caught on the real 32L GPU verification
#    (2026-07-02): torch.utils.checkpoint.CheckpointError, "a different
#    number of tensors was saved during the original forward and
#    recomputation" (1600 vs 218). Root cause: clean_latent_thread's toggles
#    were scoped to step()'s `with` block, which had already exited
#    (restoring the model's REAL config: WM on, activation_checkpointing on,
#    FiLM K self-feed engaged) by the time train_lm.py's combined
#    `.backward()` fired the checkpoint recompute — forward and recompute
#    silently ran different graphs. This only reproduces on a model whose
#    AMBIENT (pretrain) config differs from the clean-thread config, which
#    the other tests' bare `_tiny_model()` does not exercise (it already IS
#    the clean-thread config: use_memory=False, activation_checkpointing=
#    False, feedback_mode="none") — hence a dedicated model + calling
#    convention here that mirrors train_lm.py exactly: `step()` returns a
#    loss, the caller adds it to other terms and calls `.backward()`
#    strictly AFTER `step()` has returned (so any `with` block scoped
#    inside `step()` has already exited).
# ---------------------------------------------------------------------------

def _tiny_model_with_loaded_ambient_config(*, seed: int = 0) -> TinyLM:
    """A model whose PRETRAIN-TIME config (what's active outside the latent
    thread) differs from clean_latent_thread's toggled state on all three
    axes: WM on, activation_checkpointing on, FiLM K=2 self-feed engaged.
    Mirrors the real phase1_ab_B launcher's `--use_memory
    --activation_checkpointing --feedback film --feedback_self_k 3` shape,
    just tiny."""
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=3, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=256,
        output_gate=True, state_readonly_at_think=True,
        use_latent_feedback_adapter=True,
        use_memory=True, mem_size=8, thinking_token_id=THINK_ID,
        activation_checkpointing=True,
        feedback_mode="film", feedback_pairs=((0, 2),), feedback_self_k=2,
    )
    m.thinking_token_id = THINK_ID
    m.train()
    return m


def test_step_then_late_backward_matches_unchecked_with_loaded_ambient_config(
        tmp_path):
    """The exact calling pattern train_lm.py uses: `step()` returns a loss,
    the caller adds it into a combined loss and calls `.backward()`
    strictly after `step()` has returned (so clean_latent_thread's `with`
    block, wherever it lives, has already exited by backward time)."""
    prefix = str(tmp_path / "toy_ptrchase_ambient")
    _write_rung_file(f"{prefix}_n2.jsonl", 2)

    def _run(checkpoint_latent):
        m = _tiny_model_with_loaded_ambient_config(seed=0)
        reasoner = LatentReasoningCotrain(
            train_prefix=prefix, rungs=[2], tok=_FakeTok(),
            thinking_id=THINK_ID, eos_id=EOS_ID, device="cpu", max_len=64,
            no_ramp=True, seed=0, checkpoint_latent=checkpoint_latent)
        loss, rung = reasoner.step(m, step=0, total_steps=100, n_examples=2)
        loss_val = float(loss.detach())
        # Mirrors train_lm.py: combine with another (fake) loss term and
        # backward AFTER step() has returned, well outside any of its
        # internal `with` scopes.
        combined = 1.0 * loss + 0.0 * sum(p.sum() for p in m.parameters())
        combined.backward()
        return loss_val, {n: (p.grad.clone() if p.grad is not None else None)
                             for n, p in m.named_parameters()}

    # Ambient WM/activation_checkpointing/FiLM-K state must still be exactly
    # what it was before (untouched) after step() returns, on BOTH paths —
    # and, critically, checkpoint_latent=True must not raise CheckpointError
    # from this late-backward pattern.
    loss_ckpt, grads_ckpt = _run(True)
    loss_plain, grads_plain = _run(False)
    assert abs(loss_ckpt - loss_plain) < 1e-4, (loss_ckpt, loss_plain)
    assert set(grads_ckpt) == set(grads_plain)
    compared = 0
    for name in grads_ckpt:
        a, b = grads_ckpt[name], grads_plain[name]
        assert (a is None) == (b is None), name
        if a is None:
            continue
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-3), (
            name, float((a - b).abs().max()))
        compared += 1
    assert compared > 0


def test_ambient_config_restored_after_step_before_backward(tmp_path):
    """Pin the specific mechanism: right after step() returns (before
    backward), the model's ambient use_memory / activation_checkpointing
    must be back to what they were BEFORE step() was called — proving the
    toggle is properly scoped per-call, not leaked."""
    prefix = str(tmp_path / "toy_ptrchase_ambient2")
    _write_rung_file(f"{prefix}_n2.jsonl", 2)
    m = _tiny_model_with_loaded_ambient_config(seed=0)
    assert m.use_memory is True
    assert m.activation_checkpointing is True

    reasoner = LatentReasoningCotrain(
        train_prefix=prefix, rungs=[2], tok=_FakeTok(),
        thinking_id=THINK_ID, eos_id=EOS_ID, device="cpu", max_len=64,
        no_ramp=True, seed=0, checkpoint_latent=True)
    loss, _ = reasoner.step(m, step=0, total_steps=100, n_examples=1)

    assert m.use_memory is True, "WM toggle leaked past step()"
    assert m.activation_checkpointing is True, \
        "activation_checkpointing toggle leaked past step()"
    # And backward still succeeds (no CheckpointError) with these restored.
    loss.backward()
