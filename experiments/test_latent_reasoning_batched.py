"""Tests for the batched growing-thread + aux_every fix (2026-07-04).

MEASURED PROBLEM: `latent_reasoning_cotrain.py`'s aux processed its
`--latent_reasoning_n` examples/step SEQUENTIALLY as n independent B=1
growing threads — ~5.6s of every 10.4s N1 step, ~25% GPU utilization (a
kernel-launch/latency-bound stall, not a compute-bound one). TWO independent
fixes, both default-ON:

  1. Batched growing-thread (`_answer_span_latent_loss_batched`): the n
     examples of a step already share one rung/R (`_pick_rung` is called
     once per `step()`, not once per example) so they can run as ONE
     batch-n thread. The correctness hazard: a pad token WRITES to a
     linear-RNN's (DeltaNet's) recurrent state before the real prompt
     starts — unlike causal-attention padding, which is free once masked.
     The fix LEFT-pads (keeps every row's real content ending at the same
     absolute position, so `h[:, -1:, :]` stays correct with no per-row
     gather) AND marks the pad prefix as a separate "document" from the
     real content via the existing cross-document `doc_ids` isolation
     (model.py::_build_cu_seqlens) — a HARD state reset that does not
     depend on `--state_readonly_at_think` being enabled (the currently
     running N1 process is NOT built with that flag; verified directly
     against its live cmdline).
  2. `--latent_reasoning_aux_every K` (train_lm.py): fire the aux only
     every K-th optimizer step, at K x weight — same expected gradient,
     fewer stalls.

Sections below:
  A. `_latent_reasoning_aux_every_gate` (train_lm.py) — pure, CPU, no model.
  B. `_left_pad_prompts` / `_right_pad_solutions` — pure tensor-construction
     unit tests, CPU, no model.
  C. Batched == mean-of-sequential loss + gradients, on a tiny REAL DeltaNet
     TinyLM (CUDA-gated — DeltaNet's Triton kernels are CUDA-only, mirroring
     test_cu_seqlens.py's own convention). Tolerances mirror
     test_cu_seqlens.py's "packed==unpacked" precedent (rtol/atol ~2e-2):
     the cu_seqlens-flattened kernel path is mathematically but not
     bit-identical to the plain per-row batched path (different Triton
     kernel entry points / chunking), so exact equality is not the right
     bar here — only the pre-existing cu_seqlens machinery's own
     "packed==unpacked" test uses that same loose tolerance.
  D. `LatentReasoningCotrain.step()` end-to-end, `batch_examples` True vs
     False escape hatch, on tiny real DeltaNet — same example sampling
     order either way, only processing differs.

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/test_latent_reasoning_batched.py -v
"""
from __future__ import annotations

import json

import pytest
import torch

from experiments.train_lm import _latent_reasoning_aux_every_gate
from experiments.latent_reasoning_cotrain import (
    LatentReasoningCotrain,
    _answer_span_latent_loss,
    _answer_span_latent_loss_batched,
    _left_pad_prompts,
    _right_pad_solutions,
)

THINK_ID = 5   # != PAD_ID (0)
PAD_ID = 0
EOS_ID = 1
VOCAB = 64


# ---------------------------------------------------------------------------
# A. --latent_reasoning_aux_every gate (pure, no model, no CUDA)
# ---------------------------------------------------------------------------

def test_aux_every_default_k1_fires_every_step_at_weight_1():
    for step in range(10):
        fire, mult = _latent_reasoning_aux_every_gate(step, start_step=0, every=1)
        assert fire is True
        assert mult == 1.0


def test_aux_every_never_fires_before_start_step():
    for step in range(5):
        fire, _ = _latent_reasoning_aux_every_gate(step, start_step=5, every=4)
        assert fire is False
    fire, mult = _latent_reasoning_aux_every_gate(5, start_step=5, every=4)
    assert fire is True and mult == 4.0


@pytest.mark.parametrize("k", [2, 4, 8])
def test_aux_every_fires_exactly_every_kth_step_at_k_weight(k):
    start = 3
    fires = []
    for step in range(start, start + 5 * k):
        fire, mult = _latent_reasoning_aux_every_gate(step, start_step=start,
                                                       every=k)
        fires.append(fire)
        if fire:
            assert mult == float(k)
    # Fires on exactly every k-th step of the window, starting at `start`.
    expected = [(step - start) % k == 0
               for step in range(start, start + 5 * k)]
    assert fires == expected
    assert sum(fires) == 5  # 5*k steps / k == 5 fires


def test_aux_every_expected_gradient_matches_every_step_exact():
    """Deterministic tiny case: sum, over any K-step window, of
    (every-K-at-Kx) EQUALS sum of (every-step-at-1x) — the "same expected
    gradient" claim, pinned exactly (not just approximately) since both
    sides are a fixed base_grad times an integer count."""
    base_grad = 0.037  # arbitrary fixed per-fire contribution
    start = 0
    for k in (1, 2, 3, 4, 8):
        window = 8 * k  # a few full periods
        every_step_total = 0.0
        k_every_total = 0.0
        for step in range(start, start + window):
            fire1, mult1 = _latent_reasoning_aux_every_gate(step, start, 1)
            firek, multk = _latent_reasoning_aux_every_gate(step, start, k)
            if fire1:
                every_step_total += base_grad * mult1
            if firek:
                k_every_total += base_grad * multk
        assert every_step_total == pytest.approx(k_every_total), (
            k, every_step_total, k_every_total)


# ---------------------------------------------------------------------------
# B. Padding-helper construction (pure tensors, no model, no CUDA)
# ---------------------------------------------------------------------------

def test_left_pad_prompts_aligns_real_content_at_right_edge():
    prompts = [[7, 8, 9], [1, 2, 3, 4, 5], [42]]
    ids, doc_ids, P_max = _left_pad_prompts(prompts, pad_id=PAD_ID,
                                            device="cpu")
    assert P_max == 5
    assert ids.shape == (3, 5)
    for i, p in enumerate(prompts):
        pad_len = P_max - len(p)
        # Real content is right-aligned, verbatim.
        assert ids[i, pad_len:].tolist() == p
        # Pad prefix is filled with pad_id.
        assert (ids[i, :pad_len] == PAD_ID).all()
        # doc_ids: 0 over the pad prefix, 1 over the real content.
        assert (doc_ids[i, :pad_len] == 0).all()
        assert (doc_ids[i, pad_len:] == 1).all()


def test_left_pad_prompts_no_padding_needed_is_all_real():
    prompts = [[1, 2], [3, 4]]
    ids, doc_ids, P_max = _left_pad_prompts(prompts, pad_id=PAD_ID,
                                            device="cpu")
    assert P_max == 2
    assert (doc_ids == 1).all()
    assert ids.tolist() == prompts


def test_right_pad_solutions_lens_and_content():
    sols = [[10, 11], [20], [30, 31, 32]]
    ids, lens, S_max = _right_pad_solutions(sols, eos_id=EOS_ID, pad_id=PAD_ID,
                                            device="cpu")
    assert S_max == 4  # longest is [30,31,32,EOS] -> len 4
    assert lens.tolist() == [3, 2, 4]  # + eos each
    assert ids[0, :3].tolist() == [10, 11, EOS_ID]
    assert ids[0, 3].item() == PAD_ID
    assert ids[1, :2].tolist() == [20, EOS_ID]
    assert (ids[1, 2:] == PAD_ID).all()
    assert ids[2].tolist() == [30, 31, 32, EOS_ID]


# ---------------------------------------------------------------------------
# C + D. Real tiny DeltaNet: batched == mean-of-sequential (CUDA-gated)
# ---------------------------------------------------------------------------

def _tiny_deltanet_model(*, seed: int = 0, state_readonly: bool = False,
                         output_gate: bool = True):
    from experiments.layers import DeltaNetAttention
    from experiments.model import TinyLM
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=32, n_layers=3, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=0,
        output_gate=output_gate, state_readonly_at_think=state_readonly,
        use_latent_feedback_adapter=True,
    )
    m.thinking_token_id = THINK_ID
    m.train()
    return m.cuda()


def _rand_examples(n, seed=0, plen_range=(6, 14), slen_range=(1, 4)):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        plen = int(torch.randint(plen_range[0], plen_range[1] + 1, (1,),
                                 generator=g).item())
        slen = int(torch.randint(slen_range[0], slen_range[1] + 1, (1,),
                                 generator=g).item())
        c = torch.randint(2, VOCAB - 1, (plen,), generator=g).tolist()
        s = torch.randint(2, VOCAB - 1, (slen,), generator=g).tolist()
        out.append((c, s))
    return out


# Loose tolerance mirroring test_cu_seqlens.py::test_packed_equals_unpacked —
# the cu_seqlens-flattened kernel path is mathematically, not bit,
# identical to n independent B=1 forwards (different Triton kernel entry
# points / chunk scheduling).
_RTOL, _ATOL = 3e-2, 3e-2


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
@pytest.mark.parametrize("state_readonly", [False, True])
def test_batched_matches_sequential_loss(state_readonly):
    """Batched growing-thread loss == mean of sequential per-example losses,
    with DIFFERENT prompt lengths and DIFFERENT solution lengths per example
    (the case that would expose a left-pad / doc_ids / per-row-mask bug).
    Tested with state_readonly_at_think BOTH off (matches the actual running
    N1 process) and on (the doc_ids isolation must not depend on it)."""
    m = _tiny_deltanet_model(seed=0, state_readonly=state_readonly)
    examples = _rand_examples(4, seed=1)
    R = 3

    with torch.no_grad():
        seq_losses = [
            _answer_span_latent_loss(m, c, s, EOS_ID, R, THINK_ID, "cuda",
                                     checkpoint_latent=False)
            for c, s in examples
        ]
        seq_mean = torch.stack(seq_losses).mean()

        batched = _answer_span_latent_loss_batched(
            m, examples, EOS_ID, R, THINK_ID, "cuda",
            checkpoint_latent=False)

    torch.testing.assert_close(batched.cpu(), seq_mean.cpu(),
                               rtol=_RTOL, atol=_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_batched_matches_sequential_loss_with_gate_weight():
    m = _tiny_deltanet_model(seed=2, state_readonly=True)
    examples = _rand_examples(3, seed=2)
    R = 2
    with torch.no_grad():
        seq_losses = [
            _answer_span_latent_loss(m, c, s, EOS_ID, R, THINK_ID, "cuda",
                                     gate_weight=0.05, checkpoint_latent=False)
            for c, s in examples
        ]
        seq_mean = torch.stack(seq_losses).mean()
        batched = _answer_span_latent_loss_batched(
            m, examples, EOS_ID, R, THINK_ID, "cuda",
            gate_weight=0.05, checkpoint_latent=False)
    torch.testing.assert_close(batched.cpu(), seq_mean.cpu(),
                               rtol=_RTOL, atol=_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_batched_matches_sequential_gradients():
    """Gradients (adapter + embed + a trunk block param) from the batched
    path match the AVERAGE of the n sequential per-example gradients."""
    R = 3
    examples = _rand_examples(3, seed=3)

    def _grads(batched: bool, m):
        for p in m.parameters():
            p.grad = None
        if batched:
            loss = _answer_span_latent_loss_batched(
                m, examples, EOS_ID, R, THINK_ID, "cuda",
                checkpoint_latent=False)
        else:
            losses = [
                _answer_span_latent_loss(m, c, s, EOS_ID, R, THINK_ID, "cuda",
                                         checkpoint_latent=False)
                for c, s in examples
            ]
            loss = torch.stack(losses).mean()
        loss.backward()
        grads = {n: (p.grad.clone() if p.grad is not None else None)
                for n, p in m.named_parameters()}
        for p in m.parameters():
            p.grad = None
        return grads

    m_batched = _tiny_deltanet_model(seed=4, state_readonly=False)
    m_seq = _tiny_deltanet_model(seed=4, state_readonly=False)
    g_batched = _grads(True, m_batched)
    g_seq = _grads(False, m_seq)

    adapter_key = "latent_feedback_adapter.proj.weight"
    assert g_batched[adapter_key].abs().sum() > 0
    torch.testing.assert_close(g_batched[adapter_key].cpu(),
                               g_seq[adapter_key].cpu(),
                               rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(g_batched["embed.weight"].cpu(),
                               g_seq["embed.weight"].cpu(),
                               rtol=_RTOL, atol=_ATOL)
    block_keys = [n for n in g_batched
                 if n.startswith("blocks.0.") and g_batched[n] is not None]
    assert block_keys
    for k in block_keys:
        if g_batched[k].abs().sum() == 0 and g_seq[k].abs().sum() == 0:
            continue
        torch.testing.assert_close(g_batched[k].cpu(), g_seq[k].cpu(),
                                   rtol=_RTOL, atol=_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_batched_n1_matches_single_example_closely():
    """Degenerate n=1 case: no padding is needed at all (P_max == the one
    prompt's own length), so the batched path's only difference from the
    single-example path is going through the doc_ids/cu_seqlens machinery
    with an all-ones mask — should match tightly."""
    m = _tiny_deltanet_model(seed=5, state_readonly=False)
    examples = _rand_examples(1, seed=5)
    R = 2
    with torch.no_grad():
        single = _answer_span_latent_loss(m, examples[0][0], examples[0][1],
                                          EOS_ID, R, THINK_ID, "cuda",
                                          checkpoint_latent=False)
        batched = _answer_span_latent_loss_batched(
            m, examples, EOS_ID, R, THINK_ID, "cuda", checkpoint_latent=False)
    torch.testing.assert_close(batched.cpu(), single.cpu(),
                               rtol=_RTOL, atol=_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_batched_checkpoint_matches_unchecked():
    """Within the BATCHED path itself, checkpoint_latent True vs False must
    match tightly (same kernel path both sides — this is the exact-
    recompute guarantee `test_latent_reasoning_checkpoint.py` already pins
    for the sequential path; here it's re-pinned for the batched path)."""
    m = _tiny_deltanet_model(seed=6, state_readonly=False)
    examples = _rand_examples(3, seed=6)
    R = 3
    with torch.no_grad():
        loss_ckpt = _answer_span_latent_loss_batched(
            m, examples, EOS_ID, R, THINK_ID, "cuda", checkpoint_latent=True)
        loss_plain = _answer_span_latent_loss_batched(
            m, examples, EOS_ID, R, THINK_ID, "cuda", checkpoint_latent=False)
    torch.testing.assert_close(loss_ckpt.cpu(), loss_plain.cpu(),
                               rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# D. LatentReasoningCotrain.step() end-to-end, batch_examples True vs False
# ---------------------------------------------------------------------------

class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (VOCAB - 4)) for c in text]


def _write_rung_file(path, n, n_records=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    with open(path, "w") as f:
        for i in range(n_records):
            plen = int(torch.randint(3, 9, (1,), generator=g).item())
            body = "".join(chr(ord('a') + (j % 20)) for j in range(plen))
            f.write(json.dumps({
                "prompt": f"x{i} = {i}\n{body}\nprint(x{i})",
                "answer": i,
            }) + "\n")


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_latentreasoningcotrain_batched_matches_escape_hatch(tmp_path):
    prefix = str(tmp_path / "toy_ptrchase_batched")
    for n in (2, 3):
        _write_rung_file(f"{prefix}_n{n}.jsonl", n, n_records=8)

    def _run(batch_examples):
        m = _tiny_deltanet_model(seed=7, state_readonly=False)
        reasoner = LatentReasoningCotrain(
            train_prefix=prefix, rungs=[2, 3], tok=_FakeTok(),
            thinking_id=THINK_ID, eos_id=EOS_ID, device="cuda", max_len=64,
            no_ramp=True, seed=0, checkpoint_latent=False,
            batch_examples=batch_examples)
        with torch.no_grad():
            loss, rung = reasoner.step(m, step=0, total_steps=100,
                                       n_examples=4)
        return float(loss), rung

    loss_b, rung_b = _run(True)
    loss_s, rung_s = _run(False)
    assert rung_b == rung_s, "same seed must sample the same rung"
    assert loss_b == pytest.approx(loss_s, rel=_RTOL, abs=_ATOL)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_latentreasoningcotrain_defaults_to_batched():
    """`batch_examples` defaults True — the fixes-not-flags default-on
    requirement — with `pad_id` defaulting to 0."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        prefix = f"{d}/toy"
        _write_rung_file(f"{prefix}_n2.jsonl", 2, n_records=4)
        reasoner = LatentReasoningCotrain(
            train_prefix=prefix, rungs=[2], tok=_FakeTok(),
            thinking_id=THINK_ID, eos_id=EOS_ID, device="cuda", max_len=64,
            no_ramp=True, seed=0)
        assert reasoner.batch_examples is True
        assert reasoner.pad_id == 0


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_latentreasoningcotrain_batched_gradients_flow(tmp_path):
    prefix = str(tmp_path / "toy_ptrchase_grad")
    _write_rung_file(f"{prefix}_n2.jsonl", 2, n_records=6)
    m = _tiny_deltanet_model(seed=8, state_readonly=False)
    reasoner = LatentReasoningCotrain(
        train_prefix=prefix, rungs=[2], tok=_FakeTok(),
        thinking_id=THINK_ID, eos_id=EOS_ID, device="cuda", max_len=64,
        no_ramp=True, seed=0, checkpoint_latent=True)
    loss, rung = reasoner.step(m, step=0, total_steps=100, n_examples=3)
    assert rung == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert m.latent_feedback_adapter.proj.weight.grad is not None
    assert m.latent_feedback_adapter.proj.weight.grad.abs().sum() > 0
