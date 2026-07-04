"""Tests for the DENSE per-hop supervision in latent_reasoning_cotrain
(2026-07-04, the exec-trace program).

The N1 stage-1 methodology bug: the latent-reasoning co-train's loss was
ANSWER-ONLY — it never consumed the record's `intermediates` field, so the
"per-step supervision fixes latent credit assignment" hypothesis was never
actually trained (N0/N1 ran at the equivalent of --latent_reasoning_perhop_
weight 0). This adds the per-hop CE term that decodes latent step j -> the
j-th intermediate f^j(s), mirroring latent_arith_real.latent_perhop_loss's
exact position/head convention.

Sections:
  A. convention-pin — the per-hop term extracted from `_answer_span_latent_loss`
     equals `latent_arith_real.latent_perhop_loss` on identical inputs (fp32).
  B. off-by-one pin — think slot j (1-indexed, position P+j-1) supervises
     intermediates[j-1]; hand-constructed asymmetric intermediates.
  C. escape hatch — perhop_weight=0.0 is byte-identical to the answer-only path
     (loss + gradients), even when inter_ids are present.
  D. gradient isolation — the per-hop term ALONE (answer term untouched) drives
     gradient into the latent feedback adapter.
  E. `_load_rung` single-token-contract enforcement (skip+count, <50% error).
  F. batched per-hop == mean-of-sequential per-hop, with unequal prompt/solution
     lengths AND unequal intermediates counts (CUDA-gated: the left-pad +
     doc_ids state isolation the batched path uses is DeltaNet/Triton-only).

CPU sections (A-E) use SoftmaxAttention (causal, bit-exact => the trailing
answer span is masked out of the think-slot logits, so the answer-span forward
and latent_perhop_loss's shorter forward agree to fp32). Section F needs the
real DeltaNet + cu_seqlens machinery, mirroring test_latent_reasoning_batched.py.

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
    experiments/test_latent_reasoning_perhop.py -v
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.latent_reasoning_cotrain import (
    LatentReasoningCotrain,
    _answer_span_latent_loss,
    _answer_span_latent_loss_batched,
    _load_rung,
)

THINK_ID = 5   # != PAD_ID (0)
PAD_ID = 0
EOS_ID = 1
VOCAB = 64
D_MODEL = 32


def _tiny_cpu_model(*, seed: int = 0) -> TinyLM:
    """CPU SoftmaxAttention TinyLM — CLEAN config (no gist, no WM, no film,
    no activation-checkpointing) so clean_latent_thread is a no-op and the
    think-slot logits are bit-exact regardless of the trailing answer span."""
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=2, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=256,
        output_gate=True, state_readonly_at_think=True,
        use_latent_feedback_adapter=True,
    )
    m.thinking_token_id = THINK_ID
    m.train()
    return m


def _example(seed: int = 0, plen: int = 10, slen: int = 3):
    g = torch.Generator().manual_seed(seed)
    comment_ids = torch.randint(2, VOCAB - 1, (plen,), generator=g).tolist()
    sol_ids = torch.randint(2, VOCAB - 1, (slen,), generator=g).tolist()
    return comment_ids, sol_ids


def _manual_perhop(model, comment_ids, inter_ids, R):
    """Position-EXPLICIT reference for the per-hop convention: build the latent
    thread exactly like the loss helpers, then read the UNSHIFTED logits at
    think-slot position ``P + j`` (0-indexed j) and CE them against
    ``inter_ids[j]``. Writing ``P + j`` / ``inter_ids[j]`` out by hand is the
    off-by-one pin (an implementation that used ``P + j + 1`` or ``inter[j+1]``
    would mismatch this)."""
    base_ids = torch.tensor([comment_ids], dtype=torch.long)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    P = len(comment_ids)
    think = torch.full((1, 1), THINK_ID, dtype=torch.long)
    for _ in range(R):
        out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        h = out[1]
        z = h[:, -1:, :].to(cur_emb.dtype)
        z = model.apply_latent_feedback_adapter(z).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
    logits = out[0] if isinstance(out, tuple) else out
    ces = []
    for j in range(R):
        pos = P + j                        # slot j+1 (1-indexed) is at pos P+j
        tgt = torch.tensor([inter_ids[j]], dtype=torch.long)
        ces.append(F.cross_entropy(logits[:, pos, :], tgt))
    return torch.stack(ces).mean()


# ---------------------------------------------------------------------------
# A. Convention-pin: our per-hop term == latent_arith_real.latent_perhop_loss
# ---------------------------------------------------------------------------

def test_perhop_matches_latent_arith_real_convention():
    from experiments.latent_arith_real import latent_perhop_loss
    m = _tiny_cpu_model(seed=0)
    comment_ids, sol_ids = _example(seed=0, plen=9, slen=3)
    R = 3
    inter_ids = [3, 7, 11]  # single-token ids, len == R
    with torch.no_grad():
        ref = latent_perhop_loss(m, comment_ids, inter_ids, R, THINK_ID, "cpu")
        # Extract JUST the per-hop component from the answer-span loss helper.
        _, _ans, perhop = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=inter_ids, perhop_weight=1.0,
            return_components=True)
    assert torch.allclose(perhop, ref, atol=1e-5, rtol=1e-4), (
        float(perhop), float(ref))


def test_perhop_matches_manual_position_explicit_reference():
    m = _tiny_cpu_model(seed=1)
    comment_ids, sol_ids = _example(seed=1, plen=8, slen=2)
    R = 3
    inter_ids = [4, 9, 2]
    with torch.no_grad():
        manual = _manual_perhop(m, comment_ids, inter_ids, R)
        _, _ans, perhop = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=inter_ids, perhop_weight=1.0,
            return_components=True)
    assert torch.allclose(perhop, manual, atol=1e-5, rtol=1e-4), (
        float(perhop), float(manual))


# ---------------------------------------------------------------------------
# B. Off-by-one pin: order of intermediates matters (asymmetric targets).
# ---------------------------------------------------------------------------

def test_perhop_off_by_one_is_asymmetric():
    """With distinct per-step targets, reversing the intermediates order MUST
    change the loss — proving each slot is bound to a SPECIFIC intermediate
    (not e.g. an order-invariant bag), and pinning the P+j-1 <-> inter[j-1]
    direction against a shifted/reversed mis-mapping."""
    m = _tiny_cpu_model(seed=2)
    comment_ids, sol_ids = _example(seed=2, plen=8, slen=2)
    R = 3
    fwd = [3, 8, 14]
    rev = list(reversed(fwd))
    with torch.no_grad():
        m_fwd = _manual_perhop(m, comment_ids, fwd, R)
        m_rev = _manual_perhop(m, comment_ids, rev, R)
        _, _, p_fwd = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=fwd, perhop_weight=1.0,
            return_components=True)
        _, _, p_rev = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=rev, perhop_weight=1.0,
            return_components=True)
    # Order matters (asymmetric) AND our helper tracks the manual mapping for
    # both orderings.
    assert abs(float(m_fwd) - float(m_rev)) > 1e-4
    assert torch.allclose(p_fwd, m_fwd, atol=1e-5, rtol=1e-4)
    assert torch.allclose(p_rev, m_rev, atol=1e-5, rtol=1e-4)


def test_perhop_uses_available_intermediates_when_R_exceeds_len():
    """R longer than the available intermediates supervises only what exists
    (n_hops = min(R, len(inter))), matching a manual mean over those hops."""
    m = _tiny_cpu_model(seed=3)
    comment_ids, sol_ids = _example(seed=3, plen=8, slen=2)
    R = 4
    inter_ids = [6, 2]  # only 2 intermediates for a depth-4 latent thread
    with torch.no_grad():
        manual = _manual_perhop(m, comment_ids, inter_ids, len(inter_ids))
        _, _, perhop = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=inter_ids, perhop_weight=1.0,
            return_components=True)
    assert torch.allclose(perhop, manual, atol=1e-5, rtol=1e-4), (
        float(perhop), float(manual))


# ---------------------------------------------------------------------------
# C. Escape hatch: perhop_weight=0.0 is byte-identical to answer-only.
# ---------------------------------------------------------------------------

def test_perhop_weight_zero_is_byte_identical_loss_and_grads():
    comment_ids, sol_ids = _example(seed=4, plen=10, slen=3)
    R = 3
    inter_ids = [3, 7, 11]

    def _loss_and_grads(**perhop_kwargs):
        m = _tiny_cpu_model(seed=42)
        for p in m.parameters():
            p.grad = None
        loss = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, **perhop_kwargs)
        loss.backward()
        grads = {n: (p.grad.clone() if p.grad is not None else None)
                 for n, p in m.named_parameters()}
        return float(loss.detach()), grads

    # Old-style call (no per-hop args at all).
    loss_old, g_old = _loss_and_grads()
    # New call, perhop_weight=0 WITH intermediates present — must be identical.
    loss_zero, g_zero = _loss_and_grads(inter_ids=inter_ids, perhop_weight=0.0)

    assert loss_old == pytest.approx(loss_zero, abs=1e-6, rel=1e-6), (
        loss_old, loss_zero)
    assert set(g_old) == set(g_zero)
    compared = 0
    for name in g_old:
        a, b = g_old[name], g_zero[name]
        assert (a is None) == (b is None), name
        if a is None:
            continue
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), (
            name, float((a - b).abs().max()))
        compared += 1
    assert compared > 0


def test_perhop_weight_nonzero_changes_total_loss():
    """Sanity: a nonzero weight DOES move the total loss (perhop is actually
    wired into the sum, not silently dropped)."""
    m = _tiny_cpu_model(seed=5)
    comment_ids, sol_ids = _example(seed=5, plen=8, slen=2)
    R = 3
    inter_ids = [3, 7, 11]
    with torch.no_grad():
        loss0 = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=inter_ids, perhop_weight=0.0)
        loss1 = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=inter_ids, perhop_weight=1.0)
        _, ans, perhop = _answer_span_latent_loss(
            m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=inter_ids, perhop_weight=1.0,
            return_components=True)
    # total(weight=1) == total(weight=0) + 1.0 * perhop.
    assert torch.allclose(loss1, loss0 + perhop, atol=1e-5)
    assert float(perhop) > 0.0


# ---------------------------------------------------------------------------
# D. Gradient isolation: per-hop term ALONE drives the adapter.
# ---------------------------------------------------------------------------

def test_gradient_flows_to_adapter_through_perhop_alone():
    m = _tiny_cpu_model(seed=6)
    comment_ids, sol_ids = _example(seed=6, plen=9, slen=3)
    R = 3
    inter_ids = [4, 9, 2]
    for p in m.parameters():
        p.grad = None
    _total, _ans, perhop = _answer_span_latent_loss(
        m, comment_ids, sol_ids, EOS_ID, R, THINK_ID, "cpu",
        checkpoint_latent=False, inter_ids=inter_ids, perhop_weight=1.0,
        return_components=True)
    # Backprop ONLY the per-hop term (answer term never entered the graph we
    # backward through) — the adapter must still receive nonzero gradient.
    perhop.backward()
    g = m.latent_feedback_adapter.proj.weight.grad
    assert g is not None and g.abs().sum() > 0, \
        "per-hop term alone must drive gradient into the latent adapter"


# ---------------------------------------------------------------------------
# E. _load_rung single-token-contract enforcement (pure, no CUDA/model).
# ---------------------------------------------------------------------------

class _FakeTok:
    """Deterministic stand-in: each character -> one token, so str(v) is a
    SINGLE token iff v renders as one char (ints 0..9). str(42) / str(-3) are
    multi-token — the contract-violation the enforcement must catch."""

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (VOCAB - 4)) for c in text]


def _write_rung(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_rung_returns_inter_ids_when_single_token(tmp_path):
    path_prefix = str(tmp_path / "r")
    recs = [{"prompt": f"p{i}", "answer": i % 10,
             "intermediates": [i % 10, (i + 1) % 10]} for i in range(6)]
    _write_rung(f"{path_prefix}_n2.jsonl", recs)
    out = _load_rung(path_prefix, 2, _FakeTok(), max_len=64,
                     require_single_token_inter=True)
    assert len(out) == 6
    for c, s, inter in out:
        assert isinstance(inter, list) and len(inter) == 2
        assert all(isinstance(t, int) for t in inter)


def test_load_rung_skips_multitoken_and_errors_below_50pct(tmp_path):
    path_prefix = str(tmp_path / "bad")
    # 4 of 6 have a multi-token (>=10) intermediate => only 2/6 = 33% usable.
    recs = []
    for i in range(6):
        inter = [42, 3] if i < 4 else [3, 4]  # 42 -> "42" -> 2 tokens
        recs.append({"prompt": f"p{i}", "answer": 3, "intermediates": inter})
    _write_rung(f"{path_prefix}_n2.jsonl", recs)
    with pytest.raises(ValueError, match="single-token"):
        _load_rung(path_prefix, 2, _FakeTok(), max_len=64,
                   require_single_token_inter=True)


def test_load_rung_skip_count_above_threshold_ok(tmp_path):
    path_prefix = str(tmp_path / "ok")
    # 2 of 6 multi-token => 4/6 = 67% usable (>= 50%): loads the 4 usable ones.
    recs = []
    for i in range(6):
        inter = [42, 3] if i < 2 else [i % 10, (i + 1) % 10]
        recs.append({"prompt": f"p{i}", "answer": 3, "intermediates": inter})
    _write_rung(f"{path_prefix}_n2.jsonl", recs)
    out = _load_rung(path_prefix, 2, _FakeTok(), max_len=64,
                     require_single_token_inter=True)
    assert len(out) == 4


def test_load_rung_escape_hatch_keeps_all_examples(tmp_path):
    """require_single_token_inter=False (the perhop_weight=0 path) keeps EVERY
    length-passing example (byte-identical set to the old 2-tuple loader),
    inter_ids best-effort ([] for multi-token) and never enforced."""
    path_prefix = str(tmp_path / "esc")
    recs = []
    for i in range(6):
        inter = [42, 3] if i < 5 else [3, 4]  # 5/6 multi-token
        recs.append({"prompt": f"p{i}", "answer": 3, "intermediates": inter})
    _write_rung(f"{path_prefix}_n2.jsonl", recs)
    out = _load_rung(path_prefix, 2, _FakeTok(), max_len=64,
                     require_single_token_inter=False)
    assert len(out) == 6  # nothing skipped
    # Multi-token rows carry [] (best-effort, unused); the single-token row
    # carries its two ids.
    n_empty = sum(1 for _c, _s, inter in out if inter == [])
    assert n_empty == 5


def test_cotrain_class_default_is_answer_only(tmp_path):
    """Class-level default perhop_weight=0.0 => require=False => data with
    NO intermediates still loads (backwards-compat / N0-N1 reproduction)."""
    path_prefix = str(tmp_path / "nointer")
    recs = [{"prompt": f"x{i} = {i}\nprint(x{i})", "answer": i}
            for i in range(4)]
    _write_rung(f"{path_prefix}_n2.jsonl", recs)
    r = LatentReasoningCotrain(
        train_prefix=path_prefix, rungs=[2], tok=_FakeTok(),
        thinking_id=THINK_ID, eos_id=EOS_ID, device="cpu", max_len=64,
        no_ramp=True, seed=0)
    assert r.perhop_weight == 0.0
    assert len(r.data[2]) == 4


# ---------------------------------------------------------------------------
# F. Batched per-hop == mean-of-sequential per-hop (CUDA-gated DeltaNet).
# ---------------------------------------------------------------------------

def _tiny_deltanet_model(*, seed: int = 0, state_readonly: bool = False):
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=32, n_layers=3, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention, max_T=0,
        output_gate=True, state_readonly_at_think=state_readonly,
        use_latent_feedback_adapter=True,
    )
    m.thinking_token_id = THINK_ID
    m.train()
    return m.cuda()


def _perhop_examples(seed=0):
    """3 examples with unequal prompt lengths, unequal solution lengths, AND
    unequal intermediates counts (3, 1, 2) for a depth-R=3 thread."""
    g = torch.Generator().manual_seed(seed)

    def _rand(lo, hi):
        return int(torch.randint(lo, hi + 1, (1,), generator=g).item())

    inter_counts = [3, 1, 2]
    out = []
    for k in inter_counts:
        plen = _rand(6, 13)
        slen = _rand(1, 4)
        c = torch.randint(2, VOCAB - 1, (plen,), generator=g).tolist()
        s = torch.randint(2, VOCAB - 1, (slen,), generator=g).tolist()
        inter = torch.randint(2, VOCAB - 1, (k,), generator=g).tolist()
        out.append((c, s, inter))
    return out


_RTOL, _ATOL = 3e-2, 3e-2  # same loose bar as test_latent_reasoning_batched.py


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
@pytest.mark.parametrize("state_readonly", [False, True])
def test_batched_perhop_matches_mean_of_sequential(state_readonly):
    m = _tiny_deltanet_model(seed=0, state_readonly=state_readonly)
    examples = _perhop_examples(seed=1)
    R = 3
    with torch.no_grad():
        seq = [
            _answer_span_latent_loss(
                m, c, s, EOS_ID, R, THINK_ID, "cuda", checkpoint_latent=False,
                inter_ids=inter, perhop_weight=1.0, return_components=True)
            for c, s, inter in examples
        ]
        seq_total = torch.stack([t[0] for t in seq]).mean()
        seq_ans = torch.stack([t[1] for t in seq]).mean()
        seq_hop = torch.stack([t[2] for t in seq]).mean()

        b_total, b_ans, b_hop = _answer_span_latent_loss_batched(
            m, examples, EOS_ID, R, THINK_ID, "cuda", checkpoint_latent=False,
            perhop_weight=1.0, return_components=True)

    torch.testing.assert_close(b_total.cpu(), seq_total.cpu(),
                               rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(b_ans.cpu(), seq_ans.cpu(),
                               rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(b_hop.cpu(), seq_hop.cpu(),
                               rtol=_RTOL, atol=_ATOL)
    assert float(seq_hop) > 0.0  # per-hop actually contributes


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_batched_perhop_gradients_match_mean_of_sequential():
    R = 3
    examples = _perhop_examples(seed=2)

    def _grads(batched, m):
        for p in m.parameters():
            p.grad = None
        if batched:
            loss, _, _ = _answer_span_latent_loss_batched(
                m, examples, EOS_ID, R, THINK_ID, "cuda",
                checkpoint_latent=False, perhop_weight=1.0,
                return_components=True)
        else:
            losses = [
                _answer_span_latent_loss(
                    m, c, s, EOS_ID, R, THINK_ID, "cuda",
                    checkpoint_latent=False, inter_ids=inter,
                    perhop_weight=1.0)
                for c, s, inter in examples
            ]
            loss = torch.stack(losses).mean()
        loss.backward()
        return {n: (p.grad.clone() if p.grad is not None else None)
                for n, p in m.named_parameters()}

    g_b = _grads(True, _tiny_deltanet_model(seed=3, state_readonly=False))
    g_s = _grads(False, _tiny_deltanet_model(seed=3, state_readonly=False))
    ak = "latent_feedback_adapter.proj.weight"
    assert g_b[ak].abs().sum() > 0
    torch.testing.assert_close(g_b[ak].cpu(), g_s[ak].cpu(),
                               rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(g_b["embed.weight"].cpu(),
                               g_s["embed.weight"].cpu(),
                               rtol=_RTOL, atol=_ATOL)
