"""Tests for the state-cartridges eval harness (experiments/eval_state_cartridges.py).

Runs entirely on CPU. FLA's real DeltaNet is a CUDA-only Triton kernel, so the
state-injection equivalence test uses the SAME CPU-runnable stub attention as
test_meta_ttt_train.py (element-wise recurrence implementing the FLA cache
protocol) — reused directly so the injection path is validated on the identical
substrate the sibling harness is. A CUDA-guarded real-DeltaNet check is included
but skipped without a GPU (GPU use is out of scope here).

The critical test is `test_initial_state_equivalence`: running [context + task]
sequentially must equal running the task with the context's final recurrent
state injected as the initial state — this is the load-bearing correctness
guarantee of the cartridge arm.
"""
from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from experiments.eval_repo_adaptive import (
    arm_ce, context_ids, task_line_span_tokens,
)
from experiments.gen_repo_episodes import perline_ids
from experiments.eval_state_cartridges import (
    aggregate,
    arm_ce_injected,
    build_context_lines,
    cartridge_merged_state,
    eval_episode,
    ingest_segment,
    mean_merge_states,
    segment_lines,
    shuffle_segment_lines,
    _episode_seed,
)
# Reuse the validated CPU stub + fake tokenizer + episode builder.
from experiments.test_meta_ttt_train import (
    FakeCharTokenizer, VOCAB, _build_stub_model, _make_episode,
)


# =========================================================================== #
# 1. Initial-state injection equivalence (THE critical test).
# =========================================================================== #

def test_initial_state_equivalence():
    """CE(sequential [context + task]) == CE(task with initial_state =
    state(context)), at the SAME task-line positions. Validates the injection
    path end-to-end (ingest_segment -> snapshot -> states_to_cache -> injected
    block-stack forward)."""
    model = _build_stub_model(n_layers=3, d_model=16)
    model.eval()
    torch.manual_seed(7)
    context = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(50)]
    task_prefix = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(9)]
    line = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(6)]
    span = [1, 2, 3]

    seq = arm_ce(model, context + task_prefix, line, span, "cpu", bf16=False)
    st, seen = ingest_segment(model, context, "cpu", bf16=False)
    inj = arm_ce_injected(model, st, seen, task_prefix, line, span,
                          "cpu", bf16=False)

    assert seen == len(context)
    assert abs(seq["line_ce"] - inj["line_ce"]) < 1e-4, (seq, inj)
    assert abs(seq["span_ce"] - inj["span_ce"]) < 1e-4, (seq, inj)


def test_cartridge_k1_equals_sequential():
    """The full cartridge pipeline at K=1 (whole context = one segment,
    mean-merge-of-one, inject) reproduces the sequential arm — a second, higher-
    level check of the same guarantee through the production code path."""
    model = _build_stub_model(n_layers=2, d_model=16)
    model.eval()
    tok = FakeCharTokenizer()
    ep = _make_episode("e1", ["def helper(a, b):\n    return a + b\n" * 4],
                       "x = 1\ny = 2\nr = ", "helper(x, y)\n", "helper",
                       n_ctx=6000, bucket="4-8k")
    task_prefix_ids = perline_ids(ep["task_prefix"], tok)
    line_ids, span_idx = task_line_span_tokens(ep, tok)
    ctx = context_ids(ep, tok)

    seq = arm_ce(model, ctx + task_prefix_ids, line_ids, span_idx, "cpu", False)

    lines = build_context_lines(ep, tok)
    segs = segment_lines(lines, 1)
    merged, seen, n_used = cartridge_merged_state(model, segs, "cpu", False)
    cart = arm_ce_injected(model, merged, seen, task_prefix_ids, line_ids,
                           span_idx, "cpu", False)
    assert n_used == 1
    assert abs(seq["line_ce"] - cart["line_ce"]) < 1e-4, (seq, cart)


# =========================================================================== #
# 2. Merge correctness.
# =========================================================================== #

def _mk_state_list(recurrents, conv=None):
    """Build a per-layer state-dict list from a list of recurrent tensors."""
    return [{"recurrent_state": r, "conv_state": conv,
             "attn_state": None, "ffn_state": None} for r in recurrents]


def test_merge_mean_of_identical_is_that_state():
    torch.manual_seed(0)
    r = torch.randn(1, 2, 4, 4)
    states = [_mk_state_list([r.clone()]) for _ in range(4)]
    merged = mean_merge_states(states)
    assert torch.allclose(merged[0]["recurrent_state"], r, atol=1e-6)


def test_merge_mean_is_arithmetic_mean():
    torch.manual_seed(1)
    a = torch.randn(1, 2, 4, 4)
    b = torch.randn(1, 2, 4, 4)
    c = torch.randn(1, 2, 4, 4)
    merged = mean_merge_states(
        [_mk_state_list([a]), _mk_state_list([b]), _mk_state_list([c])])
    assert torch.allclose(merged[0]["recurrent_state"], (a + b + c) / 3.0,
                          atol=1e-6)


def test_merge_conv_from_last_segment():
    a_conv = (torch.randn(1, 3, 2), torch.randn(1, 3, 2), torch.randn(1, 3, 2))
    b_conv = (torch.randn(1, 3, 2), torch.randn(1, 3, 2), torch.randn(1, 3, 2))
    r = torch.randn(1, 2, 4, 4)
    sa = _mk_state_list([r.clone()], conv=a_conv)
    sb = _mk_state_list([r.clone()], conv=b_conv)
    merged = mean_merge_states([sa, sb], conv_from="last")
    for got, want in zip(merged[0]["conv_state"], b_conv):
        assert torch.equal(got, want)                    # last (b) segment's conv
    merged_first = mean_merge_states([sa, sb], conv_from="first")
    for got, want in zip(merged_first[0]["conv_state"], a_conv):
        assert torch.equal(got, want)


def test_merge_output_is_mutation_isolated():
    """Merged tensors are fresh copies — mutating an input state does not touch
    the merge (FLA updates state in place during the task forward)."""
    r = torch.ones(1, 2, 4, 4)
    states = [_mk_state_list([r.clone()]), _mk_state_list([r.clone()])]
    merged = mean_merge_states(states)
    states[0][0]["recurrent_state"].add_(100.0)
    assert torch.allclose(merged[0]["recurrent_state"], torch.ones(1, 2, 4, 4),
                          atol=1e-6)


# =========================================================================== #
# 3. Segmentation.
# =========================================================================== #

def _lines_of(counts):
    """Build per-line id chunks with the given token counts (distinct ids)."""
    lines, nxt = [], 1
    for c in counts:
        lines.append(list(range(nxt, nxt + c)))
        nxt += c
    return lines


def test_segmentation_covers_all_tokens_exactly_once():
    lines = _lines_of([5] * 20)                            # 20 lines, 5 tok each
    flat = [t for ln in lines for t in ln]
    for K in (1, 2, 3, 4, 7, 8):
        segs = segment_lines(lines, K)
        seg_flat = [t for seg in segs for ln in seg for t in ln]
        assert seg_flat == flat, K                        # order + coverage exact
        assert all(len(seg) >= 1 for seg in segs), K      # every segment non-empty
        assert len(segs) == min(K, len(lines)), K


def test_segmentation_balanced_and_respects_2k_sizing():
    # ~8000 tokens (800 lines x 10) split into K=4 -> ~2000-token segments.
    lines = _lines_of([10] * 800)
    segs = segment_lines(lines, 4)
    sizes = [sum(len(ln) for ln in seg) for seg in segs]
    assert len(segs) == 4
    for s in sizes:
        assert 1500 <= s <= 2500, sizes                   # ~2k, balanced
    assert sum(sizes) == 8000


def test_segmentation_caps_K_at_line_count():
    lines = _lines_of([4, 4, 4])                           # only 3 lines
    segs = segment_lines(lines, 8)
    assert len(segs) == 3
    assert all(len(seg) == 1 for seg in segs)


def test_segmentation_empty_context():
    assert segment_lines([], 4) == []


# =========================================================================== #
# 4. Line-shuffle control.
# =========================================================================== #

def test_shuffle_preserves_per_segment_token_multiset():
    lines = _lines_of([3, 4, 2, 5, 6, 1, 7, 8])
    segs = segment_lines(lines, 3)
    rng = random.Random(123)
    shuf = shuffle_segment_lines(segs, rng)
    assert len(shuf) == len(segs)
    for seg, seg_sh in zip(segs, shuf):
        orig = sorted(t for ln in seg for t in ln)
        got = sorted(t for ln in seg_sh for t in ln)
        assert orig == got                                # multiset preserved
        assert sum(len(ln) for ln in seg) == sum(len(ln) for ln in seg_sh)


def test_shuffle_deterministic_per_seed():
    lines = _lines_of([3, 4, 2, 5, 6, 1])
    segs = segment_lines(lines, 2)
    s1 = _episode_seed(0, "ep-abc", 2)
    s2 = _episode_seed(0, "ep-abc", 2)
    s3 = _episode_seed(1, "ep-abc", 2)
    assert s1 == s2 and s1 != s3
    a = shuffle_segment_lines(segs, random.Random(s1))
    b = shuffle_segment_lines(segs, random.Random(s2))
    assert a == b                                         # deterministic


# =========================================================================== #
# 5. none-arm == plain forward (byte-identical), and context-line partition.
# =========================================================================== #

def test_none_arm_equals_plain_forward():
    model = _build_stub_model(n_layers=2, d_model=16)
    model.eval()
    torch.manual_seed(3)
    task_prefix = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(8)]
    line = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(5)]
    span = [0, 2]
    P, L = len(task_prefix), len(line)

    # Plain forward reference.
    x = torch.tensor([task_prefix + line], dtype=torch.long)
    with torch.no_grad():
        full_logits = model(x)                            # (1, T, V)
    pred = list(range(P - 1, P + L - 1))
    ref_logits = full_logits[0, pred, :]
    tgt = torch.tensor(line, dtype=torch.long)
    ref_line_ce = F.cross_entropy(ref_logits.float(), tgt).item()

    none = arm_ce(model, task_prefix, line, span, "cpu", bf16=False)
    inj_none = arm_ce_injected(model, None, 0, task_prefix, line, span,
                               "cpu", bf16=False)
    assert abs(none["line_ce"] - ref_line_ce) < 1e-5      # none arm == plain fwd
    assert abs(inj_none["line_ce"] - ref_line_ce) < 1e-5  # injected-None == plain


def test_context_lines_flatten_equals_context_ids():
    """Segmentation partitions EXACTLY the token stream the eval scores: the
    flattened per-line chunks equal eval_repo_adaptive.context_ids()."""
    tok = FakeCharTokenizer()
    ep = _make_episode(
        "e2",
        ["import os\ndef a():\n    return 1\n",
         "class B:\n    x = 2\n    def m(self):\n        return self.x\n"],
        "y = 1\nr = ", "a() + B().m()\n", "a", n_ctx=6000, bucket="4-8k")
    lines = build_context_lines(ep, tok)
    flat = [t for ln in lines for t in ln]
    assert flat == context_ids(ep, tok)


# =========================================================================== #
# 6. End-to-end eval_episode + aggregate on the stub.
# =========================================================================== #

def test_eval_episode_and_aggregate_stub():
    model = _build_stub_model(n_layers=2, d_model=16)
    model.eval()
    tok = FakeCharTokenizer()
    ctx = "def helper(a, b):\n    return a + b\n" * 8
    results = []
    for i in range(4):
        ep = _make_episode(f"e{i}", [ctx], "x = 1\ny = 2\nr = ",
                           "helper(x, y)\n", "helper",
                           n_ctx=6000, bucket="4-8k")
        r = eval_episode(model, ep, tok, "cpu", False, [2, 4], seed=0,
                         ep_index=i)
        results.append(r)
        # every arm present with CE fields
        for arm in ("sequential", "none", "cartridge@2", "shuffled@2",
                    "cartridge@4", "shuffled@4"):
            assert arm in r["arms"]
            assert r["arms"][arm]["line_ce"] is not None
        assert r["n_segments"]["2"] in (1, 2)

    agg = aggregate(results, [2, 4], n_boot=200, seed=0)
    assert agg["n_episodes"] == 4
    assert "sanity_gate_pass" in agg["sanity"]
    for K in (2, 4):
        d = agg["retention"][str(K)]["line_ce"]
        assert d["n"] >= 1
        # retention is a finite number or None; CI is a 2-list
        assert isinstance(d["ci95"], list) and len(d["ci95"]) == 2


# =========================================================================== #
# CUDA-only: real DeltaNet injection equivalence (skipped without a GPU).
# =========================================================================== #

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="FLA DeltaNet is a CUDA-only Triton kernel")
def test_initial_state_equivalence_real_deltanet():
    from experiments.model import TinyLM
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    model = TinyLM(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                   d_head=16, d_ff=64, attention_cls=DeltaNetAttention,
                   max_T=0).cuda().eval()
    context = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(300)]
    task_prefix = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(40)]
    line = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(8)]
    span = [1, 2, 3]
    seq = arm_ce(model, context + task_prefix, line, span, "cuda", bf16=True)
    st, seen = ingest_segment(model, context, "cuda", bf16=True)
    inj = arm_ce_injected(model, st, seen, task_prefix, line, span,
                          "cuda", bf16=True)
    assert abs(seq["line_ce"] - inj["line_ce"]) < 5e-2, (seq, inj)
