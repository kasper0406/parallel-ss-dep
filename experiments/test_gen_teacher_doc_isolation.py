"""Tests for cross-document teacher-context isolation in the offline KD logit
generators (audited defect fix, 2026-07-01).

CONTEXT: the student trains with cross-document state isolation
(`MixedSourceStream(emit_doc_ids=True)` + DeltaNet state reset at every
document boundary within a packed T-length block — see AGENTS.md
"Cross-document state isolation"). Both offline teacher-logit generators
(`gen_teacher_logits.py` HF reference, `gen_teacher_logits_vllm.py` production
vLLM path) used to condition the teacher on the FULL un-isolated packed block,
so stored KD targets after the first document in a block were conditioned on
context the student structurally cannot see. This file covers the fix:

- `gen_teacher_logits._doc_segments` / `.teacher_topk_doc_isolated`: per-doc
  segment splitting + per-segment teacher forward, stitched back into packed
  order.
    (a) a position inside the FIRST document is identical whether computed
        doc-isolated or over the full packed block (same prefix either way).
    (b) a position in a LATER document DIFFERS from the full-block forward
        (proof the isolation is not a no-op — this is the actual bug).
    (c) the stitched (topv, topi) preserve packed shape/order/length exactly.
- `gen_teacher_logits_vllm._topk_row`: the vocab-filter boundary excludes
  teacher ids >= len(tok) (padding rows / the student's [THINKING] slot),
  not just >= the round64-padded student MODEL vocab (the old, too-loose
  boundary that let a [THINKING]-slot id through as a stored "prediction").

CPU-only, no GPU / no download: builds a tiny 2-layer LlamaForCausalLM from an
in-code config. `gen_teacher_logits_vllm` is imported WITHOUT vllm installed —
`_topk_row`/`_doc_segments` have no vllm/torch dependency at module-import
time (vllm is only imported inside `main()`).

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_gen_teacher_doc_isolation.py -v
"""
from __future__ import annotations

import torch
import pytest

from experiments.gen_teacher_logits import _doc_segments, teacher_topk_doc_isolated
from experiments.gen_teacher_logits_vllm import _topk_row
from experiments.gen_teacher_logits_vllm import _doc_segments as _doc_segments_vllm


def _tiny_llama(vocab_size=64, hidden=32, layers=2, heads=4):
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=vocab_size, hidden_size=hidden,
        intermediate_size=hidden * 2, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=heads,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# _doc_segments (both copies — gen_teacher_logits.py and the vllm-file's
# self-contained duplicate must agree byte-for-byte).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [_doc_segments, _doc_segments_vllm])
class TestDocSegments:
    def test_single_document(self, fn):
        assert fn([0, 0, 0, 0]) == [(0, 4)]

    def test_multi_document(self, fn):
        assert fn([0, 0, 1, 1, 1, 2]) == [(0, 2), (2, 5), (5, 6)]

    def test_single_token(self, fn):
        assert fn([0]) == [(0, 1)]

    def test_empty(self, fn):
        assert fn([]) == []

    def test_accepts_tensor(self, fn):
        assert fn(torch.tensor([0, 0, 1, 1])) == [(0, 2), (2, 4)]

    def test_non_zero_based_ids_still_split_on_change(self, fn):
        # doc_ids need not start at 0 or be contiguous integers — only
        # value-change boundaries matter.
        assert fn([5, 5, 5, 2, 2, 9]) == [(0, 3), (3, 5), (5, 6)]


# ---------------------------------------------------------------------------
# teacher_topk_doc_isolated (gen_teacher_logits.py, HF reference path)
# ---------------------------------------------------------------------------

class TestTeacherTopkDocIsolated:
    def setup_method(self):
        torch.manual_seed(0)
        self.vocab = 64
        self.top_k = 5
        self.model = _tiny_llama(vocab_size=self.vocab)

    def _full_block_topk(self, x):
        with torch.no_grad():
            out = self.model(input_ids=x)
            return torch.topk(out.logits.float(), self.top_k, dim=-1)

    def test_first_doc_matches_full_block(self):
        """A position inside the FIRST document must be IDENTICAL whether
        computed doc-isolated or over the full packed block — the first
        document's causal prefix is the same either way, so this is a sanity
        check that isolation doesn't perturb positions it doesn't need to."""
        torch.manual_seed(1)
        T = 12
        x = torch.randint(0, self.vocab, (1, T))
        doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]])
        full_v, full_i = self._full_block_topk(x)
        topv, topi = teacher_topk_doc_isolated(
            self.model, x, doc_ids, top_k=self.top_k, teacher_vocab=self.vocab)
        assert torch.allclose(topv[:, :4], full_v[:, :4], atol=1e-4)
        assert torch.equal(topi[:, :4], full_i[:, :4])

    def test_later_docs_differ_from_full_block(self):
        """Positions in the SECOND (and later) document must DIFFER from the
        full-block conditioning — this is the actual bug being fixed: without
        isolation, the stored teacher target leaks context the student
        structurally cannot see across a doc_id boundary. If this test
        failed (doc-isolated == full-block everywhere), the isolation logic
        would be a silent no-op."""
        torch.manual_seed(2)
        T = 12
        x = torch.randint(0, self.vocab, (1, T))
        doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]])
        full_v, full_i = self._full_block_topk(x)
        topv, topi = teacher_topk_doc_isolated(
            self.model, x, doc_ids, top_k=self.top_k, teacher_vocab=self.vocab)
        diverged = ~torch.isclose(topv[:, 4:], full_v[:, 4:], atol=1e-4).all(dim=-1)
        assert diverged.any(), (
            "doc-isolated logits identical to full-block conditioning at "
            "EVERY position after the first document boundary — isolation "
            "is not taking effect")

    def test_stitch_preserves_shape_order_length(self):
        """The (topv, topi) returned by the doc-isolated path must be the
        SAME shape as a plain full-block torch.topk, and a single-document
        row (isolation is then a structural no-op) must match the full-block
        forward exactly — proving the per-segment stitching writes each
        segment's result back to the correct packed positions in order,
        rather than e.g. concatenating segments in the wrong order or
        dropping/duplicating positions."""
        torch.manual_seed(3)
        B, T, k = 3, 16, 4
        x = torch.randint(0, self.vocab, (B, T))
        doc_ids = torch.zeros(B, T, dtype=torch.long)
        for b in range(B):
            cuts = sorted(torch.randperm(T - 1)[:2].tolist())
            d = 0
            prev = 0
            for c in cuts + [T]:
                doc_ids[b, prev:c] = d
                d += 1
                prev = c
        topv, topi = teacher_topk_doc_isolated(
            self.model, x, doc_ids, top_k=k, teacher_vocab=self.vocab)
        assert topv.shape == (B, T, k)
        assert topi.shape == (B, T, k)

        # Single-document row (doc_ids all-0): the segment loop takes exactly
        # one full-length span, so this must equal the plain full-block
        # forward EXACTLY at every position, in the same order.
        single_doc_ids = torch.zeros(1, T, dtype=torch.long)
        with torch.no_grad():
            full_out = self.model(input_ids=x[:1])
            full_v, full_i = torch.topk(full_out.logits.float(), k, dim=-1)
        tv1, ti1 = teacher_topk_doc_isolated(
            self.model, x[:1], single_doc_ids, top_k=k, teacher_vocab=self.vocab)
        assert torch.allclose(tv1, full_v, atol=1e-4)
        assert torch.equal(ti1, full_i)

    def test_matches_manual_per_segment_forward(self):
        """Direct ground truth: manually run the teacher on each doc segment
        (batch=1, positions restarting at 0) and compare row-by-row to
        `teacher_topk_doc_isolated`'s output at the SAME packed positions."""
        torch.manual_seed(4)
        T, k = 10, 3
        x = torch.randint(0, self.vocab, (1, T))
        doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]])
        topv, topi = teacher_topk_doc_isolated(
            self.model, x, doc_ids, top_k=k, teacher_vocab=self.vocab)
        for s0, s1 in _doc_segments(doc_ids[0]):
            with torch.no_grad():
                seg_out = self.model(input_ids=x[:, s0:s1])
                v, i = torch.topk(seg_out.logits.float(), k, dim=-1)
            assert torch.allclose(topv[:, s0:s1], v, atol=1e-4)
            assert torch.equal(topi[:, s0:s1], i)


# ---------------------------------------------------------------------------
# _topk_row vocab-filter boundary (gen_teacher_logits_vllm.py)
# ---------------------------------------------------------------------------

class _FakeLogprob:
    def __init__(self, logprob, rank):
        self.logprob = logprob
        self.rank = rank


class TestVocabFilterBoundary:
    def test_excludes_ids_at_or_above_len_tok(self):
        # Simulate a padded teacher vocab: the real tokenizer has 10 ids, but
        # the teacher's lm_head exposes padding rows up to 16 (id 10 stands
        # in for the student's [THINKING] slot, which sits at len(tok)).
        len_tok = 10
        src = {i: _FakeLogprob(logprob=-float(i), rank=i + 1) for i in range(16)}
        ids, lps = _topk_row(src, top_k=6, vocab_bound=len_tok)
        assert all(i < len_tok for i in ids)
        assert ids == [0, 1, 2, 3, 4, 5]
        assert len(ids) == len(lps) == 6

    def test_old_boundary_would_have_leaked_padding_ids(self):
        """Regression guard: reproduce the OLD (buggy) call with the padded
        MODEL vocab as the boundary and confirm it used to let padding /
        [THINKING]-slot ids through — i.e. this test would have failed to
        show a difference under the fix if the fix were a no-op."""
        len_tok = 10
        padded_model_vocab = 16  # e.g. round64(151665+1) in production
        src = {i: _FakeLogprob(logprob=-float(i), rank=i + 1) for i in range(16)}
        ids_old_boundary, _ = _topk_row(src, top_k=12, vocab_bound=padded_model_vocab)
        assert any(i >= len_tok for i in ids_old_boundary), (
            "sanity precondition failed: the old (too-loose) boundary must "
            "be ABLE to leak padding ids for this regression test to mean "
            "anything")
        ids_new_boundary, _ = _topk_row(src, top_k=12, vocab_bound=len_tok)
        assert all(i < len_tok for i in ids_new_boundary)

    def test_pads_when_fewer_than_top_k_survive_filter(self):
        # Only 3 ids are < vocab_bound; top_k=6 must pad with the last valid
        # id at a very-low logprob (near-zero softmax weight).
        src = {0: _FakeLogprob(-0.1, 1), 1: _FakeLogprob(-0.2, 2),
               2: _FakeLogprob(-0.3, 3), 50: _FakeLogprob(-0.05, 4)}
        ids, lps = _topk_row(src, top_k=6, vocab_bound=10)
        assert ids == [0, 1, 2, 2, 2, 2]
        assert lps[3:] == [-30.0, -30.0, -30.0]

    def test_thinking_slot_id_specifically_excluded(self):
        # The motivating production case: id 151665 IS the student's
        # [THINKING] slot (== len(tok) for the Qwen tokenizer in use); it
        # must never be returned as a stored teacher "prediction".
        len_tok = 151665
        thinking_id = 151665
        student_padded_vocab = 151680
        src = {thinking_id: _FakeLogprob(-0.01, 1),   # highest-rank!
               5: _FakeLogprob(-1.0, 2)}
        ids_old, _ = _topk_row(src, top_k=2, vocab_bound=student_padded_vocab)
        assert thinking_id in ids_old, "sanity: old boundary keeps the think id"
        ids_new, _ = _topk_row(src, top_k=2, vocab_bound=len_tok)
        assert thinking_id not in ids_new
        assert 5 in ids_new
