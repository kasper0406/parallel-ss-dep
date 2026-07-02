"""Tests for the live-KD teacher cross-document isolation fix (2026-07-01)
and the d_ff/tie_embeddings checkpoint-cfg persistence fix.

Task 1 — doc isolation:
  The student forward is cross-document-isolated (doc_ids -> cu_seqlens
  resets DeltaNet state at every in-block document boundary), but the LIVE
  teacher forward in the KD branch of `_nonthink_forward_loss` used to
  condition on the whole packed T=2048 block, so teacher targets right after
  a doc boundary saw context the student never gets. Covered here:
    - `_same_doc_target_mask` masks EXACTLY the last token of each document
      (its target is the first token of the next, unrelated, document).
    - `_kd_valid_mask` folds the doc mask in with the existing -100 / thinking-
      token masks (both KD branches share this one helper — the "shared
      convention" the offline generators are expected to honour too).
    - `_kd_teacher_forward_doc_isolated`, verified against a REAL tiny
      LlamaForCausalLM on CPU: logits for the first document (no preceding
      context either way) match a naive full-block forward; logits for a
      LATER document match a fresh standalone forward of that document
      alone (the ground truth for "what a doc-isolated forward looks
      like") and DIFFER from the naive full-block forward (proof the bug
      being fixed is real).
    - End-to-end through `_nonthink_forward_loss`: doc-aware KD runs, is
      finite/differentiable, and is NOT numerically identical to the naive
      (doc_ids=None) path when a boundary is present.

Task 2 — cfg persistence:
  d_ff and tie_embeddings are now saved into the ckpt cfg at every train_lm.py
  save site; `eval_bracket_structure.build_model_from_ckpt` consumes them
  (d_ff already had a shape-inference fallback; tie_embeddings did not exist
  at all before this fix -> a tied ckpt silently reconstructed UNTIED).

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_kd_doc_isolation.py -v
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.train_lm import (
    _nonthink_forward_loss,
    _same_doc_target_mask,
    _kd_valid_mask,
    _doc_segments,
    _kd_teacher_forward_doc_isolated,
)


# ===========================================================================
# _same_doc_target_mask
# ===========================================================================

def test_same_doc_target_mask_masks_exactly_doc_final_positions():
    # Row 0: docs [0,0,0 | 1,1,1,1 | 2,2] -> doc-final at index 2 (last '0')
    # and index 6 (last '1'). Row 1: docs [5,5 | 6,6,6,6,6,6] -> doc-final
    # at index 1 only.
    doc_ids = torch.tensor([
        [0, 0, 0, 1, 1, 1, 1, 2, 2],
        [5, 5, 6, 6, 6, 6, 6, 6, 6],
    ])
    mask = _same_doc_target_mask(doc_ids)
    expected = torch.ones_like(doc_ids, dtype=torch.bool)
    expected[0, 2] = False
    expected[0, 6] = False
    expected[1, 1] = False
    assert torch.equal(mask, expected)


def test_same_doc_target_mask_single_document_all_true():
    # No boundary anywhere in the block -> every position is "same doc as
    # target" (the ambiguous last column defaults True too).
    doc_ids = torch.zeros(2, 6, dtype=torch.long)
    mask = _same_doc_target_mask(doc_ids)
    assert bool(mask.all())


def test_same_doc_target_mask_last_column_defaults_true():
    # The chunk's very last input position's target doc id was never carried
    # through (x, y, doc_ids) -- documented default is "same doc" (True), so
    # a run of documents ending EXACTLY at the block boundary doesn't lose a
    # position it has no information about either way.
    doc_ids = torch.tensor([[0, 0, 1, 1, 1]])
    mask = _same_doc_target_mask(doc_ids)
    assert bool(mask[0, -1])


# ===========================================================================
# _kd_valid_mask
# ===========================================================================

def test_kd_valid_mask_combines_ignore_think_and_doc():
    B, T, think_id = 1, 6, 99
    y = torch.tensor([[1, 2, think_id, 4, -100, 6]])
    # doc boundary right after position 1 (index 1 is doc-final).
    doc_ids = torch.tensor([[0, 0, 1, 1, 1, 1]])
    mask = _kd_valid_mask(y, doc_ids, think_id)
    expected = torch.tensor([[True, False, False, True, False, True]])
    assert torch.equal(mask, expected)


def test_kd_valid_mask_doc_ids_none_is_byte_identical_to_pre_fix_mask():
    B, T, think_id = 1, 5, 99
    y = torch.tensor([[1, think_id, -100, 4, 5]])
    mask = _kd_valid_mask(y, None, think_id)
    expected = (y != -100) & (y != think_id)
    assert torch.equal(mask, expected)


def test_kd_valid_mask_no_thinking_token_id():
    y = torch.tensor([[1, -100, 3]])
    mask = _kd_valid_mask(y, None, None)
    assert torch.equal(mask, y != -100)


# ===========================================================================
# _doc_segments
# ===========================================================================

def test_doc_segments_partitions_contiguous_runs():
    doc_ids = torch.tensor([
        [0, 0, 0, 1, 1, 2],
        [7, 7, 7, 7, 7, 7],
    ])
    segs = _doc_segments(doc_ids)
    assert segs[0] == [(0, 3), (3, 2), (5, 1)]
    assert segs[1] == [(0, 6)]
    # Every row's segments must cover [0, T) exactly, in order, non-overlapping.
    for row_segs in segs:
        covered = sum(l for _, l in row_segs)
        assert covered == doc_ids.shape[1]


# ===========================================================================
# _kd_teacher_forward_doc_isolated — verified against a REAL tiny Llama (CPU)
# ===========================================================================

def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    m = LlamaForCausalLM(cfg)
    m.eval()
    return m


def test_teacher_doc_isolation_matches_first_doc_full_block_forward():
    # A document with NOTHING before it sees identical context whether or not
    # the forward is doc-isolated -- this must hold for ANY model, isolated
    # or not, and is the sanity baseline for the rest of the equivalence.
    m = _tiny_llama()
    x = torch.randint(0, 64, (2, 12))
    doc_ids = torch.zeros(2, 12, dtype=torch.long)
    doc_ids[0, 5:] = 1
    doc_ids[1, 8:] = 1

    iso = _kd_teacher_forward_doc_isolated(m, x, doc_ids)
    with torch.no_grad():
        full = m(input_ids=x).logits

    torch.testing.assert_close(iso[0, :5], full[0, :5], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(iso[1, :8], full[1, :8], atol=1e-4, rtol=1e-4)


def test_teacher_doc_isolation_matches_standalone_forward_of_second_doc():
    # The load-bearing equivalence: logits for a document AFTER a boundary
    # must match forwarding that document alone, from position 0 -- exactly
    # what the student's cu_seqlens reset does.
    m = _tiny_llama()
    x = torch.randint(0, 64, (2, 12))
    doc_ids = torch.zeros(2, 12, dtype=torch.long)
    doc_ids[0, 5:] = 1
    doc_ids[1, 8:] = 1

    iso = _kd_teacher_forward_doc_isolated(m, x, doc_ids)
    with torch.no_grad():
        alone0 = m(input_ids=x[0:1, 5:]).logits
        alone1 = m(input_ids=x[1:2, 8:]).logits

    torch.testing.assert_close(iso[0, 5:], alone0[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(iso[1, 8:], alone1[0], atol=1e-4, rtol=1e-4)


def test_teacher_doc_isolation_differs_from_naive_full_block_forward():
    # Proves the bug being fixed is real: the naive (pre-fix) forward over
    # the whole packed block gives the SECOND document logits that leaked
    # the first document's content through full self-attention.
    m = _tiny_llama()
    x = torch.randint(0, 64, (1, 12))
    doc_ids = torch.zeros(1, 12, dtype=torch.long)
    doc_ids[0, 5:] = 1

    iso = _kd_teacher_forward_doc_isolated(m, x, doc_ids)
    with torch.no_grad():
        naive_full = m(input_ids=x).logits

    assert not torch.allclose(iso[0, 5:], naive_full[0, 5:], atol=1e-4)


def test_teacher_doc_isolation_single_document_matches_full_block_forward():
    # Degenerate case: one document spanning the whole block -> isolated ==
    # naive (no boundary to isolate across, only right-padding to L_max=T
    # which is a no-op since there's nothing to pad).
    m = _tiny_llama()
    x = torch.randint(0, 64, (2, 10))
    doc_ids = torch.zeros(2, 10, dtype=torch.long)

    iso = _kd_teacher_forward_doc_isolated(m, x, doc_ids)
    with torch.no_grad():
        full = m(input_ids=x).logits
    torch.testing.assert_close(iso, full, atol=1e-4, rtol=1e-4)


# ===========================================================================
# End-to-end through _nonthink_forward_loss
# ===========================================================================

class _FakeStudent(torch.nn.Module):
    def __init__(self, B, T, V):
        super().__init__()
        self.register_buffer("fixed", torch.randn(B, T, V))
        self.scale = torch.nn.Parameter(torch.ones(()))
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, x, doc_ids=None, return_aux=False, **kw):
        return self.scale * self.fixed


def _args(distill_weight=0.5, distill_temp=2.0):
    return SimpleNamespace(
        aux_brackets=False, aux_max_depth=4, output_gate=False,
        z_loss=0.0, distill_weight=distill_weight, distill_temp=distill_temp,
    )


def test_live_kd_doc_isolated_end_to_end_finite_and_differentiable():
    m = _tiny_llama()
    B, T, Vs = 2, 10, 64
    x = torch.randint(0, 64, (B, T))
    y = torch.randint(0, 64, (B, T))
    doc_ids = torch.zeros(B, T, dtype=torch.long)
    doc_ids[0, 4:] = 1
    doc_ids[1, 7:] = 1
    student = _FakeStudent(B, T, Vs)

    out = _nonthink_forward_loss(
        student, x, y, _args(), 0, None,
        doc_ids=doc_ids, kd_teacher=m, kd_thinking_token_id=None)
    kd_loss = out[6]
    assert torch.isfinite(kd_loss)
    assert float(kd_loss.detach()) >= 0.0
    assert kd_loss.requires_grad
    kd_loss.backward()
    assert student.scale.grad is not None and torch.isfinite(student.scale.grad)


def test_live_kd_with_doc_ids_differs_from_without_when_boundary_present():
    # The whole point of the fix: doc-aware KD must diverge numerically from
    # the naive (doc_ids=None) path when there's an actual boundary to
    # isolate across.
    m = _tiny_llama()
    B, T, Vs = 1, 10, 64
    x = torch.randint(0, 64, (B, T))
    y = torch.randint(0, 64, (B, T))
    doc_ids = torch.zeros(B, T, dtype=torch.long)
    doc_ids[0, 4:] = 1
    student = _FakeStudent(B, T, Vs)

    kd_with_docs = float(_nonthink_forward_loss(
        student, x, y, _args(), 0, None,
        doc_ids=doc_ids, kd_teacher=m, kd_thinking_token_id=None)[6].detach())
    kd_without_docs = float(_nonthink_forward_loss(
        student, x, y, _args(), 0, None,
        doc_ids=None, kd_teacher=m, kd_thinking_token_id=None)[6].detach())

    assert kd_with_docs != pytest.approx(kd_without_docs, abs=1e-6)


def test_live_kd_doc_ids_none_unchanged_when_no_boundary_or_none():
    # Sanity: doc_ids=None must still work exactly as before this fix (single
    # full-block forward, no doc masking) -- the explicit backwards-compat
    # requirement.
    m = _tiny_llama()
    B, T, Vs = 1, 8, 64
    x = torch.randint(0, 64, (B, T))
    y = torch.randint(0, 64, (B, T))
    student = _FakeStudent(B, T, Vs)

    out = _nonthink_forward_loss(
        student, x, y, _args(), 0, None,
        doc_ids=None, kd_teacher=m, kd_thinking_token_id=None)
    assert torch.isfinite(out[6])


# ===========================================================================
# Task 2 — d_ff / tie_embeddings ckpt-cfg persistence round-trip
# ===========================================================================

def _make_tiny_ckpt(path: str, *, tie_embeddings: bool, d_ff: int | None,
                    include_cfg_keys: bool):
    from experiments.model import TinyLM
    from experiments.layers import DeltaNetAttention
    vocab, d_model, n_layers, n_heads, d_head = 16, 8, 2, 2, 4
    model = TinyLM(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, max_T=0,
        d_ff=d_ff, tie_embeddings=tie_embeddings,
        attention_cls=DeltaNetAttention,
    )
    cfg = {
        "vocab_size": vocab, "d_model": d_model, "n_layers": n_layers,
        "n_heads": n_heads, "d_head": d_head, "max_T": 0,
        "feedback_mode": "none", "feedback_pairs": (), "feedback_self_k": 0,
        "use_memory": False, "output_gate": False,
        "arch": "deltanet", "tokenizer": "HuggingFaceTB/SmolLM2-135M",
    }
    if include_cfg_keys:
        cfg["tie_embeddings"] = tie_embeddings
        cfg["d_ff"] = int(d_ff) if d_ff else 0
    torch.save({"state_dict": model.state_dict(), "step": 0, "config": cfg}, path)
    return model, cfg


def test_cfg_roundtrip_tie_embeddings_and_d_ff_preserved(tmp_path):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    ckpt_path = str(tmp_path / "tied.pt")
    custom_d_ff = 12   # deliberately != TinyLM's 4*d_model=32 default
    _make_tiny_ckpt(ckpt_path, tie_embeddings=True, d_ff=custom_d_ff,
                    include_cfg_keys=True)

    model, cfg = build_model_from_ckpt(ckpt_path)
    try:
        assert cfg["tie_embeddings"] is True
        assert cfg["d_ff"] == custom_d_ff
        assert model.tie_embeddings is True
        assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
        assert model.blocks[0].mlp.W_u.weight.shape[0] == custom_d_ff
    finally:
        del model
        torch.cuda.empty_cache()


def test_cfg_roundtrip_untied_and_default_d_ff(tmp_path):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    ckpt_path = str(tmp_path / "untied.pt")
    _make_tiny_ckpt(ckpt_path, tie_embeddings=False, d_ff=None,
                    include_cfg_keys=True)

    model, cfg = build_model_from_ckpt(ckpt_path)
    try:
        assert model.tie_embeddings is False
        assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr()
        # d_ff cfg key is 0 ("not set") -> falls back to shape-inference from
        # the saved MLP weight, which was built with TinyLM's own default
        # (4 * d_model = 32).
        assert model.blocks[0].mlp.W_u.weight.shape[0] == 4 * 8
    finally:
        del model
        torch.cuda.empty_cache()


def test_cfg_roundtrip_old_ckpt_without_new_keys_defaults_untied(tmp_path):
    # An OLD ckpt (saved before this fix) has no tie_embeddings/d_ff in cfg at
    # all -- must reconstruct exactly as it did before this fix: untied, d_ff
    # inferred from the state-dict shape.
    from experiments.eval_bracket_structure import build_model_from_ckpt
    ckpt_path = str(tmp_path / "old.pt")
    _make_tiny_ckpt(ckpt_path, tie_embeddings=False, d_ff=None,
                    include_cfg_keys=False)

    model, cfg = build_model_from_ckpt(ckpt_path)
    try:
        assert "tie_embeddings" not in cfg and "d_ff" not in cfg
        assert model.tie_embeddings is False
        assert model.blocks[0].mlp.W_u.weight.shape[0] == 4 * 8
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
