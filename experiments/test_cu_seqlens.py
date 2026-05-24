"""Tests for cross-document state isolation (the cu_seqlens / doc_ids path).

DeltaNet is a linear RNN; without isolation its recurrent state (and the
WorkingMemory reads) flow across documents packed into one T-length row.
These tests cover the four pieces of the fix:
  - data pipeline emits per-position doc_ids (and they survive think bursts),
  - `_build_cu_seqlens` turns doc_ids into a ragged segment index,
  - the DeltaNet kernel actually resets state at document boundaries
    (packed == unpacked), through both the plain and FiLM forward paths,
  - WorkingMemory cannot read across a document boundary.

CUDA-guarded tests need the DeltaNet Triton kernels; the rest run on CPU.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_cu_seqlens.py -v
"""
from __future__ import annotations

import torch
import pytest

from experiments.data_mix import MixedSourceStream, SourceConfig
from experiments.model import _build_cu_seqlens, TinyLM, WorkingMemory


# --------------------------------------------------------------------------
# Data pipeline: doc_ids emission
# --------------------------------------------------------------------------

class _FakeTok:
    """Minimal tokenizer stub: encode maps chars -> ids (0 avoided)."""
    eos_token_id = 99
    bos_token_id = None

    def __init__(self):
        self._vocab = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for ch in text:
            if ch not in self._vocab:
                self._vocab[ch] = len(self._vocab) + 1
            ids.append(self._vocab[ch])
        return ids


def _docid_stream(think_burst_prob: float, n_yields: int = 6,
                  block_size: int = 16):
    """MixedSourceStream(emit_doc_ids=True) over a fixed short doc."""
    src = [SourceConfig(name="fake", dataset_id="fake", split="train",
                        weight=1.0, text_field="text")]
    stream = MixedSourceStream(
        sources=src, tokenizer=_FakeTok(), block_size=block_size,
        thinking_token_id=7,
        think_burst_prob=think_burst_prob,
        think_max_bursts=2, think_max_burst_depth=4,
        base_seed=0, emit_doc_ids=True,
    )
    import experiments.data_mix as dm
    orig = dm._open_stream

    def _fake_open(src, seed=0):
        def gen():
            while True:
                yield {"text": "abcde"}
        return gen()
    dm._open_stream = _fake_open
    try:
        it = iter(stream)
        return [next(it) for _ in range(n_yields)]
    finally:
        dm._open_stream = orig


def test_doc_ids_emitted():
    """emit_doc_ids -> 3-tuple; doc_ids monotonic, increments after each EOS."""
    samples = _docid_stream(think_burst_prob=0.0)
    saw_boundary = False
    for sample in samples:
        assert len(sample) == 3, "expected (inputs, targets, doc_ids)"
        inputs, targets, doc_ids = sample
        assert doc_ids.shape == inputs.shape
        assert doc_ids.dtype == torch.long
        # Monotonic non-decreasing within the row.
        assert torch.all(doc_ids[1:] >= doc_ids[:-1])
        # Normalised to start at 0.
        assert int(doc_ids[0]) == 0
        # Every EOS in the inputs is followed by a doc-id increment.
        for i in range(len(inputs) - 1):
            if int(inputs[i]) == _FakeTok.eos_token_id:
                assert int(doc_ids[i + 1]) == int(doc_ids[i]) + 1
                saw_boundary = True
    assert saw_boundary, "test setup: no EOS-driven boundary in any chunk"


def test_doc_ids_survive_think_bursts():
    """With think bursts on, doc_ids stay aligned with inputs and monotonic."""
    samples = _docid_stream(think_burst_prob=1.0)
    saw_think = False
    for inputs, targets, doc_ids in samples:
        assert doc_ids.shape == inputs.shape
        assert torch.all(doc_ids[1:] >= doc_ids[:-1])
        if (inputs == 7).any():
            saw_think = True
    assert saw_think, "test setup: think bursts never fired"


# --------------------------------------------------------------------------
# _build_cu_seqlens
# --------------------------------------------------------------------------

def test_cu_seqlens_none():
    assert _build_cu_seqlens(None) is None


def test_cu_seqlens_construction():
    # One row, two documents of length 2.
    doc_ids = torch.tensor([[0, 0, 1, 1]])
    cu = _build_cu_seqlens(doc_ids)
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 2, 4]

    # Two rows, each a single document -> row boundary still splits them
    # even though both rows use doc-id 0.
    doc_ids = torch.tensor([[0, 0], [0, 0]])
    assert _build_cu_seqlens(doc_ids).tolist() == [0, 2, 4]

    # Mixed: row 0 has docs {0,0,1}, row 1 has docs {0,1,1}.
    doc_ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
    assert _build_cu_seqlens(doc_ids).tolist() == [0, 2, 3, 4, 6]


# --------------------------------------------------------------------------
# DeltaNet kernel: packed == unpacked (the headline correctness test)
# --------------------------------------------------------------------------

def _make_model(**kw) -> TinyLM:
    torch.manual_seed(0)
    from experiments.layers import DeltaNetAttention
    defaults = dict(
        vocab_size=64, d_model=32, n_layers=3, n_heads=2, d_head=16,
        attention_cls=DeltaNetAttention,
    )
    defaults.update(kw)
    return TinyLM(**defaults)


def _packed_vs_leaked(model):
    """Returns (b_correct, b_leaked, b_standalone): logits over document-2's
    positions in (i) a correctly doc-id'd packed run, (ii) a packed run that
    treats the whole row as one document, (iii) a standalone run of doc 2."""
    torch.manual_seed(1)
    La, Lb = 20, 28
    a = torch.randint(0, 64, (1, La), device="cuda")
    b = torch.randint(0, 64, (1, Lb), device="cuda")
    packed = torch.cat([a, b], dim=1)                       # (1, La+Lb)
    doc_correct = torch.cat([
        torch.zeros(1, La, dtype=torch.long),
        torch.ones(1, Lb, dtype=torch.long),
    ], dim=1).cuda()
    doc_one = torch.zeros(1, La + Lb, dtype=torch.long).cuda()
    with torch.no_grad():
        lc = model(packed, doc_ids=doc_correct)[0, La:]
        ll = model(packed, doc_ids=doc_one)[0, La:]
        ls = model(b, doc_ids=None)[0]
    return lc, ll, ls


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_packed_equals_unpacked():
    model = _make_model(feedback_mode="none").cuda().eval()
    b_correct, b_leaked, b_standalone = _packed_vs_leaked(model)
    # With correct doc_ids, document 2's logits match the standalone run:
    # no state leaked in from document 1.
    torch.testing.assert_close(b_correct, b_standalone, rtol=2e-2, atol=2e-2)
    # And the isolation actually does something: treating the row as one
    # document (state leaks) gives materially different logits.
    assert not torch.allclose(b_correct, b_leaked, rtol=2e-2, atol=2e-2), \
        "doc-id isolation had no effect — state leak not prevented"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_film_paths_thread_cu_seqlens():
    """Both the multi-pass FiLM forward and the _film_bypass path isolate
    documents."""
    model = _make_model(
        feedback_mode="film", feedback_pairs=((0, 2),), feedback_self_k=3,
    ).cuda().eval()
    for bypass in (False, True):
        model._film_bypass = bypass
        b_correct, b_leaked, b_standalone = _packed_vs_leaked(model)
        torch.testing.assert_close(b_correct, b_standalone,
                                   rtol=3e-2, atol=3e-2)
        assert not torch.allclose(b_correct, b_leaked, rtol=2e-2, atol=2e-2), \
            f"FiLM path (bypass={bypass}) leaked state across documents"
    model._film_bypass = False


# --------------------------------------------------------------------------
# WorkingMemory: no cross-document reads
# --------------------------------------------------------------------------

def test_memory_no_cross_doc_attention():
    """A think token in document 2 must be unaffected by document 1's
    hidden states; with doc_ids=None it IS affected (regression guard)."""
    torch.manual_seed(0)
    d_model, d_mem, T = 16, 16, 12
    mem = WorkingMemory(d_model=d_model, d_mem=d_mem, mem_size=T,
                        thinking_token_id=7).eval()
    # Document 1 = positions 0..5, document 2 = positions 6..11.
    doc_ids = torch.tensor([[0] * 6 + [1] * 6])
    # A think token at position 10 (inside document 2).
    input_ids = torch.full((1, T), 3, dtype=torch.long)
    input_ids[0, 10] = 7

    base = torch.randn(1, T, d_model)
    h1 = base.clone()
    h2 = base.clone()
    # Perturb ONLY document 1's hidden states.
    h2[0, :6] += torch.randn(6, d_model)

    with torch.no_grad():
        out1 = mem(h1, input_ids, doc_ids=doc_ids)
        out2 = mem(h2, input_ids, doc_ids=doc_ids)
        # Document 2's think position is untouched by document 1's change.
        torch.testing.assert_close(out1[0, 10], out2[0, 10])

        # Regression guard: without doc_ids the same perturbation DOES leak
        # into document 2's think position.
        out1_leak = mem(h1, input_ids, doc_ids=None)
        out2_leak = mem(h2, input_ids, doc_ids=None)
        assert not torch.allclose(out1_leak[0, 10], out2_leak[0, 10]), \
            "doc_ids=None should leak document 1 into document 2's read"


# --------------------------------------------------------------------------
# Signature back-compat
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_forward_signature_backcompat():
    """model(input_ids) still works and equals model(input_ids, doc_ids=None)."""
    model = _make_model(feedback_mode="none").cuda().eval()
    x = torch.randint(0, 64, (2, 24), device="cuda")
    with torch.no_grad():
        a = model(x)
        b = model(x, doc_ids=None)
    torch.testing.assert_close(a, b)
