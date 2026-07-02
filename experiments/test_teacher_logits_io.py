"""Tests for the sharded teacher top-k logit store (task #104).

Covers:
- Exact round-trip equality of ids / logits / input_ids across MULTIPLE shards.
- Manifest token-range integrity (shards tile [0, total) with no gaps/overlap).
- Sequential reads (next_block) concatenate to the full stream.
- Random access (get_range) matches the sequential stream.
- seek / tell / remaining behave; reading past the end raises.

CPU-only, no CUDA / model.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_teacher_logits_io.py -v
"""
from __future__ import annotations

import json

import pytest
import torch

from experiments.teacher_logits_io import LogitStoreWriter, LogitStoreReader


def _make_store(tmp_path, n_total, k, vocab, append_sizes, shard_max_tokens):
    """Write a store of `n_total` rows in chunks of `append_sizes` and return
    the directory plus the ground-truth (ids, logits_fp16, input_ids) tensors."""
    torch.manual_seed(0)
    ids = torch.randint(0, vocab, (n_total, k), dtype=torch.int64)
    logits = (torch.randn(n_total, k) * 5.0).to(torch.float16)
    input_ids = torch.randint(0, vocab, (n_total,), dtype=torch.int64)

    w = LogitStoreWriter(str(tmp_path), k=k, vocab_size=vocab,
                         teacher_model="fake/teacher",
                         tokenizer_name="fake/tok",
                         shard_max_tokens=shard_max_tokens)
    pos = 0
    for sz in append_sizes:
        w.append(ids[pos:pos + sz], logits[pos:pos + sz], input_ids[pos:pos + sz])
        pos += sz
    assert pos == n_total
    manifest = w.close()
    return manifest, ids, logits, input_ids


def test_roundtrip_multishard_exact(tmp_path):
    n, k, vocab = 250, 8, 1000
    # shard_max_tokens=100 -> 3 shards (100, 100, 50). Append in irregular chunks
    # that straddle shard boundaries.
    manifest, ids, logits, input_ids = _make_store(
        tmp_path, n, k, vocab, append_sizes=[37, 63, 90, 60], shard_max_tokens=100)
    assert len(manifest["shards"]) == 3
    r = LogitStoreReader(str(tmp_path))
    assert len(r) == n
    rid, rlog, rin = r.get_range(0, n)
    assert torch.equal(rid, ids)
    assert torch.equal(rin, input_ids)
    assert rlog.dtype == torch.float16
    assert torch.equal(rlog, logits)


def test_manifest_range_integrity(tmp_path):
    n, k, vocab = 100, 4, 50
    manifest, *_ = _make_store(tmp_path, n, k, vocab,
                               append_sizes=[100], shard_max_tokens=40)
    # 100 tokens / 40 per shard -> shards of 40, 40, 20.
    shards = manifest["shards"]
    assert [s["n"] for s in shards] == [40, 40, 20]
    expect = 0
    for s in shards:
        assert s["start"] == expect
        assert s["end"] == s["start"] + s["n"]
        expect = s["end"]
    assert expect == n == manifest["total_tokens"]
    # On-disk manifest equals returned manifest.
    with open(tmp_path / "manifest.json") as f:
        assert json.load(f)["shards"] == shards


def test_sequential_concatenates_to_full(tmp_path):
    n, k, vocab = 130, 6, 200
    _, ids, logits, input_ids = _make_store(
        tmp_path, n, k, vocab, append_sizes=[130], shard_max_tokens=50)
    r = LogitStoreReader(str(tmp_path))
    blocks_id, blocks_in = [], []
    # Irregular block sizes that cross shard boundaries.
    for bs in (13, 50, 7, 60):
        bid, blog, bin_ = r.next_block(bs)
        blocks_id.append(bid)
        blocks_in.append(bin_)
    assert r.tell() == 130
    cat_id = torch.cat(blocks_id, 0)
    cat_in = torch.cat(blocks_in, 0)
    assert torch.equal(cat_id, ids)
    assert torch.equal(cat_in, input_ids)


def test_random_access_matches_sequential(tmp_path):
    n, k, vocab = 175, 5, 300
    _, ids, logits, input_ids = _make_store(
        tmp_path, n, k, vocab, append_sizes=[80, 95], shard_max_tokens=64)
    r = LogitStoreReader(str(tmp_path))
    for (a, b) in [(0, 1), (10, 10), (0, n), (63, 65), (100, 175), (7, 130)]:
        rid, rlog, rin = r.get_range(a, b)
        assert torch.equal(rid, ids[a:b])
        assert torch.equal(rlog, logits[a:b])
        assert torch.equal(rin, input_ids[a:b])


def test_seek_and_remaining(tmp_path):
    n, k, vocab = 60, 4, 100
    _, ids, *_ = _make_store(tmp_path, n, k, vocab,
                             append_sizes=[60], shard_max_tokens=25)
    r = LogitStoreReader(str(tmp_path))
    assert r.remaining() == 60
    r.seek(50)
    assert r.tell() == 50 and r.remaining() == 10
    rid, _, _ = r.next_block(10)
    assert torch.equal(rid, ids[50:60])
    assert r.remaining() == 0


def test_read_past_end_raises(tmp_path):
    n, k, vocab = 30, 3, 50
    _make_store(tmp_path, n, k, vocab, append_sizes=[30], shard_max_tokens=20)
    r = LogitStoreReader(str(tmp_path))
    r.next_block(25)
    with pytest.raises(IndexError):
        r.next_block(10)


def test_append_shape_validation(tmp_path):
    w = LogitStoreWriter(str(tmp_path), k=4, vocab_size=50)
    with pytest.raises(ValueError):
        w.append(torch.zeros(3, 5), torch.zeros(3, 5), torch.zeros(3))  # k=5 != 4
    with pytest.raises(ValueError):
        w.append(torch.zeros(3, 4), torch.zeros(3, 4), torch.zeros(2))  # input_ids


def test_fp16_overflow_is_clamped_not_inf(tmp_path):
    # A pathologically large (or +/-inf) teacher logit must be clamped to the
    # fp16 finite range, never stored as inf (which would corrupt the KL target).
    n, k, vocab = 2, 3, 100
    ids = torch.randint(0, vocab, (n, k), dtype=torch.int64)
    input_ids = torch.randint(0, vocab, (n,), dtype=torch.int64)
    logits = torch.zeros(n, k, dtype=torch.float32)
    logits[0, 0] = 1e9                 # finite but >> fp16 max
    logits[0, 1] = float("inf")        # explicit +inf
    logits[1, 0] = -1e9                # finite but << fp16 min
    logits[1, 1] = float("-inf")       # explicit -inf
    w = LogitStoreWriter(str(tmp_path), k=k, vocab_size=vocab)
    w.append(ids, logits, input_ids)
    w.close()
    r = LogitStoreReader(str(tmp_path))
    _, rlog, _ = r.get_range(0, n)
    assert torch.isfinite(rlog).all(), rlog
    assert rlog.max().item() <= 65504.0
    assert rlog.min().item() >= -65504.0
    # The overflowing entries clamp to exactly the fp16 finite bounds.
    assert float(rlog[0, 0]) == 65504.0 and float(rlog[0, 1]) == 65504.0
    assert float(rlog[1, 0]) == -65504.0 and float(rlog[1, 1]) == -65504.0
