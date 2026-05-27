"""Tests for Fill-in-the-Middle (FIM) augmentation in experiments.data_mix.

Covers:
  - maybe_apply_fim behaviour (rate=0 noop, rate=1 always FIM, structure)
  - statistical rate check across many calls
  - Stream-level integration: SourceConfig.fim_rate propagates to FIM markers
  - Tokenizer round-trip on the three sentinels
"""
from __future__ import annotations

import os
import random
import sys
from unittest import mock

import pytest

import experiments.data_mix as dm
from experiments.data_mix import (
    MixedSourceStream,
    SourceConfig,
    _FIM_MIDDLE,
    _FIM_PREFIX,
    _FIM_SUFFIX,
    maybe_apply_fim,
)
from experiments.test_data_mix import (
    _FakeTokenizer,
    _mock_open_stream_factory,
)


# ---------------------------------------------------------------------------
# maybe_apply_fim

def test_maybe_apply_fim_rate_zero_is_noop() -> None:
    rng = random.Random(0)
    for _ in range(50):
        text = "def f(x): return x + 1"
        assert maybe_apply_fim(text, rng=rng, fim_rate=0.0) == text


def test_maybe_apply_fim_rate_one_always_reformats() -> None:
    rng = random.Random(0)
    text = "def f(x): return x + 1"
    for _ in range(20):
        out = maybe_apply_fim(text, rng=rng, fim_rate=1.0)
        assert out != text
        assert out.startswith(_FIM_PREFIX)
        # Sentinel ordering: prefix < suffix < middle.
        i_pre = out.index(_FIM_PREFIX)
        i_suf = out.index(_FIM_SUFFIX)
        i_mid = out.index(_FIM_MIDDLE)
        assert i_pre < i_suf < i_mid


def test_maybe_apply_fim_reconstruction() -> None:
    """prefix + middle + suffix (in original order) must reconstruct the input."""
    rng = random.Random(42)
    text = "def add(a, b):\n    return a + b\n"
    for _ in range(20):
        out = maybe_apply_fim(text, rng=rng, fim_rate=1.0)
        # Parse the PSM string back.
        rest = out[len(_FIM_PREFIX):]
        prefix, rest = rest.split(_FIM_SUFFIX, 1)
        suffix, middle = rest.split(_FIM_MIDDLE, 1)
        assert prefix + middle + suffix == text
        # Each piece non-empty.
        assert len(prefix) > 0 and len(middle) > 0 and len(suffix) > 0


def test_maybe_apply_fim_short_text_passes_through() -> None:
    rng = random.Random(0)
    for short in ("", "a", "ab", "abc"):
        assert maybe_apply_fim(short, rng=rng, fim_rate=1.0) == short


def test_maybe_apply_fim_statistical_rate() -> None:
    """At fim_rate=0.5, ~50% of 1000 calls reformat (±5% slack)."""
    rng = random.Random(123)
    text = "def f(x): return x * 2 + 1"
    n_fim = 0
    N = 1000
    for _ in range(N):
        out = maybe_apply_fim(text, rng=rng, fim_rate=0.5)
        if out != text:
            n_fim += 1
    ratio = n_fim / N
    assert 0.45 <= ratio <= 0.55, (
        f"FIM rate off: got {ratio:.3f} from {n_fim}/{N}, expected ~0.50"
    )


# ---------------------------------------------------------------------------
# Stream-level integration

def test_mixed_stream_propagates_fim_rate() -> None:
    """A source configured with fim_rate=0.3 should produce chunks containing
    FIM marker token-substrings at roughly the expected per-document rate."""
    tok = _FakeTokenizer()
    sources = [SourceConfig(
        name="s0", dataset_id="x", text_field="text",
        weight=1.0, fim_rate=0.3,
    )]
    # Each "document" is short enough that its tokens fit comfortably in a
    # single chunk — so chunk count ~= document count and we can measure
    # the per-document rate at the chunk level.
    text = " ".join([f"w{i}" for i in range(40)])
    streams = {"s0": [{"text": text}] * 2000}
    with mock.patch.object(dm, "_open_stream",
                            _mock_open_stream_factory(streams)):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                                block_size=64,
                                thinking_token_id=None,
                                think_burst_prob=0.0,
                                base_seed=0)
        it = iter(ds)
        # The FakeTokenizer splits on whitespace, so each FIM sentinel
        # tokenizes to a single id we can identify.
        n_chunks_with_fim = 0
        N = 400
        for _ in range(N):
            x, _y = next(it)
            xs = x.tolist()
            inv = {v: k for k, v in tok._vocab.items()}
            words = {inv[t] for t in xs if t in inv}
            if any(_FIM_PREFIX in w for w in words):
                n_chunks_with_fim += 1
        # Each chunk packs ~1.5 short documents; per-document FIM rate is
        # 0.3, so per-chunk rate should be roughly 0.3-0.5. Use a wide
        # window — the precise expectation depends on packing density.
        ratio = n_chunks_with_fim / N
        assert 0.15 <= ratio <= 0.70, (
            f"FIM marker rate in chunks off: got {ratio:.3f}, "
            f"expected ~0.30-0.50"
        )


def test_mixed_stream_fim_rate_default_zero_byte_identical() -> None:
    """A source with default (no) fim_rate produces no FIM markers."""
    tok = _FakeTokenizer()
    sources = [SourceConfig(
        name="s0", dataset_id="x", text_field="text", weight=1.0,
    )]
    assert sources[0].fim_rate == 0.0
    text = " ".join([f"w{i}" for i in range(40)])
    streams = {"s0": [{"text": text}] * 200}
    with mock.patch.object(dm, "_open_stream",
                            _mock_open_stream_factory(streams)):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                                block_size=64,
                                thinking_token_id=None,
                                think_burst_prob=0.0,
                                base_seed=0)
        it = iter(ds)
        for _ in range(50):
            x, _y = next(it)
            xs = x.tolist()
            inv = {v: k for k, v in tok._vocab.items()}
            words = {inv[t] for t in xs if t in inv}
            assert not any(_FIM_PREFIX in w for w in words)
            assert not any(_FIM_SUFFIX in w for w in words)
            assert not any(_FIM_MIDDLE in w for w in words)


# ---------------------------------------------------------------------------
# Tokenizer round-trip — real tokenizer if available; skipped otherwise.

def test_real_tokenizer_round_trip_sentinels() -> None:
    """The three FIM sentinels must encode-then-decode back to themselves.
    They will tokenize as multi-token sequences for byte-level BPE tokenizers
    that don't have these as special vocabulary; that's expected."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    try:
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception as e:  # offline, no network, missing cache, ...
        pytest.skip(f"tokenizer unavailable: {e}")
    for s in (_FIM_PREFIX, _FIM_SUFFIX, _FIM_MIDDLE):
        ids = tok.encode(s, add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=False)
        assert decoded == s, (
            f"sentinel round-trip failed: {s!r} -> {ids} -> {decoded!r}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
