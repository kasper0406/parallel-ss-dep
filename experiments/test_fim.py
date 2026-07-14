"""Tests for token-sentinel PSM Fill-in-the-Middle in experiments.data_mix.

The 2026-07-14 rewrite renders FIM at the TOKEN level with single-token
sentinels ([FIM_PRE] prefix [FIM_SUF] suffix [FIM_MID] middle), reserved via
the [THINKING]-token mechanism (ids above the real vocab) or the tokenizer's
native FIM tokens (Qwen2.5). Covered:

  - fim_split: PSM round-trip (prefix+middle+suffix == original), line-boundary
    snapping, middle-fraction bounds (unsnapped), short-text passthrough,
    single-line fallback to the char split.
  - resolve_fim_sentinel_ids: reserved [THINKING]-slot convention + native
    Qwen-style token path.
  - render_fim_psm_ids: sentinel placement + token-level PSM reassembly.
  - Stream level: rate=0 byte-identity (machinery inert), determinism under
    seed, sentinel-id plumbing (custom ids), full-loss convention (sentinels
    present in targets, not masked), statistical fim rate, doc_ids/EOS packing
    preserved, per-source accounting (fim'd fraction + tokens), read-mask
    incompatibility, think-id collision validation.

CPU-only. Run:
    PYTHONPATH=. .venv/bin/python -m pytest experiments/test_fim.py -v
"""
from __future__ import annotations

import random
import sys
from unittest import mock

import pytest
import torch

import experiments.data_mix as dm
from experiments.data_mix import (
    MixedSourceStream,
    SourceConfig,
    fim_split,
    render_fim_psm_ids,
    resolve_fim_sentinel_ids,
)
from experiments.test_data_mix import (
    _FakeTokenizer,
    _mock_open_stream_factory,
)

_SENT = (5001, 5002, 5003)   # (pre, suf, mid) — clear of _FakeTokenizer ids


def _code_doc(n_lines=8, words_per_line=5, tag="a"):
    return "\n".join(
        " ".join(f"{tag}{li}w{wi}" for wi in range(words_per_line))
        for li in range(n_lines)
    ) + "\n"


# ---------------------------------------------------------------------------
# fim_split

def test_fim_split_round_trip():
    rng = random.Random(0)
    for i in range(200):
        text = _code_doc(n_lines=3 + i % 7, tag=f"t{i}")
        parts = fim_split(text, rng=rng)
        assert parts is not None
        p, m, s = parts
        assert p + m + s == text
        assert m


def test_fim_split_line_boundary_snapping():
    rng = random.Random(1)
    text = _code_doc(n_lines=12)
    for _ in range(100):
        p, m, s = fim_split(text, rng=rng)
        # prefix ends at a line boundary (or is empty) …
        assert p == "" or p.endswith("\n")
        # … and the middle is whole lines: ends with \n unless it runs to EOF.
        assert s == "" or m.endswith("\n")


def test_fim_split_middle_frac_bounds_unsnapped():
    rng = random.Random(2)
    text = "x" * 1000
    for _ in range(200):
        p, m, s = fim_split(text, rng=rng, snap_to_lines=False)
        frac = len(m) / len(text)
        assert 0.09 <= frac <= 0.51, f"middle frac {frac} out of [0.1, 0.5]"


def test_fim_split_short_text_returns_none():
    rng = random.Random(3)
    for short in ("", "a", "ab", "abc"):
        assert fim_split(short, rng=rng) is None


def test_fim_split_single_line_falls_back_to_char_split():
    """A doc with no newline can't snap to lines; the raw char split is kept
    so single-line docs still get FIM at the configured rate."""
    rng = random.Random(4)
    text = "def f(x): return x + 1  # no newline here"
    for _ in range(50):
        parts = fim_split(text, rng=rng)
        assert parts is not None
        p, m, s = parts
        assert p + m + s == text
        assert p and m and s     # char split keeps all three non-empty


# ---------------------------------------------------------------------------
# Sentinel resolution

def test_resolve_sentinels_reserved_thinking_convention():
    tok = _FakeTokenizer()   # vocab_size=1000, len=1200
    # Think slot = max(1000, 1200) = 1200 → sentinels take the next three.
    assert resolve_fim_sentinel_ids(tok) == (1201, 1202, 1203)
    # Stable when the caller's think id IS the convention slot.
    assert resolve_fim_sentinel_ids(tok, thinking_token_id=1200) == \
        (1201, 1202, 1203)
    # A higher think id (added special tokens) pushes the sentinels above it.
    assert resolve_fim_sentinel_ids(tok, thinking_token_id=2000) == \
        (2001, 2002, 2003)


def test_resolve_sentinels_native_qwen_style_tokens():
    class _QwenLike(_FakeTokenizer):
        def get_vocab(self):
            return {"<|fim_prefix|>": 151659, "<|fim_middle|>": 151660,
                    "<|fim_suffix|>": 151661}
    # Order is (prefix, suffix, middle) — matching the PSM render.
    assert resolve_fim_sentinel_ids(_QwenLike()) == (151659, 151661, 151660)


# ---------------------------------------------------------------------------
# render_fim_psm_ids

def test_render_psm_structure_and_token_round_trip():
    tok = _FakeTokenizer()
    rng = random.Random(5)
    text = _code_doc(n_lines=10)
    plain = tok.encode(text)
    for _ in range(50):
        ids = render_fim_psm_ids(text, tok, rng=rng, sentinel_ids=_SENT)
        assert ids is not None
        pre, suf, mid = _SENT
        assert ids[0] == pre
        assert ids.count(pre) == ids.count(suf) == ids.count(mid) == 1
        i_suf, i_mid = ids.index(suf), ids.index(mid)
        assert 0 < i_suf < i_mid
        p_ids = ids[1:i_suf]
        s_ids = ids[i_suf + 1:i_mid]
        m_ids = ids[i_mid + 1:]
        # PSM → original order reassembly at the token level. Line-snapped
        # splits fall on whitespace, so the whitespace tokenizer's segment
        # encodes concatenate exactly to the plain-document encode.
        assert p_ids + m_ids + s_ids == plain


def test_render_returns_none_on_short_text():
    tok = _FakeTokenizer()
    assert render_fim_psm_ids("ab", tok, rng=random.Random(0),
                              sentinel_ids=_SENT) is None


# ---------------------------------------------------------------------------
# Stream level

def _stream(sources, streams, tok=None, **kw):
    tok = tok or _FakeTokenizer()
    patch = mock.patch.object(dm, "_open_stream",
                              _mock_open_stream_factory(streams))
    kw.setdefault("block_size", 128)
    kw.setdefault("thinking_token_id", None)
    kw.setdefault("think_burst_prob", 0.0)
    kw.setdefault("base_seed", 0)
    return patch, MixedSourceStream(sources=sources, tokenizer=tok, **kw)


def _collect(patch, ds, n):
    out = []
    with patch:
        it = iter(ds)
        for _ in range(n):
            out.append(next(it))
    return out


def test_stream_rate_zero_byte_identity_and_no_sentinels():
    """fim_rate=0 keeps the FIM machinery fully inert: no sentinel ids are
    resolved, chunks are identical whether or not explicit sentinel ids are
    plumbed, and no sentinel appears in the output."""
    text = _code_doc(n_lines=20)
    streams = {"s0": [{"text": text}] * 200}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0)]
    assert src[0].fim_rate == 0.0
    patch_a, ds_a = _stream(src, streams)
    assert ds_a.fim_sentinel_ids is None
    patch_b, ds_b = _stream(src, streams, fim_sentinel_ids=_SENT)
    chunks_a = _collect(patch_a, ds_a, 20)
    chunks_b = _collect(patch_b, ds_b, 20)
    for (xa, ya), (xb, yb) in zip(chunks_a, chunks_b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
        assert not any(int(t) in _SENT for t in xa.tolist())


def test_stream_deterministic_under_seed():
    text = _code_doc(n_lines=20)
    streams = {"s0": [{"text": text}] * 400}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=0.5)]
    patch_a, ds_a = _stream(src, streams, fim_sentinel_ids=_SENT, base_seed=7)
    patch_b, ds_b = _stream(src, streams, fim_sentinel_ids=_SENT, base_seed=7)
    patch_c, ds_c = _stream(src, streams, fim_sentinel_ids=_SENT, base_seed=8)
    a = _collect(patch_a, ds_a, 30)
    b = _collect(patch_b, ds_b, 30)
    c = _collect(patch_c, ds_c, 30)
    assert all(torch.equal(xa, xb) for (xa, _), (xb, _) in zip(a, b))
    assert any(not torch.equal(xa, xc) for (xa, _), (xc, _) in zip(a, c))


def test_stream_sentinels_present_and_in_targets_full_loss():
    """fim_rate=1.0 → every doc is PSM-rendered; sentinel ids appear in the
    inputs AND in the targets un-masked (the full-LM-loss FIM convention:
    loss on sentinels + prefix + suffix + middle, like a plain document)."""
    text = _code_doc(n_lines=20)
    streams = {"s0": [{"text": text}] * 200}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=1.0)]
    patch, ds = _stream(src, streams, fim_sentinel_ids=_SENT)
    seen_in_inputs: set[int] = set()
    seen_in_targets: set[int] = set()
    for x, y in _collect(patch, ds, 20):
        for sid in _SENT:
            if (x == sid).any():
                seen_in_inputs.add(sid)
            if (y == sid).any():
                seen_in_targets.add(sid)
        assert not (y == -100).any()      # nothing masked (no think bursts)
    # A single chunk may slice mid-document and miss a sentinel; across 20
    # chunks every sentinel must appear both as input and as an UNMASKED
    # target (the full-LM-loss convention).
    assert seen_in_inputs == set(_SENT)
    assert seen_in_targets == set(_SENT)


def test_stream_custom_sentinel_id_plumbing():
    text = _code_doc(n_lines=20)
    streams = {"s0": [{"text": text}] * 100}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=1.0)]
    custom = (7770, 7771, 7772)
    patch, ds = _stream(src, streams, fim_sentinel_ids=custom)
    assert ds.fim_sentinel_ids == custom
    (x, _y), = _collect(patch, ds, 1)
    xs = set(x.tolist())
    assert xs & set(custom)
    assert not xs & set(_SENT)


def test_stream_default_sentinels_resolved_from_tokenizer():
    text = _code_doc(n_lines=20)
    streams = {"s0": [{"text": text}] * 100}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=1.0)]
    patch, ds = _stream(src, streams)   # no explicit ids
    assert ds.fim_sentinel_ids == (1201, 1202, 1203)   # _FakeTokenizer default
    (x, _y), = _collect(patch, ds, 1)
    assert set(x.tolist()) & {1201, 1202, 1203}


def test_stream_statistical_fim_rate_and_accounting():
    text = _code_doc(n_lines=10)
    streams = {"s0": [{"text": text}] * 5000}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=0.5)]
    patch, ds = _stream(src, streams, fim_sentinel_ids=_SENT)
    _collect(patch, ds, 150)
    (st,) = ds.source_stats()
    assert st["docs"] > 200
    assert st["tokens"] > 0
    assert 0.42 <= st["fim_frac"] <= 0.58, (
        f"per-doc FIM rate off: {st['fim_frac']:.3f} over {st['docs']} docs")


def test_per_source_accounting_separates_sources():
    t0 = _code_doc(n_lines=10, tag="a")
    t1 = _code_doc(n_lines=10, tag="b")
    streams = {"s_fim": [{"text": t0}] * 3000,
               "s_plain": [{"text": t1}] * 3000}
    src = [
        SourceConfig(name="s_fim", dataset_id="x", text_field="text",
                     weight=0.5, fim_rate=1.0),
        SourceConfig(name="s_plain", dataset_id="x", text_field="text",
                     weight=0.5),
    ]
    patch, ds = _stream(src, streams, fim_sentinel_ids=_SENT)
    _collect(patch, ds, 100)
    stats = {s["name"]: s for s in ds.source_stats()}
    assert stats["s_fim"]["docs"] > 0 and stats["s_plain"]["docs"] > 0
    assert stats["s_fim"]["fim_frac"] == 1.0
    assert stats["s_plain"]["fim_frac"] == 0.0
    assert stats["s_fim"]["tokens"] > 0 and stats["s_plain"]["tokens"] > 0


def test_stream_short_docs_fall_back_to_plain():
    streams = {"s0": [{"text": "ab"}] * 2000}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=1.0,
                        filter_spec={"name": "min_content_len",
                                     "args": {"min_chars": 1}})]
    patch, ds = _stream(src, streams, fim_sentinel_ids=_SENT, block_size=32)
    chunks = _collect(patch, ds, 5)
    (st,) = ds.source_stats()
    assert st["docs"] > 0 and st["fim_docs"] == 0
    for x, _y in chunks:
        assert not any(int(t) in _SENT for t in x.tolist())


def test_stream_doc_ids_and_eos_packing_preserved():
    """The FIM transform is per-document BEFORE packing: doc_ids stay aligned
    (non-decreasing, EOS closes each doc) and every complete interior doc
    carries exactly one of each sentinel."""
    tok = _FakeTokenizer()
    text = _code_doc(n_lines=6, words_per_line=5)
    streams = {"s0": [{"text": text}] * 500}
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=1.0)]
    patch, ds = _stream(src, streams, tok=tok, fim_sentinel_ids=_SENT,
                        emit_doc_ids=True, block_size=128)
    eos = tok.eos_token_id
    interior_runs = 0
    with patch:
        it = iter(ds)
        for _ in range(15):
            x, y, doc_ids = next(it)
            assert x.shape == y.shape == doc_ids.shape
            d = doc_ids.tolist()
            xs = x.tolist()
            # non-decreasing doc ids, 0-based
            assert d[0] == 0
            assert all(d[i + 1] >= d[i] for i in range(len(d) - 1))
            # runs
            bounds = [0] + [i for i in range(1, len(d)) if d[i] != d[i - 1]] \
                + [len(d)]
            # every doc switch is preceded by the closing EOS
            for i in bounds[1:-1]:
                assert xs[i - 1] == eos, "EOS must close each packed document"
            # interior (complete) runs: exactly one of each sentinel
            for a, b in zip(bounds[1:-2], bounds[2:-1]):
                run = xs[a:b]
                for sid in _SENT:
                    assert run.count(sid) == 1
                interior_runs += 1
    assert interior_runs >= 5, "test never saw a complete interior document"


# ---------------------------------------------------------------------------
# Real tokenizer (skipped when unavailable — offline/no cache).

def test_real_tokenizer_render_round_trip():
    """With the production SmolLM2 BPE: sentinels resolve to the reserved
    [THINKING]-slot ids (49153/49154/49155, inside the 49216 padded model
    vocab) and the rendered segments DECODE back to the original document
    (PSM reassembly at the text level)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    try:
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    sids = resolve_fim_sentinel_ids(tok, thinking_token_id=49152)
    assert sids == (49153, 49154, 49155)
    rng = random.Random(11)
    text = ("def add(a, b):\n    '''Add two numbers.'''\n    return a + b\n"
            "\n\ndef mul(a, b):\n    total = 0\n    for _ in range(b):\n"
            "        total = add(total, a)\n    return total\n")
    for _ in range(20):
        ids = render_fim_psm_ids(text, tok, rng=rng, sentinel_ids=sids)
        assert ids is not None
        pre, suf, mid = sids
        i_suf, i_mid = ids.index(suf), ids.index(mid)
        p = tok.decode(ids[1:i_suf])
        s = tok.decode(ids[i_suf + 1:i_mid])
        m = tok.decode(ids[i_mid + 1:])
        assert p + m + s == text


# ---------------------------------------------------------------------------
# Validation

def test_fim_incompatible_with_read_mask_source():
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=0.5, emit_read_mask=True)]
    with pytest.raises(ValueError, match="emit_read_mask"):
        MixedSourceStream(sources=src, tokenizer=_FakeTokenizer(),
                          block_size=64, thinking_token_id=None,
                          emit_read_mask=True)


def test_sentinel_collision_with_thinking_token_raises():
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=0.5)]
    with pytest.raises(ValueError, match="collide"):
        MixedSourceStream(sources=src, tokenizer=_FakeTokenizer(),
                          block_size=64, thinking_token_id=5002,
                          fim_sentinel_ids=_SENT)


def test_sentinel_ids_must_be_three_distinct():
    src = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                        weight=1.0, fim_rate=0.5)]
    with pytest.raises(ValueError, match="distinct"):
        MixedSourceStream(sources=src, tokenizer=_FakeTokenizer(),
                          block_size=64, thinking_token_id=None,
                          fim_sentinel_ids=(5001, 5001, 5003))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------- #
# Legacy string-sentinel replay mode (2026-07-14: reinstated for lineage-
# comparability continuations — see maybe_apply_fim's docstring).
# --------------------------------------------------------------------------- #

def test_legacy_maybe_apply_fim_verbatim_semantics():
    import random as _r
    from experiments.data_mix import maybe_apply_fim, _FIM_PREFIX, _FIM_SUFFIX, _FIM_MIDDLE
    rng = _r.Random(7)
    text = "abcdefghij"
    out = maybe_apply_fim(text, rng=rng, fim_rate=1.0)
    assert out.startswith(_FIM_PREFIX)
    # reassembly: prefix + middle + suffix == original
    body = out[len(_FIM_PREFIX):]
    pfx, rest = body.split(_FIM_SUFFIX, 1)
    sfx, mid = rest.split(_FIM_MIDDLE, 1)
    assert pfx + mid + sfx == text
    # rate=0 and short docs are unchanged
    assert maybe_apply_fim(text, rng=_r.Random(0), fim_rate=0.0) is text
    assert maybe_apply_fim("ab", rng=_r.Random(0), fim_rate=1.0) == "ab"


def test_legacy_mode_stream_no_reserved_ids(tmp_path):
    """fim_legacy_strings=True: no sentinel-id resolution (works on ckpts whose
    embedding is exactly the base vocab), transform is text-level, and the
    fim-doc accounting still fires."""
    import json as _json
    from experiments.data_mix import MixedSourceStream, SourceConfig
    p = tmp_path / "docs.jsonl"
    rows = [{"text": "x = 1\ny = 2\nz = x + y\nprint(z)\n" + "pad " * 30}
            for _ in range(30)]
    p.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    tok = _CharTok() if "_CharTok" in globals() else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    src = SourceConfig(name="d", weight=1.0, dataset_id="",
                       text_field="text", jsonl_path=str(p), fim_rate=1.0)
    ms = MixedSourceStream(sources=[src], tokenizer=tok, block_size=128,
                           thinking_token_id=None, think_burst_prob=0.0,
                           base_seed=0, fim_legacy_strings=True)
    assert ms.fim_sentinel_ids is None
    it = iter(ms)
    for _ in range(4):
        next(it)
    stats = ms.source_stats()
    assert stats[0]["fim_docs"] > 0          # transform fired + accounted
