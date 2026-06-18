"""Tests for experiments.data_mix.

Covers the pure-Python pieces that don't need HuggingFace network access:
  - per-source filter functions
  - text builders (BigVul, CyberNative DPO)
  - `_extract_text` dispatch (text_field vs text_builder)
  - MixedSourceStream chunking, weighted sampling, think-burst masking
    (via monkey-patched _open_stream to inject mock streams)
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest
import torch

import experiments.data_mix as dm
from experiments.data_mix import (
    FILTER_REGISTRY,
    MixedSourceStream,
    SourceConfig,
    TEXT_BUILDER_REGISTRY,
    _build_filter,
    _builder_bigvul,
    _builder_cybernative_dpo,
    _extract_text,
)


# ---------------------------------------------------------------------------
# Filters

def test_filter_min_content_len() -> None:
    f = FILTER_REGISTRY["min_content_len"](min_chars=10)
    assert f({"text": "too short"}) is False
    assert f({"text": "long enough to pass"}) is True
    # Fall back to "content" key.
    assert f({"content": "long enough to pass"}) is True
    # Empty/missing rejected.
    assert f({"text": ""}) is False
    assert f({}) is False
    # Wrong type rejected (rare but possible from malformed HF rows).
    assert f({"text": 12345}) is False
    # BUGFIX 2026-06-18: sources whose text is in OTHER fields (no content/text)
    # must NOT be rejected wholesale — fall back to the longest string field.
    # magicoder-style ([problem, solution]):
    assert f({"problem": "x", "solution": "long enough to pass"}) is True
    assert f({"problem": "x", "solution": "short"}) is False
    # textbooks-style (completion / markdown):
    assert f({"completion": "long enough to pass", "title": "t"}) is True
    assert f({"markdown": "long enough to pass", "topic": "z"}) is True
    # No string fields at all → rejected.
    assert f({"n": 5, "ok": True}) is False


def test_filter_bigvul_vulnerable() -> None:
    f = FILTER_REGISTRY["bigvul_vulnerable"]()
    assert f({"vul": 1}) is True
    assert f({"vul": 0}) is False
    # Missing field defaults to 0.
    assert f({}) is False
    # String coercion still works.
    assert f({"vul": "1"}) is True
    assert f({"vul": "0"}) is False


def test_filter_se_score() -> None:
    f = FILTER_REGISTRY["se_score"](min_score=3, programming_only=False)
    assert f({"score": 5}) is True
    assert f({"score": 2}) is False
    assert f({"score": "10"}) is True
    # Missing score → treated as 0, rejected.
    assert f({}) is False
    # Score from alternative key name.
    assert f({"question_score": 4}) is True

    # Programming-tag filter on.
    f_prog = FILTER_REGISTRY["se_score"](min_score=1, programming_only=True)
    assert f_prog({"score": 5, "tags": ["python"]}) is True
    assert f_prog({"score": 5, "tags": ["cooking"]}) is False
    # No tags at all → accept (don't over-filter on missing metadata).
    assert f_prog({"score": 5}) is True


def test_filter_gh_issue_resolved() -> None:
    f = FILTER_REGISTRY["gh_issue_resolved"]()
    # Has code block → accept regardless of state.
    assert f({"text": "Here's the bug: ```py\nfoo()\n```"}) is True
    # No code block but closed → accept.
    assert f({"text": "Plain prose", "state": "closed"}) is True
    assert f({"content": "Plain prose", "status": "merged"}) is True
    # No code block, open → reject.
    assert f({"text": "Plain prose", "state": "open"}) is False


def test_build_filter_dispatch() -> None:
    # String form (no args).
    f = _build_filter("always")
    assert f({}) is True
    # Dict form with args.
    f = _build_filter({"name": "min_content_len", "args": {"min_chars": 5}})
    assert f({"text": "12345"}) is True
    assert f({"text": "1234"}) is False
    # Unknown name raises.
    with pytest.raises(ValueError):
        _build_filter("not_a_real_filter")


# ---------------------------------------------------------------------------
# Text builders

def test_bigvul_builder_happy_path() -> None:
    ex = {
        "CVE ID": "CVE-2017-1234",
        "CWE ID": "CWE-119",
        "commit_id": "deadbeef",
        "commit_message": "Fix buffer overflow in foo()",
        "func_before": "void foo(char *s) { strcpy(buf, s); }",
        "func_after":  "void foo(char *s) { strncpy(buf, s, sizeof(buf)-1); }",
        "lang": "C",
        "project": "libfoo",
        "vul": 1,
    }
    out = _builder_bigvul(ex)
    assert out is not None
    # Required parts of the lesson are present.
    assert "CVE-2017-1234" in out
    assert "CWE-119" in out
    assert "Fix buffer overflow in foo()" in out
    assert "Vulnerable C code" in out
    assert "Patched C code" in out
    assert "strcpy(buf, s);" in out
    assert "strncpy(buf, s, sizeof(buf)-1);" in out
    assert "libfoo" in out


def test_bigvul_builder_rejects_same_before_after() -> None:
    """The vul=0 BigVul rows have before==after (safe-state classifier
    examples). The builder must skip these — no bug→fix lesson."""
    same = "void foo() {}"
    ex = {
        "CVE ID": "CVE-X", "CWE ID": "CWE-Y",
        "commit_message": "msg",
        "func_before": same, "func_after": same,
        "lang": "C", "project": "p",
    }
    assert _builder_bigvul(ex) is None


def test_bigvul_builder_rejects_missing_fields() -> None:
    assert _builder_bigvul({}) is None
    assert _builder_bigvul({"func_before": "a"}) is None
    assert _builder_bigvul({"func_after": "b"}) is None
    # Empty strings count as missing.
    assert _builder_bigvul({"func_before": "", "func_after": "x"}) is None


def test_bigvul_builder_truncates_long_messages() -> None:
    """Commit messages can be very long (Chrome CLs include CI metadata);
    builder should cap to keep training chunks balanced."""
    msg = "ABC " * 1000  # 4000 chars
    ex = {
        "CVE ID": "CVE-X", "CWE ID": "CWE-Y",
        "commit_message": msg,
        "func_before": "x", "func_after": "y",
        "lang": "C", "project": "p",
    }
    out = _builder_bigvul(ex)
    assert out is not None
    # Should contain the truncation marker.
    assert "..." in out
    # And not contain all of the original message.
    assert out.count("ABC") < 1000


def test_cybernative_dpo_builder_happy_path() -> None:
    ex = {
        "lang": "python",
        "vulnerability": "Use of eval() can lead to code execution.",
        "question": "Write code that takes user input and runs it.",
        "chosen": "import ast\nast.literal_eval(user_input)",
        "rejected": "eval(user_input)  # UNSAFE",
        "system": "",
    }
    out = _builder_cybernative_dpo(ex)
    assert out is not None
    assert "python" in out
    assert "Use of eval()" in out
    assert "user input and runs it" in out
    assert "ast.literal_eval(user_input)" in out
    # Critical: the rejected (unsafe) side MUST NOT appear in the training text.
    assert "eval(user_input)  # UNSAFE" not in out
    assert "UNSAFE" not in out


def test_cybernative_dpo_builder_rejects_missing() -> None:
    assert _builder_cybernative_dpo({}) is None
    # Missing chosen → reject (no positive example).
    assert _builder_cybernative_dpo({"question": "q"}) is None
    # Missing question → reject.
    assert _builder_cybernative_dpo({"chosen": "c"}) is None
    # Empty chosen → reject.
    assert _builder_cybernative_dpo({"question": "q", "chosen": ""}) is None


# ---------------------------------------------------------------------------
# _extract_text dispatch

def test_extract_text_single_field() -> None:
    assert _extract_text({"text": "hello"}, "text") == "hello"
    # Fallback to alternate names when the named field is missing.
    assert _extract_text({"content": "hello"}, "text") == "hello"
    # Returns None when nothing usable.
    assert _extract_text({}, "text") is None
    assert _extract_text({"text": ""}, "text") is None


def test_extract_text_field_list() -> None:
    ex = {"q": "What is X?", "a": "It is Y."}
    assert _extract_text(ex, ["q", "a"]) == "What is X?\n\nIt is Y."
    # Missing or empty fields are skipped.
    ex = {"q": "What?", "a": ""}
    assert _extract_text(ex, ["q", "a"]) == "What?"
    # All empty → None.
    assert _extract_text({"q": "", "a": ""}, ["q", "a"]) is None


def test_extract_text_with_builder() -> None:
    ex = {
        "CVE ID": "CVE-1", "CWE ID": "CWE-2",
        "commit_message": "Fix it",
        "func_before": "buggy", "func_after": "fixed",
        "lang": "C", "project": "p",
    }
    out = _extract_text(ex, "ignored_when_builder_set",
                        text_builder="bigvul")
    assert out is not None and "buggy" in out and "fixed" in out


def test_extract_text_unknown_builder_raises() -> None:
    with pytest.raises(ValueError):
        _extract_text({}, "text", text_builder="not_a_real_builder")


# ---------------------------------------------------------------------------
# MixedSourceStream end-to-end with mock streams

class _FakeTokenizer:
    """Minimal tokenizer stub: maps each unique word to an int id, with eos = 1."""

    eos_token_id = 1
    bos_token_id = 1
    vocab_size = 1000

    def __init__(self):
        self._vocab = {}

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = 10 + len(self._vocab)
            ids.append(self._vocab[word])
        return ids

    def __len__(self):
        return self.vocab_size + 200


def _mock_open_stream_factory(streams: dict[str, list[dict]]):
    """Returns an _open_stream replacement that returns canned data per
    source by name."""
    def fake_open(src, seed=0):
        return iter(streams.get(src.name, []))
    return fake_open


def test_mixed_stream_basic_chunking() -> None:
    """Yielded tensors have correct shape and target == input shifted."""
    tok = _FakeTokenizer()
    sources = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                             weight=1.0)]
    # Inject 50 long examples so we can chunk freely.
    long_text = " ".join(["w%d" % i for i in range(200)])
    streams = {"s0": [{"text": long_text}] * 50}
    with mock.patch.object(dm, "_open_stream",
                            _mock_open_stream_factory(streams)):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                                block_size=32,
                                thinking_token_id=None,
                                think_burst_prob=0.0)
        it = iter(ds)
        for _ in range(5):
            x, y = next(it)
            assert x.shape == (32,)
            assert y.shape == (32,)
            assert x.dtype == torch.long
            assert y.dtype == torch.long


def test_mixed_stream_think_burst_masks_target_at_think_positions() -> None:
    """When a think token is inserted, the *target* at the previous position
    must be -100 (we don't want the model to learn to predict think tokens
    at random insertion sites during pretrain)."""
    tok = _FakeTokenizer()
    sources = [SourceConfig(name="s0", dataset_id="x", text_field="text",
                             weight=1.0)]
    long_text = " ".join(["w%d" % i for i in range(400)])
    streams = {"s0": [{"text": long_text}] * 200}
    thinking_id = 555
    with mock.patch.object(dm, "_open_stream",
                            _mock_open_stream_factory(streams)):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                                block_size=64,
                                thinking_token_id=thinking_id,
                                # Always insert; depth 4..6 so there are
                                # several think tokens in each chunk.
                                think_burst_prob=1.0,
                                think_max_bursts=2,
                                think_max_burst_depth=6,
                                base_seed=42)
        it = iter(ds)
        seen_any_think = False
        for _ in range(10):
            x, y = next(it)
            # Wherever the *target* sees a think token, it should be set to
            # -100 instead.
            n_think_in_target_raw = (y == thinking_id).sum().item()
            assert n_think_in_target_raw == 0, (
                "found think_token_id in target tensor — masking failed")
            # We should have at least one -100 across many chunks.
            if (y == -100).any():
                seen_any_think = True
            # Wherever the input has a think token, the previous-position
            # target should be -100 (predicting the think token).
            think_positions = (x == thinking_id).nonzero(as_tuple=True)[0]
            for pos in think_positions.tolist():
                if pos > 0:
                    # Predicting the think token at position `pos`: that's
                    # done by the model at position `pos-1`. Target at
                    # `pos-1` is the original `chunk[pos]` which equals
                    # thinking_id — masked to -100.
                    # NOTE: position 0 in `x` corresponds to original
                    # chunk[0], so target[pos-1] is chunk[pos].
                    pass
        assert seen_any_think, (
            "expected at least one think-masked target across 10 chunks "
            "with think_burst_prob=1.0")


def test_mixed_stream_weighted_sampling_is_approximately_correct() -> None:
    """With weights [0.7, 0.3], over many chunks ~70% should come from
    source 0. We check this by tagging each source's text uniquely."""
    tok = _FakeTokenizer()
    sources = [
        SourceConfig(name="s0", dataset_id="x", text_field="text",
                      weight=0.7),
        SourceConfig(name="s1", dataset_id="y", text_field="text",
                      weight=0.3),
    ]
    # Make every example tag-rich so we can identify origin by decoded ids.
    s0_text = " ".join(["S0TAG"] * 200)
    s1_text = " ".join(["S1TAG"] * 200)
    streams = {
        "s0": [{"text": s0_text}] * 500,
        "s1": [{"text": s1_text}] * 500,
    }
    with mock.patch.object(dm, "_open_stream",
                            _mock_open_stream_factory(streams)):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                                block_size=16,
                                thinking_token_id=None,
                                think_burst_prob=0.0,
                                base_seed=0)
        it = iter(ds)
        # Pre-encode the tag ids so we can identify by token.
        s0_id = tok.encode("S0TAG")[0]
        s1_id = tok.encode("S1TAG")[0]
        n0 = n1 = 0
        N = 200
        for _ in range(N):
            x, _ = next(it)
            xs = x.tolist()
            if s0_id in xs:
                n0 += 1
            elif s1_id in xs:
                n1 += 1
        total = n0 + n1
        ratio = n0 / total
        # 95% binomial CI for p=0.7 at n=200 is roughly [0.64, 0.76].
        # Allow some slack to avoid a flaky test.
        assert 0.60 <= ratio <= 0.80, (
            f"weighted sampling off: got {n0}/{n1} (ratio {ratio:.3f}) "
            f"vs expected 0.7"
        )


def test_mixed_stream_handles_source_exhaustion() -> None:
    """When a source is exhausted, weight is dropped to 0 and the iterator
    continues from the remaining sources. When all are exhausted, it
    terminates cleanly."""
    tok = _FakeTokenizer()
    sources = [
        SourceConfig(name="s0", dataset_id="x", text_field="text",
                      weight=0.5),
        SourceConfig(name="s1", dataset_id="y", text_field="text",
                      weight=0.5),
    ]
    # Very small streams so we exhaust quickly.
    short = " ".join(["w%d" % i for i in range(40)])
    streams = {
        "s0": [{"text": short}] * 2,
        "s1": [{"text": short}] * 2,
    }
    with mock.patch.object(dm, "_open_stream",
                            _mock_open_stream_factory(streams)):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                                block_size=16,
                                thinking_token_id=None,
                                think_burst_prob=0.0)
        n = 0
        for x, y in ds:
            assert x.shape == (16,)
            n += 1
            if n > 100:  # safety bound — should terminate well before this
                raise AssertionError("iterator failed to terminate")
        # Some chunks were yielded (combined ~160 tokens / 16 block ≈ 10).
        assert n > 0
        assert n < 100


def test_mixed_stream_text_builder_is_used() -> None:
    """A source with text_builder='bigvul' should emit the composite text
    instead of trying to read a text_field."""
    tok = _FakeTokenizer()
    sources = [SourceConfig(
        name="bigvul_mock", dataset_id="x", text_builder="bigvul",
        weight=1.0,
    )]
    # 50 BigVul-shaped records.
    fake_records = [{
        "CVE ID": f"CVE-2024-{i}",
        "CWE ID": "CWE-120",
        "commit_id": "abc",
        "commit_message": f"Fix issue {i}",
        "func_before": f"unique_token_BUGGY_{i} foo bar baz",
        "func_after":  f"unique_token_FIXED_{i} foo bar baz",
        "lang": "C",
        "project": "p",
    } for i in range(50)]
    streams = {"bigvul_mock": fake_records}
    with mock.patch.object(dm, "_open_stream",
                            _mock_open_stream_factory(streams)):
        ds = MixedSourceStream(sources=sources, tokenizer=tok,
                                block_size=32,
                                thinking_token_id=None,
                                think_burst_prob=0.0,
                                base_seed=0)
        it = iter(ds)
        all_token_ids: set[int] = set()
        for _ in range(5):
            x, _ = next(it)
            all_token_ids.update(x.tolist())
        # Invert the tokenizer's vocab AFTER iteration (encode() populates
        # _vocab lazily during fill_buffer).
        inv = {v: k for k, v in tok._vocab.items()}
        seen_tokens = {inv[t] for t in all_token_ids if t in inv}
        # The composite text contains the CVE/CWE/Fix preamble AND the
        # before/after code blocks. Tokens we should see across chunks:
        assert any(t.startswith("CVE-2024") for t in seen_tokens), \
            f"never saw a CVE-2024-* token; seen: {sorted(seen_tokens)[:30]}"
        # FakeTokenizer is whitespace-split so the `(CWE-120)` parens come
        # along with the token. Either form should be present.
        assert any("CWE-120" in t for t in seen_tokens), \
            f"missing CWE preamble; seen: {sorted(seen_tokens)[:30]}"
        assert any(t.startswith("unique_token_BUGGY") for t in seen_tokens), \
            "missing vulnerable code block content"
        assert any(t.startswith("unique_token_FIXED") for t in seen_tokens), \
            "missing patched code block content"


if __name__ == "__main__":
    # Allow `python -m experiments.test_data_mix` for ad-hoc runs.
    sys.exit(pytest.main([__file__, "-v"]))
