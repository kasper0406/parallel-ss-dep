"""Tests for `experiments.gen_synthetic_pyfunc_data` and the JSONL
source-type integration in `experiments.data_mix`.

These tests don't require a GPU or Qwen — they cover:
  - Topic list shape (non-empty, > 30 entries)
  - Prompt template validity across all topics
  - JSONL writer schema
  - Code-fence extraction on a known Qwen-style output
  - data_mix JSONL source: end-to-end load via YAML, stream rows
"""
from __future__ import annotations

import io
import json
import pathlib
import random
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.gen_synthetic_pyfunc_data import (
    DEFAULT_TOPICS,
    build_prompt,
    extract_code_block,
    looks_like_function,
    write_jsonl_record,
    _draw_difficulty,
)


# ---------------------------------------------------------------------------
# Topic catalog
# ---------------------------------------------------------------------------

def test_default_topics_nonempty_and_large() -> None:
    assert isinstance(DEFAULT_TOPICS, list)
    assert len(DEFAULT_TOPICS) > 30, \
        f"want > 30 topics for diversity, got {len(DEFAULT_TOPICS)}"
    # No duplicates.
    assert len(set(DEFAULT_TOPICS)) == len(DEFAULT_TOPICS)
    # All non-empty strings.
    for t in DEFAULT_TOPICS:
        assert isinstance(t, str) and t.strip()


def test_prompt_template_valid_for_all_topics() -> None:
    """The prompt template must render cleanly for every topic at both
    difficulty levels, and the rendered prompt must contain key markers
    (the topic itself, the python-fence instruction)."""
    for topic in DEFAULT_TOPICS:
        for diff in ("easy", "medium"):
            p = build_prompt(topic, diff)
            assert isinstance(p, str) and p.strip()
            assert topic in p
            assert diff in p
            assert "```python```" in p or "```python" in p


# ---------------------------------------------------------------------------
# Difficulty draw is biased toward easy
# ---------------------------------------------------------------------------

def test_draw_difficulty_biased_easy() -> None:
    rng = random.Random(0)
    draws = [_draw_difficulty(rng) for _ in range(2000)]
    n_easy = sum(1 for d in draws if d == "easy")
    n_med = sum(1 for d in draws if d == "medium")
    assert n_easy + n_med == 2000
    # ~75% easy → CI band [0.70, 0.80] for n=2000 is far inside any noise.
    assert n_easy / 2000 > 0.65, f"expected easy-biased draws, got {n_easy/2000:.3f}"


# ---------------------------------------------------------------------------
# Fence extraction
# ---------------------------------------------------------------------------

_QWEN_STYLE_OUTPUT = """\
I'll write a function that reverses a string.

First, let me think about edge cases: empty strings, single chars, unicode...

```python
def reverse_string(s: str) -> str:
    \"\"\"Return s reversed.

    >>> reverse_string("abc")
    'cba'
    \"\"\"
    return s[::-1]
```

That's the cleanest implementation.
"""


def test_extract_code_block_qwen_style() -> None:
    code = extract_code_block(_QWEN_STYLE_OUTPUT)
    assert code is not None
    assert code.startswith("def reverse_string")
    assert "return s[::-1]" in code
    # The CoT prose must NOT appear in the extracted code.
    assert "I'll write" not in code
    assert "edge cases" not in code


def test_extract_code_block_no_fence() -> None:
    assert extract_code_block("just prose, no code") is None
    assert extract_code_block("") is None


def test_extract_code_block_falls_back_to_unmarked_fence() -> None:
    txt = "Here you go:\n\n```\ndef f(): return 1\n```\n"
    code = extract_code_block(txt)
    assert code == "def f(): return 1"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_looks_like_function_accepts_valid() -> None:
    good = 'def f(x: int) -> int:\n    """Doc.\n\n    >>> f(1)\n    2\n    """\n    return x + 1\n'
    assert looks_like_function(good) is True


def test_looks_like_function_rejects_syntax_error() -> None:
    bad = "def f(:\n    return 1"
    assert looks_like_function(bad) is False


def test_looks_like_function_rejects_no_def() -> None:
    bad = "x = 5\nprint(x)\n"
    assert looks_like_function(bad) is False


def test_looks_like_function_rejects_no_docstring() -> None:
    bad = "def f(x):\n    return x\n"
    assert looks_like_function(bad) is False


# ---------------------------------------------------------------------------
# JSONL writer schema
# ---------------------------------------------------------------------------

def test_write_jsonl_record_schema() -> None:
    buf = io.StringIO()
    write_jsonl_record(buf, task_id="synth_pyfunc/list_filter/00000",
                       text="def f(x): return x\n", source="synthetic_pyfunc")
    write_jsonl_record(buf, task_id="synth_pyfunc/list_filter/00001",
                       text="def g(x): return -x\n", source="synthetic_pyfunc")
    lines = buf.getvalue().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert set(row.keys()) == {"task_id", "text", "source"}
        assert row["task_id"].startswith("synth_pyfunc/")
        assert row["source"] == "synthetic_pyfunc"
        assert row["text"].startswith("def ")


# ---------------------------------------------------------------------------
# data_mix JSONL source integration
# ---------------------------------------------------------------------------

def test_jsonl_stream_iterates_rows_and_cycles() -> None:
    from experiments.data_mix import _jsonl_stream
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for i in range(5):
            f.write(json.dumps({"task_id": f"t{i}", "text": f"body {i}",
                                "source": "synth"}) + "\n")
        p = f.name
    it = _jsonl_stream(p, seed=0)
    seen = []
    for _ in range(12):  # cycles past 5 rows so we verify forever-cycling
        seen.append(next(it))
    assert len(seen) == 12
    # All emitted rows are dicts with the expected keys.
    for row in seen:
        assert isinstance(row, dict)
        assert "text" in row and "task_id" in row
    # Each original row was emitted at least once in the first 5 draws.
    first_5 = {r["task_id"] for r in seen[:5]}
    assert first_5 == {f"t{i}" for i in range(5)}


def test_yaml_loader_accepts_jsonl_path() -> None:
    from experiments.data_mix import load_sources_from_yaml
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as jf:
        jf.write(json.dumps({"task_id": "t0", "text": "def f(): pass\n",
                             "source": "synth"}) + "\n")
        jsonl_path = jf.name
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as yf:
        yf.write(f"""sources:
  - name: synth
    jsonl_path: {jsonl_path}
    text_field: text
    weight: 1.0
    filter: always
""")
        ypath = yf.name
    srcs = load_sources_from_yaml(ypath)
    assert len(srcs) == 1
    assert srcs[0].jsonl_path == jsonl_path
    assert srcs[0].dataset_id == ""


def test_yaml_loader_rejects_source_with_neither_dataset_id_nor_jsonl() -> None:
    from experiments.data_mix import load_sources_from_yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as yf:
        yf.write("""sources:
  - name: bad
    text_field: text
    weight: 1.0
""")
        ypath = yf.name
    with pytest.raises(ValueError, match="dataset_id"):
        load_sources_from_yaml(ypath)


def test_mixed_source_stream_jsonl_end_to_end() -> None:
    """Build a MixedSourceStream from a JSONL-backed SourceConfig and
    pull a chunk through. No HuggingFace network access."""
    from experiments.data_mix import MixedSourceStream, SourceConfig
    # Fake tokenizer: id = char-ord, eos = 0.
    class FakeTok:
        eos_token_id = 0
        bos_token_id = 0
        vocab_size = 256
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 250 + 1 for c in text]  # avoid 0 (= eos)

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as jf:
        for i in range(20):
            jf.write(json.dumps(
                {"task_id": f"t{i}",
                 "text": "def f" + str(i) + "(x): return x*" + str(i) + "\n"
                         "    # body filler text " * 20,
                 "source": "synth"}) + "\n")
        jsonl_path = jf.name

    sc = SourceConfig(name="synth", jsonl_path=jsonl_path,
                      text_field="text", weight=1.0, filter_spec="always")
    ds = MixedSourceStream(
        sources=[sc], tokenizer=FakeTok(), block_size=128,
        thinking_token_id=None, think_burst_prob=0.0,
    )
    it = iter(ds)
    inp, tgt = next(it)
    assert inp.shape == (128,)
    assert tgt.shape == (128,)
    inp2, tgt2 = next(it)
    assert inp2.shape == (128,)
