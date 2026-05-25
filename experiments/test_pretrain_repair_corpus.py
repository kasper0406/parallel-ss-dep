"""Tests for `experiments.build_pretrain_repair_corpus`."""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.build_pretrain_repair_corpus import (
    build_corpus,
    convert_row,
    parse_problem_prompt,
    render_corpus_text,
)


# ---------------------------------------------------------------------------
# Synthetic minimal input row (matches the schema of
# `data/repair_triples_v3.jsonl` as produced by `gen_repair_triples.py`).

SAMPLE_PROMPT = (
    "# The previous attempt failed. Fix the code below.\n"
    "# Write a function to compute the factorial of n.\n"
    "# Example: assert factorial(5) == 120\n"
    "# Attempted solution:\n"
    "def factorial(n):\n"
    "    return n\n"
    "# Error:\n"
    "# AssertionError: factorial(5) returned 5, expected 120\n"
    "# Fix the code:\n"
)

SAMPLE_CANONICAL = (
    "def factorial(n):\n"
    "    if n <= 1:\n"
    "        return 1\n"
    "    return n * factorial(n - 1)\n"
)

SAMPLE_ROW = {
    "task_id": "repair/mbpp_train/123/4",
    "sample_idx": 4,
    "problem_prompt": SAMPLE_PROMPT,
    "qwen_completion": f"```python\n{SAMPLE_CANONICAL}```",
    "extracted_code": SAMPLE_CANONICAL,
    "has_tests": True,
    "tier": "pass",
    "score": 1.0,
    "source_tier": "exec_error",
    "source_score": 0.05,
}


# ---------------------------------------------------------------------------
# parse_problem_prompt: spec requirement #1 — extracts description correctly.

def test_parse_extracts_description():
    parsed = parse_problem_prompt(SAMPLE_PROMPT)
    assert "Write a function to compute the factorial of n." in parsed.description
    assert "Example: assert factorial(5) == 120" in parsed.description
    # The `# ` prefix must be stripped.
    assert not parsed.description.startswith("# ")
    assert "# Write a function" not in parsed.description


def test_parse_extracts_failed_code():
    parsed = parse_problem_prompt(SAMPLE_PROMPT)
    assert parsed.failed_code == "def factorial(n):\n    return n"


def test_parse_extracts_error_text():
    parsed = parse_problem_prompt(SAMPLE_PROMPT)
    assert "AssertionError: factorial(5) returned 5, expected 120" == parsed.error_text


def test_parse_multiline_error():
    """Error block may have multiple `# `-prefixed lines."""
    prompt = (
        "# The previous attempt failed. Fix the code below.\n"
        "# Sort a list.\n"
        "# Attempted solution:\n"
        "def sort_it(xs): return xs\n"
        "# Error:\n"
        "# Traceback (most recent call last):\n"
        "#   File 'x.py', line 1\n"
        "# TypeError: bad arg\n"
        "# Fix the code:\n"
    )
    parsed = parse_problem_prompt(prompt)
    assert "Traceback" in parsed.error_text
    assert "TypeError: bad arg" in parsed.error_text
    assert parsed.error_text.count("\n") == 2  # 3 lines


def test_parse_missing_marker_raises():
    with pytest.raises(ValueError):
        parse_problem_prompt("not the right format")


# ---------------------------------------------------------------------------
# render_corpus_text: spec requirement #2 — output text contains failed_code,
# error_text, and canonical_solution in the expected order.

def test_render_contains_all_parts_in_order():
    parsed = parse_problem_prompt(SAMPLE_PROMPT)
    text = render_corpus_text(parsed, SAMPLE_CANONICAL)
    # Order check.
    p_original = text.index("# Original problem:")
    p_attempt = text.index("# Attempted solution:")
    p_error = text.index("# Got this error:")
    p_fixed = text.index("# Fixed version:")
    assert p_original < p_attempt < p_error < p_fixed
    # Content check.
    assert "def factorial(n):" in text
    assert "return n" in text  # failed body
    assert "AssertionError" in text
    assert "return n * factorial(n - 1)" in text  # canonical body


def test_render_ends_with_blank_line():
    """Spec: blank line at end gives a natural document boundary."""
    parsed = parse_problem_prompt(SAMPLE_PROMPT)
    text = render_corpus_text(parsed, SAMPLE_CANONICAL)
    assert text.endswith("\n")


# ---------------------------------------------------------------------------
# Spec requirement #3 — output is valid `# ...` commented Python (parses
# cleanly when comment-only lines are stripped, the code parts must still
# form a valid Python file).

def _strip_comments(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def test_output_parses_as_python_after_stripping_comments():
    parsed = parse_problem_prompt(SAMPLE_PROMPT)
    text = render_corpus_text(parsed, SAMPLE_CANONICAL)
    stripped = _strip_comments(text)
    # Both the failed code (def factorial(n): return n) and the canonical
    # solution must be syntactically valid Python when concatenated.
    ast.parse(stripped)


# ---------------------------------------------------------------------------
# convert_row: end-to-end on the synthetic sample.

def test_convert_row_basic():
    out = convert_row(SAMPLE_ROW)
    assert out is not None
    assert out["task_id"] == "repair/mbpp_train/123/4"
    assert out["source"] == "self_debug"
    assert "def factorial(n):" in out["text"]
    assert "AssertionError" in out["text"]


def test_convert_row_empty_canonical_returns_none():
    bad = dict(SAMPLE_ROW)
    bad["extracted_code"] = ""
    assert convert_row(bad) is None


def test_convert_row_no_prompt_returns_none():
    bad = dict(SAMPLE_ROW)
    bad["problem_prompt"] = None
    assert convert_row(bad) is None


def test_convert_row_malformed_prompt_returns_none():
    bad = dict(SAMPLE_ROW)
    bad["problem_prompt"] = "garbage text without markers"
    assert convert_row(bad) is None


# ---------------------------------------------------------------------------
# Spec requirement #4 — tokenize/decode round-trip is near-identical
# (allowing tokenizer collapses, e.g. whitespace normalisation).

def test_tokenizer_roundtrip_near_identical():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")
    try:
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception as e:
        pytest.skip(f"tokenizer download failed: {e}")
    parsed = parse_problem_prompt(SAMPLE_PROMPT)
    text = render_corpus_text(parsed, SAMPLE_CANONICAL)
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=True)
    # All semantically-load-bearing tokens must survive the round-trip.
    # We don't assert byte-exact equality — tokenizers normalise whitespace,
    # collapse repeated newlines, etc.
    for needle in ["factorial", "AssertionError", "Original problem",
                   "Attempted solution", "Got this error", "Fixed version"]:
        assert needle in decoded, f"{needle!r} lost in tokenize/decode"


# ---------------------------------------------------------------------------
# End-to-end build_corpus on a tiny temp JSONL.

def test_build_corpus_end_to_end(tmp_path):
    in_path = tmp_path / "repair_in.jsonl"
    out_path = tmp_path / "repair_out.jsonl"
    rows = [SAMPLE_ROW, dict(SAMPLE_ROW, task_id="repair/mbpp_train/123/5",
                              extracted_code="")]
    with open(in_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    stats = build_corpus(in_path, out_path)
    assert stats["n_in"] == 2
    assert stats["n_out"] == 1   # the empty-canonical row was dropped
    assert stats["n_tokens_approx"] > 0
    # The written file should be valid JSONL with the expected fields.
    with open(out_path) as f:
        out_rows = [json.loads(line) for line in f]
    assert len(out_rows) == 1
    assert out_rows[0]["task_id"] == "repair/mbpp_train/123/4"
    assert out_rows[0]["source"] == "self_debug"
    assert "text" in out_rows[0]


# ---------------------------------------------------------------------------
# Real-corpus sanity (skipped if the file isn't available — it's gitignored).

def test_real_corpus_if_present():
    p = pathlib.Path("data/pretrain_repair_corpus.jsonl")
    if not p.exists():
        pytest.skip("data/pretrain_repair_corpus.jsonl not present")
    with open(p) as f:
        first = json.loads(f.readline())
    assert set(first.keys()) >= {"task_id", "text", "source"}
    assert first["source"] == "self_debug"
    assert first["task_id"].startswith("repair/")
    assert "Original problem:" in first["text"]
    assert "Fixed version:" in first["text"]
