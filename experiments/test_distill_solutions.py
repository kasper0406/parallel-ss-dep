"""Tests for distill_solutions.py — Qwen teacher generation + filter.

We don't load vLLM in the test (heavy + slow). Instead we test the
pure helpers: prompt construction, code-block extraction, grade
shimming, and JSONL emission.
"""
import json
import os
import tempfile

import pytest

from experiments.distill_solutions import (
    build_user_prompt,
    extract_code_block,
    grade_or_passthrough,
    write_jsonl_record,
)
from experiments.code_grader import Problem


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_build_user_prompt_natural_language():
    p = Problem(task_id="t", prompt="Write a function that adds two numbers.",
                tests="", entry_point="", prompt_is_code=False)
    s = build_user_prompt(p)
    # Must include the problem text and explicit instruction to think + code-block
    assert "adds two numbers" in s
    assert "step by step" in s.lower() or "think" in s.lower()
    assert "```" in s  # asks for code-block wrapping


def test_build_user_prompt_code_completion_format():
    """HumanEval-style: prompt IS code (function header + docstring).
    The instruction must tell Qwen to COMPLETE the function, not write
    a new one from scratch."""
    p = Problem(task_id="t",
                prompt='def add(a, b):\n    """Return a + b."""\n',
                tests="", entry_point="add", prompt_is_code=True)
    s = build_user_prompt(p)
    assert "def add" in s  # original code is in the prompt
    assert "complete" in s.lower() or "implement" in s.lower()


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------

def test_extract_code_block_python_fence():
    txt = ("Sure, here's the solution:\n\n"
           "```python\n"
           "def f(x):\n    return x * 2\n"
           "```\n\nThat's the answer.")
    code = extract_code_block(txt)
    assert code is not None
    assert "def f(x):" in code
    assert "return x * 2" in code
    # Prose around the block must be stripped
    assert "Sure" not in code
    assert "That's" not in code


def test_extract_code_block_bare_fence():
    txt = "```\nx = 1\n```"
    code = extract_code_block(txt)
    assert code is not None
    assert code.strip() == "x = 1"


def test_extract_code_block_no_fence_returns_none():
    """If no fenced block, return None so the caller can fall back
    to the raw text (or skip the row)."""
    code = extract_code_block("def f(x): return x + 1")
    assert code is None


def test_extract_code_block_picks_first_python_over_other_lang():
    """Qwen sometimes hedges with `bash` or `text` blocks before/after.
    We want the FIRST python (or unmarked) block."""
    txt = ("Here's a shell command first:\n```bash\nls\n```\n\n"
           "Now the Python:\n```python\ndef f(): return 1\n```\n")
    code = extract_code_block(txt)
    assert code is not None
    assert "def f()" in code
    assert "ls" not in code


# ---------------------------------------------------------------------------
# Grade-or-passthrough (no-tests problems should always "accept")
# ---------------------------------------------------------------------------

def test_grade_or_passthrough_no_tests_accepts():
    """A magicoder/codefeedback problem has tests="", so we cannot
    execute. Should return a sentinel acceptance, never call grade()."""
    p = Problem(task_id="t", prompt="...", tests="", entry_point="",
                prompt_is_code=False)
    tier, score = grade_or_passthrough(p, "def whatever(): pass")
    assert tier == "no_tests"
    assert score == 1.0  # passthrough acceptance


def test_grade_or_passthrough_with_tests_passes():
    """A real MBPP/HumanEval problem with valid completion → grade() runs."""
    p = Problem(task_id="t", prompt="", entry_point="f",
                tests="def check(candidate):\n    assert candidate(1) == 2\n",
                prompt_is_code=True)
    tier, score = grade_or_passthrough(p, "def f(x):\n    return x + 1\n")
    assert tier == "pass"
    assert score == 1.0


def test_grade_or_passthrough_with_tests_fails():
    p = Problem(task_id="t", prompt="", entry_point="f",
                tests="def check(candidate):\n    assert candidate(1) == 999\n",
                prompt_is_code=True)
    tier, score = grade_or_passthrough(p, "def f(x):\n    return x + 1\n")
    # one assert, fails → partial 0/1
    assert tier in ("partial", "syntax_error", "exec_error", "runtime_error")
    assert score < 1.0


# ---------------------------------------------------------------------------
# JSONL emission
# ---------------------------------------------------------------------------

def test_write_jsonl_record_round_trips():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "out.jsonl")
        with open(path, "w") as f:
            write_jsonl_record(f, task_id="t/1",
                               problem_prompt="solve this",
                               qwen_completion="```python\nx=1\n```",
                               extracted_code="x=1",
                               has_tests=False,
                               tier="no_tests",
                               score=1.0,
                               sample_idx=0)
            write_jsonl_record(f, task_id="t/2",
                               problem_prompt="other",
                               qwen_completion="```python\nx=2\n```",
                               extracted_code="x=2",
                               has_tests=True,
                               tier="pass",
                               score=1.0,
                               sample_idx=3)
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 2
        assert rows[0]["task_id"] == "t/1"
        assert rows[0]["tier"] == "no_tests"
        assert rows[0]["sample_idx"] == 0
        assert rows[1]["task_id"] == "t/2"
        assert rows[1]["tier"] == "pass"
        assert rows[1]["has_tests"] is True


# ---------------------------------------------------------------------------
# sft_code.py JSONL consumer
# ---------------------------------------------------------------------------

def test_sft_load_distilled_jsonl_full_completion(tmp_path):
    """load_distilled_jsonl returns (problem, solution) pairs where
    solution is the FULL qwen completion by default (CoT + code)."""
    from experiments.sft_code import load_distilled_jsonl

    jp = tmp_path / "d.jsonl"
    rows = [
        # Row 1: passing problem with extracted code
        {"task_id": "t/1", "sample_idx": 0, "problem_prompt": "P1",
         "qwen_completion": "Let me think...\n```python\nx=1\n```",
         "extracted_code": "x=1", "has_tests": True, "tier": "pass",
         "score": 1.0},
        # Row 2: distillation-only problem (no tests)
        {"task_id": "t/2", "sample_idx": 0, "problem_prompt": "P2",
         "qwen_completion": "```python\ny=2\n```",
         "extracted_code": "y=2", "has_tests": False, "tier": "no_tests",
         "score": 1.0},
        # Row 3: with-tests, FAILED → kept by default, dropped if keep_only_passing
        {"task_id": "t/3", "sample_idx": 0, "problem_prompt": "P3",
         "qwen_completion": "```python\nz=3\n```",
         "extracted_code": "z=3", "has_tests": True, "tier": "exec_error",
         "score": 0.05},
        # Row 4: no extracted code (truncated mid-CoT) → dropped
        {"task_id": "t/4", "sample_idx": 0, "problem_prompt": "P4",
         "qwen_completion": "Let me think a long thought...",
         "extracted_code": None, "has_tests": False, "tier": "no_tests",
         "score": 1.0},
    ]
    import json
    with open(jp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Default: prefer full completion, require extracted code, no
    # rejection sampling. Should keep rows 1, 2, 3 (row 4 has no code).
    pairs = load_distilled_jsonl(str(jp))
    assert len(pairs) == 3
    # First pair has FULL qwen completion (with CoT)
    assert "Let me think" in pairs[0][1]
    assert "```python" in pairs[0][1]


def test_sft_load_distilled_jsonl_code_only(tmp_path):
    """With prefer_full_completion=False, solution is just the extracted code."""
    from experiments.sft_code import load_distilled_jsonl
    jp = tmp_path / "d.jsonl"
    import json
    rows = [
        {"task_id": "t/1", "sample_idx": 0, "problem_prompt": "P1",
         "qwen_completion": "Let me think...\n```python\nx=1\n```",
         "extracted_code": "x=1", "has_tests": True, "tier": "pass",
         "score": 1.0},
    ]
    with open(jp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pairs = load_distilled_jsonl(str(jp), prefer_full_completion=False)
    assert len(pairs) == 1
    # Solution is the bare extracted code, no CoT
    assert pairs[0][1] == "x=1"
    assert "Let me think" not in pairs[0][1]


def test_sft_load_distilled_jsonl_keep_only_passing(tmp_path):
    """keep_only_passing drops failed-test rows but keeps no-tests rows."""
    from experiments.sft_code import load_distilled_jsonl
    jp = tmp_path / "d.jsonl"
    import json
    rows = [
        # has_tests + pass → kept
        {"task_id": "t/1", "sample_idx": 0, "problem_prompt": "P1",
         "qwen_completion": "```python\nx=1\n```",
         "extracted_code": "x=1", "has_tests": True, "tier": "pass",
         "score": 1.0},
        # has_tests + fail → DROPPED
        {"task_id": "t/2", "sample_idx": 0, "problem_prompt": "P2",
         "qwen_completion": "```python\ny=2\n```",
         "extracted_code": "y=2", "has_tests": True, "tier": "exec_error",
         "score": 0.05},
        # no_tests → KEPT regardless
        {"task_id": "t/3", "sample_idx": 0, "problem_prompt": "P3",
         "qwen_completion": "```python\nz=3\n```",
         "extracted_code": "z=3", "has_tests": False, "tier": "no_tests",
         "score": 1.0},
    ]
    with open(jp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pairs = load_distilled_jsonl(str(jp), keep_only_passing=True)
    # Only t/1 (passed) and t/3 (no tests) kept
    assert len(pairs) == 2
    problems = [p for p, _ in pairs]
    assert "P1" in problems and "P3" in problems
    assert "P2" not in problems
