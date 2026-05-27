"""Tests for the Phase-4 CoT distillation pipeline:
- experiments/gen_cot_distill_data.py (prompt + parser + validator)
- experiments/build_cot_sft_data.py (triple -> distill-shaped row)
"""
from __future__ import annotations

import json
import pathlib

from experiments.code_grader import Problem
from experiments.gen_cot_distill_data import (
    build_prompt,
    parse_cot_response,
    count_thinking_steps,
    validate_triple,
    extract_code_block,
)
from experiments.build_cot_sft_data import build_sft_row


# ---------------------------------------------------------------------------
# Sample fixtures.
# ---------------------------------------------------------------------------

_PROBLEM_NL = Problem(
    task_id="mbpp/3",
    prompt="Write a function that adds two numbers.",
    tests="def check(c):\n    assert c(1, 2) == 3\n",
    entry_point="add",
    prompt_is_code=False,
)

_PROBLEM_CODE = Problem(
    task_id="humaneval/0",
    prompt="def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
    tests="def check(c):\n    assert c(1, 2) == 3\n",
    entry_point="add",
    prompt_is_code=True,
)

_GOOD_QWEN_RESPONSE = """\
THINKING:
1. We need a function that takes two numbers.
2. It should return their sum using the `+` operator.
3. Handle both ints and floats — Python's `+` is generic.

SOLUTION:
```python
def add(a, b):
    return a + b
```
"""

_BAD_NO_STEPS = """\
THINKING:
Just add the two numbers together.

SOLUTION:
```python
def add(a, b):
    return a + b
```
"""

_BAD_NO_SOLUTION_HEADER = """\
THINKING:
1. Add them.
2. Return the result.

```python
def add(a, b):
    return a + b
```
"""

_BAD_BROKEN_CODE = """\
THINKING:
1. Add them.
2. Return result.

SOLUTION:
```python
def add(a, b
    return a + b
```
"""


# ---------------------------------------------------------------------------
# Prompt template tests.
# ---------------------------------------------------------------------------

def test_build_prompt_nl_includes_problem():
    out = build_prompt(_PROBLEM_NL)
    assert "Write a function that adds two numbers." in out
    assert "THINKING:" in out
    assert "SOLUTION:" in out
    assert "```python" in out


def test_build_prompt_code_wraps_in_python_fence():
    out = build_prompt(_PROBLEM_CODE)
    # When prompt_is_code, the problem body should be inside a python fence.
    assert "```python\ndef add(a, b):" in out


# ---------------------------------------------------------------------------
# Parser tests.
# ---------------------------------------------------------------------------

def test_parse_good_response():
    cot, code = parse_cot_response(_GOOD_QWEN_RESPONSE)
    assert cot is not None
    assert "1." in cot and "2." in cot
    assert code is not None
    assert "def add(a, b)" in code
    assert "return a + b" in code


def test_parse_no_solution_header_returns_none():
    cot, code = parse_cot_response(_BAD_NO_SOLUTION_HEADER)
    assert cot is None and code is None


def test_parse_no_thinking_header_returns_none():
    text = "SOLUTION:\n```python\ndef f(): pass\n```"
    cot, code = parse_cot_response(text)
    assert cot is None and code is None


def test_count_thinking_steps():
    assert count_thinking_steps("1. a\n2. b\n3. c") == 3
    assert count_thinking_steps("Just one paragraph here.") == 0
    assert count_thinking_steps("1) a\n2) b") == 2


def test_extract_code_block_python_fence():
    assert extract_code_block("```python\nx = 1\n```") == "x = 1"


def test_extract_code_block_falls_back_to_any_fence():
    assert extract_code_block("```\ny = 2\n```") == "y = 2"


def test_extract_code_block_none_when_missing():
    assert extract_code_block("no fences here") is None


# ---------------------------------------------------------------------------
# Validator tests.
# ---------------------------------------------------------------------------

def test_validator_accepts_good_triple():
    cot, code = parse_cot_response(_GOOD_QWEN_RESPONSE)
    assert validate_triple(cot, code) is True


def test_validator_rejects_no_thinking_steps():
    cot, code = parse_cot_response(_BAD_NO_STEPS)
    assert validate_triple(cot, code) is False


def test_validator_rejects_unparseable_code():
    cot, code = parse_cot_response(_BAD_BROKEN_CODE)
    # parser DOES extract the (syntactically broken) code; validator drops it.
    assert validate_triple(cot, code) is False


def test_validator_rejects_empty():
    assert validate_triple(None, "def f(): pass") is False
    assert validate_triple("1. a\n2. b", None) is False
    assert validate_triple("", "def f(): pass") is False


def test_validator_rejects_single_step():
    assert validate_triple("1. only one step", "def f(): pass") is False


# ---------------------------------------------------------------------------
# build_sft_row: triple -> distill-shaped row.
# ---------------------------------------------------------------------------

def test_build_sft_row_schema():
    triple = {
        "task_id": "mbpp/42",
        "sample_idx": 1,
        "problem_text": "Write a function that adds two numbers.",
        "cot_text": "1. Add a and b.\n2. Return the result.",
        "solution_code": "def add(a, b):\n    return a + b",
        "source": "mbpp_combined",
    }
    row = build_sft_row(triple)
    # Required keys for the distill-shaped reader.
    for key in ("task_id", "problem_prompt", "qwen_completion",
                "extracted_code", "has_tests", "tier", "score",
                "sample_idx"):
        assert key in row, f"missing {key}"
    # Phase-4 metadata.
    assert row["prepare_for_thinking"] is True
    assert row["cot_text"] == triple["cot_text"]
    assert row["source_tier"] == "cot"
    assert row["source_score"] == 1.0
    # qwen_completion reconstructs CoT + fenced python.
    assert "1. Add a and b" in row["qwen_completion"]
    assert "```python\ndef add(a, b):" in row["qwen_completion"]
    # extracted_code is just the solution.
    assert row["extracted_code"] == "def add(a, b):\n    return a + b"


def test_build_sft_row_jsonl_roundtrip(tmp_path: pathlib.Path):
    triple = {
        "task_id": "t",
        "sample_idx": 0,
        "problem_text": "p",
        "cot_text": "1. a\n2. b",
        "solution_code": "def f(): return 0",
        "source": "mbpp_combined",
    }
    row = build_sft_row(triple)
    s = json.dumps(row)
    rt = json.loads(s)
    assert rt["task_id"] == "t"
    assert rt["prepare_for_thinking"] is True
