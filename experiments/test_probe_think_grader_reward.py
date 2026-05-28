"""Tests for the PRIORITY-1 think-grader-reward probe helpers.

These cover the pure logic (completion-prep for grading, dataset loader
dispatch) without needing a GPU or a model — the generation path itself is
exercised by the live run logged to runs/probe_think_grader_reward.log.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from experiments import code_grader
from experiments.probe_think_grader_reward import (
    _build_completion_for_grading,
    _load_problems,
)


def _humaneval_like_problem() -> code_grader.Problem:
    # The grader's granular runner exec's each assert statement against a
    # namespace where the entry_point function name (here `f`) is in scope,
    # so the check references `f` directly (mirrors HumanEval's convention).
    return code_grader.Problem(
        task_id="t/0",
        prompt="def f(x):\n    \"\"\"doc\"\"\"\n",
        tests="def check(candidate):\n    assert f(1) == 2\n",
        entry_point="f",
        prompt_is_code=True,
    )


def test_extract_code_block_grades_standalone():
    """With extract_code_block, the fenced code is graded as a standalone
    program (prompt emptied, prompt_is_code=False) so the prompt is not
    double-prepended."""
    prob = _humaneval_like_problem()
    gen = "Here is the solution:\n```python\ndef f(x):\n    return x + 1\n```"
    graded_prob, completion = _build_completion_for_grading(
        prob, gen, extract_code_block=True)
    assert graded_prob.prompt == ""
    assert graded_prob.prompt_is_code is False
    assert completion == "def f(x):\n    return x + 1"
    # The graded problem keeps the tests + entry point.
    assert graded_prob.tests == prob.tests
    assert graded_prob.entry_point == prob.entry_point


def test_extract_code_block_falls_back_to_raw_text():
    """No fence present → grade the raw text (will likely syntax_error, but
    keeps the probe honest rather than silently skipping)."""
    prob = _humaneval_like_problem()
    gen = "def f(x):\n    return x + 1"  # no fence
    graded_prob, completion = _build_completion_for_grading(
        prob, gen, extract_code_block=True)
    assert completion == gen
    assert graded_prob.prompt == ""


def test_no_extract_passes_through_to_grade():
    """Without extract_code_block, the original problem is returned
    unchanged and grade() does its own prompt+truncate concat."""
    prob = _humaneval_like_problem()
    gen = "    return x + 1\n"
    graded_prob, completion = _build_completion_for_grading(
        prob, gen, extract_code_block=False)
    assert graded_prob is prob
    assert completion == gen


def test_load_problems_unknown_dataset_raises():
    with pytest.raises(ValueError):
        _load_problems("not_a_dataset", max_problems=1)


def test_load_problems_caps_count(monkeypatch):
    fake = [code_grader.Problem(task_id=f"x/{i}", prompt="", tests="",
                                entry_point="f")
            for i in range(10)]
    monkeypatch.setattr(code_grader, "load_humaneval", lambda: fake)
    out = _load_problems("humaneval", max_problems=3)
    assert len(out) == 3
    out_all = _load_problems("humaneval", max_problems=None)
    assert len(out_all) == 10


def test_extracted_block_grades_pass_end_to_end():
    """Sanity: a correct extracted block actually grades as pass through
    the real grader (exercises the standalone-exec path)."""
    prob = _humaneval_like_problem()
    gen = "```python\ndef f(x):\n    return x + 1\n```"
    graded_prob, completion = _build_completion_for_grading(
        prob, gen, extract_code_block=True)
    res = code_grader.grade(graded_prob, completion, timeout_s=7)
    assert res.passed is True
    assert res.tier == "pass"
