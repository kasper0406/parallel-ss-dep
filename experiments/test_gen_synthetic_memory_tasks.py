"""Tests for gen_synthetic_memory_tasks.py.

These tasks are the "make thinking matter" training data: each example
sets a value early, then has distractor text, then asks for the value.
The point is that SFT loss on the final answer can ONLY drop if the
model uses its WorkingMemory to recall the value (greedy attention over
short contexts isn't enough once the distractor pushes the value out of
the trunk's effective range).

Properties we verify:
 1. Every family generates plausible, executable Python that prints the
    declared answer (sanity: if we run the embedded code, we get the
    claimed answer).
 2. The problem prompt contains a distractor block between the
    binding and the recall (otherwise WM isn't needed).
 3. JSONL records match the distill_solutions.py schema so
    sft_code.load_distilled_jsonl consumes them.
"""
import contextlib
import io
import json
import random

import pytest

from experiments.gen_synthetic_memory_tasks import (
    _gen_var_binding, _gen_chain_arithmetic, _gen_list_index_recall,
    _gen_dict_lookup, _gen_multi_step_arithmetic,
    _to_jsonl_record, _FAMILIES,
)


_FENCED_PY = __import__("re").compile(
    r"```python\s*\n(.*?)\n?```", __import__("re").DOTALL)


def _extract_program(problem_prompt: str) -> str:
    """Pull the embedded python program from the prompt."""
    m = _FENCED_PY.search(problem_prompt)
    assert m is not None, f"no ```python block in {problem_prompt[:200]}"
    return m.group(1)


def _execute_and_capture_stdout(program: str) -> str:
    """Exec the program in a fresh namespace, return whatever print()
    emits."""
    buf = io.StringIO()
    ns: dict = {}
    with contextlib.redirect_stdout(buf):
        exec(compile(program, "<test>", "exec"), ns)
    return buf.getvalue().strip()


# ---------------------------------------------------------------------------
# Family correctness: executing the embedded program prints the claimed answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,gen", _FAMILIES.items())
def test_embedded_program_prints_claimed_answer(family, gen):
    """For every family, running the embedded code must produce the
    stated answer. This catches generator bugs (off-by-one in arithmetic,
    wrong index, etc.)."""
    rng = random.Random(42)
    for _ in range(20):  # sample many to catch corner cases
        ex = gen(rng)
        prog = _extract_program(ex["problem"])
        out = _execute_and_capture_stdout(prog)
        assert out == ex["answer"], (
            f"{family}: embedded program printed {out!r} but generator "
            f"claimed answer={ex['answer']!r}. Program:\n{prog}"
        )


# ---------------------------------------------------------------------------
# Distractor presence: every family must interpose at least 1 line of
# unrelated text between value-binding and the print.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,gen", _FAMILIES.items())
def test_distractor_separates_binding_and_recall(family, gen):
    rng = random.Random(7)
    ex = gen(rng)
    prog = _extract_program(ex["problem"])
    lines = prog.strip().split("\n")
    # The first line is a binding (assignment), the last is print(),
    # there must be at least one distractor line in between.
    assert len(lines) > 2, (
        f"{family} program too short to require WM: {lines!r}"
    )
    # The last line should be a print() call
    assert lines[-1].strip().startswith("print"), (
        f"{family} last line should be print(): {lines[-1]!r}"
    )


# ---------------------------------------------------------------------------
# JSONL record schema matches distill_solutions output
# ---------------------------------------------------------------------------

def test_jsonl_record_schema():
    """Required keys + types must match what sft_code.load_distilled_jsonl
    expects: {task_id, sample_idx, problem_prompt, qwen_completion,
    extracted_code, has_tests, tier, score}."""
    rng = random.Random(0)
    ex = _gen_var_binding(rng)
    rec = _to_jsonl_record("synthmem/test/0", 0, ex)
    assert set(rec.keys()) == {
        "task_id", "sample_idx", "problem_prompt", "qwen_completion",
        "extracted_code", "has_tests", "tier", "score",
    }
    assert rec["task_id"].startswith("synthmem/")
    assert isinstance(rec["sample_idx"], int)
    assert isinstance(rec["problem_prompt"], str)
    assert isinstance(rec["qwen_completion"], str)
    assert isinstance(rec["extracted_code"], str)
    assert rec["has_tests"] is False
    assert rec["tier"] == "synthetic_memory"
    assert rec["score"] == 1.0


def test_jsonl_extracted_code_executes_to_answer():
    """The extracted_code field of every synthetic record should be a
    minimal `print(<answer>)` program — when executed, it produces the
    stated answer. This lets the existing distill JSONL consumer use it
    as the SFT target without further processing."""
    rng = random.Random(1)
    for _ in range(10):
        ex = _gen_chain_arithmetic(rng)
        rec = _to_jsonl_record("synthmem/chain/x", 0, ex)
        # extracted_code is a print(...) — run it
        out = _execute_and_capture_stdout(rec["extracted_code"])
        assert out == ex["answer"], (
            f"extracted_code printed {out!r}, expected {ex['answer']!r}"
        )


def test_consumable_by_sft_code():
    """End-to-end: write a small batch to a temp file, then
    sft_code.load_distilled_jsonl reads it and returns (problem, solution)
    pairs without error."""
    from experiments.sft_code import load_distilled_jsonl
    import tempfile
    rng = random.Random(99)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for family, gen in _FAMILIES.items():
            for i in range(3):
                rec = _to_jsonl_record(f"synthmem/{family}/{i}", 0, gen(rng))
                f.write(json.dumps(rec) + "\n")
        path = f.name
    pairs = load_distilled_jsonl(path)
    # 5 families × 3 examples = 15
    assert len(pairs) == 15
    # Each pair is (problem, solution)
    for p, s in pairs:
        assert isinstance(p, str) and len(p) > 0
        assert isinstance(s, str) and len(s) > 0
