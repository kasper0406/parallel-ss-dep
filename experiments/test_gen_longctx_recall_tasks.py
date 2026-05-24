"""Tests for gen_longctx_recall_tasks.py — the long-context recall
synthetic data that's supposed to force WM to do real work.

We verify (a) the embedded program prints the claimed answer, (b) the
distance between binding and recall is plausibly long enough to defeat
short-range attention, (c) JSONL schema matches the distill_solutions
consumer.
"""
import contextlib
import io
import json
import random

import pytest

from experiments.gen_longctx_recall_tasks import (
    _gen_var_binding_long, _to_jsonl_record, _build_distractor_block,
)


_FENCED_PY = __import__("re").compile(
    r"```python\s*\n(.*?)\n?```", __import__("re").DOTALL)


def _execute(program: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(program, "<test>", "exec"), {})
    return buf.getvalue().strip()


def test_program_prints_claimed_answer():
    """For multiple random seeds, executing the embedded program must
    output exactly the claimed answer."""
    rng = random.Random(42)
    for _ in range(15):
        dist = rng.randint(500, 2000)
        ex = _gen_var_binding_long(rng, dist)
        m = _FENCED_PY.search(ex["problem"])
        assert m is not None
        out = _execute(m.group(1))
        assert out == ex["answer"], (
            f"program printed {out!r}, expected {ex['answer']!r}"
        )


def test_distance_grows_with_target():
    """Larger target_distance_tokens should produce larger actual
    distractor blocks (approx_distance_tokens)."""
    rng = random.Random(0)
    ex_small = _gen_var_binding_long(rng, target_distance_tokens=200)
    ex_large = _gen_var_binding_long(rng, target_distance_tokens=2000)
    assert ex_large["approx_distance_tokens"] > ex_small["approx_distance_tokens"] * 5


def test_distractor_block_self_contained():
    """Each distractor line must be standalone Python (since we sample
    a random subset). Run the whole block — no NameError, no
    SyntaxError."""
    rng = random.Random(7)
    block = _build_distractor_block(rng, target_lines=50)
    # Exec in isolation; should not raise.
    exec(compile(block, "<test_distractor>", "exec"), {})


def test_jsonl_record_schema():
    """Must match distill_solutions schema so load_distilled_jsonl
    consumes it without changes."""
    rng = random.Random(11)
    ex = _gen_var_binding_long(rng, 1000)
    rec = _to_jsonl_record("longctx/test/0", ex)
    required = {"task_id", "sample_idx", "problem_prompt", "qwen_completion",
                "extracted_code", "has_tests", "tier", "score"}
    assert required.issubset(rec.keys())
    assert rec["task_id"].startswith("longctx/")
    assert rec["tier"] == "longctx_recall"
    assert rec["has_tests"] is False
    # Extracted code is a minimal print
    assert rec["extracted_code"].startswith("print(")
    # Solution prose references the answer
    assert ex["answer"] in rec["qwen_completion"]


def test_consumable_by_sft_loader(tmp_path):
    """Round-trip: write JSONL → sft_code.load_distilled_jsonl reads
    it → returns (problem, solution) pairs."""
    from experiments.sft_code import load_distilled_jsonl
    rng = random.Random(99)
    path = tmp_path / "longctx.jsonl"
    with open(path, "w") as f:
        for i in range(10):
            ex = _gen_var_binding_long(rng, target_distance_tokens=800)
            rec = _to_jsonl_record(f"longctx/{i}", ex)
            f.write(json.dumps(rec) + "\n")
    pairs = load_distilled_jsonl(str(path))
    assert len(pairs) == 10
    for p, s in pairs:
        assert "print(" in p
        assert isinstance(s, str) and len(s) > 0
