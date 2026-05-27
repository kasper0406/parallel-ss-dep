"""Tests for gen_synthetic_reasoning_tasks.py + the synth_reasoning loader.

The reasoning tasks live alongside gen_synthetic_memory_tasks (recall
without reasoning). What we verify here:

  1. Each family generates a non-empty set of distinct tasks.
  2. Every generated task's gold_solution actually passes the grader
     against its own check() block — that's the contract.
  3. The synth_reasoning code_grader loader round-trips JSONL back into
     Problem objects.
  4. The tasks REQUIRE multi-step reasoning (heuristic: gold body has
     >= 3 statements, OR the prompt mentions step-by-step structure).
"""
from __future__ import annotations

import ast
import json
import pathlib
import tempfile

import pytest

from experiments.code_grader import (
    LOADERS, Problem, grade,
    load_synth_reasoning,
    load_synth_reasoning_heldout,
    load_synth_reasoning_train,
)
from experiments.gen_synthetic_reasoning_tasks import (
    _FAMILIES, generate, problem_to_record, record_to_problem,
)


# ---------------------------------------------------------------------------
# 1) Each family generates a non-empty distinct set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", list(_FAMILIES))
def test_family_generates_distinct_tasks(family):
    problems = generate(
        families=[family],
        n_per_family=8,
        seed=42,
        validate=False,
    )
    assert len(problems) == 8, f"{family} produced {len(problems)} tasks"
    # All task_ids unique
    ids = {p.task_id for p in problems}
    assert len(ids) == len(problems), f"{family} duplicate task_ids"
    # Prompts mostly distinct (allow some collisions in tiny RNG runs
    # but require at least half to differ).
    prompts = {p.prompt for p in problems}
    assert len(prompts) >= len(problems) // 2 + 1, (
        f"{family} produced too many duplicate prompts: "
        f"{len(prompts)}/{len(problems)}"
    )


# ---------------------------------------------------------------------------
# 2) Every gold_solution passes its own tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", list(_FAMILIES))
def test_gold_solution_passes_grader(family):
    """The contract: generate() with validate=True yields only tasks
    whose gold solution passes. Regenerate a small batch and verify by
    running each gold solution through the grader directly."""
    problems = generate(
        families=[family],
        n_per_family=5,
        seed=7,
        validate=False,
    )
    for p in problems:
        res = grade(p, p.gold_solution)
        assert res.passed, (
            f"{family} task {p.task_id} gold failed: tier={res.tier} "
            f"err={res.error} text={res.error_text!r}\n"
            f"--- gold:\n{p.gold_solution}\n--- tests:\n{p.tests}"
        )


def test_validate_cull_drops_buggy_tasks():
    """Sanity: validate=True returns the requested count when the
    generators are well-formed. If any family ever drops below 80% yield
    after validation, the test fails — that's a generator regression."""
    for fam in _FAMILIES:
        problems = generate(
            families=[fam],
            n_per_family=10,
            seed=11,
            validate=True,
        )
        assert len(problems) == 10, (
            f"{fam} validation cull dropped to {len(problems)}/10"
        )


# ---------------------------------------------------------------------------
# 3) Loader round-trip
# ---------------------------------------------------------------------------

def test_loader_round_trip(tmp_path: pathlib.Path):
    problems = generate(
        families=list(_FAMILIES),
        n_per_family=2,
        seed=3,
        validate=False,
    )
    out_path = tmp_path / "synth_reasoning.jsonl"
    with open(out_path, "w") as f:
        for p in problems:
            f.write(json.dumps(problem_to_record(p)) + "\n")

    loaded = load_synth_reasoning(str(out_path))
    assert len(loaded) == len(problems)
    for orig, back in zip(problems, loaded):
        assert isinstance(back, Problem)
        assert back.task_id == orig.task_id
        assert back.prompt == orig.prompt
        assert back.tests == orig.tests
        assert back.entry_point == orig.entry_point
        assert back.prompt_is_code == orig.prompt_is_code
        assert back.gold_solution == orig.gold_solution


def test_loader_is_registered():
    assert "synth_reasoning" in LOADERS
    # Calling it without a generated file should raise a clear error,
    # not crash with a JSON decode somewhere downstream.
    with pytest.raises(FileNotFoundError, match="not found"):
        LOADERS["synth_reasoning"](path="/tmp/does_not_exist_synth.jsonl")


def test_train_and_heldout_loaders_are_registered():
    """The stochastic-gate discovery RL experiment depends on the
    `synth_reasoning_train` and `synth_reasoning_heldout` LOADERS keys.
    Lock the contract."""
    assert "synth_reasoning_train" in LOADERS
    assert "synth_reasoning_heldout" in LOADERS
    # Both raise a clear FileNotFoundError when pointed at a missing
    # path, mirroring the legacy loader.
    with pytest.raises(FileNotFoundError, match="not found"):
        load_synth_reasoning_train(path="/tmp/does_not_exist_train.jsonl")
    with pytest.raises(FileNotFoundError, match="not found"):
        load_synth_reasoning_heldout(
            path="/tmp/does_not_exist_heldout.jsonl"
        )


def test_train_and_heldout_loaders_smoke(tmp_path: pathlib.Path):
    """Generate tiny train + heldout JSONLs with disjoint seeds and
    verify both loaders read them back as Problems. Mirrors the
    pipeline used by the stochastic-gate discovery experiment."""
    train_problems = generate(
        families=list(_FAMILIES), n_per_family=2, seed=0, validate=True,
    )
    held_problems = generate(
        families=list(_FAMILIES), n_per_family=2, seed=1000, validate=True,
    )
    train_path = tmp_path / "synth_reasoning_train.jsonl"
    held_path = tmp_path / "synth_reasoning_heldout.jsonl"
    for path, probs in [(train_path, train_problems),
                        (held_path, held_problems)]:
        with open(path, "w") as f:
            for p in probs:
                f.write(json.dumps(problem_to_record(p)) + "\n")

    train_loaded = load_synth_reasoning_train(path=str(train_path))
    held_loaded = load_synth_reasoning_heldout(path=str(held_path))

    assert len(train_loaded) == len(train_problems) == 2 * len(_FAMILIES)
    assert len(held_loaded) == len(held_problems) == 2 * len(_FAMILIES)
    for prob in train_loaded + held_loaded:
        assert isinstance(prob, Problem)
        assert prob.task_id.startswith("synth_reason/")
        assert prob.gold_solution is not None


def test_record_to_problem_inverts_to_record():
    """problem_to_record / record_to_problem are exact inverses."""
    problems = generate(
        families=["multi_step_arith"], n_per_family=3, seed=1,
        validate=False,
    )
    for p in problems:
        rec = problem_to_record(p)
        back = record_to_problem(rec)
        assert back == p


# ---------------------------------------------------------------------------
# 4) Tasks require multi-step reasoning
# ---------------------------------------------------------------------------

def _gold_body_statements(gold: str) -> int:
    """Count top-level statements inside the entry-point function body."""
    tree = ast.parse(gold)
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef)), None,
    )
    assert fn is not None, f"no top-level function in:\n{gold}"
    return len(fn.body)


@pytest.mark.parametrize("family", list(_FAMILIES))
def test_tasks_are_multi_step(family):
    """Heuristic: a reasoning task should either have a multi-statement
    gold solution OR a prompt that explicitly describes multiple steps
    (numbered list, the word 'step', 'then', or a chain of 'Let'
    statements). One or the other must hold across every instance.

    For the 'closed-form' families whose gold is `return <answer>` (the
    task delegates the reasoning to the model, the gold encodes only
    the final number), the prompt must carry the multi-step structure.
    """
    problems = generate(
        families=[family], n_per_family=5, seed=99, validate=False,
    )
    for p in problems:
        body_n = _gold_body_statements(p.gold_solution)
        prompt_l = p.prompt.lower()
        multi_step_prompt = (
            "step" in prompt_l
            or "then" in prompt_l
            or prompt_l.count("\n  - ") >= 2  # bulleted rules
            or prompt_l.count("let ") >= 2    # arithmetic chain
            or "next term" in prompt_l        # pattern_next
            or "midpoint" in prompt_l         # binary search trace
            or "postfix" in prompt_l          # rpn
            or "count " in prompt_l           # count_with_offset
        )
        assert body_n >= 3 or multi_step_prompt, (
            f"{family} task {p.task_id} looks single-step: "
            f"gold body has {body_n} stmts, prompt didn't hit any "
            f"multi-step marker.\nPROMPT:\n{p.prompt}\nGOLD:\n{p.gold_solution}"
        )
