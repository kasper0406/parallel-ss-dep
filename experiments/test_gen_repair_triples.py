"""Tests for experiments/gen_repair_triples.py (repair-triple corpus miner).

CPU-only; the grader integration tests actually execute attempts in the
code_grader subprocess sandbox (no model, no GPU). Run with:

    PYTHONPATH=. .venv/bin/python -m pytest experiments/test_gen_repair_triples.py -v
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.code_grader import grade  # noqa: E402
from experiments.gen_repair_triples import (  # noqa: E402
    MUTATION_FAMILIES,
    _problem_from_dict,
    build_problem_text,
    cap_tier,
    corpus_stats,
    enumerate_mutants,
    is_heldout_problem,
    mine_dump_problem_worker,
    mutate_problem_worker,
    render_document,
    split_records,
)

# ---------------------------------------------------------------------------
# A self-contained synthetic problem (no HF datasets needed).
# ---------------------------------------------------------------------------

GOLD = (
    "def add_positive(nums):\n"
    "    total = 0\n"
    "    for i in range(len(nums)):\n"
    "        if nums[i] > 0 and nums[i] < 100:\n"
    "            total = total + nums[i]\n"
    "    return total\n"
)

PROB = {
    "problem_key": "synth/1",
    "task_id": "synth/1",
    "prompt": "Write a function to sum the positive numbers below 100 "
              "in a list.\n",
    "tests": ("def check(candidate):\n"
              "    assert add_positive([1, -2, 3]) == 4\n"
              "    assert add_positive([-1]) == 0\n"
              "    assert add_positive([2, 200, 2]) == 4\n"),
    "entry_point": "add_positive",
    "prompt_is_code": False,
    "gold_solution": GOLD,
}

BROKEN_ATTEMPT = (
    "def add_positive(nums):\n"
    "    total = 0\n"
    "    for i in range(len(nums)):\n"
    "        total = total + nums[i]\n"
    "    return total\n"
)

SYNTAX_ATTEMPT = "def add_positive(nums:\n    return total\n"


def _dump_row(code: str, tier: str, idx: int = 0) -> dict:
    return {"task_id": f"reject/synth/1/{idx}", "sample_idx": idx,
            "extracted_code": code, "tier": tier}


# ---------------------------------------------------------------------------
# Mutator.
# ---------------------------------------------------------------------------

def test_mutants_parseable_failing_shaped_and_distinct():
    muts = enumerate_mutants(GOLD, "0:synth/1", max_candidates=48)
    assert len(muts) >= 10
    gold_norm = ast.unparse(ast.parse(GOLD))
    seen = set()
    for family, code in muts:
        assert family in MUTATION_FAMILIES
        ast.parse(code)                    # parseable, always
        assert code != gold_norm           # semantically mutated
        assert code not in seen            # unique
        seen.add(code)


def test_mutator_all_families_fire_on_rich_source():
    muts = enumerate_mutants(GOLD, "0:synth/1", max_candidates=48)
    families = {f for f, _ in muts}
    # GOLD contains int constants + len() (off_by_one), comparisons +
    # BinOp + BoolOp (operator_swap), >=2 locals (wrong_variable), an if
    # (dropped_condition) and `total = 0` (wrong_init).
    assert families == set(MUTATION_FAMILIES)


def test_mutator_deterministic():
    a = enumerate_mutants(GOLD, "0:synth/1", max_candidates=48)
    b = enumerate_mutants(GOLD, "0:synth/1", max_candidates=48)
    assert a == b


def test_mutator_seed_key_changes_selection():
    a = enumerate_mutants(GOLD, "0:synth/1", max_candidates=8)
    b = enumerate_mutants(GOLD, "1:synth/1", max_candidates=8)
    assert a != b   # different seed -> different site sample/order


def test_mutator_unparseable_gold_returns_empty():
    assert enumerate_mutants("def broken(:\n    pass", "0:x") == []


# ---------------------------------------------------------------------------
# Grader integration (real subprocess execution).
# ---------------------------------------------------------------------------

def test_grader_returns_real_error_text_for_semantic_failure():
    res = grade(_problem_from_dict(PROB), BROKEN_ATTEMPT, timeout_s=4)
    assert res.tier == "partial"
    assert res.error_text and "AssertionError" in res.error_text
    # the granular grader re-evals operands: real got/expected diagnosis
    assert "got" in res.error_text and "expected" in res.error_text


def test_grader_returns_real_error_text_for_syntax_failure():
    res = grade(_problem_from_dict(PROB), SYNTAX_ATTEMPT, timeout_s=4)
    assert res.tier == "syntax_error"
    assert res.error_text and "SyntaxError" in res.error_text


def test_mutate_worker_keeps_only_failing_with_diagnostics():
    recs = mutate_problem_worker(PROB, seed=0, max_keep=12,
                                 max_candidates=48, timeout_s=4)
    assert len(recs) >= 5
    gold_norm = GOLD.strip()
    for r in recs:
        assert r["tier"] != "pass"
        assert r["error_text"]
        assert r["provenance"] == "synthetic_mutant"
        assert r["mutation"] in MUTATION_FAMILIES
        assert r["fix"] == gold_norm          # fix = verified gold
        assert r["attempt"] != gold_norm
        ast.parse(r["attempt"])               # failing but parseable
    assert len({r["task_id"] for r in recs}) == len(recs)


def test_mutate_worker_refuses_unverifiable_gold():
    bad = dict(PROB)
    bad["gold_solution"] = "def add_positive(nums):\n    return 0\n"
    assert mutate_problem_worker(bad, seed=0, max_keep=8,
                                 max_candidates=16, timeout_s=4) == []


# ---------------------------------------------------------------------------
# On-policy dump mining.
# ---------------------------------------------------------------------------

def test_mine_dumps_prefers_passing_sibling_as_fix():
    rows = [_dump_row(BROKEN_ATTEMPT, "partial", 0),
            _dump_row(GOLD, "pass", 1)]
    recs = mine_dump_problem_worker(PROB, rows, max_keep=4, timeout_s=4)
    assert len(recs) == 1
    r = recs[0]
    assert r["provenance"] == "on_policy"
    assert r["fix_source"] == "sibling_rollout"
    assert r["fix"] == GOLD.strip()
    assert r["attempt"] == BROKEN_ATTEMPT.strip()
    assert r["tier"] == "partial"
    assert "AssertionError" in r["error_text"]   # re-graded, real diagnostics


def test_mine_dumps_falls_back_to_gold_fix_and_mixes_tiers():
    rows = [_dump_row(BROKEN_ATTEMPT, "partial", 0),
            _dump_row(SYNTAX_ATTEMPT, "syntax_error", 1)]
    recs = mine_dump_problem_worker(PROB, rows, max_keep=4, timeout_s=4)
    assert len(recs) == 2
    assert all(r["fix_source"] == "gold_solution" for r in recs)
    assert {r["tier"] for r in recs} == {"partial", "syntax_error"}


def test_mine_dumps_skips_attempts_that_actually_pass():
    rows = [_dump_row(GOLD, "partial", 0)]   # dump mislabeled; re-grade wins
    recs = mine_dump_problem_worker(PROB, rows, max_keep=4, timeout_s=4)
    assert recs == []


# ---------------------------------------------------------------------------
# Split / tier cap / stats.
# ---------------------------------------------------------------------------

def _fake_rec(key: str, tier: str = "partial", i: int = 0) -> dict:
    return {"task_id": f"repair_triple/{key}/synthetic_mutant/{i}",
            "problem_key": key, "problem": "p", "attempt": "a",
            "error_text": "e", "fix": "f", "provenance": "synthetic_mutant",
            "tier": tier, "mutation": "off_by_one",
            "fix_source": "gold_solution", "text": "t"}


def test_split_disjoint_by_problem_id():
    recs = [_fake_rec(f"mbpp_train/{i}", i=j)
            for i in range(200) for j in range(3)]
    train, heldout = split_records(recs, heldout_frac=0.2, seed=0)
    tr_keys = {r["problem_key"] for r in train}
    ho_keys = {r["problem_key"] for r in heldout}
    assert tr_keys.isdisjoint(ho_keys)
    assert len(train) + len(heldout) == len(recs)
    assert 0 < len(ho_keys) < 200
    # every record of one problem lands on the same side
    for key in ho_keys:
        assert all(is_heldout_problem(key, 0.2, 0) for _ in range(2))


def test_split_deterministic_across_calls():
    recs = [_fake_rec(f"k/{i}") for i in range(100)]
    a = split_records(recs, 0.3, seed=7)
    b = split_records(recs, 0.3, seed=7)
    assert a == b


def test_tier_cap_enforced_and_deterministic():
    recs = ([_fake_rec(f"k/{i}", "syntax_error", i) for i in range(80)]
            + [_fake_rec(f"k/{i}", "partial", i) for i in range(80, 100)])
    capped = cap_tier(recs, "syntax_error", 0.30, seed=0)
    n_syntax = sum(r["tier"] == "syntax_error" for r in capped)
    assert n_syntax / len(capped) <= 0.30 + 1e-9
    assert sum(r["tier"] == "partial" for r in capped) == 20  # others intact
    assert capped == cap_tier(recs, "syntax_error", 0.30, seed=0)
    # under the cap -> untouched
    assert cap_tier(recs[70:], "syntax_error", 0.50, seed=0) == recs[70:]


# ---------------------------------------------------------------------------
# Document rendering.
# ---------------------------------------------------------------------------

def test_render_document_exact():
    doc = render_document(
        problem="Write a function to add two numbers.\nExample: assert add(1, 2) == 3",
        attempt="def add(a, b):\n    return a - b",
        error_text="1/1 assertions failed:\n  assert add(1, 2) == 3  ->  "
                   "AssertionError (got -1, expected 3)",
        fix="def add(a, b):\n    return a + b",
    )
    assert doc == (
        "# Original problem:\n"
        "# Write a function to add two numbers.\n"
        "# Example: assert add(1, 2) == 3\n"
        "# Attempted solution:\n"
        "def add(a, b):\n"
        "    return a - b\n"
        "\n"
        "# Got this error:\n"
        "# 1/1 assertions failed:\n"
        "#   assert add(1, 2) == 3  ->  AssertionError (got -1, expected 3)\n"
        "\n"
        "# Fixed version:\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )


def test_render_document_house_style_and_comment_safety():
    recs = mutate_problem_worker(PROB, seed=0, max_keep=3,
                                 max_candidates=16, timeout_s=4)
    assert recs
    for r in recs:
        t = r["text"]
        assert t.startswith("# Original problem:\n")
        for marker in ("# Attempted solution:\n", "# Got this error:\n",
                       "# Fixed version:\n"):
            assert marker in t
        assert t.endswith("\n")
        # problem + error blocks stay inside comments (parse-safe format)
        err_block = t.split("# Got this error:\n")[1].split(
            "\n\n# Fixed version:")[0]
        assert all(line.startswith("#") for line in err_block.splitlines())


def test_build_problem_text_includes_example_assert():
    txt = build_problem_text(PROB["prompt"], PROB["tests"])
    assert txt.startswith("Write a function to sum")
    assert "Example: assert add_positive([1, -2, 3]) == 4" in txt


# ---------------------------------------------------------------------------
# End-to-end determinism (mutants pipeline, no pool scheduling involved).
# ---------------------------------------------------------------------------

def test_end_to_end_determinism():
    runs = []
    for _ in range(2):
        recs = mutate_problem_worker(PROB, seed=0, max_keep=8,
                                     max_candidates=32, timeout_s=4)
        train, heldout = split_records(recs, 0.5, seed=0)
        train = cap_tier(train, "syntax_error", 0.3, seed=0)
        runs.append(json.dumps(train, sort_keys=True))
    assert runs[0] == runs[1]


def test_corpus_stats_shape():
    recs = [_fake_rec(f"k/{i}", t, i) for i, t in
            enumerate(["partial", "partial", "syntax_error"])]
    s = corpus_stats(recs)
    assert s["n"] == 3 and s["n_problems"] == 3
    assert s["tiers"] == {"partial": 2, "syntax_error": 1}
    assert s["provenance"] == {"synthetic_mutant": 3}
    assert set(s["text_chars"]) == {"p10", "p50", "p90", "mean"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
