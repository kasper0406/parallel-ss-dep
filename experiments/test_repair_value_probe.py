"""Tests for the Phase-3a natural-code transfer probe
(`experiments/repair_value_probe.py`, SEARCH_NATIVE_PLAN_2026_07_19.md 3a).

CPU-only and fast: the assert-parsing, answer-parsing, ground-truth-execution,
ranking, and aggregation machinery is pure Python + the code_grader subprocess
sandbox (no CUDA / FLA / HF / model). The log-prob scorer is checked against a
tiny CPU nn.Module (reusing the Phase-2 byte-tokenizer fixture pattern).

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_repair_value_probe.py -v
"""
from __future__ import annotations

import math
import random

import torch

from experiments.repair_value_probe import (
    parse_assert, first_literal_assert, expected_type, build_exec_program,
    parse_final_answer, answer_matches, build_single_assert_check, verify_pair,
    resolve_pair, prepare_pairs, score_pair_logprob, score_pair_random,
    aggregate, TRUE_INDEX, _stable_hash,
)
from experiments.value_function import (
    top_indices, success_strict, success_random, teacher_forced_mean_logprob,
)


# --------------------------------------------------------------------------- #
# Assert parsing.
# --------------------------------------------------------------------------- #

def test_parse_assert_basic_int():
    fname, args, expected, call_src, assert_src = parse_assert(
        "assert add(1, 2) == 3")
    assert fname == "add" and args == [1, 2] and expected == 3
    assert call_src == "add(1, 2)"
    assert assert_src == "assert add(1, 2) == 3"


def test_parse_assert_string_and_list_expected():
    p = parse_assert('assert first_repeated_char("abcabc") == "a"')
    assert p[0] == "first_repeated_char" and p[1] == ["abcabc"] and p[2] == "a"
    p = parse_assert("assert dedup([1, 1, 2]) == [1, 2]")
    assert p[0] == "dedup" and p[1] == [[1, 1, 2]] and p[2] == [1, 2]


def test_parse_assert_reversed_operand_order():
    # `assert <literal> == f(<literals>)`
    fname, args, expected, call_src, _ = parse_assert("assert 5 == sq(2) ")
    assert fname == "sq" and args == [2] and expected == 5
    assert call_src == "sq(2)"


def test_parse_assert_rejects_non_literal_args():
    # constructor call inside an arg -> not a literal -> skip
    assert parse_assert("assert max_chain([Pair(5, 24)], 4) == 3") is None


def test_parse_assert_rejects_non_eq_and_keywords():
    assert parse_assert("assert abs(f(1) - 0.5) < 1e-06") is None  # not ==
    assert parse_assert("assert f(1, k=2) == 3") is None           # keyword arg
    assert parse_assert("assert f(*xs) == 3") is None              # starred
    assert parse_assert("assert g(1) == h(2)") is None             # both calls
    assert parse_assert("assert 1 == 2") is None                   # no call


def test_parse_assert_negative_and_tuple_literal():
    fname, args, expected, _, _ = parse_assert("assert neg(3) == -3")
    assert expected == -3
    # a tuple literal expected still parses (bucketed 'other')
    p = parse_assert("assert pair(1) == (1, 2)")
    assert p is not None and p[2] == (1, 2)


def test_first_literal_assert_scans_past_non_literal():
    tests = ("def check(candidate):\n"
             "    assert max_chain([Pair(5, 24)], 4) == 3\n"   # non-literal, skip
             "    assert width(\"ab\") == 2\n")                 # first literal
    p = first_literal_assert(tests)
    assert p is not None and p[0] == "width" and p[2] == 2


def test_first_literal_assert_none_when_all_non_literal():
    tests = ("def check(candidate):\n"
             "    assert abs(f(1) - 0.5) < 1e-06\n"
             "    assert g([Pair(1, 2)]) == 3\n")
    assert first_literal_assert(tests) is None


def test_expected_type_buckets():
    assert expected_type(4) == "int"
    assert expected_type("a") == "str"
    assert expected_type([1, 2]) == "list"
    assert expected_type(True) == "other"     # bool is NOT int here
    assert expected_type(1.5) == "other"
    assert expected_type((1, 2)) == "other"
    assert expected_type(None) == "other"


# --------------------------------------------------------------------------- #
# Executor program rendering.
# --------------------------------------------------------------------------- #

def test_build_exec_program_appends_tracked_call():
    prog = build_exec_program("def f(a):\n    return a + 1\n", "f(2)")
    assert prog == "def f(a):\n    return a + 1\nx = f(2)"


# --------------------------------------------------------------------------- #
# Answer parsing (ast.literal_eval fallback paths).
# --------------------------------------------------------------------------- #

def test_parse_final_answer_int_str_list():
    assert parse_final_answer("# step 1: x = 4\n# final: 4\n") == {
        "found": True, "raw": "4", "value": 4, "parsed": True}
    p = parse_final_answer("# final: 'abc'\n")
    assert p["parsed"] and p["value"] == "abc"
    p = parse_final_answer("# final: [1, 2, 3]\n")
    assert p["parsed"] and p["value"] == [1, 2, 3]


def test_parse_final_answer_not_found():
    p = parse_final_answer("# step 1: x = 4\n(no final line here)")
    assert p == {"found": False, "raw": None, "value": None, "parsed": False}


def test_parse_final_answer_trailing_comment_fallback():
    p = parse_final_answer("# final: 7  # done\n")
    assert p["parsed"] and p["value"] == 7


def test_parse_final_answer_unparseable_keeps_raw():
    p = parse_final_answer("# final: four\n")
    assert p["found"] and not p["parsed"] and p["raw"] == "four"


def test_answer_matches_value_not_string():
    assert answer_matches({"found": True, "raw": "[1, 2]", "value": [1, 2],
                           "parsed": True}, [1, 2]) == 1
    # correct value, wrong type is NOT credited (anti-inflation): 1 vs True
    assert answer_matches({"found": True, "raw": "1", "value": 1,
                           "parsed": True}, True) == 0
    # 0 vs 0.0 not credited (avoids Python coincidence)
    assert answer_matches({"found": True, "raw": "0", "value": 0,
                           "parsed": True}, 0.0) == 0


def test_answer_matches_str_fallback_for_unquoted_literal():
    # executor emits `abc` (a Name, unparseable) for expected 'abc'
    assert answer_matches({"found": True, "raw": "abc", "value": None,
                           "parsed": False}, "abc") == 1
    assert answer_matches({"found": False, "raw": None, "value": None,
                           "parsed": False}, "abc") == 0


# --------------------------------------------------------------------------- #
# Ground-truth execution check (subprocess sandbox).
# --------------------------------------------------------------------------- #

_GOOD = "def add_one(v):\n    return v + 1\n"
_BAD = "def add_one(v):\n    return v + 2\n"


def test_verify_pair_confirms_good_and_refutes_bad():
    fp, bf, ft, bt = verify_pair("add_one", "assert add_one(1) == 2",
                                 _GOOD, _BAD, timeout_s=5)
    assert fp and bf and ft == "pass" and bt != "pass"


def test_verify_pair_label_noise_buggy_actually_passes():
    # a "buggy" candidate that happens to pass THIS assert -> not a droppable
    # buggy_fail: buggy_fail is False, so the pair would be dropped upstream.
    fp, bf, _, _ = verify_pair("add_one", "assert add_one(1) == 2",
                               _GOOD, _GOOD, timeout_s=5)
    assert fp and not bf


def test_verify_pair_sandbox_timeout():
    loop = "def add_one(v):\n    while True:\n        pass\n"
    # buggy loops forever -> sandbox must not hang and must report it as a fail.
    fp, bf, ft, bt = verify_pair("add_one", "assert add_one(1) == 2",
                                 _GOOD, loop, timeout_s=3)
    assert fp and bf and bt in ("timeout", "exec_error")


def test_build_single_assert_check_indents():
    blk = build_single_assert_check("assert f(1) == 2")
    assert blk == "def check(candidate):\n    assert f(1) == 2\n"


# --------------------------------------------------------------------------- #
# resolve_pair + prepare_pairs (workers=1 to avoid multiprocessing in tests).
# --------------------------------------------------------------------------- #

def _triple(task_id, key, fix, attempt, problem="Problem.\nExample: x"):
    return {"task_id": task_id, "problem_key": key, "tier": "partial",
            "provenance": "synthetic_mutant", "mutation": "const",
            "problem": problem, "fix": fix, "attempt": attempt}


def test_resolve_pair_skips_and_resolves():
    tbk = {"k/1": "def check(candidate):\n    assert add_one(1) == 2\n"}
    kind, payload = resolve_pair(_triple("t/1", "k/1", _GOOD, _BAD), tbk)
    assert kind == "pair" and payload["fname"] == "add_one"
    assert payload["expected"] == 2 and payload["expected_type"] == "int"
    # missing problem key
    assert resolve_pair(_triple("t/2", "nope", _GOOD, _BAD), tbk)[1] == "no_problem"
    # no literal assert in the test block
    tbk2 = {"k/1": "def check(candidate):\n    assert abs(f(1)-0.5) < 1e-6\n"}
    assert resolve_pair(_triple("t/3", "k/1", _GOOD, _BAD), tbk2)[1] == \
        "no_literal_assert"
    # empty candidate
    assert resolve_pair(_triple("t/4", "k/1", "", _BAD), tbk)[1] == \
        "empty_candidate"


def test_prepare_pairs_verifies_and_drops_label_noise():
    tbk = {"k/good": "def check(candidate):\n    assert add_one(1) == 2\n",
           "k/noise": "def check(candidate):\n    assert add_one(1) == 2\n"}
    triples = [
        _triple("t/good", "k/good", _GOOD, _BAD),     # verified pair
        _triple("t/noise", "k/noise", _GOOD, _GOOD),  # buggy passes -> dropped
    ]
    pairs, stats = prepare_pairs(triples, tbk, workers=1, timeout_s=5)
    assert stats["n_verified"] == 1
    assert pairs[0]["task_id"] == "t/good"
    assert stats["drops"].get("buggy_did_not_fail") == 1


# --------------------------------------------------------------------------- #
# Ranking / tie math (2-way; reuse Phase-2 conventions, true index = fixed = 0).
# --------------------------------------------------------------------------- #

def test_two_way_ranking_fixed_wins_and_ties_fail():
    # executor: fixed match=1, buggy match=0 -> fixed unique max
    keys = [(1, -0.2), (0, float("-inf"))]
    assert top_indices(keys) == [TRUE_INDEX]
    assert success_strict(keys, TRUE_INDEX) == 1
    # both wrong, tie on match bit and logprob -> tie -> failure
    tie = [(0, float("-inf")), (0, float("-inf"))]
    assert success_strict(tie, TRUE_INDEX) == 0
    # tie-broken-random: ~1/2 over the 2-way tie
    wins = sum(success_random(tie, TRUE_INDEX, random.Random(s))
               for s in range(2000))
    assert 0.4 < wins / 2000 < 0.6


def test_two_way_ranking_logprob_scalar_keys():
    assert success_strict([(-0.5,), (-1.2,)], TRUE_INDEX) == 1
    assert success_strict([(-1.2,), (-0.5,)], TRUE_INDEX) == 0


def test_executor_secondary_tiebreak_on_logprob():
    # both match=1, higher answer-logprob (fixed) wins uniquely
    assert success_strict([(1, -0.1), (1, -0.9)], TRUE_INDEX) == 1
    assert success_strict([(1, -0.9), (1, -0.1)], TRUE_INDEX) == 0


# --------------------------------------------------------------------------- #
# Determinism.
# --------------------------------------------------------------------------- #

def test_random_keys_deterministic_under_seed():
    pair = {"task_id": "t/x"}
    a = score_pair_random(pair, random.Random(_stable_hash("t/x") ^ 999))
    b = score_pair_random(pair, random.Random(_stable_hash("t/x") ^ 999))
    assert a == b and len(a) == 2


def test_stable_hash_is_stable():
    assert _stable_hash("abc") == _stable_hash("abc")
    assert _stable_hash("abc") != _stable_hash("abd")


def _fake_scored(n, exec_fixed=1, exec_buggy=0, lp_fixed=True, etype="int"):
    """Build a scored-pair list with deterministic successes for aggregate()."""
    out = []
    for i in range(n):
        ex = [(exec_fixed, -0.1), (exec_buggy, -0.5)]
        lp = [(-0.1,), (-0.5,)] if lp_fixed else [(-0.5,), (-0.1,)]
        out.append({
            "task_id": f"t/{i}", "expected_type": etype,
            "tier": "partial", "provenance": "x",
            "success": {
                "executor": {"strict": success_strict(ex, TRUE_INDEX),
                             "random": success_random(ex, TRUE_INDEX,
                                                      random.Random(i))},
                "logprob": {"strict": success_strict(lp, TRUE_INDEX),
                            "random": success_random(lp, TRUE_INDEX,
                                                     random.Random(i))},
                "random": {"strict": 0, "random": 0},
            },
            "executor_fixed_match": exec_fixed,
            "expected": repr(4),
        })
    return out


def test_aggregate_verdict_and_determinism():
    # executor: fixed always ranks #1 (acc 1.0); logprob: buggy ranks #1 (acc 0)
    scored = _fake_scored(50, exec_fixed=1, exec_buggy=0, lp_fixed=False)
    r1 = aggregate(scored, seed=0, n_boot=200)
    r2 = aggregate(scored, seed=0, n_boot=200)
    assert r1 == r2  # deterministic
    assert r1["pooled"]["executor"]["ties_as_failures"]["acc"] == 1.0
    assert r1["pooled"]["logprob"]["ties_as_failures"]["acc"] == 0.0
    v = r1["verdict"]
    assert v["executor_minus_logprob_pp_ties_as_failures"] == 100.0
    assert v["pass"] is True
    assert r1["executor_fixed_match_rate"]["pooled"] == 1.0


def test_aggregate_fail_when_executor_below_bar():
    # executor and logprob both rank fixed #1 -> delta 0 -> FAIL the +10pp bar
    scored = _fake_scored(40, exec_fixed=1, exec_buggy=0, lp_fixed=True)
    r = aggregate(scored, seed=0, n_boot=200)
    assert r["verdict"]["executor_minus_logprob_pp_ties_as_failures"] == 0.0
    assert r["verdict"]["pass"] is False


# --------------------------------------------------------------------------- #
# Log-prob scorer via a tiny CPU model (byte tokenizer, concatenation-safe).
# --------------------------------------------------------------------------- #

class _ByteTok:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class _TinyModel(torch.nn.Module):
    def __init__(self, vocab=256, d=16):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids))


def test_score_pair_logprob_two_keys_and_matches_manual():
    tok, model = _ByteTok(), _TinyModel()
    model.eval()
    pair = {"problem": "Return v+1.", "fixed_code": "def f(v):\n    return v+1\n",
            "buggy_code": "def f(v):\n    return v+2\n"}
    keys = score_pair_logprob(model, tok, pair)
    assert len(keys) == 2 and all(len(k) == 1 for k in keys)
    prefix = "Return v+1.\n"
    manual = teacher_forced_mean_logprob(model, tok, prefix,
                                         prefix + pair["fixed_code"])
    assert math.isclose(keys[0][0], manual, rel_tol=1e-6, abs_tol=1e-6)
