"""Tests for the CRUXEval-O transfer probe (eval_cruxeval_transfer.py).

CPU-only, no network, no GPU: the pure parse/score/prompt/selection helpers
are exercised directly; the arm-dispatch test monkeypatches the module-level
generator references (greedy_generate / latent_greedy_answer) so no model or
dataset is ever touched.

Run: PYTHONPATH=. .venv/bin/python -m pytest \
        experiments/test_eval_cruxeval_transfer.py -v
"""
from __future__ import annotations

import pytest

import experiments.eval_cruxeval_transfer as ect
from experiments.eval_cruxeval_transfer import (
    DEFAULT_ARMS,
    _has_complete_literal,
    _has_complete_trace_answer,
    _latent_arm_r,
    aggregate,
    build_direct_prompt,
    build_trace_prompt,
    extract_direct_answer,
    extract_literal_prefix,
    extract_trace_answer,
    score_answer,
    select_problems,
    try_literal,
)


# --------------------------------------------------------------------------- #
# Scorer: literal equality + string-strip fallback.
# --------------------------------------------------------------------------- #

def test_score_ints():
    assert score_answer("17", "17") == (True, True)
    assert score_answer(" 17 ", "17") == (True, True)      # whitespace
    assert score_answer("18", "17") == (False, True)
    assert score_answer("-3", "-3") == (True, True)


def test_score_strings_quote_style():
    # different repr quote styles, same value
    assert score_answer('"abc"', "'abc'") == (True, True)
    assert score_answer("'abc'", "'abd'") == (False, True)


def test_score_lists_whitespace():
    assert score_answer("[1,2]", "[1, 2]") == (True, True)
    assert score_answer("[1, 2, 3]", "[1, 2]") == (False, True)


def test_score_tuple_vs_string_not_equal():
    # a STRING that spells a tuple must not equal the tuple
    assert score_answer("'(1, 2)'", "(1, 2)") == (False, True)
    assert score_answer("(1, 2)", "(1, 2)") == (True, True)
    # and list vs tuple are distinct values under Python ==
    assert score_answer("[1, 2]", "(1, 2)") == (False, True)


def test_score_none_candidate_and_none_value():
    assert score_answer(None, "17") == (False, False)
    # literal None as a legitimate value
    assert score_answer("None", "None") == (True, True)


def test_score_string_strip_fallback():
    # neither side literal-evals -> string-strip equality, parsed=False
    assert score_answer("foo(bar", "foo(bar") == (True, False)
    assert score_answer("  foo(bar \n", "foo(bar") == (True, False)
    assert score_answer("foo(bar", "foo(baz") == (False, False)


def test_try_literal_never_raises():
    assert try_literal("")[1] is False
    assert try_literal("f(3)")[1] is False       # function call: not a literal
    assert try_literal("{'a': 1}") == ({"a": 1}, True)


# --------------------------------------------------------------------------- #
# Prompt templates — exactness.
# --------------------------------------------------------------------------- #

REC = {
    "id": "sample_0",
    "code": "def f(x):\n    return x + 1\n",
    "input": "[1, 1, 3]",
    "output": "[1, 1, 3]",
}


def test_direct_prompt_exact():
    assert build_direct_prompt(REC) == (
        "def f(x):\n    return x + 1\nassert f([1, 1, 3]) == ")


def test_trace_prompt_exact():
    # trailing space after '==' dropped, then the trained cue
    assert build_trace_prompt(REC) == (
        "def f(x):\n    return x + 1\nassert f([1, 1, 3]) ==\n# trace:\n")


# --------------------------------------------------------------------------- #
# Parse rules.
# --------------------------------------------------------------------------- #

def test_extract_literal_prefix():
    assert extract_literal_prefix("17\nassert f(2) == 3") == "17"
    # multi-line literal kept intact, junk after cut
    assert extract_literal_prefix("[1,\n 2]\njunk") == "[1,\n 2]"
    assert extract_literal_prefix("no literal here") is None
    assert extract_literal_prefix("") is None


def test_extract_direct_answer_fallback_first_line():
    assert extract_direct_answer("[(4, 1), (2, 3)]\nmore") == "[(4, 1), (2, 3)]"
    # non-literal completion -> first non-empty line for the string fallback
    assert extract_direct_answer("\n  garbage output  \nx") == "garbage output"
    assert extract_direct_answer("\n \n") is None


def test_extract_trace_answer_final_line():
    txt = "# step 1: x = 2\n# step 2: x = 9\n# final: [1, 2]\n"
    assert extract_trace_answer(txt) == "[1, 2]"


def test_extract_trace_answer_assert_completion():
    txt = "# step 1: x = 2\nassert f([1, 1, 3]) == 9\n"
    assert extract_trace_answer(txt) == "9"


def test_extract_trace_answer_priority_final_over_assert():
    txt = "assert f(3) == 8\n# final: 9\n"
    assert extract_trace_answer(txt) == "9"


def test_extract_trace_answer_leading_literal():
    # model answered immediately after the latent slots
    assert extract_trace_answer("'abc'\nblah") == "'abc'"


def test_extract_trace_answer_garbage_is_none():
    # NO first-line fallback for trace arms ('# step' lines are not answers)
    assert extract_trace_answer("# step 1: x = 2\n# step 2: x = 5\n") is None
    assert extract_trace_answer("total garbage") is None


def test_stop_predicates():
    assert _has_complete_literal("17\n") is True
    assert _has_complete_literal("17") is False          # maybe mid-number
    assert _has_complete_literal("[1,\n 2]\n") is True
    assert _has_complete_literal("junk\n") is False
    assert _has_complete_trace_answer("# final: 9\n") is True
    assert _has_complete_trace_answer("# final: 9") is False   # unterminated
    assert _has_complete_trace_answer("assert f(1) == [2]\n") is True
    assert _has_complete_trace_answer("# step 1: x = 2\n") is False


# --------------------------------------------------------------------------- #
# Seeded shuffle determinism.
# --------------------------------------------------------------------------- #

def test_select_problems_deterministic():
    recs = [{"id": f"sample_{i}"} for i in range(50)]
    a = select_problems(recs, 10, seed=0)
    b = select_problems(recs, 10, seed=0)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 10
    c = select_problems(recs, 10, seed=1)
    assert [r["id"] for r in a] != [r["id"] for r in c]
    # n<=0 or n>=len -> all records (still shuffled deterministically)
    assert len(select_problems(recs, 0, seed=0)) == 50
    assert len(select_problems(recs, 999, seed=0)) == 50
    assert {r["id"] for r in select_problems(recs, 0, seed=0)} == \
        {r["id"] for r in recs}


# --------------------------------------------------------------------------- #
# Arm dispatch with a mocked model (no torch model, no GPU, no dataset).
# --------------------------------------------------------------------------- #

class _FakeTok:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_arm_dispatch_mocked(monkeypatch):
    calls = {"greedy": [], "latent": []}

    def fake_greedy(model, tok, prompt_ids, max_gen, eos_id, stop_fn=None,
                    grace=4):
        calls["greedy"].append((len(prompt_ids), max_gen, stop_fn))
        # direct arm gets the literal-stop predicate; trace the final-stop
        if stop_fn is ect._has_complete_literal:
            return "[1, 1, 3]\n"                       # correct direct answer
        return "# step 1: x = 1\n# final: [1, 1, 3]\n"  # correct trace answer

    def fake_latent(model, prompt_ids, R, thinking_id, tok, max_gen, device,
                    grace=4):
        calls["latent"].append((R, thinking_id, max_gen))
        return "# final: 'wrong'\n"                     # parseable but wrong

    monkeypatch.setattr(ect, "greedy_generate", fake_greedy)
    monkeypatch.setattr(ect, "latent_greedy_answer", fake_latent)

    res = ect.eval_record(object(), _FakeTok(), eos_id=0, thinking_id=0,
                          rec=REC, device="cpu", arms=DEFAULT_ARMS,
                          max_gen_direct=64, max_gen_trace=256,
                          max_gen_latent=64)

    assert res["direct"]["correct"] == 1 and res["direct"]["parsed"] == 1
    assert res["text_trace"]["correct"] == 1
    assert res["text_trace"]["pred"] == "[1, 1, 3]"
    for arm in ("latent_R4", "latent_R8"):
        assert res[arm]["correct"] == 0 and res[arm]["parsed"] == 1
        assert res[arm]["pred"] == "'wrong'"
    # both text arms hit greedy_generate; latent arms hit latent with R=4, 8
    assert len(calls["greedy"]) == 2
    assert [c[0] for c in calls["latent"]] == [4, 8]
    # budgets threaded through
    assert calls["greedy"][0][1] == 64 and calls["greedy"][1][1] == 256
    assert all(c[2] == 64 for c in calls["latent"])


def test_latent_arm_name_parsing():
    assert _latent_arm_r("latent_R4") == 4
    assert _latent_arm_r("latent_R12") == 12
    assert _latent_arm_r("direct") is None
    assert _latent_arm_r("latent_Rx") is None


def test_unknown_arm_raises(monkeypatch):
    monkeypatch.setattr(ect, "greedy_generate", lambda *a, **k: "")
    with pytest.raises(ValueError):
        ect.eval_record(object(), _FakeTok(), 0, 0, REC, "cpu",
                        arms=("bogus_arm",))


def test_aggregate():
    per_record = [
        {"id": "a",
         "direct": {"pred": "1", "correct": 1, "parsed": 1, "text": "1"},
         "latent_R4": {"pred": None, "correct": 0, "parsed": 0, "text": ""}},
        {"id": "b",
         "direct": {"pred": "2", "correct": 0, "parsed": 1, "text": "2"},
         "latent_R4": {"pred": "2", "correct": 0, "parsed": 1, "text": "2"}},
    ]
    agg = aggregate(per_record, arms=("direct", "latent_R4", "text_trace"))
    assert agg["direct"] == {"n": 2, "acc": 0.5, "parse_rate": 1.0}
    assert agg["latent_R4"] == {"n": 2, "acc": 0.0, "parse_rate": 0.5}
    assert agg["text_trace"] == {"n": 0, "acc": None, "parse_rate": None}
