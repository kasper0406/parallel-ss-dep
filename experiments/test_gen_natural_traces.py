"""CPU tests for the natural-code exec-trace corpus generator
(experiments/gen_natural_traces.py, SEARCH_NATIVE_PLAN "REVIVAL ATTEMPT A").

No GPU / FLA / model needed: everything under test is the pure tracer +
renderer + split + contamination guard. Trusted hand-written functions are
traced in-process (the forked sandbox is only for untrusted corpus code).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from experiments.gen_natural_traces import (
    _trace_once, _comprehension_targets, build_record, render_trace,
    assert_no_contamination, split_heldout, resolve_specs,
    _bounded_repr, _assert_unique_task_ids,
)

STAGEA_DATA = pathlib.Path("data/exec_trace_text_train.jsonl")


def _run_ok(events, ret_repr, match=True, ret_too_long=False, ret_bad=False,
            event_overflow=False, repr_overflow=False):
    """A worker-shaped run dict (mirrors `_trace_once`'s return)."""
    return {"events": events, "ret_repr": ret_repr, "ret_too_long": ret_too_long,
            "ret_bad": ret_bad, "event_overflow": event_overflow,
            "repr_overflow": repr_overflow, "match": match}


def _sp(fix, call_src, fname="f", expected_repr="0", problem_key="pk",
        task_id="t/0", provenance="synthetic_mutant"):
    return {"fix": fix, "call_src": call_src, "fname": fname,
            "expected_repr": expected_repr, "problem_key": problem_key,
            "task_id": task_id, "provenance": provenance}


# --------------------------------------------------------------------------- #
# 1. Tracer: exact (name, value) event sequence on a hand-written function.
# --------------------------------------------------------------------------- #

def test_tracer_exact_event_sequence():
    # Params a,b are the initial state (excluded). c is created (5), d is
    # created (10), then c is REASSIGNED (9) — the tracer must emit both the
    # creation and the later re-change of the same name, in line order.
    fix = ("def foo(a, b):\n"
           "    c = a + b\n"
           "    d = c * 2\n"
           "    c = d - 1\n"
           "    return c\n")
    r = _trace_once(fix, "foo(2, 3)", "foo", 9, set(), 40, 48)
    assert r["events"] == [("c", "5"), ("d", "10"), ("c", "9")]
    assert r["ret_repr"] == "9"
    assert r["match"] is True          # return 9 == expected 9
    assert r["event_overflow"] is False
    assert r["repr_overflow"] is False


def test_tracer_only_traces_target_function_frame():
    # A helper call must NOT pollute the target function's trace.
    fix = ("def helper(z):\n"
           "    w = z + 100\n"
           "    return w\n"
           "def f(a):\n"
           "    b = helper(a)\n"
           "    return b\n")
    r = _trace_once(fix, "f(1)", "f", 101, set(), 40, 48)
    # only f's own local b, never helper's w
    assert r["events"] == [("b", "101")]


def test_comprehension_target_excluded_pep709():
    # 3.12 inlines the comprehension var `i` into the frame; without the static
    # exclude it leaks as phantom steps, with it only the real local remains.
    comp = ("def h(lst):\n"
            "    out = [i * 2 for i in lst]\n"
            "    return out\n")
    assert _comprehension_targets(comp) == {"i"}
    leaked = _trace_once(comp, "h([1, 2, 3])", "h", [2, 4, 6], set(), 40, 48)
    assert ("i", "1") in leaked["events"]           # leaks without exclude
    clean = _trace_once(comp, "h([1, 2, 3])", "h", [2, 4, 6],
                        _comprehension_targets(comp), 40, 48)
    assert clean["events"] == [("out", "[2, 4, 6]")]


def test_generator_expression_target_not_excluded():
    # A genexp gets its own frame on 3.12 (no leak), so its target name must NOT
    # be excluded — otherwise a genuine same-named local's events are dropped.
    g = ("def g(n):\n"
         "    i = n * 100\n"
         "    s = sum(i for i in range(n))\n"
         "    return i\n")
    assert _comprehension_targets(g) == set()          # genexp `i` NOT excluded
    r = _trace_once(g, "g(3)", "g", 300, _comprehension_targets(g), 40, 48)
    assert ("i", "300") in r["events"]                 # real local preserved
    assert r["ret_repr"] == "300"


# --------------------------------------------------------------------------- #
# 2. Loop-heavy function hits the event cap and is skipped.
# --------------------------------------------------------------------------- #

def test_loop_heavy_hits_event_cap_and_is_skipped():
    loopy = ("def loopy(n):\n"
             "    total = 0\n"
             "    for i in range(n):\n"
             "        total = total + i + 1\n"
             "    return total\n")
    r1 = _trace_once(loopy, "loopy(100)", "loopy", 5050, set(), 40, 48)
    r2 = _trace_once(loopy, "loopy(100)", "loopy", 5050, set(), 40, 48)
    assert r1["event_overflow"] is True
    assert len(r1["events"]) > 40
    rec, reason = build_record(_sp(loopy, "loopy(100)"), "ok", r1, r2,
                               40, 48, 4000)
    assert rec is None and reason == "too_many_events"


# --------------------------------------------------------------------------- #
# 3. Rendered format byte-matches the Stage-A conventions.
# --------------------------------------------------------------------------- #

def test_rendered_format_byte_matches_stageA_conventions():
    # Reference record: FIRST line of data/exec_trace_text_train.jsonl (the real
    # Stage-A text-scratchpad training data). Its `text` ends with the exact
    # trailing convention we reproduce: "...value of x.\n# trace:\n# step 1: x =
    # 8\n# step 2: x = 4\n# final: 4\n". We ground the byte tokens against it,
    # then assert an exact-match on a hand-checked render.
    assert STAGEA_DATA.exists(), f"missing reference data: {STAGEA_DATA}"
    ref = json.loads(STAGEA_DATA.open().readline())["text"]
    assert "\n# trace:\n" in ref            # cue separator
    assert "\n# step 1: " in ref            # numbered step lines
    assert "\n# final: " in ref             # labelled final
    assert ref.endswith("\n")               # trailing newline
    # framing shared with repair_value_probe's eval prompt (render_exec_prompt):
    assert ref.startswith("You are given the following Python program.\n")
    assert "After running the program, what is the final value of x?" in ref
    assert ("Write a Python function `solve()` that takes no arguments and "
            "returns the final value of x.") in ref

    fix = ("def add(a, b):\n"
           "    c = a + b\n"
           "    return c\n")
    got = render_trace(fix, "add(2, 3)", [("c", "5")], "5")
    expected = (
        "You are given the following Python program.\n"
        "def add(a, b):\n"
        "    c = a + b\n"
        "    return c\n"
        "x = add(2, 3)\n"
        "\n"
        "After running the program, what is the final value of x?\n"
        "\n"
        "Write a Python function `solve()` that takes no arguments and "
        "returns the final value of x.\n"
        "# trace:\n"
        "# step 1: c = 5\n"
        "# final: 5\n"
    )
    assert got == expected


# --------------------------------------------------------------------------- #
# 4. `# final:` == the actual return value; expected-match recorded.
# --------------------------------------------------------------------------- #

def test_final_equals_actual_return_and_records_expected_match():
    fix = ("def add(a, b):\n"
           "    c = a + b\n"
           "    return c\n")
    r1 = _trace_once(fix, "add(2, 3)", "add", 5, set(), 40, 48)
    r2 = _trace_once(fix, "add(2, 3)", "add", 5, set(), 40, 48)
    assert r1["ret_repr"] == "5"            # == repr(actual return)
    rec, reason = build_record(_sp(fix, "add(2, 3)", fname="add",
                                   expected_repr="5"), "ok", r1, r2, 40, 48, 4000)
    assert reason is None and rec is not None
    assert rec["final_repr"] == "5"
    assert rec["text"].endswith("# final: 5\n")
    assert rec["expected_match"] is True


def test_expected_mismatch_kept_but_flagged():
    # The trace is self-consistent (final == actual return) even when the first
    # literal assert's expected differs from what THIS call returns; we KEEP it
    # and record expected_match=False.
    fix = ("def add(a, b):\n"
           "    c = a + b\n"
           "    return c\n")
    r = _trace_once(fix, "add(2, 3)", "add", 999, set(), 40, 48)  # expect 999
    assert r["ret_repr"] == "5" and r["match"] is False
    rec, reason = build_record(_sp(fix, "add(2, 3)", fname="add",
                                   expected_repr="999"), "ok", r, r, 40, 48, 4000)
    assert rec is not None and rec["expected_match"] is False
    assert rec["final_repr"] == "5"          # still the real return


# --------------------------------------------------------------------------- #
# 5. Nondeterministic function is skipped.
# --------------------------------------------------------------------------- #

def test_nondeterministic_run_pair_skipped():
    # Directly: two runs whose events differ must be dropped.
    r1 = _run_ok([("a", "1")], "1")
    r2 = _run_ok([("a", "2")], "2")
    rec, reason = build_record(_sp("x", "f()"), "ok", r1, r2, 40, 48, 4000)
    assert rec is None and reason == "nondeterministic"


def test_nondeterministic_function_skipped_end_to_end():
    # Functionally: an unseeded random draw traces to two different values.
    fix = ("import random\n"
           "def g():\n"
           "    r = random.random()\n"
           "    return r\n")
    r1 = _trace_once(fix, "g()", "g", 0.0, set(), 40, 48)
    r2 = _trace_once(fix, "g()", "g", 0.0, set(), 40, 48)
    assert r1["ret_repr"] != r2["ret_repr"]      # genuinely nondeterministic
    rec, reason = build_record(_sp(fix, "g()", fname="g"), "ok", r1, r2,
                               40, 48, 4000)
    assert rec is None and reason == "nondeterministic"


def test_short_address_repr_caught_by_double_run_not_length_cap():
    # The real corpus case (heapq.merge -> generator): a repr that embeds a
    # memory address is < max_repr chars, so the LENGTH cap misses it, but it
    # differs run-to-run so the double-run determinism check drops it. Uses the
    # exact repr strings observed on this box (an object()-address unit test
    # would be flaky: CPython can reuse a freed address).
    v1 = "<generator object merge at 0x76cd477853f0>"
    v2 = "<generator object merge at 0x76cd47785e70>"
    assert len(v1) < 48 and len(v2) < 48          # length cap would NOT catch it
    r1 = _run_ok([("result", v1)], "[1, 2]")
    r2 = _run_ok([("result", v2)], "[1, 2]")
    rec, reason = build_record(_sp("x", "f()"), "ok", r1, r2, 40, 48, 4000)
    assert rec is None and reason == "nondeterministic"


# --------------------------------------------------------------------------- #
# 6. Oversized repr is skipped (no truncation-ellipsis in training data).
# --------------------------------------------------------------------------- #

def test_oversized_value_repr_skipped():
    fix = ("def g():\n"
           "    s = 'a' * 100\n"
           "    return s\n")
    r1 = _trace_once(fix, "g()", "g", "z", set(), 40, 48)
    r2 = _trace_once(fix, "g()", "g", "z", set(), 40, 48)
    assert r1["repr_overflow"] is True or r1["ret_too_long"] is True
    rec, reason = build_record(_sp(fix, "g()", fname="g"), "ok", r1, r2,
                               40, 48, 4000)
    assert rec is None and reason == "repr_too_long"


def test_bounded_repr_fastpath_on_huge_container():
    # A big list is rejected by len() without materialising its full repr.
    r, too_long, bad = _bounded_repr(list(range(1000)), 48)
    assert r is None and too_long is True and bad is False
    # a small value passes.
    r, too_long, bad = _bounded_repr([1, 2, 3], 48)
    assert r == "[1, 2, 3]" and too_long is False and bad is False


def test_oversized_final_return_skipped():
    r1 = _run_ok([("a", "1")], None, ret_too_long=True)
    rec, reason = build_record(_sp("x", "f()"), "ok", r1, r1, 40, 48, 4000)
    assert rec is None and reason == "repr_too_long"


def test_trace_too_long_skipped():
    # Under-cap events but a rendered document over max_chars -> skip.
    events = [("v", "1")] * 5
    r = _run_ok(events, "1")
    rec, reason = build_record(_sp("def f():\n    return 1\n", "f()"),
                               "ok", r, r, 40, 48, max_chars=10)
    assert rec is None and reason == "trace_too_long"


def test_exec_error_and_timeout_skipped():
    assert build_record(_sp("x", "f()"), "error", "boom", None,
                        40, 48, 4000) == (None, "exec_error")
    assert build_record(_sp("x", "f()"), "timeout", None, None,
                        40, 48, 4000) == (None, "timeout")


# --------------------------------------------------------------------------- #
# 7. Contamination guard fires on any overlap.
# --------------------------------------------------------------------------- #

def test_contamination_guard():
    assert_no_contamination({"pk1", "pk2"}, {"pk3", "pk4"})   # disjoint: OK
    with pytest.raises(AssertionError):
        assert_no_contamination({"pk1", "pk2"}, {"pk2", "pk9"})  # overlap fires


def test_unique_task_id_guard():
    _assert_unique_task_ids([{"task_id": "a"}, {"task_id": "b"}])   # unique: OK
    with pytest.raises(AssertionError):
        _assert_unique_task_ids([{"task_id": "a"}, {"task_id": "a"}])  # dup fires


# --------------------------------------------------------------------------- #
# 8. Split determinism + problem-disjointness.
# --------------------------------------------------------------------------- #

def _fake_records():
    # 24 problems, multiplicity 1..3, 48 records total.
    recs = []
    for p in range(24):
        for j in range((p % 3) + 1):
            recs.append({"problem_key": f"pk{p:02d}", "task_id": f"pk{p:02d}/{j}"})
    return recs


def test_split_deterministic_and_problem_disjoint():
    recs = _fake_records()
    t1, h1, pk1 = split_heldout(recs, heldout_n=10, seed=0)
    t2, h2, pk2 = split_heldout(recs, heldout_n=10, seed=0)
    assert [r["task_id"] for r in t1] == [r["task_id"] for r in t2]
    assert [r["task_id"] for r in h1] == [r["task_id"] for r in h2]
    assert pk1 == pk2
    # problem-disjoint: no problem_key on both sides
    assert not ({r["problem_key"] for r in t1} & {r["problem_key"] for r in h1})
    # reaches the target (stops at the boundary that first meets it)
    assert len(h1) >= 10
    assert len(t1) + len(h1) == len(recs)


def test_split_changes_with_seed():
    recs = _fake_records()
    _, _, pk_a = split_heldout(recs, heldout_n=10, seed=0)
    _, _, pk_b = split_heldout(recs, heldout_n=10, seed=1)
    assert pk_a != pk_b       # a different seed selects a different holdback


# --------------------------------------------------------------------------- #
# 9. Assert resolution counts a non-literal-assert problem as a skip.
# --------------------------------------------------------------------------- #

def test_resolve_specs_skips_missing_and_nonliteral():
    triples = [
        {"problem_key": "k1", "task_id": "k1/0", "fix": "def f():\n    return 1\n",
         "provenance": "x"},
        {"problem_key": "kX", "task_id": "kX/0", "fix": "def f():\n    return 1\n",
         "provenance": "x"},                       # no test block -> no_problem
        {"problem_key": "k3", "task_id": "k3/0", "fix": "",  # empty fix
         "provenance": "x"},
    ]
    tests_by_key = {
        "k1": "def check(c):\n    assert f() == 1\n",
        "k3": "def check(c):\n    assert f() == 1\n",
        "kX_absent": "",
    }
    specs, skips = resolve_specs(triples, tests_by_key)
    assert len(specs) == 1 and specs[0]["problem_key"] == "k1"
    assert specs[0]["call_src"] == "f()" and specs[0]["expected_repr"] == "1"
    assert skips["empty_fix"] == 1 and skips["no_problem"] == 1
