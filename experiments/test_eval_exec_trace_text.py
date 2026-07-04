"""Tests for the Stage-A text-scratchpad executor harness
(`eval_exec_trace_text.py`, EXEC_TRACE_LATENT_PLAN.md phase N3 text-staging,
2026-07-04).

CPU-only, pure-Python -- no CUDA / FLA / HF / model needed. These pin the
parsers (the load-bearing, error-prone part of a text-scratchpad eval: a
model's freeform generation is never guaranteed well-formed) plus the
aggregation / verdict math.
"""
from __future__ import annotations

from experiments.eval_exec_trace_text import (
    aggregate_rung,
    build_prompts,
    compute_verdict,
    _has_complete_direct_value,
    _has_complete_final_line,
    load_rung,
    parse_direct,
    parse_final,
    parse_step_lines,
)


# --------------------------------------------------------------------------- #
# parse_step_lines
# --------------------------------------------------------------------------- #

def test_parse_step_lines_well_formed():
    text = "# step 1: x = 8\n# step 2: x = 4\n# final: 4\n"
    assert parse_step_lines(text) == {1: 8, 2: 4}


def test_parse_step_lines_extra_whitespace_and_case_tolerant():
    text = "#  Step  1  :   x=8\n#step2:x = 4\n"
    assert parse_step_lines(text) == {1: 8, 2: 4}


def test_parse_step_lines_missing_line_absent_from_dict():
    # Step 2's line never appears (model skipped it) -- must NOT show up as
    # a spurious key, and callers score it as wrong via a missing lookup,
    # never a crash.
    text = "# step 1: x = 8\n# final: 4\n"
    parsed = parse_step_lines(text)
    assert parsed == {1: 8}
    assert 2 not in parsed


def test_parse_step_lines_malformed_line_ignored():
    # No '=' at all -- the whole line fails to match, contributes nothing.
    text = "# step 1: x is 8\n# step 2: x = 4\n"
    assert parse_step_lines(text) == {2: 4}


def test_parse_step_lines_duplicate_step_first_wins():
    text = "# step 1: x = 8\n# step 1: x = 9\n"
    assert parse_step_lines(text) == {1: 8}


def test_parse_step_lines_negative_values():
    text = "# step 1: x = -3\n"
    assert parse_step_lines(text) == {1: -3}


def test_parse_step_lines_empty_text():
    assert parse_step_lines("") == {}


# --------------------------------------------------------------------------- #
# parse_final
# --------------------------------------------------------------------------- #

def test_parse_final_normal():
    assert parse_final("# step 1: x = 8\n# final: 4\n") == 4


def test_parse_final_missing_returns_none():
    assert parse_final("# step 1: x = 8\n") is None


def test_parse_final_malformed_returns_none():
    assert parse_final("# final: banana\n") is None


def test_parse_final_takes_first_occurrence():
    # Degenerate repeated-cue generation -- first "# final:" wins.
    assert parse_final("# final: 4\n# final: 7\n") == 4


def test_parse_final_tolerant_of_case_and_spacing():
    assert parse_final("#Final:7") == 7


# --------------------------------------------------------------------------- #
# parse_direct
# --------------------------------------------------------------------------- #

def test_parse_direct_leading_value():
    assert parse_direct("4\n") == 4


def test_parse_direct_leading_whitespace():
    assert parse_direct("  4\n") == 4


def test_parse_direct_no_digits_returns_none():
    assert parse_direct("the answer is unclear") is None


def test_parse_direct_negative_value():
    assert parse_direct("-3 is the value") == -3


def test_parse_direct_picks_first_int_even_if_prose_precedes_it():
    assert parse_direct("well, x = 8 at the end") == 8


# --------------------------------------------------------------------------- #
# Early-stop predicates.
# --------------------------------------------------------------------------- #

def test_has_complete_final_line_true_with_trailing_newline():
    assert _has_complete_final_line("# step 1: x = 8\n# final: 4\n")


def test_has_complete_final_line_false_without_trailing_char():
    # Value looks like it might still be mid-emission (no char follows yet).
    assert not _has_complete_final_line("# final: 4")


def test_has_complete_final_line_false_when_no_final_line():
    assert not _has_complete_final_line("# step 1: x = 8\n")


def test_has_complete_direct_value_true_and_false():
    assert _has_complete_direct_value("4\n")
    assert not _has_complete_direct_value("4")
    assert not _has_complete_direct_value("no digits here")


# --------------------------------------------------------------------------- #
# build_prompts
# --------------------------------------------------------------------------- #

def test_build_prompts_exact_cue_suffixes():
    rec = {"prompt": "line one\nline two.\n"}
    with_trace, direct = build_prompts(rec)
    assert with_trace == "line one\nline two.\n# trace:\n"
    assert direct == "line one\nline two.\n# final: "


def test_build_prompts_rstrips_trailing_whitespace_before_cue():
    rec = {"prompt": "line one\nline two.   \n\n"}
    with_trace, direct = build_prompts(rec)
    # No blank-line / trailing-space artifacts between the prompt body and
    # the cue -- exactly one '\n' separates them, matching the trained text.
    assert with_trace == "line one\nline two.\n# trace:\n"
    assert direct == "line one\nline two.\n# final: "


# --------------------------------------------------------------------------- #
# aggregate_rung
# --------------------------------------------------------------------------- #

def _rec(step_found, step_correct, trace_exact, ans_pred, ans_correct,
        direct_pred, direct_correct):
    return {"step_found": step_found, "step_correct": step_correct,
           "trace_exact": trace_exact, "ans_pred": ans_pred,
           "ans_correct": ans_correct, "direct_pred": direct_pred,
           "direct_correct": direct_correct}


def test_aggregate_rung_basic_micro_average():
    # 2 examples, K=2: example 1 both steps right, example 2 one right/one
    # missing -> trace_state_acc = 3/4, full_format_rate = 1/2 (only ex1 has
    # both step lines present).
    recs = [
        _rec([1, 1], [1, 1], 1, 5, 1, 5, 1),
        _rec([1, 0], [1, 0], 0, None, 0, None, 0),
    ]
    agg = aggregate_rung(2, recs)
    assert agg["K"] == 2 and agg["n"] == 2
    assert agg["trace_state_acc"] == 3 / 4
    assert agg["full_format_rate"] == 0.5
    assert agg["trace_exact"] == 0.5
    assert agg["answer_exact"] == 0.5
    assert agg["answer_parse_rate"] == 0.5
    assert agg["answer_direct"] == 0.5
    assert agg["direct_parse_rate"] == 0.5
    assert agg["lift_pp"] == 0.0


def test_aggregate_rung_empty_is_all_none():
    agg = aggregate_rung(4, [])
    assert agg["n"] == 0
    assert agg["trace_state_acc"] is None
    assert agg["full_format_rate"] is None
    assert agg["answer_exact"] is None
    assert agg["lift_pp"] is None


def test_aggregate_rung_lift_pp_sign_and_magnitude():
    # answer_exact=1.0 (both correct via trace), answer_direct=0.0 (both
    # wrong without it) -> lift_pp = +100.
    recs = [
        _rec([1], [1], 1, 3, 1, None, 0),
        _rec([1], [1], 1, 7, 1, 2, 0),
    ]
    agg = aggregate_rung(1, recs)
    assert agg["answer_exact"] == 1.0
    assert agg["answer_direct"] == 0.0
    assert agg["lift_pp"] == 100.0


# --------------------------------------------------------------------------- #
# compute_verdict
# --------------------------------------------------------------------------- #

def _rung(K, trace_state_acc, lift_pp):
    return {"K": K, "trace_state_acc": trace_state_acc, "lift_pp": lift_pp}


def test_compute_verdict_passes_when_both_criteria_met():
    rungs = [
        _rung(2, 0.9, 20.0),
        _rung(4, 0.6, 10.0),
        _rung(6, 0.55, 8.0),
    ]
    v = compute_verdict(rungs)
    assert v["trace_state_acc_at_K4"] == 0.6
    assert v["trace_state_acc_at_K4_pass (>=0.5)"] is True
    assert v["lift_pass_all_K_ge_4 (>=5pp each)"] is True
    assert v["overall_pass"] is True


def test_compute_verdict_fails_on_state_acc_at_k4():
    rungs = [_rung(4, 0.3, 10.0), _rung(6, 0.6, 10.0)]
    v = compute_verdict(rungs)
    assert v["trace_state_acc_at_K4_pass (>=0.5)"] is False
    assert v["overall_pass"] is False


def test_compute_verdict_fails_if_any_k_ge_4_rung_misses_lift():
    rungs = [_rung(4, 0.9, 10.0), _rung(6, 0.9, 2.0)]  # K=6 lift below 5pp
    v = compute_verdict(rungs)
    assert v["lift_pass_all_K_ge_4 (>=5pp each)"] is False
    assert v["overall_pass"] is False


def test_compute_verdict_missing_k4_fails_state_criterion():
    rungs = [_rung(2, 0.9, 20.0), _rung(6, 0.9, 10.0)]  # no K=4 rung at all
    v = compute_verdict(rungs)
    assert v["trace_state_acc_at_K4"] is None
    assert v["trace_state_acc_at_K4_pass (>=0.5)"] is False
    assert v["overall_pass"] is False


def test_compute_verdict_no_applicable_k_ge_4_rungs_fails_lift():
    rungs = [_rung(2, 0.9, 20.0), _rung(4, 0.6, 10.0)]
    # only K=4 is >=4 here and it passes, so lift should still pass (uses
    # whatever K>=4 rungs are present)
    v = compute_verdict(rungs)
    assert v["lift_pass_all_K_ge_4 (>=5pp each)"] is True
    rungs_none_ge4 = [_rung(2, 0.9, 20.0), _rung(3, 0.9, 20.0)]
    v2 = compute_verdict(rungs_none_ge4)
    assert v2["lift_pass_all_K_ge_4 (>=5pp each)"] is False  # vacuous -> FAIL, not PASS


# --------------------------------------------------------------------------- #
# load_rung
# --------------------------------------------------------------------------- #

def test_load_rung_reads_jsonl(tmp_path):
    import json
    p = tmp_path / "toy_n3.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"task_id": "a", "prompt": "p", "answer": 1,
                            "intermediates": [1, 2, 3], "rung": 3}) + "\n")
        f.write("\n")  # blank line must be skipped, not crash
        f.write(json.dumps({"task_id": "b", "prompt": "p2", "answer": 4,
                            "intermediates": [4, 5, 6], "rung": 3}) + "\n")
    recs = load_rung(str(tmp_path / "toy"), 3)
    assert len(recs) == 2
    assert recs[0]["task_id"] == "a" and recs[1]["task_id"] == "b"


def test_load_rung_missing_file_returns_empty(tmp_path):
    assert load_rung(str(tmp_path / "nope"), 5) == []
