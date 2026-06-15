"""Tests for the verifier-arbitrated best-of-{no-think, think} mechanism.

The load-bearing property is the Pareto invariant: the returned score is
ALWAYS >= the no-think score, for every possible (no-think, think) grade pair.
"""
import itertools

import pytest

from experiments.best_of_think import arbitrate, best_of_think


def _const(x):
    return lambda: x


def test_arbitrate_picks_max_and_breaks_ties_to_prefer():
    assert arbitrate([0.0, 1.0]) == 1
    assert arbitrate([1.0, 0.0]) == 0
    # tie -> prefer (default 0)
    assert arbitrate([0.5, 0.5]) == 0
    assert arbitrate([0.5, 0.5], prefer=1) == 1
    with pytest.raises(ValueError):
        arbitrate([])


@pytest.mark.parametrize("s0,sT", itertools.product(
    [0.0, 0.05, 0.2, 0.5, 0.7, 1.0], repeat=2))
def test_strictly_geq_nothink_for_every_grade_pair(s0, sT):
    """THE invariant: best-of result score >= no-think score, always."""
    grades = {"NT": s0, "T": sT}
    res = best_of_think(_const("NT"), _const("T"), lambda c: grades[c],
                        skip_think_if_passed=False)
    assert res.score >= s0 - 1e-12
    assert res.score == max(s0, sT)
    # the returned text matches the chosen branch's score
    assert grades[res.text] == res.score


def test_ties_prefer_nothink_branch():
    """Equal scores must keep no-think (cheaper / structurally safe default)."""
    res = best_of_think(_const("NT"), _const("T"), lambda c: 0.4,
                        skip_think_if_passed=False)
    assert res.used_think is False
    assert res.text == "NT"


def test_think_adopted_only_when_strictly_better():
    res = best_of_think(_const("NT"), _const("T"),
                        lambda c: 1.0 if c == "T" else 0.2,
                        skip_think_if_passed=False)
    assert res.used_think is True
    assert res.score == 1.0


def test_skip_think_when_nothink_already_passes():
    """Cost lever: a perfect no-think result must skip the think branch."""
    calls = {"think": 0}

    def gen_think():
        calls["think"] += 1
        return "T"

    res = best_of_think(_const("NT"), gen_think, lambda c: 1.0,
                        max_score=1.0, skip_think_if_passed=True)
    assert res.think_evaluated is False
    assert res.score_think is None
    assert calls["think"] == 0          # think branch never generated
    assert res.score == 1.0


def test_think_runs_when_nothink_fails():
    calls = {"think": 0}

    def gen_think():
        calls["think"] += 1
        return "T"

    res = best_of_think(_const("NT"), gen_think,
                        lambda c: 1.0 if c == "T" else 0.0,
                        max_score=1.0, skip_think_if_passed=True)
    assert calls["think"] == 1
    assert res.used_think is True
    assert res.score == 1.0


def test_think_worse_falls_back_to_nothink():
    """The documented code hurt-case: thinking is WORSE -> keep no-think."""
    res = best_of_think(_const("NT"), _const("T"),
                        lambda c: 0.65 if c == "NT" else 0.13,
                        skip_think_if_passed=False)
    assert res.used_think is False
    assert res.score == 0.65
    assert res.score_think == 0.13      # recorded for diagnostics
