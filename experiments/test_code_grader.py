"""Tests for the dense execution-grounded grader (experiments/code_grader.py).

Covers the tier ladder: syntax_error / exec_error / runtime_error /
partial (fractional) / pass. The point of the dense grader is that a
weak model gets a differentiated signal before it can fully solve a
task — these tests pin the score for each rung.
"""
from experiments.code_grader import Problem, grade, _compute_score


def _problem(tests: str, entry_point: str = "f") -> Problem:
    """HumanEval-style problem: prompt is the function header, the
    completion finishes the body. Here the prompt is empty and we pass
    the whole solution as the completion."""
    return Problem(task_id="t", prompt="", tests=tests,
                   entry_point=entry_point, prompt_is_code=True)


# Four independent assertions so fractional scoring is observable.
_FOUR_ASSERT_CHECK = (
    "def check(candidate):\n"
    "    assert candidate(1) == 2\n"
    "    assert candidate(2) == 3\n"
    "    assert candidate(3) == 4\n"
    "    assert candidate(10) == 11\n"
)


def test_pass_all():
    res = grade(_problem(_FOUR_ASSERT_CHECK), "def f(x):\n    return x + 1\n")
    assert res.tier == "pass"
    assert res.passed is True
    assert res.n_tests == 4 and res.n_passed == 4
    assert res.score == 1.0


def test_partial_half():
    # `x + 1` only when x is odd, else wrong → asserts 1 and 3 pass (1,3 odd),
    # asserts 2 and 10 fail.
    sol = "def f(x):\n    return x + 1 if x % 2 == 1 else x + 99\n"
    res = grade(_problem(_FOUR_ASSERT_CHECK), sol)
    assert res.tier == "partial"
    assert res.passed is False
    assert res.n_tests == 4 and res.n_passed == 2
    # 0.2 + 0.7 * 0.5 = 0.55
    assert abs(res.score - 0.55) < 1e-9


def test_partial_none():
    # Runs fine, defines f, but every assertion fails.
    res = grade(_problem(_FOUR_ASSERT_CHECK), "def f(x):\n    return -1\n")
    assert res.tier == "partial"
    assert res.n_tests == 4 and res.n_passed == 0
    assert abs(res.score - 0.2) < 1e-9  # 0.2 + 0.7 * 0.0


def test_syntax_error():
    res = grade(_problem(_FOUR_ASSERT_CHECK), "def f(x)\n    return x + 1\n")
    assert res.tier == "syntax_error"
    assert res.passed is False
    assert res.score == 0.0


def test_exec_error_entry_point_undefined():
    # Compiles and runs, but never defines `f`.
    res = grade(_problem(_FOUR_ASSERT_CHECK), "g = 1\n")
    assert res.tier == "exec_error"
    assert res.score == 0.05


def test_exec_error_solution_raises():
    # Compiles, but exec'ing the module body raises before f is usable.
    res = grade(_problem(_FOUR_ASSERT_CHECK),
                "raise ValueError('boom')\ndef f(x):\n    return x + 1\n")
    assert res.tier == "exec_error"
    assert res.score == 0.05


def test_runtime_error_in_check_setup():
    # A non-assert setup statement in check() crashes.
    check = (
        "def check(candidate):\n"
        "    assert candidate(1) == 2\n"
        "    boom = 1 / 0\n"
        "    assert candidate(2) == 3\n"
    )
    res = grade(_problem(check), "def f(x):\n    return x + 1\n")
    assert res.tier == "runtime_error"
    # one assert ran before the crash
    assert res.n_passed == 1
    assert abs(res.score - 0.2) < 1e-9


def test_failed_assert_does_not_mask_later_ones():
    # First assert fails, but the remaining three still get counted.
    check = (
        "def check(candidate):\n"
        "    assert candidate(1) == 999\n"   # fails
        "    assert candidate(2) == 3\n"     # passes
        "    assert candidate(3) == 4\n"     # passes
        "    assert candidate(10) == 11\n"   # passes
    )
    res = grade(_problem(check), "def f(x):\n    return x + 1\n")
    assert res.tier == "partial"
    assert res.n_tests == 4 and res.n_passed == 3


def test_score_ladder_monotonic():
    # The tier ladder must be monotonic in "how correct" the output is.
    s_syntax = _compute_score("syntax_error", 0, 0)
    s_exec = _compute_score("exec_error", 0, 0)
    s_runtime = _compute_score("runtime_error", 0, 0)
    s_partial_lo = _compute_score("partial", 4, 0)
    s_partial_hi = _compute_score("partial", 4, 3)
    s_pass = _compute_score("pass", 4, 4)
    assert s_syntax < s_exec < s_runtime <= s_partial_lo < s_partial_hi < s_pass
