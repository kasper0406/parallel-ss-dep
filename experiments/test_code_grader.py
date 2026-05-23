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


# --- error_text: the diagnosis fed back into the self-repair loop (#80) ---

def test_error_text_none_on_pass():
    res = grade(_problem(_FOUR_ASSERT_CHECK), "def f(x):\n    return x + 1\n")
    assert res.error_text is None


def test_error_text_syntax_error_has_line():
    res = grade(_problem(_FOUR_ASSERT_CHECK), "def f(x)\n    return x + 1\n")
    assert res.error_text is not None
    assert "SyntaxError" in res.error_text
    assert "line" in res.error_text


def test_error_text_exec_error_has_traceback():
    res = grade(_problem(_FOUR_ASSERT_CHECK),
                "raise ValueError('boom')\ndef f(x):\n    return x + 1\n")
    assert res.error_text is not None
    assert "ValueError" in res.error_text and "boom" in res.error_text


def test_error_text_partial_lists_failed_asserts():
    # `return -1` fails every assertion — error_text should list them.
    res = grade(_problem(_FOUR_ASSERT_CHECK), "def f(x):\n    return -1\n")
    assert res.tier == "partial"
    assert res.error_text is not None
    assert "4/4 assertions failed" in res.error_text
    # the actual failed assertion source must appear, so the model can diagnose
    assert "candidate(1) == 2" in res.error_text
    assert "AssertionError" in res.error_text


def test_error_text_entry_point_undefined():
    res = grade(_problem(_FOUR_ASSERT_CHECK), "g = 1\n")
    assert res.error_text is not None
    # Message is "could not resolve entry_point `f`: NameError ..."
    assert "f" in res.error_text and "NameError" in res.error_text


def test_error_text_runtime_error_shows_statement():
    check = (
        "def check(candidate):\n"
        "    assert candidate(1) == 2\n"
        "    boom = 1 / 0\n"
    )
    res = grade(_problem(check), "def f(x):\n    return x + 1\n")
    assert res.tier == "runtime_error"
    assert res.error_text is not None
    assert "ZeroDivisionError" in res.error_text
    assert "1 / 0" in res.error_text


# ---------------------------------------------------------------------------
# Expression-style entry_point (LeetCode "Solution().method" pattern)
# ---------------------------------------------------------------------------

def test_expression_entry_point_class_instantiation():
    """LeetCode-style: entry_point is `Solution().twoSum`, an expression
    that instantiates a class and extracts a bound method. The fix in
    _exec_target uses `eval(entry_point, ns)` instead of `ns[entry_point]`
    so this resolves correctly.
    """
    code = (
        "class Solution:\n"
        "    def twoSum(self, nums, target):\n"
        "        d = {}\n"
        "        for i, x in enumerate(nums):\n"
        "            if (y := target - x) in d:\n"
        "                return [d[y], i]\n"
        "            d[x] = i\n"
    )
    tests = (
        "def check(candidate):\n"
        "    assert candidate(nums=[2, 7, 11, 15], target=9) == [0, 1]\n"
        "    assert candidate(nums=[3, 2, 4], target=6) == [1, 2]\n"
        "    assert candidate(nums=[3, 3], target=6) == [0, 1]\n"
    )
    p = Problem(task_id="t", prompt="", tests=tests,
                entry_point="Solution().twoSum", prompt_is_code=True)
    res = grade(p, code)
    assert res.tier == "pass", f"expected pass, got {res.tier}: {res.error_text}"
    assert res.n_passed == 3 and res.n_tests == 3
    assert res.score == 1.0


def test_expression_entry_point_resolution_failure():
    """Expression entry_point that references an undefined class still
    surfaces a clean exec_error / NameError, not a KeyError or crash."""
    tests = "def check(candidate):\n    assert candidate(1) == 1\n"
    p = Problem(task_id="t", prompt="", tests=tests,
                entry_point="DoesNotExist().method", prompt_is_code=True)
    res = grade(p, "x = 1\n")
    assert res.tier == "exec_error"
    assert res.error_text is not None
    assert "DoesNotExist" in res.error_text


# ---------------------------------------------------------------------------
# Expanded-corpus loaders (MBPP variants + LeetCode)
# ---------------------------------------------------------------------------

def test_load_mbpp_all_full_size():
    """The expanded MBPP loader should return all 974 problems
    (train 374 + validation 90 + test 500 + prompt 10).
    """
    from experiments.code_grader import load_mbpp_all
    probs = load_mbpp_all()
    assert len(probs) == 974
    # Every problem should have natural-language prompt + assert-style tests
    sample = probs[0]
    assert sample.task_id.startswith("mbpp_")
    assert sample.prompt_is_code is False
    assert "def check" in sample.tests
    assert sample.entry_point  # parsed function name


def test_load_mbpp_plus_full_size():
    from experiments.code_grader import load_mbpp_plus
    probs = load_mbpp_plus()
    assert len(probs) == 378
    sample = probs[0]
    assert sample.task_id.startswith("mbppplus/")


def test_load_leetcode_loads_problems():
    """LeetCode loader gets ≥2000 problems with expression entry_points
    and HumanEval-style test blocks (`def check(candidate)`).
    """
    from experiments.code_grader import load_leetcode
    probs = load_leetcode()
    assert len(probs) >= 2000
    sample = probs[0]
    assert sample.task_id.startswith("leetcode/")
    assert sample.prompt_is_code is True
    # LeetCode entry_points are expressions like "Solution().method"
    assert "Solution" in sample.entry_point and "." in sample.entry_point
    assert "def check(candidate)" in sample.tests


def test_load_super_combined_size():
    """super_combined = mbpp_all + mbpp_plus + leetcode."""
    from experiments.code_grader import load_super_combined
    probs = load_super_combined()
    # mbpp_all (974) + mbpp_plus (378) + leetcode (≥2000) → at least 3352
    assert len(probs) >= 3352


def test_loader_registry_has_new_keys():
    from experiments.code_grader import LOADERS
    for k in ("mbpp", "mbpp_all", "mbpp_plus", "mbpp_combined",
              "leetcode", "super_combined"):
        assert k in LOADERS, f"LOADERS missing key {k!r}"
