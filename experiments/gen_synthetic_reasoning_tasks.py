"""Generate synthetic reasoning-required tasks.

Complement to gen_synthetic_memory_tasks.py: those tasks need MEMORY
(recall a value past a distractor) but not multi-step reasoning. The
families here need >=2 sequential mental steps that the model CANNOT
compress into a single forward pass — exactly the regime where the
thinking gate should provide the difference between solving and not.

Each generated task is a `Problem` (experiments.code_grader.Problem)
serialised as JSONL, so the existing grader can score the model's
generation directly. The fields:

  task_id        : "synth_reason/<family>/<idx>"
  prompt         : natural-language statement asking for a Python
                   function with a specific entry_point
  tests          : a `def check(candidate): ...` block of asserts
  entry_point    : the function name the model must define
  prompt_is_code : False (MBPP-style — completion is a standalone program)
  gold_solution  : a reference Python function known to pass the tests

Each family is parametrised on `random.Random` so we can produce
thousands of unique instances with `--seed`.

Families:
  1. multi_step_arith        — 5-8 step chain: x=a+b; y=x*c; z=y-d; ...
  2. conditional_rule        — 2-3 nested if/elif rules over an input.
  3. count_with_offset       — count items satisfying a predicate, then
                                add a fixed offset.
  4. binary_search_trace     — return the sequence of midpoint indices
                                visited while binary-searching a target.
  5. stack_machine_eval      — evaluate a postfix arithmetic expression.
  6. pattern_next            — given a numeric sequence, infer the
                                generator (arith/geometric/fib) and
                                return the next term.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_synthetic_reasoning_tasks.py \\
      --out data/synth_reasoning_tasks_small.jsonl --n_per_family 100

The generator runs every task's gold_solution through the same grader
the model will face, and refuses to emit any task whose gold doesn't
pass — generator bugs surface here, not in training.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random
import sys
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.code_grader import Problem, grade


# ---------------------------------------------------------------------------
# Family generators. Each returns a Problem with gold_solution set.
# ---------------------------------------------------------------------------

def _gen_multi_step_arith(
    rng: random.Random,
    idx: int,
    n_steps_min: int = 5,
    n_steps_max: int = 8,
) -> Problem:
    """`n_steps_min`-`n_steps_max` sequential assignments composing
    intermediate values.

    Task: define solve() returning the final value of a chain like
        x = a + b; y = x * c; z = y - d; w = z + e; return w
    The model must propagate intermediate values through each step.

    The default 5-8 range matches the historical generator. The
    difficulty-ladder uses fixed single-value ranges (e.g. n_steps_min ==
    n_steps_max == 3) to isolate a single rung.
    """
    n_steps = rng.randint(n_steps_min, n_steps_max)
    vals = [rng.randint(2, 9) for _ in range(n_steps + 1)]
    ops = [rng.choice(["+", "-", "*"]) for _ in range(n_steps)]

    chain_lines: list[str] = []
    var_names = [f"v{i}" for i in range(n_steps + 1)]
    chain_lines.append(f"Let {var_names[0]} = {vals[0]}.")
    running = vals[0]
    for i in range(n_steps):
        op = ops[i]
        if op == "+":
            new = running + vals[i + 1]
        elif op == "-":
            new = running - vals[i + 1]
        else:
            new = running * vals[i + 1]
        chain_lines.append(
            f"Let {var_names[i+1]} = {var_names[i]} {op} {vals[i+1]}."
        )
        running = new
    answer = running

    prompt = (
        "Solve the following chain of arithmetic operations step by step.\n\n"
        + "\n".join(chain_lines)
        + f"\n\nWrite a Python function `solve()` that takes no arguments and "
        f"returns the final value of {var_names[-1]}.\n"
    )

    # Gold: just hard-code the result. The model is graded on whether the
    # returned value matches; the means by which it got there are up to it.
    gold = f"def solve():\n    return {answer}\n"

    tests = (
        "def check(candidate):\n"
        f"    assert candidate() == {answer}\n"
    )
    return Problem(
        task_id=f"synth_reason/multi_step_arith/{idx}",
        prompt=prompt,
        tests=tests,
        entry_point="solve",
        prompt_is_code=False,
        gold_solution=gold,
    )


def _gen_conditional_rule(rng: random.Random, idx: int) -> Problem:
    """Nested if/elif over an integer input.

    Task: define classify(n) that applies a 2- or 3-level conditional
    rule and returns an integer transform. The model must trace the
    branches against several test inputs.
    """
    # Choose a 3-arm rule on n based on (n % m) ranges.
    m = rng.choice([3, 4, 5])
    arms = []
    for r in range(m):
        # arm: returns one of n*k+c, n-k, n+k, or a constant
        kind = rng.choice(["mul_add", "sub", "add", "const"])
        if kind == "mul_add":
            k = rng.randint(2, 5)
            c = rng.randint(-3, 3)
            expr = f"n * {k} + ({c})"
            fn = lambda n, k=k, c=c: n * k + c
        elif kind == "sub":
            k = rng.randint(1, 10)
            expr = f"n - {k}"
            fn = lambda n, k=k: n - k
        elif kind == "add":
            k = rng.randint(1, 10)
            expr = f"n + {k}"
            fn = lambda n, k=k: n + k
        else:
            c = rng.randint(-5, 20)
            expr = f"{c}"
            fn = lambda n, c=c: c
        arms.append((r, expr, fn))

    bullets: list[str] = []
    for r, expr, _ in arms:
        bullets.append(f"  - If n % {m} == {r}, return {expr}.")
    prompt = (
        "Implement a Python function `classify(n)` that takes an integer "
        "and applies the following rules:\n"
        + "\n".join(bullets)
        + "\nAssume the input is a non-negative integer.\n"
    )

    # Gold solution: a literal if/elif cascade.
    arms_src = []
    for r, expr, _ in arms:
        if r == 0:
            arms_src.append(f"    if n % {m} == {r}:\n        return {expr}")
        else:
            arms_src.append(f"    elif n % {m} == {r}:\n        return {expr}")
    gold = "def classify(n):\n" + "\n".join(arms_src) + "\n"

    # Tests: pick a handful of n covering every residue.
    test_ns = rng.sample(range(0, 60), k=min(6, m * 2))
    assert_lines = []
    for n in test_ns:
        r = n % m
        for arm_r, _, fn in arms:
            if arm_r == r:
                expected = fn(n)
                assert_lines.append(
                    f"    assert candidate({n}) == {expected}"
                )
                break
    tests = "def check(candidate):\n" + "\n".join(assert_lines) + "\n"
    return Problem(
        task_id=f"synth_reason/conditional_rule/{idx}",
        prompt=prompt,
        tests=tests,
        entry_point="classify",
        prompt_is_code=False,
        gold_solution=gold,
    )


def _gen_count_with_offset(rng: random.Random, idx: int) -> Problem:
    """Count list elements matching a predicate, then add an offset.

    Task: define solve() returning the count of elements satisfying
    a condition, plus a fixed offset. The list and offset are
    instance-specific.
    """
    n = rng.randint(8, 15)
    L = [rng.randint(0, 20) for _ in range(n)]
    threshold = rng.randint(5, 15)
    cond = rng.choice(["gt", "lt", "eq", "even", "odd"])
    offset = rng.randint(-3, 7)

    if cond == "gt":
        desc = f"greater than {threshold}"
        match = lambda x, t=threshold: x > t
    elif cond == "lt":
        desc = f"less than {threshold}"
        match = lambda x, t=threshold: x < t
    elif cond == "eq":
        desc = f"equal to {threshold}"
        match = lambda x, t=threshold: x == t
    elif cond == "even":
        desc = "even"
        match = lambda x: x % 2 == 0
    else:
        desc = "odd"
        match = lambda x: x % 2 == 1

    count = sum(1 for x in L if match(x))
    answer = count + offset

    prompt = (
        f"Given the list L = {L!r}.\n\n"
        f"Count how many elements of L are {desc}, then add {offset} "
        f"to that count.\n\n"
        f"Write a Python function `solve()` that takes no arguments and "
        f"returns the resulting integer.\n"
    )
    gold = f"def solve():\n    return {answer}\n"
    tests = (
        "def check(candidate):\n"
        f"    assert candidate() == {answer}\n"
    )
    return Problem(
        task_id=f"synth_reason/count_with_offset/{idx}",
        prompt=prompt,
        tests=tests,
        entry_point="solve",
        prompt_is_code=False,
        gold_solution=gold,
    )


def _gen_binary_search_trace(rng: random.Random, idx: int) -> Problem:
    """Return the sequence of midpoint indices visited by binary search.

    Task: implement search_trace(arr, target) that returns the list of
    midpoint indices (in order) considered by standard binary search,
    until the target is found or the range collapses.
    """
    n = rng.randint(7, 15)
    arr = sorted(rng.sample(range(1, 100), k=n))
    if rng.random() < 0.7:
        target = rng.choice(arr)
    else:
        # absent target — generate a number not in arr
        cands = [x for x in range(1, 100) if x not in arr]
        target = rng.choice(cands) if cands else arr[0]

    # Reference implementation we both grade against and embed as gold.
    def _ref(a, t):
        lo, hi = 0, len(a) - 1
        trace = []
        while lo <= hi:
            mid = (lo + hi) // 2
            trace.append(mid)
            if a[mid] == t:
                return trace
            if a[mid] < t:
                lo = mid + 1
            else:
                hi = mid - 1
        return trace

    trace_present = _ref(arr, target)
    # Build a couple test cases for the same algorithm but different
    # (arr, target) pairs, so the model must implement search_trace
    # generally, not just memorise this instance.
    extra_cases: list[tuple[list[int], int]] = []
    for _ in range(2):
        m = rng.randint(5, 10)
        a2 = sorted(rng.sample(range(1, 80), k=m))
        # Bias toward an in-array target.
        t2 = rng.choice(a2)
        extra_cases.append((a2, t2))

    prompt = (
        "Implement standard binary search over a sorted list of integers, "
        "and return the SEQUENCE of midpoint indices considered (in the "
        "order they are visited), terminating as soon as the target is "
        "found or the search range becomes empty.\n\n"
        "Write a Python function `search_trace(arr, target)` that returns "
        "this list of midpoint indices. Use `mid = (lo + hi) // 2` and the "
        "standard half-open update rules.\n\n"
        f"Example: search_trace({arr!r}, {target}) should return "
        f"{trace_present!r}.\n"
    )
    gold = (
        "def search_trace(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    trace = []\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        trace.append(mid)\n"
        "        if arr[mid] == target:\n"
        "            return trace\n"
        "        if arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return trace\n"
    )
    assert_lines = [f"    assert candidate({arr!r}, {target}) == {trace_present!r}"]
    for a2, t2 in extra_cases:
        expected = _ref(a2, t2)
        assert_lines.append(f"    assert candidate({a2!r}, {t2}) == {expected!r}")
    tests = "def check(candidate):\n" + "\n".join(assert_lines) + "\n"
    return Problem(
        task_id=f"synth_reason/binary_search_trace/{idx}",
        prompt=prompt,
        tests=tests,
        entry_point="search_trace",
        prompt_is_code=False,
        gold_solution=gold,
    )


def _gen_stack_machine_eval(rng: random.Random, idx: int) -> Problem:
    """Evaluate a small postfix (RPN) arithmetic expression.

    Task: implement eval_rpn(tokens) -> int taking a list of string
    tokens ("3", "+", "-", "*") and returning the final value of the
    stack-machine evaluation. We generate expressions with 3-6 binary
    operations such that all intermediate values stay integers and small.
    """
    n_ops = rng.randint(3, 6)
    # Build expression by alternating pushes and operations such that
    # the stack never underflows. Start with two pushes, then either
    # push or op (op only allowed when stack depth >= 2).
    tokens: list[str] = []
    stack: list[int] = []
    # initial two pushes
    for _ in range(2):
        v = rng.randint(1, 9)
        tokens.append(str(v))
        stack.append(v)
    ops_done = 0
    while ops_done < n_ops:
        # choose: push or op
        if len(stack) >= 2 and (rng.random() < 0.55 or len(stack) >= 4):
            op = rng.choice(["+", "-", "*"])
            b = stack.pop(); a = stack.pop()
            if op == "+":
                r = a + b
            elif op == "-":
                r = a - b
            else:
                r = a * b
            # Keep intermediates bounded so the model has a chance.
            if abs(r) > 500:
                stack.append(a); stack.append(b)
                continue
            tokens.append(op)
            stack.append(r)
            ops_done += 1
        else:
            v = rng.randint(1, 9)
            tokens.append(str(v))
            stack.append(v)
    # Drain remaining stack with ops (so single final value).
    while len(stack) >= 2:
        op = rng.choice(["+", "-", "*"])
        b = stack.pop(); a = stack.pop()
        if op == "+":
            r = a + b
        elif op == "-":
            r = a - b
        else:
            r = a * b
        if abs(r) > 5000:
            # If it overflows the cap mid-drain, just use addition.
            r = a + b
            op = "+"
        tokens.append(op)
        stack.append(r)
    answer = stack[0]

    prompt = (
        "Evaluate the following postfix (Reverse Polish Notation) "
        "expression, returning the final integer value on the stack.\n\n"
        f"Tokens: {tokens!r}\n\n"
        "Write a Python function `eval_rpn(tokens)` that takes a list of "
        "string tokens (operands as decimal strings, operators as one of "
        "'+', '-', '*') and returns the resulting integer.\n"
    )
    gold = (
        "def eval_rpn(tokens):\n"
        "    stack = []\n"
        "    for tok in tokens:\n"
        "        if tok in ('+', '-', '*'):\n"
        "            b = stack.pop(); a = stack.pop()\n"
        "            if tok == '+':\n"
        "                stack.append(a + b)\n"
        "            elif tok == '-':\n"
        "                stack.append(a - b)\n"
        "            else:\n"
        "                stack.append(a * b)\n"
        "        else:\n"
        "            stack.append(int(tok))\n"
        "    return stack[0]\n"
    )
    # Also test the model on the instance tokens plus a couple of
    # short hand-crafted expressions.
    extra_cases = [
        (["3", "4", "+"], 7),
        (["5", "1", "2", "+", "4", "*", "+"], 17),
        (["2", "3", "*", "4", "+"], 10),
    ]
    assert_lines = [f"    assert candidate({tokens!r}) == {answer}"]
    for toks, exp in extra_cases:
        assert_lines.append(f"    assert candidate({toks!r}) == {exp}")
    tests = "def check(candidate):\n" + "\n".join(assert_lines) + "\n"
    return Problem(
        task_id=f"synth_reason/stack_machine_eval/{idx}",
        prompt=prompt,
        tests=tests,
        entry_point="eval_rpn",
        prompt_is_code=False,
        gold_solution=gold,
    )


def _gen_pattern_next(rng: random.Random, idx: int) -> Problem:
    """Identify the rule behind a numeric sequence and predict the next term.

    Families:
      - arithmetic:  a, a+d, a+2d, ...
      - geometric:   a, a*r, a*r^2, ... (small ratios)
      - fibonacci-like:  a, b, a+b, a+2b, 2a+3b, ...
      - modular:     n_i = (n_{i-1} + k) % m
    """
    kind = rng.choice(["arith", "geom", "fib", "mod"])
    n = rng.randint(5, 7)
    if kind == "arith":
        a = rng.randint(1, 20)
        d = rng.randint(1, 9)
        seq = [a + i * d for i in range(n)]
        nxt = a + n * d
        hint = "an arithmetic progression"
    elif kind == "geom":
        a = rng.randint(1, 5)
        r = rng.randint(2, 4)
        seq = [a * (r ** i) for i in range(n)]
        nxt = a * (r ** n)
        hint = "a geometric progression"
    elif kind == "fib":
        a = rng.randint(1, 5)
        b = rng.randint(1, 5)
        seq = [a, b]
        for _ in range(n - 2):
            seq.append(seq[-1] + seq[-2])
        nxt = seq[-1] + seq[-2]
        hint = "a Fibonacci-like sequence where each term is the sum of the two preceding terms"
    else:
        m = rng.randint(7, 13)
        k = rng.randint(1, m - 1)
        a = rng.randint(0, m - 1)
        seq = []
        cur = a
        for _ in range(n):
            seq.append(cur)
            cur = (cur + k) % m
        nxt = (seq[-1] + k) % m
        hint = f"a modular sequence with step {k} mod {m}"

    prompt = (
        f"The following sequence follows a simple rule ({hint}):\n\n"
        f"{seq!r}\n\n"
        "Write a Python function `solve()` that takes no arguments and "
        "returns the NEXT term in the sequence (as an integer).\n"
    )
    gold = f"def solve():\n    return {nxt}\n"
    tests = (
        "def check(candidate):\n"
        f"    assert candidate() == {nxt}\n"
    )
    return Problem(
        task_id=f"synth_reason/pattern_next/{idx}",
        prompt=prompt,
        tests=tests,
        entry_point="solve",
        prompt_is_code=False,
        gold_solution=gold,
    )


_FAMILIES: dict[str, Callable[[random.Random, int], Problem]] = {
    "multi_step_arith":      _gen_multi_step_arith,
    "conditional_rule":      _gen_conditional_rule,
    "count_with_offset":     _gen_count_with_offset,
    "binary_search_trace":   _gen_binary_search_trace,
    "stack_machine_eval":    _gen_stack_machine_eval,
    "pattern_next":          _gen_pattern_next,
}


# ---------------------------------------------------------------------------
# Serialisation + validation
# ---------------------------------------------------------------------------

def problem_to_record(p: Problem) -> dict:
    """Serialise a Problem to a JSONL-safe dict."""
    d = dataclasses.asdict(p)
    return d


def record_to_problem(rec: dict) -> Problem:
    """Inverse of problem_to_record."""
    return Problem(
        task_id=rec["task_id"],
        prompt=rec["prompt"],
        tests=rec["tests"],
        entry_point=rec["entry_point"],
        prompt_is_code=rec.get("prompt_is_code", False),
        gold_solution=rec.get("gold_solution"),
    )


def _gold_passes(p: Problem) -> bool:
    """Run p.gold_solution against p.tests; return True iff `tier == 'pass'`.

    Used to drop any task whose gold the grader can't pass — that's a
    generator bug, not training signal.
    """
    if p.gold_solution is None:
        return False
    res = grade(p, p.gold_solution)
    return res.passed


def generate(
    families: list[str],
    n_per_family: int,
    seed: int,
    validate: bool = True,
    arith_n_steps: int | None = None,
) -> list[Problem]:
    """Generate `n_per_family` tasks for each family in `families`.

    If `validate` is True (default), each generated task's gold solution
    is graded; tasks whose gold doesn't pass are dropped silently (the
    caller can inspect the returned list length to detect attrition).

    `arith_n_steps`, when set, fixes the chain length of the
    `multi_step_arith` family to exactly that value (both min and max),
    used to build a single difficulty rung of the ladder. It has no
    effect on the other families.
    """
    rng = random.Random(seed)
    out: list[Problem] = []
    for fam in families:
        gen = _FAMILIES[fam]
        if fam == "multi_step_arith" and arith_n_steps is not None:
            gen = lambda r, i, _n=arith_n_steps: _gen_multi_step_arith(
                r, i, n_steps_min=_n, n_steps_max=_n)
        n_emitted = 0
        idx = 0
        # Up to 3x sampling budget per family — generators are tight, the
        # multiplier is a small insurance margin for the validation cull.
        max_attempts = n_per_family * 3
        while n_emitted < n_per_family and idx < max_attempts:
            p = gen(rng, idx)
            idx += 1
            if validate and not _gold_passes(p):
                continue
            out.append(p)
            n_emitted += 1
        print(f"  {fam:<24} emitted {n_emitted}/{n_per_family} "
              f"(attempts={idx})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=str,
                    help="Output JSONL path. Overwrites.")
    ap.add_argument("--n_per_family", type=int, default=1000,
                    help="Number of tasks per family.")
    ap.add_argument("--families", type=str, default="",
                    help="Comma-separated family names; empty = all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_validate", action="store_true",
                    help="Skip the gold-solution grading pass (faster).")
    ap.add_argument("--arith_n_steps", type=int, default=None,
                    help="If set, fix the multi_step_arith chain length to "
                         "exactly this many steps (single difficulty rung).")
    args = ap.parse_args()

    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
        for f in families:
            if f not in _FAMILIES:
                raise SystemExit(
                    f"unknown family {f!r}; choices: {list(_FAMILIES)}"
                )
    else:
        families = list(_FAMILIES)

    problems = generate(
        families=families,
        n_per_family=args.n_per_family,
        seed=args.seed,
        validate=not args.no_validate,
        arith_n_steps=args.arith_n_steps,
    )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in problems:
            f.write(json.dumps(problem_to_record(p)) + "\n")
    print(f"\ntotal: {len(problems)} tasks -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
