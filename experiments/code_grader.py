"""Programmatic grader for code-RL and code-eval.

Provides a uniform `grade(problem, generated_code) -> GradingResult` interface
on top of unit-test-style datasets (HumanEval, MBPP, MBPP+). The actual test
execution runs in a sandboxed subprocess (no model imports, hard timeout)
so it's safe to call from a hot RL loop.

Pulled out of `experiments/eval_humaneval.py:_run_test_in_subprocess` so
both the eval script and the RL trainer share the same execution backend.

CLI usage (sanity-check a checkpoint):

    python -m experiments.code_grader --ckpt CKPT --dataset humaneval \\
        --max_problems 20 --max_gen 256
"""
from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import pathlib
import signal
import sys
from contextlib import contextmanager
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# Stop sequences typical of function-end at column 0. We trim model
# generations here so any natural-language continuation past the function
# body doesn't break the test exec.
_STOP_SEQUENCES = ["\nclass ", "\ndef ", "\nif __name__", "\n#", "\nprint("]


@contextmanager
def _time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@dataclasses.dataclass
class Problem:
    """A single code-grading problem, normalised across dataset families."""
    task_id: str
    prompt: str                # what the model sees
    tests: str                 # a Python block that, when exec'd against the solution, raises on failure
    entry_point: str           # function name the tests will look up in the exec'd namespace
    prompt_is_code: bool = True   # If True (HumanEval), the prompt is the start of the
                                  # solution and we concat prompt+completion for exec.
                                  # If False (MBPP), the prompt is natural-language
                                  # instructions and we exec the completion alone.
    gold_solution: str | None = None


@dataclasses.dataclass
class GradingResult:
    passed: bool               # True iff the test block ran to completion without raising
    error: str | None = None   # exception class name if it raised; "timeout" on timeout
    elapsed_s: float = 0.0


def truncate_at_stop(text: str) -> str:
    """Trim generated text at first natural function-boundary."""
    earliest = len(text)
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx >= 0 and idx < earliest:
            earliest = idx
    return text[:earliest]


def _exec_target(code: str, test: str, entry_point: str, q: mp.Queue):
    """Subprocess body: exec the solution, then the test block, then check()."""
    try:
        ns: dict = {}
        # The HumanEval test block defines `check(candidate)`; calling
        # check(ns[entry_point]) is the actual assertion.
        with _time_limit(5):
            exec(code, ns)
            exec(test, ns)
            ns["check"](ns[entry_point])
        q.put(("ok", None))
    except Exception as exc:
        q.put(("err", type(exc).__name__))


def grade(problem: Problem, completion: str, timeout_s: int = 7) -> GradingResult:
    """Run `completion` against `problem`'s test block in a subprocess.

    `completion` is the model's raw output text *after* the prompt. We
    concatenate `problem.prompt + completion` (so the function header
    from the prompt is in scope), then `truncate_at_stop` to keep only
    the function body.
    """
    import time

    if problem.prompt_is_code:
        # HumanEval-style: prompt is the function header + docstring,
        # the completion finishes the body. Truncate at the next top-level
        # `def` / `class` / `#` etc. so trailing natural-language doesn't
        # break the exec.
        full_code = problem.prompt + truncate_at_stop(completion)
    else:
        # MBPP-style: prompt is natural-language instructions; the
        # completion must be a complete program (possibly multiple
        # top-level defs/classes). Don't truncate — exec the whole thing.
        full_code = completion

    q = mp.Queue()
    p = mp.Process(target=_exec_target, args=(full_code, problem.tests,
                                              problem.entry_point, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout=timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return GradingResult(passed=False, error="timeout",
                             elapsed_s=time.perf_counter() - t0)
    try:
        status, err = q.get_nowait()
    except Exception:
        return GradingResult(passed=False, error="no-result",
                             elapsed_s=time.perf_counter() - t0)
    return GradingResult(
        passed=(status == "ok"), error=err,
        elapsed_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_humaneval() -> list[Problem]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    out: list[Problem] = []
    for x in ds:
        out.append(Problem(
            task_id=x["task_id"],
            prompt=x["prompt"],
            tests=x["test"],
            entry_point=x["entry_point"],
            gold_solution=x.get("canonical_solution"),
        ))
    return out


def load_mbpp() -> list[Problem]:
    """MBPP — Microsoft's 974 short Python problems with assert tests.

    MBPP problems look like: `Write a function to ...`. We construct a
    HumanEval-style `check(candidate)` block from the assert list so the
    grader subprocess interface is uniform.
    """
    from datasets import load_dataset
    ds = load_dataset("mbpp", split="train")
    out: list[Problem] = []
    for x in ds:
        # Each problem ships ~3 asserts in `test_list`. Build a check()
        # that runs them via the model's generated function. The MBPP
        # asserts already name the function directly, so we don't need
        # to substitute an `entry_point` — but the HumanEval-style grader
        # wants a candidate-callable. We wrap.
        asserts = "\n    ".join(x["test_list"])
        test_block = f"def check(candidate):\n    {asserts}\n"
        # MBPP doesn't reliably name the function in the prompt the way
        # HumanEval does — the function name is whatever appears in the
        # asserts. Parse the first assert to recover it.
        first = x["test_list"][0]
        # heuristic: `assert <fn>(...) == ...`
        try:
            fn_name = first.split("assert", 1)[1].split("(", 1)[0].strip()
        except IndexError:
            fn_name = "candidate"
        # MBPP "text" is the prompt; "code" is the gold solution.
        prompt = f"{x['text']}\n"
        out.append(Problem(
            task_id=f"mbpp/{x['task_id']}",
            prompt=prompt,
            tests=test_block,
            entry_point=fn_name,
            prompt_is_code=False,
            gold_solution=x.get("code"),
        ))
    return out


LOADERS: dict[str, Callable[[], list[Problem]]] = {
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,
}


# ---------------------------------------------------------------------------
# Standalone CLI (smoke + per-model sanity check)
# ---------------------------------------------------------------------------

def _self_test_with_gold(problems: list[Problem], max_problems: int) -> None:
    """Sanity-check the grader by running each problem's `gold_solution`."""
    n = 0
    n_pass = 0
    for prob in problems[:max_problems]:
        if prob.gold_solution is None:
            continue
        # Gold solutions are function bodies that follow the prompt.
        res = grade(prob, prob.gold_solution)
        n += 1
        if res.passed:
            n_pass += 1
        else:
            print(f"  ✗ {prob.task_id}  err={res.error}")
    print(f"\nGold solution check: {n_pass}/{n} pass "
          f"({100 * n_pass / max(1, n):.1f}%)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(LOADERS), default="humaneval")
    p.add_argument("--max_problems", type=int, default=20)
    p.add_argument("--gold_check", action="store_true",
                   help="Run each problem's gold_solution through the grader. "
                        "Verifies the grader pipeline; should be ~100%.")
    args = p.parse_args()

    print(f"loading dataset: {args.dataset}")
    problems = LOADERS[args.dataset]()
    print(f"  loaded {len(problems)} problems")

    if args.gold_check:
        _self_test_with_gold(problems, args.max_problems)
        return 0

    print("nothing else to do without --gold_check (use it for grader smoke)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
