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
import ast
import dataclasses
import json
import multiprocessing as mp
import pathlib
import signal
import sys
import traceback
from contextlib import contextmanager
from typing import Callable

# Cap on the formatted error text fed back into a re-added (self-repair)
# prompt — long enough to diagnose, short enough not to bloat the prompt.
_ERROR_TEXT_CAP = 1000

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


# Dense reward ladder. Each tier is a qualitatively different failure
# mode; `score` is the GRPO reward in [0, 1]. The point is that a weak
# model gets a non-zero, *differentiated* signal long before it can
# fully solve a task — so GRPO groups have advantage variance to learn
# from. `partial` interpolates on the fraction of unit tests passed;
# the gap between partial-at-max (0.9) and pass (1.0) is the
# "fully-correct" boost.
_TIER_BASE_SCORE: dict[str, float] = {
    "syntax_error":  0.0,   # generated code doesn't compile
    "grader_error":  0.0,   # the dataset's own test block is broken (not the model's fault)
    "exec_error":    0.05,  # compiles, but exec'ing the solution raises / entry_point undefined
    "timeout":       0.05,  # ran but hit the wall-clock limit (e.g. infinite loop)
    "runtime_error": 0.2,   # check() ran but a setup statement crashed before/between asserts
    "partial":       0.2,   # + 0.7 * (n_passed / n_tests)
    "pass":          1.0,   # every assertion passed
}


def _compute_score(tier: str, n_tests: int, n_passed: int) -> float:
    if tier == "partial" and n_tests > 0:
        return 0.2 + 0.7 * (n_passed / n_tests)
    return _TIER_BASE_SCORE.get(tier, 0.0)


@dataclasses.dataclass
class GradingResult:
    """Dense grading outcome.

    `passed` is kept as a field for backward compatibility (eval scripts
    read it) but is fully determined by `tier == "pass"`. `score` is the
    dense reward — that's what the GRPO trainer should consume.

    `error_text` is the human-readable diagnosis — SyntaxError message +
    offending line, the exec traceback, the failed-assertion source
    lines, etc. It's what the self-repair loop (task #80) feeds back into
    a re-added prompt so the model learns to diagnose. None on a pass.
    """
    passed: bool               # True iff every assertion passed (tier == "pass")
    tier: str = "syntax_error"  # one of _TIER_BASE_SCORE keys
    score: float = 0.0         # dense reward in [0, 1]
    n_tests: int = 0           # total assertions found in check() (0 if not splittable)
    n_passed: int = 0          # assertions that passed
    error: str | None = None   # exception class name if it raised; "timeout" on timeout
    error_text: str | None = None  # formatted diagnosis for the self-repair loop
    elapsed_s: float = 0.0


def truncate_at_stop(text: str) -> str:
    """Trim generated text at first natural function-boundary."""
    earliest = len(text)
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx >= 0 and idx < earliest:
            earliest = idx
    return text[:earliest]


def _assert_detail(test: ast.expr, ns: dict) -> str:
    """Best-effort ``(got X, expected Y)`` suffix for a failed
    ``assert LHS == RHS``. Returns '' for any assert shape we can't cleanly
    introspect (so the caller falls back to a bare ``AssertionError``). The
    operands are re-eval'd in the test namespace — fully guarded, since a
    side-effecting / raising / slow operand must never break grading. Only
    plain ``==`` comparisons are enriched; everything else falls back."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)):
        return ""
    try:
        got = eval(ast.unparse(test.left), ns)            # noqa: re-runs call
        expected = eval(ast.unparse(test.comparators[0]), ns)
    except Exception:
        return ""

    def _short(v: object) -> str:
        try:
            s = repr(v)
        except Exception:
            return ""
        return s if len(s) <= 120 else s[:117] + "..."

    g, e = _short(got), _short(expected)
    if not g and not e:
        return ""
    return f" (got {g}, expected {e})"


def _run_check_granular(check_fn: ast.FunctionDef, ns: dict, candidate):
    """Run a `check(candidate)` body statement-by-statement so a single
    failing assert doesn't mask the rest. Returns (tier, error, n_tests,
    n_passed, error_text).

    - `assert` statements are counted individually (pass/fail); a failed
      one does NOT stop the run. The source line of each failed assert
      (plus the exception, if it wasn't a plain AssertionError) is
      collected into `error_text`.
    - non-assert statements are setup (assignments, loops, imports); if
      one raises, the test harness itself broke past that point ->
      `runtime_error`, with the statement source + traceback.
    """
    local_ns = dict(ns)
    local_ns["candidate"] = candidate
    n_tests = 0
    n_passed = 0
    failures: list[str] = []  # "  <assert src>  ->  <exc>"
    for stmt in check_fn.body:
        src = ast.unparse(stmt)
        if isinstance(stmt, ast.Assert):
            n_tests += 1
            try:
                exec(src, local_ns)
                n_passed += 1
            except TimeoutError:
                raise
            except AssertionError:
                failures.append(
                    f"  {src}  ->  AssertionError"
                    f"{_assert_detail(stmt.test, local_ns)}")
            except Exception as exc:
                failures.append(f"  {src}  ->  {type(exc).__name__}: {exc}")
        else:
            try:
                exec(src, local_ns)
            except TimeoutError:
                raise
            except Exception as exc:
                tb = traceback.format_exc(limit=3)
                txt = (f"test-harness setup statement crashed:\n"
                       f"  {src}\n{tb}")[:_ERROR_TEXT_CAP]
                return ("runtime_error", type(exc).__name__,
                        n_tests, n_passed, txt)
    if n_tests == 0:
        # check() had no assert statements at all — nothing to grade on.
        return ("grader_error", "no-asserts", 0, 0,
                "check() contained no assert statements")
    if n_passed == n_tests:
        return ("pass", None, n_tests, n_passed, None)
    # partial: format the failed assertions for the self-repair prompt.
    n_fail = n_tests - n_passed
    lines = [f"{n_fail}/{n_tests} assertions failed:"]
    lines.extend(failures[:8])
    if len(failures) > 8:
        lines.append(f"  ... and {len(failures) - 8} more")
    txt = "\n".join(lines)[:_ERROR_TEXT_CAP]
    return ("partial", "assertion-failed", n_tests, n_passed, txt)


def _exec_target(code: str, test: str, entry_point: str, q: mp.Queue):
    """Subprocess body: compile + exec the solution, exec the test block,
    then run check() granularly. Puts
    (tier, error, n_tests, n_passed, error_text)."""
    try:
        with _time_limit(5):
            # 1) Does the generated code even compile?
            try:
                compiled = compile(code, "<solution>", "exec")
            except SyntaxError as exc:
                txt = f"SyntaxError: {exc.msg}"
                if exc.lineno is not None:
                    txt += f" (line {exc.lineno})"
                if exc.text:
                    txt += f"\n  {exc.text.rstrip()}"
                q.put(("syntax_error", type(exc).__name__, 0, 0,
                       txt[:_ERROR_TEXT_CAP]))
                return
            # 2) Does exec'ing the solution itself raise?
            ns: dict = {}
            try:
                exec(compiled, ns)
            except Exception as exc:
                q.put(("exec_error", type(exc).__name__, 0, 0,
                       traceback.format_exc(limit=4)[:_ERROR_TEXT_CAP]))
                return
            # Resolve entry_point. For simple names (HumanEval/MBPP:
            # "two_sum") it's a dict lookup. For expressions (LeetCode:
            # "Solution().twoSum") we eval it so the Solution class can
            # be instantiated and the bound method extracted.
            try:
                entry_callable = eval(entry_point, ns)
            except Exception as exc:
                q.put(("exec_error", "entry-point-undefined", 0, 0,
                       f"could not resolve entry_point `{entry_point}`: "
                       f"{type(exc).__name__}: {exc}"))
                return
            # 3) Exec the dataset's test block to define check(). A failure
            # here is the dataset's fault, not the model's.
            try:
                exec(test, ns)
                check_tree = ast.parse(test)
            except Exception as exc:
                q.put(("grader_error", type(exc).__name__, 0, 0,
                       f"dataset test block failed to load: "
                       f"{type(exc).__name__}: {exc}"))
                return
            check_fn = next(
                (n for n in ast.walk(check_tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "check"),
                None,
            )
            if check_fn is None:
                q.put(("grader_error", "no-check-fn", 0, 0,
                       "dataset test block defined no check() function"))
                return
            # 4) Granular run.
            q.put(_run_check_granular(check_fn, ns, entry_callable))
    except TimeoutError:
        q.put(("timeout", "timeout", 0, 0,
               "execution exceeded the time limit (possible infinite loop)"))
    except Exception as exc:
        q.put(("grader_error", type(exc).__name__, 0, 0,
               f"grader internal error: {type(exc).__name__}: {exc}"))


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

    import queue as _queue

    q = mp.Queue()
    p = mp.Process(target=_exec_target, args=(full_code, problem.tests,
                                              problem.entry_point, q))
    t0 = time.perf_counter()
    p.start()
    # Read the result with a BOUNDED wait, then guarantee the child is reaped
    # without ever blocking on an unbounded join. A subprocess that ignores
    # SIGALRM *and* SIGTERM (e.g. a child forked from a CUDA-initialised parent,
    # or one stuck in a C extension) previously hung the unbounded p.join()
    # here for hours — the wall-clock guard alone is not enough. q.get(timeout)
    # caps the wait; SIGKILL (p.kill) cannot be ignored, so cleanup is bounded.
    result = None
    try:
        result = q.get(timeout=timeout_s)
    except _queue.Empty:
        result = None
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
        if p.is_alive():
            try:
                p.kill()            # SIGKILL — cannot be caught/ignored
            except Exception:
                pass
            p.join(timeout=2)
        q.cancel_join_thread()      # don't block on the queue feeder thread
    if result is None:
        return GradingResult(passed=False, tier="timeout", score=0.05,
                             error="timeout",
                             error_text="execution exceeded the wall-clock "
                                        "limit (possible infinite loop) or "
                                        "produced no result",
                             elapsed_s=time.perf_counter() - t0)
    tier, err, n_tests, n_passed, error_text = result
    return GradingResult(
        passed=(tier == "pass"), tier=tier,
        score=_compute_score(tier, n_tests, n_passed),
        n_tests=n_tests, n_passed=n_passed, error=err,
        error_text=error_text,
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


def _mbpp_problem_from_row(x: dict, source_tag: str) -> Problem:
    """Build a Problem from an MBPP-style row (works for mbpp + mbppplus)."""
    asserts = "\n    ".join(x["test_list"])
    test_block = f"def check(candidate):\n    {asserts}\n"
    first = x["test_list"][0]
    try:
        fn_name = first.split("assert", 1)[1].split("(", 1)[0].strip()
    except IndexError:
        fn_name = "candidate"
    prompt = f"{x['text'] if 'text' in x else x['prompt']}\n"
    return Problem(
        task_id=f"{source_tag}/{x['task_id']}",
        prompt=prompt,
        tests=test_block,
        entry_point=fn_name,
        prompt_is_code=False,
        gold_solution=x.get("code"),
    )


def load_mbpp(splits: tuple[str, ...] = ("train",)) -> list[Problem]:
    """MBPP. Default: 374-problem train split (backwards-compat).
    Pass splits=("train","validation","test","prompt") for the full 974.
    """
    from datasets import load_dataset
    out: list[Problem] = []
    for split in splits:
        ds = load_dataset("mbpp", split=split)
        for x in ds:
            out.append(_mbpp_problem_from_row(x, source_tag=f"mbpp_{split}"))
    return out


def load_mbpp_all() -> list[Problem]:
    """All 974 MBPP problems (train+validation+test+prompt). Use as the
    expanded RL training set — the test split was NEVER used by us for
    eval (we evaluate on HumanEval), so it's safe to train on.
    """
    return load_mbpp(splits=("train", "validation", "test", "prompt"))


def load_mbpp_plus() -> list[Problem]:
    """MBPP+ (evalplus/mbppplus): 378 MBPP problems with more rigorous
    test_list (the "+" augmentation). Disjoint task_id namespace from
    plain MBPP — fine to mix during training.
    """
    from datasets import load_dataset
    ds = load_dataset("evalplus/mbppplus", split="test")
    return [_mbpp_problem_from_row(x, source_tag="mbppplus") for x in ds]


def _legacy_load_mbpp() -> list[Problem]:
    """Kept for the body that used to live here, for reference only."""
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


def load_mbpp_combined() -> list[Problem]:
    """Expanded RL training set: all MBPP splits + MBPP+. ~1352 problems,
    3.6x the legacy load_mbpp. Disjoint task_id namespaces so duplicates
    in the underlying corpus don't matter for sampling.
    """
    return load_mbpp_all() + load_mbpp_plus()


def load_leetcode() -> list[Problem]:
    """LeetCode (newfacade/LeetCodeDataset): 2641 train problems with
    HumanEval-style `check(candidate)` test blocks. Entry-point format is
    `Solution().method_name` (LeetCode's class-based wrapper). Drop-in
    compatible with our existing grade() — the test block does the
    instantiation for us, the entry_point string is what `check(candidate)`
    receives as `candidate`.
    """
    from datasets import load_dataset
    ds = load_dataset("newfacade/LeetCodeDataset", split="train")
    out: list[Problem] = []
    for x in ds:
        # `prompt` already contains the imports + starter signature; we
        # use it as-is. Note: prompt_is_code=True because the prompt IS
        # code (imports + Solution class scaffold + signature), unlike
        # MBPP where the prompt is natural language.
        out.append(Problem(
            task_id=f"leetcode/{x['task_id']}",
            prompt=x["prompt"],
            tests=x["test"],
            entry_point=x["entry_point"],
            prompt_is_code=True,
            gold_solution=x.get("completion"),
        ))
    return out


def load_super_combined() -> list[Problem]:
    """Biggest pool: all MBPP splits + MBPP+ + LeetCode. ~3993 problems,
    10.7x the legacy load_mbpp.
    """
    return load_mbpp_combined() + load_leetcode()


def load_magicoder_oss_python(max_n: int | None = None) -> list[Problem]:
    """Magicoder-OSS-Instruct-75K, filtered to Python.

    Each row has (problem, solution) — a natural-language problem
    statement and the gold Python solution. There are NO executable
    tests, so these problems are for DISTILLATION ONLY (teacher
    generates solutions, student imitates) — they cannot be used as the
    RL-grader dataset. Returned with `tests=""` as a sentinel.

    ~22k Python problems after filtering.
    """
    from datasets import load_dataset
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    out: list[Problem] = []
    for x in ds:
        if x.get("lang") != "python":
            continue
        out.append(Problem(
            task_id=f"magicoder/{x['raw_index']}",
            prompt=x["problem"],
            tests="",                  # no tests → cannot grade
            entry_point="",
            prompt_is_code=False,
            gold_solution=x.get("solution"),
        ))
        if max_n is not None and len(out) >= max_n:
            break
    return out


def load_codefeedback_python(max_n: int | None = None) -> list[Problem]:
    """CodeFeedback-Filtered-Instruction, filtered to Python.

    Each row has (query, answer). Like Magicoder above, NO executable
    tests — distillation only.

    ~50k+ Python rows after filtering.
    """
    from datasets import load_dataset
    ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    out: list[Problem] = []
    for i, x in enumerate(ds):
        if x.get("lang") != "python":
            continue
        out.append(Problem(
            task_id=f"codefeedback/{i}",
            prompt=x["query"],
            tests="",
            entry_point="",
            prompt_is_code=False,
            gold_solution=x.get("answer"),
        ))
        if max_n is not None and len(out) >= max_n:
            break
    return out


_SYNTH_REASONING_DEFAULT_PATH = "data/synth_reasoning_tasks_small.jsonl"
_SYNTH_REASONING_TRAIN_PATH = "data/synth_reasoning_train.jsonl"
_SYNTH_REASONING_HELDOUT_PATH = "data/synth_reasoning_heldout.jsonl"


def load_synth_reasoning(path: str | None = None) -> list[Problem]:
    """Synthetic reasoning-required tasks (gen_synthetic_reasoning_tasks.py).

    Reads a JSONL produced by `experiments.gen_synthetic_reasoning_tasks`
    where each line is a serialised `Problem`. By default looks at
    `data/synth_reasoning_tasks_small.jsonl`; override via the
    `SYNTH_REASONING_PATH` env var or the `path` arg.

    For the larger train/held-out splits used by the stochastic-gate
    discovery RL experiment, use `load_synth_reasoning_train` /
    `load_synth_reasoning_heldout` (registered under
    `"synth_reasoning_train"` / `"synth_reasoning_heldout"` in LOADERS).
    """
    import os
    p = (path
         or os.environ.get("SYNTH_REASONING_PATH")
         or _SYNTH_REASONING_DEFAULT_PATH)
    path_obj = pathlib.Path(p)
    if not path_obj.is_absolute():
        path_obj = pathlib.Path(__file__).resolve().parents[1] / path_obj
    if not path_obj.exists():
        raise FileNotFoundError(
            f"synth_reasoning JSONL not found at {path_obj}. "
            f"Generate it with "
            f"`python experiments/gen_synthetic_reasoning_tasks.py "
            f"--out {p}`."
        )
    out: list[Problem] = []
    with open(path_obj) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(Problem(
                task_id=rec["task_id"],
                prompt=rec["prompt"],
                tests=rec["tests"],
                entry_point=rec["entry_point"],
                prompt_is_code=rec.get("prompt_is_code", False),
                gold_solution=rec.get("gold_solution"),
            ))
    return out


def load_synth_reasoning_train(path: str | None = None) -> list[Problem]:
    """Larger synth_reasoning train split (3000 tasks, seed=0).

    Used by the stochastic-gate discovery RL experiment as the
    RL-training pool. Same JSONL format as `load_synth_reasoning`;
    only the default path differs.
    """
    return load_synth_reasoning(path=path or _SYNTH_REASONING_TRAIN_PATH)


def load_synth_reasoning_heldout(path: str | None = None) -> list[Problem]:
    """Held-out synth_reasoning eval split (300 tasks, seed=1000).

    Disjoint-seed sibling of `load_synth_reasoning_train`. Used to
    measure generalisation of the stochastic-gate discovery RL run.
    """
    return load_synth_reasoning(path=path or _SYNTH_REASONING_HELDOUT_PATH)


def load_distill_corpus(
    max_magicoder: int | None = None,
    max_codefeedback: int | None = None,
) -> list[Problem]:
    """Full distillation problem pool: all gradeable problems +
    Magicoder + CodeFeedback (the latter two are distillation-only since
    they lack executable tests).
    """
    return (load_mbpp_combined()
            + load_leetcode()
            + load_magicoder_oss_python(max_n=max_magicoder)
            + load_codefeedback_python(max_n=max_codefeedback))


LOADERS: dict[str, Callable[[], list[Problem]]] = {
    "humaneval": load_humaneval,
    "mbpp": load_mbpp,            # legacy: train split only (374)
    "mbpp_all": load_mbpp_all,    # full 974
    "mbpp_plus": load_mbpp_plus,  # 378 EvalPlus
    "mbpp_combined": load_mbpp_combined,  # 1352
    "leetcode": load_leetcode,    # 2641 LeetCode
    "super_combined": load_super_combined,  # 3993 = mbpp_combined + leetcode
    # Distillation-only (no executable tests; cannot RL-train):
    "magicoder_oss": load_magicoder_oss_python,
    "codefeedback": load_codefeedback_python,
    "distill_corpus": load_distill_corpus,
    # Synthetic reasoning-required tasks (gen_synthetic_reasoning_tasks.py).
    "synth_reasoning": load_synth_reasoning,             # legacy small (504)
    "synth_reasoning_train": load_synth_reasoning_train,   # 3000 train pool
    "synth_reasoning_heldout": load_synth_reasoning_heldout,  # 300 held-out
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
            print(f"  ✗ {prob.task_id}  tier={res.tier} "
                  f"score={res.score:.2f} "
                  f"tests={res.n_passed}/{res.n_tests} err={res.error}")
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
