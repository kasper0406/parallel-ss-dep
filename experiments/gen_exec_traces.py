"""REAL Python execution traces as per-step latent supervision (2026-07-04).

EXEC_TRACE_LATENT_PLAN.md phase N1 ("the neural interpreter"). The bet: code
is the one domain where dense per-step ground truth is FREE and INFINITE —
execute the program and every intermediate interpreter state is a supervised
target for a latent-thinking step. This is the REAL-execution counterpart to
`experiments/gen_exec_trace.py` (singular — the OLD, pre-fair-control
generator): that script computes its `x` chain by re-implementing the update
rule in a parallel Python loop INSIDE the generator and never actually runs
the emitted program text, i.e. it is templated, not executed. This script
never hand-computes an answer: every `intermediates`/`answer` value here is
read off a REAL `sys.settrace` execution of the exact program text shown in
the prompt (synthetic) or of the exact MBPP gold solution + test-derived call
(mbpp), independently re-verified by `--validate`.

Sources (both produce ordered STATE-CHANGE EVENTS for one tracked variable —
the `intermediates` field is exactly that list, `answer` is its last entry):
  (a) synthetic-but-real Python: assignments / augmented ops / if-else on the
      tracked variable's own value / small for-loops with accumulators /
      table lookups, with distractor variables (scalar + list + dict
      mutations) interleaved among the tracked-variable updates (the
      `gen_state_track.py` interleaving pattern). Values are mod-10 by
      construction so every intermediate is a guaranteed single BPE token.
  (b) MBPP gold solutions run on a real test-list call: a heuristic picks the
      function's most-assigned LOCAL variable, traces its value through the
      real function call, and keeps the example only if every recorded value
      is a real int (non-bool) that encodes to a SINGLE token under the
      tokenizer (which for SmolLM2's digit-split BPE means exactly the
      one-digit non-negatives 0..9) and 2 <= K <= 12. Expected keep-rate is
      LOW (most MBPP locals are lists, strings, or multi-digit ints) —
      that's fine, this script reports it rather than templating around it.

Execution is NEVER in-process: every trace (synthetic batch or one MBPP
call) runs inside a forked `multiprocessing.Process`, hard-timed via
`signal.alarm` (mirrors `experiments/code_grader.py`'s subprocess pattern),
with an address-space rlimit sized RELATIVE to the parent's current VSZ (see
`_apply_resource_limits`'s docstring for the footgun this avoids: an
absolute cap breaks immediately once `transformers`/`datasets` are imported
in this same process, since a forked child inherits that VSZ at ~zero extra
cost via COW).

Schema (deliberately IDENTICAL to `gen_state_track.py` / `latent_arith_real.py`
/ `gen_exec_trace.py`'s output — a drop-in for `--latent_reasoning_train_prefix`
/ `latent_arith_real.py --per_hop`): task_id, prompt, tests, entry_point,
prompt_is_code, gold_solution, rung, answer, intermediates, horizon. Additive
fields not read by any consumer today, kept for `--validate` + provenance:
source, program_text, tracked_var, plus mbpp-only call_expr/is_param/mbpp_task_id
and synthetic-only setup_src/traced_src.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/gen_exec_traces.py \\
      --out_prefix data/exec_trace --rungs 2,3,4,5,6,7,8 \\
      --n_train 4000 --n_heldout 300
  PYTHONPATH=. .venv/bin/python experiments/gen_exec_traces.py --validate
  PYTHONPATH=. .venv/bin/python experiments/gen_exec_traces.py --smoke
"""
from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import pathlib
import queue as _queue
import random
import resource
import signal
import statistics
import sys
import time
import warnings
from contextlib import contextmanager

# Real MBPP gold solutions routinely contain unescaped backslashes in string
# literals (regex-flavoured docstrings etc.) — a genuine property of that
# dataset, not a bug here. `compile`/`ast.parse` on arbitrary real code
# raises SyntaxWarning for those; silence it so the mining log isn't noise.
warnings.filterwarnings("ignore", category=SyntaxWarning)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MOD = 10  # single-token digit range (0..9) for every synthetic tracked value
SOLVE_TAIL = ("\n\nWrite a Python function `solve()` that takes no arguments and "
              "returns the final value of {q}.\n")  # verbatim gen_state_track.py tail
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DISTRACTOR_VARS = ["a", "b", "c", "d", "e", "g", "h", "k"]  # excl. f/i/j/l (confusable), x (tracked)
_SENTINEL = object()


# ---------------------------------------------------------------------------
# Sandbox primitives — subprocess isolation, hard wall-clock timeout, memory
# bound. Same shape as experiments/code_grader.py's grade(): bounded q.get(),
# guaranteed reap (terminate -> join -> kill -> join), never an unbounded join.
# ---------------------------------------------------------------------------

def _apply_resource_limits(headroom_mb: int = 400, cpu_s: int = 5) -> None:
    """Bound the CHILD's address space to (VSZ at fork-time + headroom) and
    its CPU time, as a defense-in-depth backstop alongside the SIGALRM
    wall-clock guard below.

    An ABSOLUTE RLIMIT_AS (e.g. 256 MB) is a footgun in this script:
    `transformers` alone leaves a process at ~6.6 GB VSZ (mostly reserved
    shared-lib address space, not RSS) and `datasets` (used for MBPP
    mining) adds another ~2.7 GB. A forked child inherits that VSZ at
    essentially zero extra RSS cost (copy-on-write) — an absolute cap
    smaller than the PARENT's already-mapped VSZ makes every child violate
    the limit before it executes a single traced line, crashing 100% of
    calls the moment either library has been imported anywhere in this
    process. Sizing the cap RELATIVE to the current VSZ avoids that while
    still catching genuine runaway allocation inside the traced code
    (e.g. an accidental unbounded list-append loop).
    """
    try:
        vsz_kb = 0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    vsz_kb = int(line.split()[1])
                    break
        if vsz_kb:
            limit = vsz_kb * 1024 + headroom_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except Exception:
        pass


@contextmanager
def _time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("timed out")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _run_in_subprocess(target, args, timeout_s: float):
    """Run `target(*args, q)` in a fresh forked process; return whatever it
    `q.put(...)`, or None on timeout/crash. Never blocks unboundedly."""
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=target, args=(*args, q))
    p.start()
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
                p.kill()
            except Exception:
                pass
            p.join(timeout=2)
        q.cancel_join_thread()
    return result


# ---------------------------------------------------------------------------
# Generic per-frame state-change tracer (runs INSIDE a sandboxed child).
# ---------------------------------------------------------------------------

def _make_tracer(filename, funcname, tracked_var, events, state):
    """`state` = {'target': None, 'last': _SENTINEL}, mutated in place.

    Appends `frame.f_locals[tracked_var]` to `events` every time it differs
    from the last-seen value, restricted to exactly ONE frame: the first
    'call' whose code object matches (filename, funcname). `funcname=None`
    matches the module-level frame (synthetic-program path, where
    `frame.f_locals is frame.f_globals` and already carries anything an
    earlier untraced `exec` set up — verified empirically, see
    EXEC_TRACE_LATENT_PLAN.md session notes); a concrete name selects one
    function's OUTERMOST invocation (MBPP path) so nested/recursive calls
    into the same function don't pollute the trace. 'line' fires BEFORE the
    upcoming statement, i.e. it reflects the state AFTER every statement
    already executed in this frame — checking on 'line' (plus 'return', to
    catch a change made by the very last statement, which has no following
    'line' event) is therefore exactly "the value after each line that
    changed it".
    """
    def tracer(frame, event, arg):
        code = frame.f_code
        if event == "call":
            if (state["target"] is None and code.co_filename == filename
                    and (funcname is None or code.co_name == funcname)):
                state["target"] = frame
                state["last"] = frame.f_locals.get(tracked_var, _SENTINEL)
                return tracer
            return None
        if event in ("line", "return") and frame is state["target"]:
            v = frame.f_locals.get(tracked_var, _SENTINEL)
            if v is not _SENTINEL and v != state["last"]:
                events.append(v)
                state["last"] = v
        return tracer
    return tracer


# ---------------------------------------------------------------------------
# (a) Synthetic-but-real programs.
# ---------------------------------------------------------------------------

def _chunk_assign_const(rng):
    return ([f"x = {rng.randrange(MOD)}"], 1)


def _chunk_aug(rng, op):
    if op == "mul":
        d = rng.randint(2, MOD - 1)
        return ([f"x = (x * {d}) % {MOD}"], 1)
    d = rng.randint(1, MOD - 1)
    sign = "+" if op == "add" else "-"
    return ([f"x = (x {sign} {d}) % {MOD}"], 1)


def _chunk_cond(rng):
    thresh = rng.randint(1, MOD - 1)
    d1 = rng.randint(1, MOD - 1)
    d2 = rng.randint(1, MOD - 1)
    lines = [f"if x >= {thresh}:",
             f"    x = (x - {d1}) % {MOD}",
             "else:",
             f"    x = (x + {d2}) % {MOD}"]
    return (lines, 1)


def _chunk_lookup(rng, table_name):
    return ([f"x = {table_name}[x]"], 1)


def _chunk_loop(rng, remaining):
    length = min(remaining, rng.choice([2, 3]))
    d = rng.randint(1, MOD - 1)
    loopvar = f"_i{rng.randrange(10 ** 6)}"
    lines = [f"for {loopvar} in range({length}):",
             f"    x = (x + {d}) % {MOD}"]
    return (lines, length)


def _distractor_assign(rng, nm):
    return [f"{nm} = {rng.randrange(MOD)}"]


def _distractor_aug(rng, nm):
    d = rng.randint(1, MOD - 1)
    sign = rng.choice(["+", "-"])
    return [f"{nm} = ({nm} {sign} {d}) % {MOD}"]


def _distractor_list(rng, list_name, length):
    return [f"{list_name}[{rng.randrange(length)}] = {rng.randrange(MOD)}"]


def _distractor_dict(rng, dict_name, keys):
    return [f"{dict_name}['{rng.choice(keys)}'] = {rng.randrange(MOD)}"]


def _gen_synthetic_program(rng, K: int, n_distractor_vars: int = 3,
                           distractor_density: float = 2.5):
    """Build (setup_src, traced_src, tracked_var='x'). `setup_src` is exec'd
    UNTRACED (it's the initial state, excluded from `intermediates` — mirrors
    `gen_state_track.py` not counting the `c = 1` init line); `traced_src` is
    exec'd under `_make_tracer`, real execution producing exactly K events
    (asserted by the caller against the tracer's actual output, never assumed).
    """
    x0 = rng.randrange(MOD)
    distractor_names = rng.sample(DISTRACTOR_VARS, n_distractor_vars)
    table_vals = [rng.randrange(MOD) for _ in range(MOD)]
    list_len = 5
    list_vals = [rng.randrange(MOD) for _ in range(list_len)]
    dict_keys = ["p", "q", "r"]

    setup_lines = [f"x = {x0}"]
    setup_lines += [f"{nm} = {rng.randrange(MOD)}" for nm in distractor_names]
    setup_lines.append("tbl = [" + ", ".join(str(v) for v in table_vals) + "]")
    setup_lines.append("lst = [" + ", ".join(str(v) for v in list_vals) + "]")
    setup_lines.append("dct = {" + ", ".join(f"'{k}': {rng.randrange(MOD)}"
                                              for k in dict_keys) + "}")

    tracked_chunks = []
    remaining = K
    # "mul" is down-weighted: x*d mod 10 is a NO-OP (produces the same value,
    # so the tracer correctly records zero events for that line) whenever
    # x=0 for ANY d, and additionally at x=5 for any d coprime with 10.
    # ("assign" redrawing the current value ~10% of the time and "lookup"
    # hitting a table fixed point share the same coincidental-no-op risk at
    # lower rates — mul is just the worst offender.) Unavoidable without
    # knowing x at generation time, which would mean pre-computing the
    # trace instead of really executing it, defeating the point; the retry
    # loop in `_gen_synthetic_rung` re-checks the REAL executed event count
    # regardless of cause. Lower mul weight just reduces how often it has
    # to discard a program for landing short of K real events.
    kinds = ["lookup", "lookup", "assign", "add", "add", "sub", "sub",
             "mul", "cond", "cond", "loop", "loop"]
    guard = 0
    while remaining > 0:
        guard += 1
        if guard > 10 * (K + 5):
            raise RuntimeError("synthetic chunk-budgeting failed to terminate")
        kind = rng.choice(kinds)
        if kind == "loop":
            if remaining < 2:
                continue
            lines, n_ev = _chunk_loop(rng, remaining)
        elif kind == "assign":
            lines, n_ev = _chunk_assign_const(rng)
        elif kind in ("add", "sub", "mul"):
            lines, n_ev = _chunk_aug(rng, kind)
        elif kind == "cond":
            lines, n_ev = _chunk_cond(rng)
        else:
            lines, n_ev = _chunk_lookup(rng, "tbl")
        tracked_chunks.append(lines)
        remaining -= n_ev

    n_distractor_chunks = max(1, int(round(distractor_density * K)))
    distractor_chunks = []
    for _ in range(n_distractor_chunks):
        kind = rng.choice(["assign", "aug", "list", "dict"])
        nm = rng.choice(distractor_names)
        if kind == "assign":
            distractor_chunks.append(_distractor_assign(rng, nm))
        elif kind == "aug":
            distractor_chunks.append(_distractor_aug(rng, nm))
        elif kind == "list":
            distractor_chunks.append(_distractor_list(rng, "lst", list_len))
        else:
            distractor_chunks.append(_distractor_dict(rng, "dct", dict_keys))

    # Interleave: preserve tracked_chunks' relative order (the chain depends
    # on it), scatter distractor_chunks at random positions among them —
    # the gen_state_track.py interleaving pattern.
    total = tracked_chunks + distractor_chunks
    n_total = len(total)
    tracked_positions = set(rng.sample(range(n_total), len(tracked_chunks)))
    order = []
    ti = di = 0
    for i in range(n_total):
        if i in tracked_positions:
            order.append(tracked_chunks[ti]); ti += 1
        else:
            order.append(distractor_chunks[di]); di += 1
    traced_lines = [ln for chunk in order for ln in chunk]

    setup_src = "\n".join(setup_lines) + "\n"
    traced_src = "\n".join(traced_lines) + "\n"
    return setup_src, traced_src, "x"


def _synthetic_batch_worker(batch, q):
    """batch: list of (setup_src, traced_src, tracked_var). Puts a list of
    (status, events, err) tuples, one per item, in the SAME order.

    RLIMIT_CPU is scaled with the batch size: the limit applies to the
    WHOLE child process, and this worker runs up to `batch_size` programs
    per fork — a flat per-call cap (5s) could SIGXCPU-kill the process
    mid-batch before `q.put`, silently discarding the entire batch's
    results (review-caught granularity mismatch, 2026-07-04; the per-item
    wall-clock `_time_limit(2)` below is the real per-program guard)."""
    _apply_resource_limits(cpu_s=max(5, 2 * len(batch)))
    out = []
    for setup_src, traced_src, tracked_var in batch:
        try:
            with _time_limit(2):
                ns: dict = {}
                exec(compile(setup_src, "<setup>", "exec"), ns)
                events: list = []
                state = {"target": None, "last": _SENTINEL}
                tracer = _make_tracer("<traced>", None, tracked_var, events, state)
                sys.settrace(tracer)
                try:
                    exec(compile(traced_src, "<traced>", "exec"), ns)
                finally:
                    sys.settrace(None)
            out.append(("ok", events, None))
        except TimeoutError:
            out.append(("timeout", [], "timeout"))
        except Exception as exc:
            out.append(("error", [], f"{type(exc).__name__}: {exc}"))
    q.put(out)


def _gen_synthetic_rung(rng, K: int, n: int, batch_size: int = 200,
                        max_attempts_mult: int = 8):
    """Generate + REALLY EXECUTE synthetic programs at rung K until `n` of
    them land on exactly K real state-change events (or a bounded retry
    budget is exhausted). Returns (records, stats) where each record is
    (setup_src, traced_src, tracked_var, events).

    A generated program is DISCARDED (not "fixed up") whenever real
    execution produces something other than exactly K events — this
    happens whenever a mod-10 op coincidentally lands on a no-op (e.g.
    `x * d % 10` when x=0, or an `assign_const` redrawing the current
    value): the generator can't know x's value at construction time
    without hand-computing the trace, which would defeat the point of
    this script (real execution decides `intermediates`, never the
    generator). Retrying with fresh random parameters is the correct
    response to that, not templating around it — so this loops rather
    than accepting a shortfall, bounded by `max_attempts_mult * n` so a
    pathological K still fails loudly instead of hanging.
    """
    records = []
    n_ok = n_bad = 0
    attempted = 0
    max_attempts = max(n * max_attempts_mult, n + 200)
    while len(records) < n and attempted < max_attempts:
        this_n = min(batch_size, max_attempts - attempted, max(n - len(records), 1) * 2)
        specs = [_gen_synthetic_program(rng, K) for _ in range(this_n)]
        attempted += this_n
        result = _run_in_subprocess(_synthetic_batch_worker, (specs,),
                                    timeout_s=2 * len(specs) + 10)
        if result is None:
            n_bad += len(specs)
            continue
        for (setup_src, traced_src, var), (status, events, _err) in zip(specs, result):
            if status != "ok" or len(events) != K:
                n_bad += 1
                continue
            records.append((setup_src, traced_src, var, events))
            n_ok += 1
    return records[:n], dict(ok=n_ok, bad=n_bad, attempted=attempted,
                             shortfall=max(0, n - len(records)))


def _build_synthetic_record(setup_src, traced_src, tracked_var, events, K, idx):
    program_text = setup_src.rstrip("\n") + "\n" + traced_src.rstrip("\n")
    answer = events[-1]
    prompt = ("You are given the following Python program.\n"
              f"{program_text}\n\n"
              f"After running the program, what is the final value of {tracked_var}?"
              + SOLVE_TAIL.format(q=tracked_var))
    return {
        "task_id": f"exec_synth/n{K}/{idx}",
        "prompt": prompt,
        "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {answer}\n",
        "rung": K,
        "answer": answer,
        "intermediates": list(events),
        "horizon": len(program_text.splitlines()),
        "source": "synthetic",
        "program_text": program_text,
        "tracked_var": tracked_var,
        "setup_src": setup_src,
        "traced_src": traced_src,
    }


# ---------------------------------------------------------------------------
# (b) MBPP-derived real functions.
# ---------------------------------------------------------------------------

def _extract_call_exprs(test_src: str, fn_name: str) -> list:
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return []
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == fn_name):
            try:
                calls.append(ast.unparse(node))
            except Exception:
                continue
    return calls


def _find_function_def(tree, fn_name: str):
    """LAST matching def wins — mirrors runtime shadowing semantics (if a
    gold solution defines the same name twice, the second binding is the
    one the extracted test call actually invokes)."""
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            found = node
    return found


def _target_names(target) -> set:
    """All Name identifiers bound by an assignment/for/comprehension target
    (handles tuple/list destructuring and starred targets)."""
    out: set = set()
    for n in ast.walk(target):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _rank_candidate_vars(fn_node) -> list:
    """[(var_name, is_param, drop_first_event), ...] ranked by static
    assignment-count desc. Skips functions with nested def/class/lambda —
    ambiguous local scoping (a nested frame's locals would be a different
    `co_name`/frame anyway, but keeping the candidate-variable heuristic
    simple: don't offer names that might belong to an inner scope).

    TWO review-caught exclusions/refinements (2026-07-04, both verified
    empirically against CPython 3.12 tracer behaviour):

    1. COMPREHENSION NAME COLLISION (data corruption, not just noise):
       PEP 709 inlines list/set/dict comprehensions into the enclosing
       frame on 3.12+, so a comprehension's scoped iteration variable is
       visible through `frame.f_locals` UNDER THE OUTER VARIABLE'S NAME if
       the names collide — the tracer would record the comprehension's
       whole iteration as phantom "state changes" of the tracked variable
       (verified: `x=1; y=[x for x in range(5)]; return x` traces x as
       [1,0,1,2,3,4,1] even though real scoping keeps x==1 throughout,
       and the trace is DETERMINISTIC, so --validate reproduces rather
       than catches it). Generator expressions still get their own frame
       on 3.12 (verified clean) and need no exclusion. Fix: refuse any
       candidate whose name is also a ListComp/SetComp/DictComp target
       anywhere in the function.

    2. FOR-LOOP CONTROL VARIABLES must NOT get the "drop first event ==
       init" treatment (`drop_first_event=False`): their first-iteration
       binding is a genuine per-iteration value, not an init placeholder
       (verified: `for x in [10,20,30]` traces [10,20,30]; dropping 10
       would emit a silently-off-by-one K=2 record). The drop is only
       correct for the accumulator pattern (`total = 0` then updates),
       i.e. names whose bindings are all plain/aug assignments.
    """
    for n in ast.walk(fn_node):
        if n is fn_node:
            continue
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return []
    counts: dict = {}
    param_names = {a.arg for a in fn_node.args.args}
    for_targets: set = set()
    comp_targets: set = set()
    for node in ast.walk(fn_node):
        if node is fn_node:
            continue
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    counts[t.id] = counts.get(t.id, 0) + 1
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            counts[node.target.id] = counts.get(node.target.id, 0) + 1
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            counts[node.target.id] = counts.get(node.target.id, 0) + 1
            for_targets.add(node.target.id)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            for gen in node.generators:
                comp_targets |= _target_names(gen.target)
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(name, name in param_names,
             (name not in param_names) and (name not in for_targets))
            for name, _cnt in ranked if name not in comp_targets]


def _mbpp_trace_worker(code_src, call_expr, fn_name, tracked_var,
                       drop_first_event, q):
    try:
        _apply_resource_limits()
        with _time_limit(2):
            ns: dict = {}
            exec(compile(code_src, "<mbpp_solution>", "exec"), ns)
            events: list = []
            state = {"target": None, "last": _SENTINEL}
            tracer = _make_tracer("<mbpp_solution>", fn_name, tracked_var, events, state)
            sys.settrace(tracer)
            try:
                eval(compile(call_expr, "<call>", "eval"), ns)
            finally:
                sys.settrace(None)
        if drop_first_event and events:
            # First assignment of an accumulator-pattern local == its init,
            # not a hop. `drop_first_event` is decided STATICALLY by
            # _rank_candidate_vars: False for params (their initial binding
            # happens at 'call' time and is never recorded as an event
            # anyway) AND for for-loop control variables (whose first
            # binding is a genuine first-iteration value — review-caught
            # off-by-one, 2026-07-04).
            events = events[1:]
        q.put(("ok", events, None))
    except TimeoutError:
        q.put(("timeout", [], "timeout"))
    except Exception as exc:
        q.put(("error", [], f"{type(exc).__name__}: {exc}"))


def _mine_mbpp(problems, per_problem_max_vars: int = 2, call_timeout_s: float = 3.0):
    """One pass over `problems`; returns (by_k: dict[K] -> list[record-ish
    tuple], stats). Each mined item: (prob, gold_src, call_expr, var_name,
    is_param, events)."""
    by_k: dict = {}
    stats = dict(problems_tried=0, problems_yielding=0, examples=0,
                 fail_no_gold=0, fail_no_call=0, fail_no_candidates=0,
                 fail_exec=0, fail_bad_k=0, fail_not_int=0)
    for prob in problems:
        stats["problems_tried"] += 1
        gold = prob.gold_solution
        if not gold:
            stats["fail_no_gold"] += 1
            continue
        try:
            tree = ast.parse(gold)
        except SyntaxError:
            stats["fail_no_gold"] += 1
            continue
        fn_node = _find_function_def(tree, prob.entry_point)
        if fn_node is None:
            stats["fail_no_gold"] += 1
            continue
        calls = _extract_call_exprs(prob.tests, prob.entry_point)
        if not calls:
            stats["fail_no_call"] += 1
            continue
        candidates = _rank_candidate_vars(fn_node)
        if not candidates:
            stats["fail_no_candidates"] += 1
            continue
        call_expr = calls[0]
        got_one = False
        for var_name, is_param, drop_first in candidates[:per_problem_max_vars]:
            result = _run_in_subprocess(
                _mbpp_trace_worker,
                (gold, call_expr, prob.entry_point, var_name, drop_first),
                timeout_s=call_timeout_s)
            if result is None:
                stats["fail_exec"] += 1
                continue
            status, events, _err = result
            if status != "ok" or not events:
                stats["fail_exec"] += 1
                continue
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in events):
                stats["fail_not_int"] += 1
                continue
            K = len(events)
            if K < 2 or K > 12:
                stats["fail_bad_k"] += 1
                continue
            by_k.setdefault(K, []).append(
                (prob, gold, call_expr, var_name, is_param, drop_first, events))
            stats["examples"] += 1
            got_one = True
        if got_one:
            stats["problems_yielding"] += 1
    return by_k, stats


def _build_mbpp_record(prob, gold_src, call_expr, var_name, is_param,
                       drop_first_event, events, idx):
    K = len(events)
    answer = events[-1]
    prompt = ("You are given the following Python function.\n"
              f"{gold_src.rstrip()}\n\n"
              f"It is called as: {call_expr}\n\n"
              "After running the function body (just before it returns), what "
              f"is the final value of the local variable `{var_name}`?"
              + SOLVE_TAIL.format(q=var_name))
    return {
        "task_id": f"exec_mbpp/{prob.task_id}/{var_name}/{idx}",
        "prompt": prompt,
        "tests": f"def check(candidate):\n    assert candidate() == {answer}\n",
        "entry_point": "solve",
        "prompt_is_code": False,
        "gold_solution": f"def solve():\n    return {answer}\n",
        "rung": K,
        "answer": answer,
        "intermediates": list(events),
        "horizon": len(gold_src.strip().splitlines()),
        "source": "mbpp",
        "program_text": gold_src,
        "tracked_var": var_name,
        "call_expr": call_expr,
        "is_param": is_param,
        "drop_first_event": drop_first_event,
        "mbpp_task_id": prob.task_id,
        "mbpp_fn_name": prob.entry_point,
    }


# ---------------------------------------------------------------------------
# Tokenizer single-token filter. Loading `transformers` inflates this
# process's VSZ by several GB (COW-shared mappings), which every LATER fork
# (the per-rung synthetic batches in run_generate's loop) inherits —
# harmless only because _apply_resource_limits sizes RLIMIT_AS relative to
# the child's own fork-time VSZ rather than absolutely (see its docstring).
# ---------------------------------------------------------------------------

def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _is_single_token(tok, v) -> bool:
    return len(tok.encode(str(v), add_special_tokens=False)) == 1


def _passes_single_token(tok, rec) -> bool:
    return all(_is_single_token(tok, v) for v in list(rec["intermediates"]) + [rec["answer"]])


# ---------------------------------------------------------------------------
# Generation driver.
# ---------------------------------------------------------------------------

def _mbpp_pool_split(seed: int, heldout_frac: float = 0.15):
    from experiments.code_grader import load_mbpp_all
    problems = load_mbpp_all()
    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)
    n_heldout = int(round(len(shuffled) * heldout_frac))
    return shuffled[n_heldout:], shuffled[:n_heldout]  # (train_pool, heldout_pool)


def run_generate(args) -> None:
    t0 = time.time()
    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    out_path = pathlib.Path(args.out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mbpp_by_k_train: dict = {}
    mbpp_by_k_heldout: dict = {}
    mbpp_stats_train = mbpp_stats_heldout = None
    if not args.skip_mbpp:
        train_pool, heldout_pool = _mbpp_pool_split(args.seed, args.mbpp_heldout_frac)
        if args.mbpp_max_problems:
            train_pool = train_pool[:args.mbpp_max_problems]
            heldout_pool = heldout_pool[:max(1, args.mbpp_max_problems // 6)]
        print(f"[mbpp] pool split: train={len(train_pool)} heldout={len(heldout_pool)} "
              f"(disjoint problems)", flush=True)
        mbpp_by_k_train, mbpp_stats_train = _mine_mbpp(train_pool)
        mbpp_by_k_heldout, mbpp_stats_heldout = _mine_mbpp(heldout_pool)
        print(f"[mbpp] train-pool mining: {mbpp_stats_train}", flush=True)
        print(f"[mbpp] heldout-pool mining: {mbpp_stats_heldout}", flush=True)

    tok = _load_tokenizer()
    meta = {"rungs": {}, "mbpp_mining": {"train_pool": mbpp_stats_train,
                                          "heldout_pool": mbpp_stats_heldout},
            "seed": args.seed, "n_train_target": args.n_train,
            "n_heldout_target": args.n_heldout}

    for K in rungs:
        rung_meta = {}
        for split, target, mbpp_by_k, seed_offset in (
                ("train", args.n_train, mbpp_by_k_train, 0),
                ("heldout", args.n_heldout, mbpp_by_k_heldout, 99991)):
            # NOT `hash((K, split, args.seed))` (the gen_state_track.py
            # convention this generator otherwise mirrors): Python randomizes
            # str hashing per-process (PYTHONHASHSEED) unless pinned, so that
            # formula is silently NON-deterministic across runs of the same
            # --seed — caught empirically here (two back-to-back invocations
            # with identical --seed produced different synthetic programs).
            # Pure-integer arithmetic keeps this generator actually
            # deterministic given --seed, as required.
            rng = random.Random(args.seed * 1_000_003 + K * 1009 + seed_offset)

            mbpp_candidates = mbpp_by_k.get(K, [])
            mbpp_recs = []
            for i, (prob, gold, call_expr, var_name, is_param, drop_first,
                    events) in enumerate(mbpp_candidates):
                rec = _build_mbpp_record(prob, gold, call_expr, var_name,
                                         is_param, drop_first, events, i)
                if _passes_single_token(tok, rec):
                    mbpp_recs.append(rec)
            mbpp_recs = mbpp_recs[:target]

            n_synth_needed = max(0, target - len(mbpp_recs))
            synth_raw, synth_stats = _gen_synthetic_rung(rng, K, n_synth_needed)
            synth_recs = []
            seen_prompts = set()
            for i, (setup_src, traced_src, var, events) in enumerate(synth_raw):
                rec = _build_synthetic_record(setup_src, traced_src, var, events, K, i)
                if rec["prompt"] in seen_prompts:
                    continue
                if not _passes_single_token(tok, rec):
                    continue
                seen_prompts.add(rec["prompt"])
                synth_recs.append(rec)

            combined = mbpp_recs + synth_recs
            rng.shuffle(combined)
            out_file = f"{args.out_prefix}_{split}_n{K}.jsonl"
            with open(out_file, "w") as fh:
                for r in combined:
                    fh.write(json.dumps(r) + "\n")

            horizons = [r["horizon"] for r in combined] or [0]
            rung_meta[split] = dict(
                total=len(combined), mbpp=len(mbpp_recs), synthetic=len(synth_recs),
                mbpp_mined_at_K=len(mbpp_candidates),
                synthetic_ok=synth_stats["ok"], synthetic_bad=synth_stats["bad"],
                synthetic_attempted=synth_stats["attempted"],
                synthetic_shortfall=synth_stats["shortfall"],
                median_horizon=statistics.median(horizons))
            shortfall = target - len(combined)
            print(f"  K={K:>2} {split:>7}: total={len(combined):>5} "
                  f"(mbpp={len(mbpp_recs):>3} synth={len(synth_recs):>5}/"
                  f"{synth_stats['attempted']:>5} attempted) "
                  f"median_horizon={statistics.median(horizons):.0f}"
                  + (f"  [SHORT {shortfall} of target {target}]" if shortfall > 0 else ""),
                  flush=True)
        meta["rungs"][K] = rung_meta

    meta["elapsed_s"] = round(time.time() - t0, 1)
    with open(f"{args.out_prefix}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[done] {time.time() - t0:.0f}s. meta -> {args.out_prefix}_meta.json", flush=True)


# ---------------------------------------------------------------------------
# --validate: independent re-execution of a sample of EMITTED examples.
# ---------------------------------------------------------------------------

def _revalidate_one(rec) -> tuple:
    """Re-run rec's program in a FRESH sandbox; return (fresh_events, ok_exec)."""
    if rec["source"] == "synthetic":
        result = _run_in_subprocess(
            _synthetic_batch_worker,
            ([(rec["setup_src"], rec["traced_src"], rec["tracked_var"])],),
            timeout_s=5)
        if result is None:
            return None, False
        status, events, _err = result[0]
    else:
        result = _run_in_subprocess(
            _mbpp_trace_worker,
            (rec["program_text"], rec["call_expr"], rec["mbpp_fn_name"],
             rec["tracked_var"], rec["drop_first_event"]),
            timeout_s=5)
        if result is None:
            return None, False
        status, events, _err = result
    if status != "ok":
        return None, False
    return list(events), True


def run_validate(args) -> None:
    out_path = pathlib.Path(args.out_prefix)
    files = sorted(out_path.parent.glob(f"{out_path.name}_*_n*.jsonl"))
    if not files:
        raise SystemExit(f"no emitted files matching {args.out_prefix}_*_n*.jsonl "
                          f"— run generation first")
    all_recs = []
    for p in files:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                r["_file"] = p.name
                all_recs.append(r)
    rng = random.Random(args.seed + 777)
    sample = rng.sample(all_recs, min(100, len(all_recs)))

    tok = _load_tokenizer()
    n_exec_ok = n_match = n_single_token = 0
    mismatches = []
    for rec in sample:
        fresh_events, exec_ok = _revalidate_one(rec)
        if not exec_ok:
            mismatches.append((rec, "re-execution failed/timed out"))
            continue
        n_exec_ok += 1
        stored = list(rec["intermediates"])
        if fresh_events == stored and (fresh_events[-1] if fresh_events else None) == rec["answer"]:
            n_match += 1
        else:
            mismatches.append((rec, f"fresh={fresh_events} stored={stored} "
                                    f"answer={rec['answer']}"))
        if _passes_single_token(tok, rec):
            n_single_token += 1
        else:
            mismatches.append((rec, "single-token check FAILED"))

    print(f"\n=== VALIDATE ({len(sample)} sampled from {len(all_recs)} emitted) ===")
    print(f"  re-execution succeeded: {n_exec_ok}/{len(sample)}")
    print(f"  intermediates+answer match fresh execution: {n_match}/{len(sample)}")
    print(f"  single-token (tokenizer={TOKENIZER_NAME}): {n_single_token}/{len(sample)}")
    if mismatches:
        print(f"\n  {len(mismatches)} PROBLEM(S):")
        for rec, why in mismatches[:20]:
            print(f"    {rec['task_id']} ({rec['_file']}): {why}")

    # Per-rung counts + source mix from the meta file, if present.
    meta_path = f"{args.out_prefix}_meta.json"
    if pathlib.Path(meta_path).exists():
        meta = json.load(open(meta_path))
        print(f"\n=== PER-RUNG STATS (from {meta_path}) ===")
        for K, rm in sorted(meta.get("rungs", {}).items(), key=lambda kv: int(kv[0])):
            for split in ("train", "heldout"):
                s = rm.get(split, {})
                if s:
                    print(f"  K={K:>2} {split:>7}: total={s['total']:>5} "
                          f"mbpp={s['mbpp']:>4}/{s['mbpp_mined_at_K']:>4} mined  "
                          f"synth={s['synthetic']:>5}/{s['synthetic_attempted']:>5} attempted  "
                          f"median_horizon={s['median_horizon']:.0f}"
                          + (f"  [SHORT {s['synthetic_shortfall']}]"
                             if s.get("synthetic_shortfall") else ""))
        mm = meta.get("mbpp_mining", {})
        for pool in ("train_pool", "heldout_pool"):
            st = mm.get(pool)
            if st:
                rate = st["examples"] / max(1, st["problems_tried"])
                print(f"  mbpp {pool}: {st} (raw keep-rate {rate:.3f} examples/problem, "
                      f"pre single-token filter)")

    if n_exec_ok < len(sample) or n_match < n_exec_ok or n_single_token < len(sample):
        raise SystemExit(1)
    print("\nVALIDATE PASSED.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_prefix", default="data/exec_trace")
    ap.add_argument("--rungs", default="2,3,4,5,6,7,8")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_heldout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mbpp_max_problems", type=int, default=0,
                    help="cap on MBPP problems scanned per pool (0 = all ~974)")
    ap.add_argument("--mbpp_heldout_frac", type=float, default=0.15,
                    help="fraction of the MBPP problem pool reserved for heldout "
                         "(disjoint from train — different PROBLEMS, not just "
                         "different seeds)")
    ap.add_argument("--skip_mbpp", action="store_true",
                    help="synthetic-only (fast path; MBPP mining is the slow part)")
    ap.add_argument("--validate", action="store_true",
                    help="skip generation; re-execute a sample of already-emitted "
                         "examples in a FRESH sandbox and assert they still match "
                         "+ are single-token")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run (few rungs, small counts, capped MBPP)")
    args = ap.parse_args()
    if args.smoke:
        args.rungs = "2,3,4"
        args.n_train = 20
        args.n_heldout = 10
        args.mbpp_max_problems = 80
    if args.validate:
        run_validate(args)
        return
    run_generate(args)


if __name__ == "__main__":
    main()
