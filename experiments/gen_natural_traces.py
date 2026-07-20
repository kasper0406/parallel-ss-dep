"""Natural-code execution-trace corpus (SEARCH_NATIVE_PLAN_2026_07_19.md,
"REVIVAL ATTEMPT A", step 1).

Phase 3a killed the search-native program on a HOSTILE distribution: the
executor trained on synthetic mod-10 integer traces simulates its training
distribution, not Python (exact-match on natural FIXED candidates = 1.6%
pooled). The registered revival: teach the interpreter REAL Python by
`sys.settrace`-tracing actual MBPP-style functions (CWM did this at 32B; the
open, cheap question is whether a 402M model learns messy real code).

This script builds that natural-trace corpus. Source: the repair-corpus
TRAIN split (`data/repair_triples_train.jsonl`, `gen_repair_triples.py`) —
each triple's VERIFIED-PASSING `fix` program, called on the FIRST parseable
literal assert's arguments (`repair_value_probe.first_literal_assert`, the
exact assert-parsing the 3a gate uses). We line-trace `fname(*args)` and
record, per executed line INSIDE the target function, the (name, repr(value))
of every local that CHANGED on that line — then render it in the Stage-A
"# trace:" text format so the protocol transfers to the executor untouched.

Render format (byte-identical prefix to `repair_value_probe`'s eval prompt so
the trained format == the scored format), with the Stage-A `# step N:` /
`# final:` byte conventions read off `data/exec_trace_text_train.jsonl`:

    You are given the following Python program.
    <fix source>
    x = fname(<args>)

    After running the program, what is the final value of x?

    Write a Python function `solve()` that takes no arguments and returns the final value of x.
    # trace:
    # step 1: <name> = <literal>
    # step 2: <name> = <literal>
    ...
    # final: <repr(return value)>

The synthetic Stage-A steps all track a single `x`; the natural steps track
the function's INTERNAL locals and `# final:` is the return value bound to x
(the CWM-style "trace internals, then the answer" protocol). The eval gate
(`repair_value_probe`) parses only `# final:`, so the differing step-variable
names are intended, not a mismatch.

Caps (all CLI-overridable; a program is SKIPPED, never truncated, when it
exceeds one — "no truncation-ellipsis inside training data"):
  --max_events 40   max trace events / program
  --max_repr   48   max chars per recorded value repr AND per final repr
  --max_chars  4000 max rendered trace-text chars
  --timeout_s  2    per-run wall-clock execution timeout (subprocess sandbox)
  --min_events 1    skip trivial 0-event traces (no state change to teach)

Determinism / correctness guards:
  * every trace runs in a forked, RLIMIT-bounded, SIGALRM-timed subprocess
    (the `gen_exec_traces` / `code_grader` sandbox), TWICE — a program whose
    two consecutive traces differ is nondeterministic and dropped.
  * `# final:` == the ACTUAL return value by construction; we additionally
    check it equals the assert's expected literal and RECORD the match rate
    (expected-match ≈ 100% is expected, since the fix passed the full block —
    a low rate would signal a parsing/consistency bug).
  * PEP 709 (3.12) inlines list/set/dict comprehension iteration variables
    into the function frame; their names are statically excluded so a
    comprehension's throwaway loop var never becomes a phantom trace step
    (the `gen_exec_traces._rank_candidate_vars` precedent).

Contamination guard (MANDATORY, asserted before writing): ZERO problem_key
overlap with the 3a heldout triples (`data/repair_triples_heldout.jsonl`).
The train/heldout split is disjoint by construction; we assert it.

Output (fields matched to the Stage-A mix's exec-trace stream — `text_field:
text` — so `configs/pretrain_mix_stageA_natural.yaml` needs only a path swap):
  data/natural_traces_train.jsonl
  data/natural_traces_heldout.jsonl   (train-split holdback, ~1,500 examples,
                                        split by problem_key hash, deterministic)

Usage (CPU only, parallel workers):
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/gen_natural_traces.py --workers 16
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/gen_natural_traces.py --dry_scan --workers 16
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/gen_natural_traces.py --smoke
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import pathlib
import statistics
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", category=SyntaxWarning)

# Byte-reproducibility guard. str/frozenset hashing is PYTHONHASHSEED-randomized,
# so a set-of-STRINGS local (e.g. {'red', 'green', 'black'}) reprs in a
# different order each interpreter run — stable WITHIN one process (so the
# same-process double-run determinism check cannot see it) but not ACROSS runs.
# Pin the seed so the emitted corpus is byte-identical given --seed. Done here,
# BEFORE the heavy imports and ONLY when run as a script (never on import — so
# `import experiments.gen_natural_traces` from the tests is unaffected); the
# forked pool/sandbox children inherit the pinned seed. Re-exec is a one-time
# cost that also avoids a double heavy-import.
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# Sandbox primitives reused VERBATIM from the exec-trace generator (forked
# subprocess + relative RLIMIT_AS + SIGALRM wall-clock guard).
from experiments.gen_exec_traces import (  # noqa: E402
    _apply_resource_limits, _time_limit, _run_in_subprocess, _SENTINEL,
)
# Assert parsing + prompt rendering reused (NOT duplicated) from the 3a gate,
# so the trained format is byte-identical to the scored format.
from experiments.repair_value_probe import (  # noqa: E402
    first_literal_assert, build_tests_by_key, build_exec_program,
)
from experiments.value_function import render_exec_prompt  # noqa: E402

TRACKED_VAR = "x"
_OVERFLOW = object()  # marker in the last-seen snapshot for an over-cap value


# --------------------------------------------------------------------------- #
# Value rendering (pure). Bounded so a giant object never has its full repr
# materialised just to discover it is over-cap.
# --------------------------------------------------------------------------- #

def _bounded_repr(v, max_repr: int):
    """(repr_str_or_None, too_long, unreprable).

    Fast-path: a sized non-string container with more elements than `max_repr`
    cannot possibly repr within `max_repr` chars (each element is >=1 char plus
    separators) — reject it without building the (potentially huge) string.
    """
    try:
        if (hasattr(v, "__len__") and not isinstance(v, (str, bytes, bytearray))
                and len(v) > max_repr):
            return None, True, False
    except Exception:
        pass
    try:
        r = repr(v)
    except Exception:
        return None, False, True
    return r, (len(r) > max_repr), False


def _comprehension_targets(fix_code: str) -> set:
    """Names bound as list/set/dict comprehension targets. On 3.12 (PEP 709)
    list/set/dict comprehensions are inlined into the enclosing frame, so their
    iteration variable leaks into `frame.f_locals` — excluding those names keeps
    a comprehension's throwaway loop var from becoming a phantom trace step
    (mirrors `gen_exec_traces._rank_candidate_vars`).

    Generator expressions are DELIBERATELY excluded from this set: they still get
    their own frame on 3.12 (verified clean in gen_exec_traces), so their target
    never leaks — adding their names here would only DROP a genuine same-named
    local's events (review-caught, 2026-07-20). Residual (inherent, name-based):
    a list/set/dict comp target that collides with a real local also drops that
    local's events — lower incidence, same failure mode, accepted as
    conservative (the `# final:` return is unaffected)."""
    try:
        tree = ast.parse(fix_code)
    except SyntaxError:
        return set()
    out: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            for gen in node.generators:
                for n in ast.walk(gen.target):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
    return out


# --------------------------------------------------------------------------- #
# Multi-variable line tracer (extends gen_exec_traces._make_tracer from ONE
# tracked variable to "every local that changed on this line"). Restricted to
# the OUTERMOST frame of `funcname` in `filename`; nested/helper calls are not
# traced (matches the single-frame convention and keeps traces focused).
# --------------------------------------------------------------------------- #

def _make_multi_tracer(filename, funcname, events, state, exclude, max_events,
                       max_repr):
    """`state` = {'target', 'last', 'event_overflow', 'repr_overflow'}; `events`
    is appended (name, repr) tuples in `frame.f_locals` (insertion) order per
    line. The parameters bound at 'call' time are the initial state, snapshotted
    and excluded from events (mirrors gen_exec_traces not counting the setup)."""
    def tracer(frame, event, arg):
        code = frame.f_code
        if event == "call":
            if (state["target"] is None and code.co_filename == filename
                    and code.co_name == funcname):
                state["target"] = frame
                base = {}
                for k, v in frame.f_locals.items():
                    if k in exclude:
                        continue
                    r, too_long, bad = _bounded_repr(v, max_repr)
                    base[k] = _OVERFLOW if (too_long or bad) else r
                state["last"] = base
                return tracer
            return None
        if event in ("line", "return") and frame is state["target"]:
            if state["event_overflow"] or state["repr_overflow"]:
                return None  # stop tracing this frame; program will be skipped
            last = state["last"]
            for k in list(frame.f_locals.keys()):
                if k in exclude:
                    continue
                v = frame.f_locals[k]
                r, too_long, bad = _bounded_repr(v, max_repr)
                key = _OVERFLOW if (too_long or bad) else r
                if last.get(k, _SENTINEL) == key:
                    continue
                if too_long or bad:
                    state["repr_overflow"] = True
                    last[k] = key
                    return None
                events.append((k, r))
                last[k] = key
                if len(events) > max_events:
                    state["event_overflow"] = True
                    return None
        return tracer
    return tracer


def _trace_once(fix_code, call_src, fname, expected, exclude, max_events,
                max_repr):
    """Execute `fname(*args)` (via `eval(call_src)`) on `fix_code` once under
    the multi-tracer. Returns a dict; raises on compile/exec failure (the caller
    treats a raise as an invalid trace, never a silent pass)."""
    ns: dict = {}
    exec(compile(fix_code, "<fix>", "exec"), ns)
    events: list = []
    state = {"target": None, "last": {}, "event_overflow": False,
             "repr_overflow": False}
    tracer = _make_multi_tracer("<fix>", fname, events, state, exclude,
                                max_events, max_repr)
    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        ret = eval(compile(call_src, "<call>", "eval"), ns)
    finally:
        sys.settrace(old)
    ret_repr, ret_too_long, ret_bad = _bounded_repr(ret, max_repr)
    try:
        match = bool(ret == expected) and type(ret) is type(expected)
    except Exception:
        match = False
    return {
        "events": events,
        "ret_repr": ret_repr,
        "ret_too_long": ret_too_long,
        "ret_bad": ret_bad,
        "event_overflow": state["event_overflow"],
        "repr_overflow": state["repr_overflow"],
        "match": match,
    }


def _trace_batch_worker(specs, max_events, max_repr, per_timeout, q):
    """Trace a chunk of specs in ONE forked child (per-spec SIGALRM). Puts a
    list of (status, run1, run2) in the SAME order. RLIMIT_CPU scales with the
    chunk size so a per-spec cap can't SIGXCPU-kill the whole batch mid-run
    (the gen_exec_traces._synthetic_batch_worker granularity fix)."""
    _apply_resource_limits(cpu_s=max(5, (2 * per_timeout + 1) * len(specs)))
    out = []
    for sp in specs:
        try:
            with _time_limit(per_timeout):
                r1 = _trace_once(sp["fix"], sp["call_src"], sp["fname"],
                                 sp["expected"], sp["exclude"], max_events,
                                 max_repr)
            with _time_limit(per_timeout):
                r2 = _trace_once(sp["fix"], sp["call_src"], sp["fname"],
                                 sp["expected"], sp["exclude"], max_events,
                                 max_repr)
            out.append(("ok", r1, r2))
        except TimeoutError:
            out.append(("timeout", None, None))
        except Exception as exc:  # noqa: BLE001 - any exec failure is a skip
            out.append(("error", f"{type(exc).__name__}: {exc}", None))
    q.put(out)


# --------------------------------------------------------------------------- #
# Rendering (pure).
# --------------------------------------------------------------------------- #

def render_trace(fix_code: str, call_src: str, events, ret_repr: str) -> str:
    """The Stage-A "# trace:" text document. The header (program + call binding
    + question framing + "# trace:") is byte-identical to `repair_value_probe`'s
    eval prompt (`build_exec_program` + `render_exec_prompt` + "\\n# trace:\\n");
    the continuation is one "# step N: <name> = <literal>" per event, then
    "# final: <repr(return)>" — the Stage-A byte conventions from
    data/exec_trace_text_train.jsonl (numbered steps, labelled final)."""
    prog = build_exec_program(fix_code, call_src, TRACKED_VAR)
    header = render_exec_prompt(prog, TRACKED_VAR).rstrip() + "\n# trace:\n"
    steps = "".join(f"# step {j}: {name} = {val}\n"
                    for j, (name, val) in enumerate(events, 1))
    return header + steps + f"# final: {ret_repr}\n"


# --------------------------------------------------------------------------- #
# Parent-side assert resolution (pure; no execution).
# --------------------------------------------------------------------------- #

def load_triples(path: str) -> list:
    return [json.loads(l) for l in pathlib.Path(path).open() if l.strip()]


def load_problem_keys(path: str) -> set:
    p = pathlib.Path(path)
    if not p.exists():
        return set()
    return {json.loads(l)["problem_key"] for l in p.open() if l.strip()}


def assert_no_contamination(used_keys, heldout_keys) -> None:
    """Hard guard: the natural-trace corpus must not touch any problem the 3a
    gate scores. Raises AssertionError (fails generation) on any overlap."""
    overlap = set(used_keys) & set(heldout_keys)
    assert not overlap, (
        f"CONTAMINATION: {len(overlap)} problem_key(s) overlap with the 3a "
        f"heldout ({sorted(overlap)[:10]}...); the natural-trace corpus must be "
        f"disjoint from data/repair_triples_heldout.jsonl")


def _assert_unique_task_ids(specs) -> None:
    """Traces are stored/read back keyed by task_id; a duplicate would misassign
    one spec's trace to another spec's program (silent corpus corruption)."""
    ids = [s["task_id"] for s in specs]
    dupes = [k for k, c in collections.Counter(ids).items() if c > 1]
    assert not dupes, (f"{len(dupes)} duplicate task_id(s) would misassign "
                       f"traces: {dupes[:5]}")


def resolve_specs(triples, tests_by_key):
    """Turn each triple into a trace spec, or count a skip. Deterministic, pure.
    Returns (specs, skips): a spec carries everything the worker needs plus the
    parent-side bookkeeping (problem_key, task_id, provenance, expected_repr)."""
    specs, skips = [], collections.Counter()
    for rec in triples:
        fix = (rec.get("fix") or "").strip()
        if not fix:
            skips["empty_fix"] += 1
            continue
        tests = tests_by_key.get(rec["problem_key"])
        if tests is None:
            skips["no_problem"] += 1
            continue
        parsed = first_literal_assert(tests)
        if parsed is None:
            skips["no_literal_assert"] += 1
            continue
        fname, args, expected, call_src, _assert_src = parsed
        specs.append({
            "task_id": rec["task_id"],
            "problem_key": rec["problem_key"],
            "provenance": rec.get("provenance"),
            "fix": fix,
            "fname": fname,
            "call_src": call_src,
            "expected": expected,
            "expected_repr": repr(expected),
            "exclude": _comprehension_targets(fix),
        })
    return specs, skips


# --------------------------------------------------------------------------- #
# Record construction from a completed (run1, run2) trace pair.
# --------------------------------------------------------------------------- #

def build_record(sp, status, run1, run2, max_events, max_repr, max_chars):
    """Return (record | None, skip_reason | None). `record` has the mix's
    `text` field plus provenance/diagnostics."""
    if status != "ok":
        return None, ("timeout" if status == "timeout" else "exec_error")
    # determinism: two consecutive traces must agree exactly.
    if (run1["events"] != run2["events"] or run1["ret_repr"] != run2["ret_repr"]
            or run1["event_overflow"] != run2["event_overflow"]
            or run1["repr_overflow"] != run2["repr_overflow"]
            or run1["match"] != run2["match"]):
        return None, "nondeterministic"
    if run1["event_overflow"]:
        return None, "too_many_events"
    if run1["repr_overflow"] or run1["ret_too_long"] or run1["ret_bad"]:
        return None, "repr_too_long"
    events = run1["events"]
    if len(events) < 1:  # min_events applied by caller via this floor
        return None, "no_events"
    ret_repr = run1["ret_repr"]
    text = render_trace(sp["fix"], sp["call_src"], events, ret_repr)
    if len(text) > max_chars:
        return None, "trace_too_long"
    rec = {
        "text": text,
        "problem_key": sp["problem_key"],
        "task_id": f"natural_trace/{sp['task_id']}",
        "source_task_id": sp["task_id"],
        "provenance": sp["provenance"],
        "fname": sp["fname"],
        "call_src": sp["call_src"],
        "n_events": len(events),
        "n_chars": len(text),
        "final_repr": ret_repr,
        "expected_repr": sp["expected_repr"],
        "expected_match": bool(run1["match"]),
    }
    return rec, None


# --------------------------------------------------------------------------- #
# Deterministic problem-hash holdback split.
# --------------------------------------------------------------------------- #

def _pk_score(problem_key: str, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{problem_key}".encode()).hexdigest()[:8], 16)


def split_heldout(records, heldout_n: int, seed: int):
    """Split by problem_key hash so all traces of a problem land on ONE side
    (disjoint by problem). Greedily add problems in ascending hash order until
    the holdback reaches `heldout_n` EXAMPLES. Deterministic; returns
    (train, heldout, heldout_pks)."""
    by_pk = collections.defaultdict(list)
    for r in records:
        by_pk[r["problem_key"]].append(r)
    ranked = sorted(by_pk, key=lambda pk: (_pk_score(pk, seed), pk))
    heldout_pks, count = set(), 0
    for pk in ranked:
        if count >= heldout_n:
            break
        heldout_pks.add(pk)
        count += len(by_pk[pk])
    train = [r for r in records if r["problem_key"] not in heldout_pks]
    heldout = [r for r in records if r["problem_key"] in heldout_pks]
    return train, heldout, heldout_pks


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def run(args) -> None:
    t0 = time.time()
    triples = load_triples(args.triples)
    if args.limit:
        triples = triples[:args.limit]
    # The MANDATORY contamination guard must not be able to pass vacuously: a
    # missing/misnamed/empty reference file would leave heldout_keys empty and
    # make assert_no_contamination a no-op. Fail loudly instead (review-caught).
    if not pathlib.Path(args.contam_check).exists():
        raise SystemExit(f"[FATAL] --contam_check {args.contam_check} not found; "
                         f"the MANDATORY contamination guard cannot run")
    heldout_keys = load_problem_keys(args.contam_check)
    if not heldout_keys:
        raise SystemExit(f"[FATAL] --contam_check {args.contam_check} yielded 0 "
                         f"problem_keys; the contamination guard would pass "
                         f"vacuously — refusing to proceed")
    print(f"[load] {len(triples)} train triples <- {args.triples}", flush=True)
    print(f"[load] {len(heldout_keys)} 3a-heldout problem_keys <- "
          f"{args.contam_check} (contamination reference)", flush=True)

    tests_by_key = build_tests_by_key(args.dataset)
    print(f"[load] {len(tests_by_key)} test blocks <- LOADERS[{args.dataset}]",
          flush=True)

    specs, resolve_skips = resolve_specs(triples, tests_by_key)
    # Deterministic worker order.
    specs.sort(key=lambda s: s["task_id"])
    # Traces are keyed by task_id (results[...] read-back builds each record);
    # a duplicate would render one spec's trace against another's program.
    _assert_unique_task_ids(specs)
    print(f"[resolve] {len(specs)} specs with a literal assert  "
          f"skips={dict(resolve_skips)}", flush=True)

    if args.dry_scan:
        by_prov = collections.Counter(s["provenance"] for s in specs)
        n_pk = len({s["problem_key"] for s in specs})
        n_fix = len({(s["problem_key"], s["fix"]) for s in specs})
        print(f"[dry_scan] resolvable={len(specs)} problems={n_pk} "
              f"unique(problem,fix)={n_fix} provenance={dict(by_prov)}")
        print(f"[dry_scan] done {time.time()-t0:.1f}s")
        return

    # Trace every spec (chunked; one forked child per chunk, per-spec SIGALRM).
    results: dict = {}  # task_id -> (status, run1, run2)
    chunks = list(_chunks(specs, args.chunk_size))
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for ch in chunks:
            fut = ex.submit(_trace_chunk, ch, args.max_events, args.max_repr,
                            args.timeout_s)
            futs[fut] = ch
        for fut in as_completed(futs):
            ch = futs[fut]
            for sp, res in zip(ch, fut.result()):
                results[sp["task_id"]] = res
            done += 1
            if done % 50 == 0 or done == len(chunks):
                print(f"  [trace] {done}/{len(chunks)} chunks "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # Build records in the deterministic spec order.
    records, skips = [], collections.Counter(resolve_skips)
    n_expected_match = 0
    for sp in specs:
        status, run1, run2 = results[sp["task_id"]]
        rec, reason = build_record(sp, status, run1, run2, args.max_events,
                                   args.max_repr, args.max_chars)
        if rec is None:
            skips[reason] += 1
            continue
        if rec["n_events"] < args.min_events:
            skips["too_few_events"] += 1
            continue
        records.append(rec)
        n_expected_match += int(rec["expected_match"])

    expected_match_rate = (n_expected_match / len(records)) if records else 0.0
    print(f"[build] {len(records)} traces kept  skips={dict(skips)}", flush=True)
    print(f"[build] expected-match rate {expected_match_rate:.4f} "
          f"({n_expected_match}/{len(records)})", flush=True)

    # ---- MANDATORY contamination guard (assert before writing anything). ----
    used_keys = {r["problem_key"] for r in records}
    assert_no_contamination(used_keys, heldout_keys)
    print(f"[guard] contamination check PASSED: 0 overlap with "
          f"{args.contam_check} (used {len(used_keys)} problem_keys)", flush=True)

    # ---- Deterministic holdback split. ----
    train, heldout, heldout_pks = split_heldout(records, args.heldout_n,
                                                 args.seed)
    assert not ({r["problem_key"] for r in train}
                & {r["problem_key"] for r in heldout}), "split not disjoint"
    print(f"[split] train={len(train)} heldout={len(heldout)} "
          f"(heldout spans {len(heldout_pks)} problems; target {args.heldout_n})",
          flush=True)

    _write_jsonl(args.out_train, train)
    _write_jsonl(args.out_heldout, heldout)

    _report(records, train, heldout, skips, resolve_skips,
            expected_match_rate, len(triples), time.time() - t0)


def _trace_chunk(chunk, max_events, max_repr, timeout_s):
    """ProcessPool job: trace one chunk inside a forked sandbox child. On a hard
    child crash (result None) fall back to per-spec isolation so one pathological
    spec costs a single skip, not the whole chunk."""
    res = _run_in_subprocess(_trace_batch_worker,
                             (chunk, max_events, max_repr, timeout_s),
                             timeout_s=(2 * timeout_s + 5) * len(chunk) + 10)
    if res is not None and len(res) == len(chunk):
        return res
    out = []
    for sp in chunk:
        one = _run_in_subprocess(_trace_batch_worker,
                                 ([sp], max_events, max_repr, timeout_s),
                                 timeout_s=2 * timeout_s + 10)
        out.append(one[0] if one else ("error", "worker_crash", None))
    return out


def _write_jsonl(path, records):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[write] {path}: {len(records)} records", flush=True)


def _report(records, train, heldout, skips, resolve_skips, match_rate,
            n_triples, elapsed):
    n_events = sorted(r["n_events"] for r in records) or [0]
    n_chars = sorted(r["n_chars"] for r in records) or [0]
    n_unique_text = len({r["text"] for r in records})

    def pct(sorted_vals, p):
        return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]

    print("\n=== NATURAL-TRACE CORPUS REPORT ===")
    print(f"  input triples            {n_triples}")
    print(f"  resolve skips            {dict(resolve_skips)}")
    print(f"  trace/cap skips          "
          f"{ {k: v for k, v in skips.items() if k not in resolve_skips} }")
    print(f"  kept traces              {len(records)}")
    print(f"  unique rendered texts    {n_unique_text}  "
          f"(multiplicity {len(records)/max(1,n_unique_text):.1f}x)")
    print(f"  expected-match rate      {match_rate:.4f}")
    print(f"  train / heldout          {len(train)} / {len(heldout)}")
    print(f"  events  p50/p90/max      {pct(n_events,.5)}/{pct(n_events,.9)}/"
          f"{n_events[-1]}")
    print(f"  chars   p50/p90/max      {pct(n_chars,.5)}/{pct(n_chars,.9)}/"
          f"{n_chars[-1]}")
    print(f"  elapsed                  {elapsed:.0f}s")


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triples", default="data/repair_triples_train.jsonl")
    ap.add_argument("--contam_check", default="data/repair_triples_heldout.jsonl",
                    help="3a heldout triples; corpus must be problem-disjoint")
    ap.add_argument("--dataset", default="mbpp_combined",
                    help="code_grader LOADERS key for the test blocks (must "
                         "match gen_repair_triples' --dataset)")
    ap.add_argument("--out_train", default="data/natural_traces_train.jsonl")
    ap.add_argument("--out_heldout", default="data/natural_traces_heldout.jsonl")
    ap.add_argument("--heldout_n", type=int, default=1500,
                    help="target #examples in the holdback (problem-hash split; "
                         "stops at the problem boundary that reaches this)")
    ap.add_argument("--max_events", type=int, default=40)
    ap.add_argument("--max_repr", type=int, default=48)
    ap.add_argument("--max_chars", type=int, default=4000)
    ap.add_argument("--min_events", type=int, default=1)
    ap.add_argument("--timeout_s", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: only the first N triples")
    ap.add_argument("--dry_scan", action="store_true",
                    help="resolve asserts + count only (no execution)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run (200 triples, small holdback)")
    return ap


def main():
    args = build_parser().parse_args()
    if args.smoke:
        args.limit = 200
        args.heldout_n = 20
        args.out_train = "data/natural_traces_smoke_train.jsonl"
        args.out_heldout = "data/natural_traces_smoke_heldout.jsonl"
    run(args)


if __name__ == "__main__":
    main()
