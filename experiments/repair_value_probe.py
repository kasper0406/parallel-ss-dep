"""Phase 3a natural-code transfer probe for the executor value function
(SEARCH_NATIVE_PLAN_2026_07_19.md, "PHASE 3 GATES" -> 3a).

The pre-registered gate (frozen bar): does an EXECUTOR-derived score rank a
repair triple's FIXED program above its BUGGY program better than a mean
log-prob baseline does, on a HOSTILE (natural-code) distribution?

    **Bar: executor - logprob >= +10pp two-way ranking accuracy**
    (ties-as-failures headline column).

This is the transfer counterpart to Phase 2 (`value_function.py`, +73pp on the
executor's synthetic home distribution). The CRUXEval mechanism-transfer
failure is the prior: this is exactly where the search-native program is
expected to die if it dies. If 3a FAILS, the search program is killed for the
cost of one eval and the on-distribution +73pp result stands alone as a scoped
finding.

--------------------------------------------------------------------------- #
DATA. The heldout repair corpus (`gen_repair_triples.py`, 2,389 records in
`data/repair_triples_heldout.jsonl`): each record is a natural MBPP-style task
with a `fix` (verified-passing program), an `attempt` (buggy program), the
grader `error_text`, tier and provenance. The record does NOT carry the test
block; we re-resolve it from the same loader `gen_repair_triples` used
(`code_grader.load_mbpp_combined`, keyed on `problem_key` -- verified 129/129
heldout keys present offline) and extract the FIRST parseable assert of form

    assert f(<literal args>) == <literal expected>

(ast.parse; literal args + literal expected only; no keywords/starred args;
`abs(...) < eps` and constructor-arg asserts like `f([Pair(..)] , 4)` are
skipped). 97.8% of triples yield such an assert.

TWO CANDIDATES per triple: {fixed, buggy}. Scored under:

  (a) EXECUTOR (`--ckpt`, default checkpoints/executor_longctx.pt). We render a
      Stage-A-style execution prompt: the candidate program's source with a
      trailing `x = f(<args>)` call line, framed by `value_function.
      render_exec_prompt` (the EXACT synthetic executor template the model was
      co-trained on) plus the "# trace:\n" cue, and greedy-decode the trace +
      answer via the true incremental prefill/forward_step path.

      Why `x = f(<args>)` and not `print(f(<args>))`: the executor's trained
      protocol reports the final value of a tracked *variable* (`# final: V`),
      never stdout. Binding the call result to a variable and asking "what is
      the final value of x?" is therefore the in-distribution rendering; a
      `print` would put the answer on stdout, which the executor was never
      trained to emit or trace. The tracked variable name is `x`, matching every
      synthetic training program. Documented as the exact rendering choice.

      Score = 1 iff the parsed `# final:` answer equals the expected literal
      (compared as PYTHON VALUES via ast.literal_eval, not strings, when both
      parse; str fallback otherwise), else 0. Ties broken by the executor's
      first-answer-token log-prob (mirrors `value_function.executor_generate`).

      OOD handling: the executor was trained on synthetic mod-10 INTEGER traces.
      Natural MBPP values (str/list/large int) are OOD. We still RUN every case
      (the probe MEASURES transfer -- skipping hard cases would bias it), parse
      the generated answer with an ast.literal_eval fallback, and record
      per-expected-type breakdowns (int/str/list/other) so the report shows
      WHERE transfer holds.

  (b) LOGPROB: mean per-token log-prob of the candidate program given the
      task/prompt context (the triple's NL `problem` field), teacher-forced,
      identical framing for both candidates (`value_function.
      teacher_forced_mean_logprob`).

  (c) RANDOM: 0.5 by construction (control floor).

GROUND-TRUTH VERIFICATION. For every parseable triple we actually EXECUTE both
candidates against the parsed assert in a `code_grader.grade` subprocess sandbox
and CONFIRM fixed PASSES and buggy FAILS that assert. Pairs where that does not
hold are DROPPED (label-noise for a ranking probe: the buggy attempt may fail a
DIFFERENT assert, or the fix may be for a different test); the drop count is
reported.

METRIC. Two-way ranking accuracy = fraction of pairs where fixed outranks buggy
strictly. Reported ties-as-failures (headline) AND ties-broken-random, pooled +
per-expected-type, with bootstrap 95% CIs (1000 resamples). Ranking/tie/CI math
is reused verbatim from Phase 2 (`value_function.top_indices / success_strict /
success_random / bootstrap_ci`).

Usage:
  # dry scan (CPU only, no model): parseable + verified pair counts per type
  PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/repair_value_probe.py --dry_scan --workers 16

  # 10-pair GPU smoke on the default ckpt
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/repair_value_probe.py --limit 10 \\
      --out results/repair_value_probe_smoke.json

  # full run (launched after review)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/repair_value_probe.py \\
      --ckpt checkpoints/executor_longctx.pt \\
      --out results/repair_value_probe.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import pathlib
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.code_grader import Problem, grade
# Ranking / tie / CI / log-prob / prompt machinery reused VERBATIM from Phase 2
# so the two probes share identical conventions.
from experiments.value_function import (
    teacher_forced_mean_logprob, render_exec_prompt,
    top_indices, success_strict, success_random, bootstrap_ci,
)

EXPECTED_TYPES = ("int", "str", "list", "other")


# --------------------------------------------------------------------------- #
# Assert parsing (pure; no torch/model/HF). `assert f(<literal args>) ==
# <literal expected>` in either operand order.
# --------------------------------------------------------------------------- #

def _try_literal(node: ast.AST):
    """(value, ok): ast.literal_eval of one AST node, guarded."""
    try:
        return ast.literal_eval(node), True
    except Exception:
        return None, False


def _literal_call(node: ast.AST):
    """(fname, [arg_values]) if `node` is `Name(...)` with ALL-literal args and
    no keywords/starred args, else None. A constructor/nested call inside an arg
    (e.g. `Pair(5, 24)`) makes that arg non-literal -> the whole call is
    rejected (returns None)."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return None
    if node.keywords:
        return None
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    args = []
    for a in node.args:
        v, ok = _try_literal(a)
        if not ok:
            return None
        args.append(v)
    return node.func.id, args


def parse_assert(assert_src: str):
    """Parse one `assert f(<literals>) == <literal>` statement.

    Returns (fname, args, expected, call_src, assert_src_norm) or None if the
    statement is not a single `==` compare between a literal-arg call and a
    literal. `call_src` is the unparsed call node (`f(a, b)`), used to render the
    executor prompt; `assert_src_norm` is the unparsed assert, used as the
    single-assert ground-truth check body."""
    try:
        mod = ast.parse(assert_src.strip())
    except SyntaxError:
        return None
    if len(mod.body) != 1 or not isinstance(mod.body[0], ast.Assert):
        return None
    test = mod.body[0].test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)):
        return None
    left, right = test.left, test.comparators[0]
    lc, rc = _literal_call(left), _literal_call(right)
    # exactly one side is the literal-arg call; the other must be a literal
    if lc is not None and rc is None:
        exp, ok = _try_literal(right)
        call_node = left
        fname, args = lc
    elif rc is not None and lc is None:
        exp, ok = _try_literal(left)
        call_node = right
        fname, args = rc
    else:
        return None
    if not ok:
        return None
    return fname, args, exp, ast.unparse(call_node), ast.unparse(mod.body[0])


def first_literal_assert(tests: str):
    """First `assert f(<literals>) == <literal>` in a test block (ast.walk
    order), or None. Scans ALL asserts so a non-literal first assert doesn't
    hide a literal one later."""
    try:
        tree = ast.parse(tests)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            p = parse_assert(ast.unparse(node))
            if p is not None:
                return p
    return None


def expected_type(v) -> str:
    """int / str / list / other. bool is 'other' (it is not the integer case the
    executor was trained on and would confound the int bucket)."""
    if isinstance(v, bool):
        return "other"
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    return "other"


# --------------------------------------------------------------------------- #
# Executor prompt rendering + answer parsing (pure).
# --------------------------------------------------------------------------- #

def build_exec_program(candidate_code: str, call_src: str,
                       tracked_var: str = "x") -> str:
    """Full program the executor is asked to simulate: the candidate program,
    then a module-level `x = f(<args>)` binding the call result to the tracked
    variable. Rendered by `render_exec_prompt` (the synthetic executor
    template)."""
    return (candidate_code.rstrip("\n") + "\n"
            + f"{tracked_var} = {call_src}")


# "# final: 4" / "# final: 'a'" / "# final: [1, 2, 3]" / "# final: None"
_FINAL_LINE_RE = re.compile(r"#\s*final\s*:\s*([^\n]*)", re.IGNORECASE)
# "# final:" prefix only (used for generation early-stop + tie-break capture).
_FINAL_PREFIX_RE = re.compile(r"#\s*final\s*:", re.IGNORECASE)


def _safe_literal_eval(s: str):
    """(value, ok): ast.literal_eval of a string, with a trailing-inline-comment
    fallback (`4  # note` -> 4)."""
    s = (s or "").strip()
    if not s:
        return None, False
    try:
        return ast.literal_eval(s), True
    except Exception:
        pass
    # trim a trailing inline comment and retry
    cut = s.split("#", 1)[0].strip()
    if cut and cut != s:
        try:
            return ast.literal_eval(cut), True
        except Exception:
            pass
    return None, False


def parse_final_answer(text: str) -> dict:
    """Parse the executor's `# final: VALUE` line from generated text.

    Returns {found, raw, value, parsed}: `found` iff a `# final:` line exists,
    `raw` the trimmed text after it, `parsed` whether ast.literal_eval succeeded,
    `value` the parsed Python object (or None)."""
    m = _FINAL_LINE_RE.search(text)
    if not m:
        return {"found": False, "raw": None, "value": None, "parsed": False}
    raw = m.group(1).strip()
    val, ok = _safe_literal_eval(raw)
    return {"found": True, "raw": raw, "value": val if ok else None,
            "parsed": ok}


def answer_matches(parsed: dict, expected) -> int:
    """1 iff the parsed executor answer equals `expected`. Value-equality when
    the answer parses (so [1,2,3]==[1,2,3], 'a'=='a', 4==4); a str fallback
    (raw == str(expected)) covers an unquoted literal like `abc` for expected
    'abc'."""
    if not parsed["found"]:
        return 0
    if parsed["parsed"]:
        try:
            return int(bool(parsed["value"] == expected) and
                       type(parsed["value"]) == type(expected))
        except Exception:
            return 0
    return int(parsed["raw"] == str(expected))


# --------------------------------------------------------------------------- #
# Ground-truth verification (subprocess sandbox via code_grader.grade).
# --------------------------------------------------------------------------- #

def build_single_assert_check(assert_src: str) -> str:
    """A `check(candidate)` block containing just the one parsed assert. The
    assert references the function by its real name (defined by exec'ing the
    candidate), so `candidate` is unused -- but grade() needs the wrapper."""
    body = "\n".join("    " + ln for ln in assert_src.splitlines())
    return f"def check(candidate):\n{body}\n"


def verify_pair(fname: str, assert_src: str, fixed_code: str,
                buggy_code: str, timeout_s: int = 4):
    """Execute both candidates against the single parsed assert in the sandbox.
    Returns (fixed_pass, buggy_fail, fixed_tier, buggy_tier). A pair is kept iff
    fixed_pass and buggy_fail."""
    tests = build_single_assert_check(assert_src)
    prob = Problem(task_id="verify", prompt="", tests=tests,
                   entry_point=fname, prompt_is_code=False)
    fixed_res = grade(prob, fixed_code, timeout_s=timeout_s)
    buggy_res = grade(prob, buggy_code, timeout_s=timeout_s)
    return (fixed_res.tier == "pass", buggy_res.tier != "pass",
            fixed_res.tier, buggy_res.tier)


def _grade_job(job):
    """ProcessPool worker: verify one triple. `job` = (idx, fname, assert_src,
    fixed_code, buggy_code, timeout_s)."""
    idx, fname, assert_src, fixed_code, buggy_code, timeout_s = job
    fp, bf, ft, bt = verify_pair(fname, assert_src, fixed_code, buggy_code,
                                 timeout_s)
    return idx, fp, bf, ft, bt


# --------------------------------------------------------------------------- #
# Pair preparation (parse in parent, grade in a ProcessPool). CPU-only; runs
# BEFORE the model / any CUDA context is created.
# --------------------------------------------------------------------------- #

def load_triples(path: str) -> list[dict]:
    out = []
    for line in pathlib.Path(path).open():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def resolve_pair(rec: dict, tests_by_key: dict):
    """Parse a triple into a scorable pair, or return ('skip', reason).
    Deterministic, no execution."""
    tests = tests_by_key.get(rec["problem_key"])
    if tests is None:
        return ("skip", "no_problem")
    parsed = first_literal_assert(tests)
    if parsed is None:
        return ("skip", "no_literal_assert")
    fname, args, expected, call_src, assert_src = parsed
    fixed_code = (rec.get("fix") or "").strip()
    buggy_code = (rec.get("attempt") or "").strip()
    if not fixed_code or not buggy_code:
        return ("skip", "empty_candidate")
    return ("pair", {
        "task_id": rec["task_id"], "problem_key": rec["problem_key"],
        "tier": rec.get("tier"), "provenance": rec.get("provenance"),
        "mutation": rec.get("mutation"),
        "problem": rec.get("problem", ""),
        "fname": fname, "call_src": call_src, "assert_src": assert_src,
        "expected": expected, "expected_type": expected_type(expected),
        "fixed_code": fixed_code, "buggy_code": buggy_code,
    })


def prepare_pairs(triples: list[dict], tests_by_key: dict, workers: int,
                  timeout_s: int):
    """Resolve + ground-truth-verify all triples. Returns (verified_pairs,
    stats). `verified_pairs` is sorted by task_id (deterministic scoring order).
    """
    skips = collections.Counter()
    resolved = []  # (idx, meta)
    for rec in triples:
        kind, payload = resolve_pair(rec, tests_by_key)
        if kind == "pair":
            resolved.append(payload)
        else:
            skips[payload] += 1

    # Verify (fixed passes / buggy fails the assert) in parallel.
    jobs = [(i, m["fname"], m["assert_src"], m["fixed_code"], m["buggy_code"],
             timeout_s) for i, m in enumerate(resolved)]
    verdicts = {}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_grade_job, j) for j in jobs]
            for f in as_completed(futs):
                idx, fp, bf, ft, bt = f.result()
                verdicts[idx] = (fp, bf, ft, bt)
    else:
        for j in jobs:
            idx, fp, bf, ft, bt = _grade_job(j)
            verdicts[idx] = (fp, bf, ft, bt)

    drops = collections.Counter()
    verified = []
    for i, m in enumerate(resolved):
        fp, bf, ft, bt = verdicts[i]
        if fp and bf:
            verified.append(m)
        elif not fp:
            drops["fixed_did_not_pass"] += 1
        else:  # fp and not bf
            drops["buggy_did_not_fail"] += 1

    verified.sort(key=lambda m: m["task_id"])
    stats = {
        "n_triples": len(triples),
        "n_resolved": len(resolved),
        "n_verified": len(verified),
        "skips": dict(skips),
        "drops": dict(drops),
        "resolved_by_type": dict(collections.Counter(
            m["expected_type"] for m in resolved)),
        "verified_by_type": dict(collections.Counter(
            m["expected_type"] for m in verified)),
    }
    return verified, stats


# --------------------------------------------------------------------------- #
# Executor generation (GPU; mirrors value_function.executor_generate but with a
# GENERAL final-line detector + ast.literal_eval-friendly early stop, since
# natural answers are str/list/large-int, not the int-only synthetic case).
# --------------------------------------------------------------------------- #

def _final_answer_started(text: str) -> bool:
    m = _FINAL_PREFIX_RE.search(text)
    return bool(m) and text[m.end():].strip() != ""


def _final_line_complete(text: str) -> bool:
    """True once a `# final:` line has some value AND a following newline (so a
    multi-token value has finished emitting)."""
    m = _FINAL_PREFIX_RE.search(text)
    if not m:
        return False
    rest = text[m.end():]
    return "\n" in rest and rest.strip() != ""


@torch.no_grad()
def executor_generate_natural(model, tok, prompt_ids: list[int], max_gen: int,
                              eos_id: int, grace: int = 4):
    """Greedy '# trace:' decode via the incremental prefill/forward_step path
    (mirrors value_function.executor_generate) returning (text,
    answer_logprob). `answer_logprob` is the log-prob of the first answer-value
    token (secondary tie-break); None if no `# final:` value is ever emitted."""
    device = next(model.parameters()).device
    t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen_ids: list[int] = []
    answer_logprob = None
    stop_at = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cache, last_logits = model.prefill(t)
        step_logits = last_logits[:, -1, :].float()
        nxt = step_logits.argmax(-1, keepdim=True)
        for step in range(max_gen):
            tid = int(nxt.item())
            if tid == eos_id:
                break
            tok_logprob = float(torch.log_softmax(step_logits, dim=-1)[0, tid])
            gen_ids.append(tid)
            text = tok.decode(gen_ids, skip_special_tokens=False)
            if answer_logprob is None and _final_answer_started(text):
                answer_logprob = tok_logprob
            if _final_line_complete(text):
                if stop_at is None:
                    stop_at = step
                if step - stop_at >= grace:
                    break
            logits, cache = model.forward_step(nxt, cache)
            step_logits = logits[:, -1, :].float()
            nxt = step_logits.argmax(-1, keepdim=True)
    return tok.decode(gen_ids, skip_special_tokens=False), answer_logprob


# --------------------------------------------------------------------------- #
# Per-pair value-function scorers. Candidate order is [fixed, buggy], so the
# "true" (should-rank-#1) index is always 0.
# --------------------------------------------------------------------------- #

TRUE_INDEX = 0  # fixed


def score_pair_executor(model, tok, eos_id, pair: dict, max_gen: int):
    """Keys (match_bit, answer_logprob) for [fixed, buggy]. match_bit=1 iff the
    executor's simulated `# final:` answer equals the expected literal."""
    keys, dbg = [], []
    for role, code in (("fixed", pair["fixed_code"]),
                       ("buggy", pair["buggy_code"])):
        prog = build_exec_program(code, pair["call_src"])
        prompt = render_exec_prompt(prog, "x").rstrip() + "\n# trace:\n"
        ids = tok.encode(prompt, add_special_tokens=False)
        text, alp = executor_generate_natural(model, tok, ids, max_gen, eos_id)
        parsed = parse_final_answer(text)
        match = answer_matches(parsed, pair["expected"])
        keys.append((match, alp if alp is not None else float("-inf")))
        dbg.append({"role": role, "raw": parsed["raw"],
                    "parsed": parsed["parsed"],
                    "value": repr(parsed["value"]) if parsed["parsed"] else None,
                    "match": match})
    return keys, dbg


def score_pair_logprob(model, tok, pair: dict):
    """Keys (mean_program_logprob,) for [fixed, buggy], given the NL problem as
    a shared prefix (identical framing for both candidates)."""
    prefix = (pair["problem"] or "").rstrip() + "\n"
    keys = []
    for code in (pair["fixed_code"], pair["buggy_code"]):
        full = prefix + code
        keys.append((teacher_forced_mean_logprob(model, tok, prefix, full),))
    return keys


def score_pair_random(pair: dict, rng: random.Random):
    return [(rng.random(),) for _ in range(2)]


# --------------------------------------------------------------------------- #
# Aggregation (pure). Pooled + per-expected-type, both tie policies, bootstrap.
# --------------------------------------------------------------------------- #

VALUE_FNS = ("executor", "logprob", "random")


def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def aggregate(scored: list[dict], seed: int, n_boot: int) -> dict:
    """`scored` = per-pair dicts with keys {expected_type, success{vf}{policy}}.
    Returns pooled + per-type ranking accuracy with bootstrap CIs, plus the
    per-type executor exact-match rate on the FIXED candidate (a diagnostic of
    where the executor can compute the value at all)."""
    def _collect(pairs):
        out = {vf: {"strict": [], "random": []} for vf in VALUE_FNS}
        for p in pairs:
            for vf in VALUE_FNS:
                out[vf]["strict"].append(p["success"][vf]["strict"])
                out[vf]["random"].append(p["success"][vf]["random"])
        return out

    def _block(pairs, tag):
        col = _collect(pairs)
        blk = {}
        for vf in VALUE_FNS:
            s, r = col[vf]["strict"], col[vf]["random"]
            ms, los, his = bootstrap_ci(s, n_boot, _stable_hash(f"{tag}-{vf}-s"))
            mr, lor, hir = bootstrap_ci(r, n_boot, _stable_hash(f"{tag}-{vf}-r"))
            blk[vf] = {
                "ties_as_failures": {"acc": ms, "ci95": [los, his], "n": len(s)},
                "ties_broken_random": {"acc": mr, "ci95": [lor, hir],
                                       "n": len(r)},
            }
        return blk

    def _delta(block, col):
        e = block["executor"][col]["acc"]
        l = block["logprob"][col]["acc"]
        return (100.0 * (e - l)) if (e is not None and l is not None) else None

    pooled = _block(scored, "pooled")
    per_type = {}
    for t in EXPECTED_TYPES:
        sub = [p for p in scored if p["expected_type"] == t]
        if sub:
            per_type[t] = _block(sub, f"type-{t}")

    # executor exact-match on the FIXED candidate, per type (transfer diagnostic)
    fixed_match = collections.defaultdict(lambda: [0, 0])  # type -> [hit, n]
    for p in scored:
        fixed_match[p["expected_type"]][0] += p["executor_fixed_match"]
        fixed_match[p["expected_type"]][1] += 1
    fixed_match["pooled"] = [sum(p["executor_fixed_match"] for p in scored),
                             len(scored)]
    exec_fixed_acc = {k: (v[0] / v[1] if v[1] else None)
                      for k, v in fixed_match.items()}

    d_strict = _delta(pooled, "ties_as_failures")
    d_random = _delta(pooled, "ties_broken_random")
    verdict = {
        "bar_pp": 10.0,
        "executor_minus_logprob_pp_ties_as_failures": d_strict,
        "executor_minus_logprob_pp_ties_broken_random": d_random,
        "headline_column": "ties_as_failures",
        "pass": bool(d_strict is not None and d_strict >= 10.0),
    }
    return {"pooled": pooled, "per_type": per_type,
            "executor_fixed_match_rate": exec_fixed_acc, "verdict": verdict}


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

class _PairTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _PairTimeout()


def score_all(model, tok, eos_id, pairs: list[dict], seed: int,
              max_gen: int, per_pair_timeout_s: int = 60) -> list[dict]:
    # Per-pair SIGALRM watchdog: one pathological pair (the 2026-07-19 full
    # run CPU-wedged after model load with zero progress for 13.5h) must cost
    # 60s + a recorded skip, not the run. Progress every 50 pairs.
    import signal
    signal.signal(signal.SIGALRM, _alarm_handler)
    scored = []
    n_timeout = 0
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"[score] pair {i}/{len(pairs)} (timeouts so far: {n_timeout})",
                  flush=True)
        try:
            signal.alarm(per_pair_timeout_s)
            ex_keys, ex_dbg = score_pair_executor(model, tok, eos_id, pair,
                                                  max_gen)
            lp_keys = score_pair_logprob(model, tok, pair)
        except _PairTimeout:
            n_timeout += 1
            print(f"[score] TIMEOUT ({per_pair_timeout_s}s) on pair {i} "
                  f"task_id={pair['task_id']!r} — skipped", flush=True)
            continue
        finally:
            signal.alarm(0)
        rand_rng = random.Random(_stable_hash(pair["task_id"]) ^ (seed + 999))
        rd_keys = score_pair_random(pair, rand_rng)
        keys_by_vf = {"executor": ex_keys, "logprob": lp_keys, "random": rd_keys}
        success = {}
        for vf in VALUE_FNS:
            keys = keys_by_vf[vf]
            success[vf] = {
                "strict": success_strict(keys, TRUE_INDEX),
                "random": success_random(
                    keys, TRUE_INDEX,
                    random.Random(_stable_hash(pair["task_id"] + vf)
                                  ^ (seed + 777))),
            }
        scored.append({
            "task_id": pair["task_id"], "expected_type": pair["expected_type"],
            "tier": pair["tier"], "provenance": pair["provenance"],
            "success": success,
            "executor_fixed_match": ex_dbg[0]["match"],
            "executor": ex_dbg,
            "expected": repr(pair["expected"]),
        })
    return scored


def print_table(results: dict):
    def row(vf, block):
        s = block[vf]["ties_as_failures"]
        r = block[vf]["ties_broken_random"]
        sa = f"{s['acc']:.3f}" if s["acc"] is not None else "  n/a"
        ra = f"{r['acc']:.3f}" if r["acc"] is not None else "  n/a"
        sci = (f"[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]"
               if s["acc"] is not None else "")
        return f"    {vf:>9} {sa:>7} {sci:>17} | {ra:>7}"

    print("\n=== TWO-WAY RANKING ACCURACY (fixed ranked above buggy) ===")
    print(f"    {'':>9} {'ties=fail':>7} {'95% CI':>17} | {'tie=rand':>7}")
    print(f"--- POOLED (n={results['pooled']['executor']['ties_as_failures']['n']}) ---")
    for vf in VALUE_FNS:
        print(row(vf, results["pooled"]))
    for t in EXPECTED_TYPES:
        if t in results["per_type"]:
            n = results["per_type"][t]["executor"]["ties_as_failures"]["n"]
            print(f"--- expected-type={t} (n={n}) ---")
            for vf in VALUE_FNS:
                print(row(vf, results["per_type"][t]))

    print("\n=== EXECUTOR EXACT-MATCH RATE on the FIXED candidate (transfer) ===")
    efm = results["executor_fixed_match_rate"]
    for k in ("pooled",) + EXPECTED_TYPES:
        if k in efm and efm[k] is not None:
            print(f"    {k:>8}: {efm[k]:.3f}")

    v = results["verdict"]
    d = v["executor_minus_logprob_pp_ties_as_failures"]
    dr = v["executor_minus_logprob_pp_ties_broken_random"]
    print("\n=== PHASE 3a PRE-REGISTERED VERDICT ===")
    print(f"executor - logprob: {d:+.1f}pp (ties=fail, HEADLINE)  |  "
          f"{dr:+.1f}pp (tie=rand)")
    print(f"bar: >= {v['bar_pp']:.0f}pp   ->   "
          f"{'PASS' if v['pass'] else 'FAIL'}")


def build_tests_by_key(dataset: str) -> dict:
    from experiments.code_grader import LOADERS
    return {p.task_id: p.tests for p in LOADERS[dataset]()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/executor_longctx.pt")
    ap.add_argument("--triples", default="data/repair_triples_heldout.jsonl")
    ap.add_argument("--dataset", default="mbpp_combined",
                    help="code_grader LOADERS key to re-resolve each triple's "
                         "test block (must match gen_repair_triples' --dataset)")
    ap.add_argument("--out", default="results/repair_value_probe.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N verified pairs (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=16,
                    help="processes for the CPU ground-truth-verification pass")
    ap.add_argument("--timeout_s", type=int, default=4)
    ap.add_argument("--max_gen", type=int, default=64,
                    help="executor trace+answer token budget per candidate")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--dry_scan", action="store_true",
                    help="prep only (no model): parseable + verified pair counts "
                         "per expected-type, then exit")
    args = ap.parse_args()
    t0 = time.time()

    print(f"[load] triples <- {args.triples}", flush=True)
    triples = load_triples(args.triples)
    print(f"[load] tests   <- LOADERS[{args.dataset}]", flush=True)
    tests_by_key = build_tests_by_key(args.dataset)

    print(f"[prep] resolving + verifying {len(triples)} triples "
          f"(workers={args.workers}) ...", flush=True)
    pairs, stats = prepare_pairs(triples, tests_by_key, args.workers,
                                 args.timeout_s)
    print(f"[prep] resolved={stats['n_resolved']}  verified={stats['n_verified']}"
          f"  skips={stats['skips']}  drops={stats['drops']}", flush=True)
    print(f"[prep] resolved_by_type={stats['resolved_by_type']}", flush=True)
    print(f"[prep] verified_by_type={stats['verified_by_type']}", flush=True)

    if args.dry_scan:
        out = {"mode": "dry_scan", "triples": args.triples, "stats": stats,
               "runtime_s": round(time.time() - t0, 1)}
        if args.out:
            pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2, default=str)
            print(f"[saved] {args.out}", flush=True)
        print(f"[done] dry scan {time.time() - t0:.1f}s", flush=True)
        return

    if args.limit and args.limit > 0:
        pairs = pairs[:args.limit]
    print(f"[score] scoring {len(pairs)} verified pairs with the model ...",
          flush=True)

    from experiments.eval_exec_trace_text import load_eval_model
    model, cfg, tok, eos_id = load_eval_model(args.ckpt)
    print(f"[model] {args.ckpt} ({cfg.get('n_layers')}L x {cfg.get('d_model')}d, "
          f"eos_id={eos_id})", flush=True)

    scored = score_all(model, tok, eos_id, pairs, args.seed, args.max_gen)
    per_type_counts = dict(collections.Counter(p["expected_type"] for p in scored))
    results = aggregate(scored, args.seed, args.n_boot)
    print_table(results)

    out = {
        "ckpt": args.ckpt, "triples": args.triples, "dataset": args.dataset,
        "seed": args.seed, "max_gen": args.max_gen, "n_boot": args.n_boot,
        "limit": args.limit, "n_scored": len(scored),
        "scored_by_type": per_type_counts,
        "prep_stats": stats, "results": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n[saved] {args.out}", flush=True)
    print(f"[done] {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
