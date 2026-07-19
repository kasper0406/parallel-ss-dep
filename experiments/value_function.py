"""Interpreter-as-value-function kill-test harness
(SEARCH_NATIVE_PLAN_2026_07_19.md, Phase 2).

The pre-registered claim (Phase-2 bar): the trained Stage-A/B text-scratchpad
EXECUTOR, used as a *value function on non-executable program prefixes*, ranks
candidate continuations better than a mean-log-prob baseline — by >= 10pp
ranking accuracy. If the interpreter can't outrank log-prob on its home
distribution, the whole search-native program dies here (cheap kill, no search
infra written).

Setup (per eval item, on HELD-OUT exec-trace SYNTHETIC programs — the schema
`gen_exec_traces.py` emits into `data/exec_trace_heldout_n{K}.jsonl`):

  1. Truncate a K-step program at a tracked-event boundary k in [1, K-1]
     (deterministic per-item seed). The prefix is a partial program with a
     KNOWN true final answer (== the original program's answer); the rest of
     the program is the TRUE continuation.
  2. Build N_DISTRACT=3 distractor continuations by MUTATING the true
     continuation (change one arithmetic constant, or swap one +/- operation).
     Every mutation is re-EXECUTED with the SAME sys.settrace machinery
     `gen_exec_traces.py` uses; a mutation is kept only if the mutated
     program's actual final answer DIFFERS from the true answer (and is
     syntactically valid + distinct). This makes the candidates
     surface-plausible but semantically-wrong.
  3. Score each of the 4 candidates under three value functions:
       (a) EXECUTOR  — run the Stage-A "# trace:" scratchpad protocol on the
                       full [prefix + candidate] program, parse the executor's
                       simulated "# final:" answer, score 1 iff it equals the
                       TRUE program's answer (ties broken by the executor's
                       answer-token log-prob when available).
       (b) LOGPROB   — mean per-token log-prob of the candidate continuation
                       given the prefix, teacher-forced (no generation).
       (c) RANDOM    — uniform random key (control; ~1/(N+1) accuracy).
  4. Ranking accuracy = fraction of items where the TRUE continuation is
     ranked strictly #1 among the 4 candidates. Reported under two tie
     policies (ties-as-failures AND ties-broken-random), per-K and pooled,
     with bootstrap 95% CIs.

WHY SYNTHETIC-ONLY: the heldout rungs are 96-100% synthetic (the MBPP minority
lacks `setup_src`/`traced_src` and has no clean constant/operator mutation
surface with a verifiable different answer). The harness filters to
`source == "synthetic"`; there are still >=287 items at every rung (300 at
K=8), well above the pre-registered n>=100 for K in {4,6,8}.

Usage (real run, GPU — once Phase 1 lands `checkpoints/executor_longctx.pt`):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/value_function.py \\
      --ckpt checkpoints/executor_longctx.pt \\
      --ks 4,6,8 --n_per_k 300 --out results/value_function_eval.json

  # tiny end-to-end GPU smoke (3 items) on the ckpt that exists NOW:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/value_function.py --ckpt checkpoints/stageA_executor.pt --smoke
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pathlib
import random
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.gen_exec_traces import _SENTINEL, _make_tracer, SOLVE_TAIL
from experiments.eval_exec_trace_text import (
    build_prompts, parse_final, _has_complete_final_line, load_rung,
)

N_DISTRACT = 3


# --------------------------------------------------------------------------- #
# In-process execution (trusted, bounded synthetic programs only).
#
# `gen_exec_traces.py` sandboxes execution in forked subprocesses because it
# mines arbitrary MBPP gold solutions. Here every program is one of OUR OWN
# synthetic mod-10 records (or a single-constant/operator mutation of one):
# bounded `for _i in range(2..3)` loops, no `while`, no I/O, no division. That
# is safe to run in-process with the SAME `_make_tracer` state-change tracer,
# which is what the mutation-verification correctness demands (identical event
# semantics to the generator). Mutations only ever swap an integer literal or a
# +/- operator inside an existing statement, so loop bounds — and therefore
# termination — are never touched.
# --------------------------------------------------------------------------- #

def execute_synthetic(setup_src: str, traced_src: str,
                      tracked_var: str = "x") -> list[int]:
    """Return the ordered list of tracked-variable state-change events for
    `setup_src` (untraced init) + `traced_src` (traced body), using the exact
    `_make_tracer` semantics `gen_exec_traces._synthetic_batch_worker` uses.
    Raises on a program that fails to compile/execute (callers treat that as an
    invalid candidate, never a silent pass)."""
    ns: dict = {}
    exec(compile(setup_src, "<setup>", "exec"), ns)
    events: list = []
    state = {"target": None, "last": _SENTINEL}
    tracer = _make_tracer("<traced>", None, tracked_var, events, state)
    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        exec(compile(traced_src, "<traced>", "exec"), ns)
    finally:
        sys.settrace(old)
    return events


# --------------------------------------------------------------------------- #
# Program-text chunking + truncation (pure).
# --------------------------------------------------------------------------- #

def split_top_level(traced_src: str) -> list[str]:
    """Split `traced_src` into its top-level statement chunks, preserving the
    EXACT source text (so ''.join(chunks) == traced_src). Multi-line
    constructs (if/else, for-loops) stay in one chunk. Any leading blank/comment
    lines attach to the first chunk; trailing lines to the last."""
    lines = traced_src.splitlines(keepends=True)
    nodes = ast.parse(traced_src).body
    if not nodes:
        return [traced_src] if traced_src else []
    starts = [n.lineno - 1 for n in nodes]
    starts[0] = 0  # absorb any leading blank lines into chunk 0
    chunks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(lines)
        chunks.append("".join(lines[s:e]))
    return chunks


def choose_split(setup_src: str, traced_src: str, tracked_var: str, K: int,
                 target_k: int) -> tuple[str, str, int] | None:
    """Truncate `traced_src` at a top-level chunk boundary so the prefix
    contains as close to `target_k` tracked events as reachable (loops emit
    multiple events per chunk, so exact k is not always hittable). Returns
    (prefix_traced, cont_traced, actual_k) with 1 <= actual_k <= K-1, or None
    if no boundary yields a non-empty prefix AND a non-empty continuation of
    tracked events."""
    chunks = split_top_level(traced_src)
    # events after chunks[:m], for m = 0..len(chunks)
    cum = [0]
    for m in range(1, len(chunks) + 1):
        cum.append(len(execute_synthetic(setup_src, "".join(chunks[:m]),
                                         tracked_var)))
    # valid boundary: prefix has >=1 event and continuation has >=1 event.
    valid = [m for m in range(1, len(chunks)) if 1 <= cum[m] <= K - 1]
    if not valid:
        return None
    best = min(valid, key=lambda m: (abs(cum[m] - target_k), m))
    prefix_traced = "".join(chunks[:best])
    cont_traced = "".join(chunks[best:])
    return prefix_traced, cont_traced, cum[best]


# --------------------------------------------------------------------------- #
# Mutation surface (pure): change one arithmetic constant OR swap one +/- op,
# on a single `x = ...` statement of the continuation.
# --------------------------------------------------------------------------- #

# `x = (x + 5) % 10`  (optionally indented, e.g. inside if/for bodies)
AUG_RE = re.compile(r"^(\s*)x\s*=\s*\(\s*x\s*([+\-*])\s*(\d+)\s*\)\s*%\s*(\d+)\s*$")
# `x = 5`  (plain constant assign; NOT `x = tbl[x]` / `x = (x+..)`)
ASSIGN_RE = re.compile(r"^(\s*)x\s*=\s*(\d+)\s*$")


def line_mutations(line_body: str) -> list[tuple[str, str]]:
    """All single-edit mutations of one statement `line_body` (no trailing
    newline). Returns [(mutated_line_body, mutation_class), ...]; empty if the
    line has no clean constant/operator to edit (e.g. a `tbl[x]` lookup or an
    `if x >= t:` header — those are never directly mutated)."""
    out: list[tuple[str, str]] = []
    m = AUG_RE.match(line_body)
    if m:
        indent, op, d, mod = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        # constant edit: pick a different valid magnitude (mul needs >=2)
        lo = 2 if op == "*" else 1
        for nd in range(lo, 10):
            if nd != d:
                out.append((f"{indent}x = (x {op} {nd}) % {mod}", "const"))
        # operator swap: +/- flip (both valid with the SAME magnitude);
        # a '*' line is swapped to '+' (a valid alternative op).
        if op == "+":
            out.append((f"{indent}x = (x - {d}) % {mod}", "opswap"))
        elif op == "-":
            out.append((f"{indent}x = (x + {d}) % {mod}", "opswap"))
        else:  # '*'
            out.append((f"{indent}x = (x + {d}) % {mod}", "opswap"))
        return out
    m = ASSIGN_RE.match(line_body)
    if m:
        indent, c = m.group(1), int(m.group(2))
        for nc in range(0, 10):
            if nc != c:
                out.append((f"{indent}x = {nc}", "const"))
    return out


def enumerate_distractors(setup_src: str, prefix_traced: str, cont_traced: str,
                          tracked_var: str, true_answer: int, rng: random.Random,
                          n_needed: int = N_DISTRACT):
    """Build up to `n_needed` distractor continuations by mutating ONE `x = ...`
    statement of `cont_traced`. Each accepted distractor: compiles, executes
    cleanly, has a final answer != `true_answer`, and is textually distinct
    from the true continuation and from the other distractors. Deterministic
    given `rng`. Returns list of dicts {cont_text, answer, mutation}."""
    lines = cont_traced.splitlines(keepends=True)
    candidates = []  # (line_idx, mutated_line_with_newline, mutation_class)
    for i, ln in enumerate(lines):
        nl = ln[len(ln.rstrip("\n")):]  # preserve exact trailing newline(s)
        body = ln.rstrip("\n")
        for mut_body, cls in line_mutations(body):
            candidates.append((i, mut_body + nl, cls))
    rng.shuffle(candidates)

    accepted = []
    seen_text = {cont_traced}
    for line_idx, mut_line, cls in candidates:
        if len(accepted) >= n_needed:
            break
        mut_lines = list(lines)
        mut_lines[line_idx] = mut_line
        mut_cont = "".join(mut_lines)
        if mut_cont in seen_text:
            continue
        full_traced = prefix_traced + mut_cont
        try:
            compile(full_traced, "<mut>", "exec")  # syntactic validity guard
            events = execute_synthetic(setup_src, full_traced, tracked_var)
        except Exception:
            continue
        if not events or events[-1] == true_answer:
            continue
        seen_text.add(mut_cont)
        accepted.append({"cont_text": mut_cont, "answer": events[-1],
                         "mutation": {"line_idx": line_idx, "class": cls}})
    return accepted


# --------------------------------------------------------------------------- #
# Per-item construction (pure; no model). One item = 1 true + 3 distractor
# continuations of a truncated program, in a shuffled order with true_index.
# --------------------------------------------------------------------------- #

def _stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def build_item(rec: dict, seed: int, n_distract: int = N_DISTRACT):
    """Turn one heldout synthetic record into a ranking item, or return
    ('skip', reason). Deterministic given (rec['task_id'], seed)."""
    if rec.get("source") != "synthetic":
        return ("skip", "non_synthetic")
    setup_src = rec["setup_src"]
    traced_src = rec["traced_src"]
    tracked_var = rec.get("tracked_var", "x")
    K = int(rec["rung"])
    true_answer = rec["answer"]
    if K < 2:
        return ("skip", "K_lt_2")

    rng = random.Random((_stable_hash(rec["task_id"])
                         ^ (seed * 0x9E3779B97F4A7C15)) & ((1 << 63) - 1))

    # sanity: the stored trace must reproduce under our execution machinery
    true_events = execute_synthetic(setup_src, traced_src, tracked_var)
    if true_events != list(rec["intermediates"]) or (
            true_events and true_events[-1] != true_answer):
        return ("skip", "exec_mismatch")

    target_k = rng.randint(1, K - 1)
    split = choose_split(setup_src, traced_src, tracked_var, K, target_k)
    if split is None:
        return ("skip", "no_valid_split")
    prefix_traced, cont_traced, actual_k = split

    distractors = enumerate_distractors(setup_src, prefix_traced, cont_traced,
                                        tracked_var, true_answer, rng, n_distract)
    if len(distractors) < n_distract:
        return ("skip", "insufficient_distractors")

    candidates = [{"cont_text": cont_traced, "answer": true_answer,
                   "is_true": True, "mutation": None}]
    for d in distractors:
        candidates.append({"cont_text": d["cont_text"], "answer": d["answer"],
                           "is_true": False, "mutation": d["mutation"]})
    order = list(range(len(candidates)))
    rng.shuffle(order)
    candidates = [candidates[i] for i in order]
    true_index = next(i for i, c in enumerate(candidates) if c["is_true"])

    return ("item", {
        "task_id": rec["task_id"], "K": K, "actual_k": actual_k,
        "tracked_var": tracked_var, "setup_src": setup_src,
        "prefix_traced": prefix_traced, "true_answer": true_answer,
        "candidates": candidates, "true_index": true_index,
    })


def load_items(ks, n_per_k, prefix, seed, n_distract=N_DISTRACT):
    """Build items for every K. Returns (items_by_k, skip_by_k)."""
    items_by_k, skip_by_k = {}, {}
    for K in ks:
        recs = load_rung(prefix, K)
        items, skips = [], {}
        for rec in recs:
            if len(items) >= n_per_k:
                break
            kind, payload = build_item(rec, seed, n_distract)
            if kind == "item":
                items.append(payload)
            else:
                skips[payload] = skips.get(payload, 0) + 1
        items_by_k[K] = items
        skip_by_k[K] = skips
    return items_by_k, skip_by_k


# --------------------------------------------------------------------------- #
# Value-function scorers.
# --------------------------------------------------------------------------- #

def render_exec_prompt(program_text: str, tracked_var: str) -> str:
    """Reproduce `gen_exec_traces._build_synthetic_record`'s prompt template
    verbatim for a (possibly mutated) full program."""
    return ("You are given the following Python program.\n"
            f"{program_text}\n\n"
            f"After running the program, what is the final value of {tracked_var}?"
            + SOLVE_TAIL.format(q=tracked_var))


def _full_program_text(item: dict, cont_text: str) -> str:
    """Normalized full program text (setup + prefix + candidate), matching
    `_build_synthetic_record`'s `program_text` normalization exactly."""
    full_traced = item["prefix_traced"] + cont_text
    return item["setup_src"].rstrip("\n") + "\n" + full_traced.rstrip("\n")


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _mean_cont_logprob(logits: torch.Tensor, ids: list[int], start: int) -> float:
    """Mean teacher-forced log-prob of ids[start:] under `logits` (1,T,V).
    Token at position j is predicted by logits[:, j-1]. `start` must be >=1."""
    if start >= len(ids):
        return 0.0
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    total, n = 0.0, 0
    for j in range(max(start, 1), len(ids)):
        total += float(logp[j - 1, ids[j]])
        n += 1
    return total / n if n else 0.0


def teacher_forced_mean_logprob(model, tok, prefix_str: str, full_str: str) -> float:
    """Mean per-token log-prob of the continuation (full_str beyond prefix_str)
    given the prefix, teacher-forced. Model-agnostic: calls `model(input_ids)`
    and reads logits; works with the bf16 TinyLM (cuda) and any tiny CPU
    nn.Module returning (B,T,V) logits."""
    pid = tok.encode(prefix_str, add_special_tokens=False)
    fid = tok.encode(full_str, add_special_tokens=False)
    start = _longest_common_prefix_len(pid, fid)
    if start < 1:
        start = 1  # never score the very first token (no preceding context)
    device = next(model.parameters()).device
    inp = torch.tensor([fid], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(inp)
    if isinstance(logits, tuple):
        logits = logits[0]
    return _mean_cont_logprob(logits, fid, start)


@torch.no_grad()
def executor_generate(model, tok, prompt_ids, max_gen: int, eos_id: int,
                      grace: int = 4):
    """Greedy '# trace:' decode (mirrors eval_exec_trace_text.greedy_generate)
    that ALSO returns the log-prob of the token that first completes the
    '# final:' answer (the tie-break signal; None if never parsed). GPU path
    (prefill/forward_step)."""
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
            if answer_logprob is None and parse_final(text) is not None:
                answer_logprob = tok_logprob
            if _has_complete_final_line(text):
                if stop_at is None:
                    stop_at = step
                if step - stop_at >= grace:
                    break
            logits, cache = model.forward_step(nxt, cache)
            step_logits = logits[:, -1, :].float()
            nxt = step_logits.argmax(-1, keepdim=True)
    return tok.decode(gen_ids, skip_special_tokens=False), answer_logprob


def score_item_executor(model, tok, eos_id, item):
    """Score keys for the EXECUTOR value function: (match_bit, answer_logprob).
    match_bit = 1 iff the executor's simulated '# final:' answer equals the
    TRUE program's answer. answer_logprob (or -inf) is the secondary tie-break."""
    keys, dbg = [], []
    for c in item["candidates"]:
        prog = _full_program_text(item, c["cont_text"])
        prompt = render_exec_prompt(prog, item["tracked_var"]).rstrip() + "\n# trace:\n"
        ids = tok.encode(prompt, add_special_tokens=False)
        max_gen = item["K"] * 12 + 24
        text, alp = executor_generate(model, tok, ids, max_gen, eos_id)
        pred = parse_final(text)
        match = int(pred is not None and pred == item["true_answer"])
        keys.append((match, alp if alp is not None else float("-inf")))
        dbg.append({"exec_answer": pred, "match": match})
    return keys, dbg


def score_item_logprob(model, tok, item):
    """Score keys for the LOGPROB value function: (mean_continuation_logprob,)."""
    prefix_str = ("You are given the following Python program.\n"
                  + item["setup_src"].rstrip("\n") + "\n" + item["prefix_traced"])
    keys = []
    for c in item["candidates"]:
        full_str = prefix_str + c["cont_text"]
        keys.append((teacher_forced_mean_logprob(model, tok, prefix_str, full_str),))
    return keys


def score_item_random(item, rng: random.Random):
    """Score keys for the RANDOM control: one uniform key per candidate."""
    return [(rng.random(),) for _ in item["candidates"]]


# --------------------------------------------------------------------------- #
# Ranking + aggregation (pure).
# --------------------------------------------------------------------------- #

def top_indices(keys) -> list[int]:
    """Indices whose key is the (lexicographic) maximum."""
    best = max(keys)
    return [i for i, k in enumerate(keys) if k == best]


def success_strict(keys, true_index: int) -> int:
    """Ties-as-failures: true is #1 only if it is the UNIQUE maximum."""
    top = top_indices(keys)
    return int(len(top) == 1 and top[0] == true_index)


def success_random(keys, true_index: int, rng: random.Random) -> int:
    """Ties-broken-random: among the tied maxima, a random one is picked."""
    top = top_indices(keys)
    if true_index not in top:
        return 0
    return int(rng.choice(top) == true_index)


def bootstrap_ci(vals: list[int], n_boot: int, seed: int, alpha: float = 0.05):
    """(mean, lo, hi) 95% percentile bootstrap CI over a 0/1 vector."""
    if not vals:
        return (None, None, None)
    mean = sum(vals) / len(vals)
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return (mean, lo, hi)


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

VALUE_FNS = ("executor", "logprob", "random")


def evaluate(model, tok, eos_id, items_by_k, ks, seed, n_boot):
    """Score every item under all three value functions; aggregate per-K and
    pooled with both tie policies + bootstrap CIs. Returns (results, per_item)."""
    # per (K, vf, policy) -> list of 0/1 successes, in item order
    succ = {vf: {"strict": {}, "random": {}} for vf in VALUE_FNS}
    per_item_dbg = {}
    for K in ks:
        for vf in VALUE_FNS:
            succ[vf]["strict"][K] = []
            succ[vf]["random"][K] = []
        for it in items_by_k.get(K, []):
            # per-item deterministic RNG for the random value-function keys
            rand_rng = random.Random(_stable_hash(it["task_id"]) ^ (seed + 999))
            ex_keys, ex_dbg = score_item_executor(model, tok, eos_id, it)
            lp_keys = score_item_logprob(model, tok, it)
            rd_keys = score_item_random(it, rand_rng)
            keys_by_vf = {"executor": ex_keys, "logprob": lp_keys, "random": rd_keys}
            for vf in VALUE_FNS:
                keys = keys_by_vf[vf]
                succ[vf]["strict"][K].append(success_strict(keys, it["true_index"]))
                # a fresh deterministic draw per (vf) so policies are independent
                succ[vf]["random"][K].append(
                    success_random(keys, it["true_index"],
                                   random.Random(_stable_hash(it["task_id"] + vf)
                                                 ^ (seed + 777))))
            per_item_dbg[it["task_id"]] = {
                "K": K, "actual_k": it["actual_k"], "true_index": it["true_index"],
                "executor": ex_dbg, "true_answer": it["true_answer"],
            }

    def agg(vals, tag):
        m, lo, hi = bootstrap_ci(vals, n_boot, _stable_hash(tag))
        return {"acc": m, "ci95": [lo, hi], "n": len(vals)}

    results = {"per_k": {}, "pooled": {}}
    pooled = {vf: {"strict": [], "random": []} for vf in VALUE_FNS}
    for K in ks:
        results["per_k"][K] = {}
        for vf in VALUE_FNS:
            s, r = succ[vf]["strict"][K], succ[vf]["random"][K]
            pooled[vf]["strict"] += s
            pooled[vf]["random"] += r
            results["per_k"][K][vf] = {
                "ties_as_failures": agg(s, f"{vf}-strict-{K}"),
                "ties_broken_random": agg(r, f"{vf}-random-{K}"),
            }
    for vf in VALUE_FNS:
        results["pooled"][vf] = {
            "ties_as_failures": agg(pooled[vf]["strict"], f"{vf}-strict-pooled"),
            "ties_broken_random": agg(pooled[vf]["random"], f"{vf}-random-pooled"),
        }

    # verdict: executor - logprob, reported on BOTH columns; headline the
    # stricter ties-as-failures column to avoid inflating the executor's edge
    # (ties, which happen only when the executor is WRONG, count as failures).
    def delta(col):
        e = results["pooled"]["executor"][col]["acc"]
        l = results["pooled"]["logprob"][col]["acc"]
        return (100.0 * (e - l)) if (e is not None and l is not None) else None
    d_strict = delta("ties_as_failures")
    d_random = delta("ties_broken_random")
    results["verdict"] = {
        "bar_pp": 10.0,
        "executor_minus_logprob_pp_ties_as_failures": d_strict,
        "executor_minus_logprob_pp_ties_broken_random": d_random,
        "headline_column": "ties_as_failures",
        "pass": bool(d_strict is not None and d_strict >= 10.0),
    }
    return results, per_item_dbg


def print_table(results, ks):
    def row(vf, block):
        s = block[vf]["ties_as_failures"]
        r = block[vf]["ties_broken_random"]
        sa = f"{s['acc']:.3f}" if s["acc"] is not None else " n/a"
        ra = f"{r['acc']:.3f}" if r["acc"] is not None else " n/a"
        sci = (f"[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]"
               if s["acc"] is not None else "")
        return f"{vf:>9} {sa:>7} {sci:>15} | {ra:>7}"
    print("\n=== RANKING ACCURACY (true continuation ranked #1 of 4) ===")
    print(f"{'':>9} {'ties=fail':>7} {'95% CI':>15} | {'tie=rand':>7}")
    for K in ks:
        b = results["per_k"][K]
        n = b["executor"]["ties_as_failures"]["n"]
        print(f"--- K={K}  (n={n}) ---")
        for vf in VALUE_FNS:
            print(row(vf, b))
    print(f"--- POOLED ---")
    for vf in VALUE_FNS:
        print(row(vf, results["pooled"]))
    v = results["verdict"]
    print(f"\nexecutor - logprob: {v['executor_minus_logprob_pp_ties_as_failures']:+.1f}pp "
          f"(ties=fail, headline)  |  "
          f"{v['executor_minus_logprob_pp_ties_broken_random']:+.1f}pp (tie=rand)")
    print(f"bar: >= {v['bar_pp']:.0f}pp   ->   "
          f"{'PASS' if v['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="executor ckpt (checkpoints/executor_longctx.pt once "
                         "phase 1 lands; smoke with checkpoints/stageA_executor.pt)")
    ap.add_argument("--heldout_prefix", default="data/exec_trace_heldout")
    ap.add_argument("--ks", default="4,6,8")
    ap.add_argument("--n_per_k", type=int, default=300)
    ap.add_argument("--n_distract", type=int, default=N_DISTRACT)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/value_function_eval.json")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run (3 items) for a fast self-check")
    args = ap.parse_args()

    if args.smoke:
        args.ks = "4,6,8"
        args.n_per_k = 3

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    t0 = time.time()

    print(f"[build] items from {args.heldout_prefix} ks={ks} "
          f"n_per_k={args.n_per_k} seed={args.seed}", flush=True)
    items_by_k, skip_by_k = load_items(ks, args.n_per_k, args.heldout_prefix,
                                       args.seed, args.n_distract)
    for K in ks:
        print(f"  K={K}: {len(items_by_k[K])} items  skips={dict(skip_by_k[K])}",
              flush=True)

    from experiments.eval_exec_trace_text import load_eval_model
    model, cfg, tok, eos_id = load_eval_model(args.ckpt)
    print(f"[model] {args.ckpt} ({cfg.get('n_layers')}L x {cfg.get('d_model')}d, "
          f"eos_id={eos_id})", flush=True)

    results, per_item = evaluate(model, tok, eos_id, items_by_k, ks,
                                 args.seed, args.n_boot)
    print_table(results, ks)

    out = {
        "ckpt": args.ckpt, "ks": ks, "n_per_k": args.n_per_k,
        "n_distract": args.n_distract, "seed": args.seed, "n_boot": args.n_boot,
        "counts": {K: len(items_by_k[K]) for K in ks},
        "skips": {K: dict(skip_by_k[K]) for K in ks},
        "results": results, "runtime_s": round(time.time() - t0, 1),
    }
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n[saved] {args.out}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
