"""SCOREBOARD — the project's north-star metric: long-context coding-recall
QUALITY x decode COST, vs distance, for a bounded-state DeltaNet vs a
transformer. This is the ONE board that makes the cost moat the metric.

WHY this exists (2026-06-30). HumanEval-164 greedy is the WRONG headline for
the committed thesis (a cheap, bounded-state, long-context coding agent whose
moat is COST at long horizon): it is short-context (the moat is invisible) and
noisy. We already proved the cost moat in isolation (DECODE_COST_BENCH.md:
constant decode memory + batched-throughput win). This board JOINS the two
axes — does bounded state hold recall accuracy where a fixed-window transformer
goes blind, and at what tok/s + peak-memory cost — so a single table shows the
quality-where-it-matters AND the price to deliver it, into the regime where the
transformer is expensive.

TASK (leak-free, first-occurrence recall; reused from flagship_recall_probe):
  - Bind K facts near the START of a Python program as `vN = <4-digit int>`.
  - Distractor-fill (no 4-digit literals, never clobbers a vN) to a controlled
    token length L (the binding->query distance ~= L).
  - Query ONE key at the END via a raw-Python output-comment continuation
    `print(vQ)  # ` (no instruction-following needed; best-shot for BOTH base
    LMs — see prompt_pyout).
  - The bound value is NEVER restated before the query -> a recency copy cannot
    answer; the model must RECALL across the full distance. Values are 4-digit
    and distractors carry no 4-digit literal, so the FIRST isolated 4-digit
    number in the continuation is the recalled value (identical extractor for
    every arm). This is the repo's mandated leak-free / first-occurrence
    protocol (NOT the leaky `eval_code_recall --mode teacher_forced` that scores
    a restated answer).

ARMS (identical task text / values / extractor across all three):
  - ours              : the lean linearized DeltaNet (bounded recurrent state).
                        No positional limit (max_T=0) -> processes arbitrary L.
                        Scored with the TRUE incremental decode (prefill +
                        forward_step) — the same O(1) path decode_bench costs.
  - control_full      : the transformer at FULL context (perfect-but-O(L)-KV
                        attention; degrades past its 8192 training window).
  - control_window{W} : the SAME transformer but the context is TRUNCATED to the
                        last W tokens before the query — a fixed-COST agent.
                        Past L=W the early bindings leave the window -> recall
                        MUST collapse to ~chance. This is the foil bounded state
                        is supposed to beat at long range.

COST columns (per L, from decode_bench machinery, bf16 eager, single-stream):
  steady-state ms/tok (median over G decode steps after warmup) and decode-phase
  peak MiB (peak reset AFTER prefill so the O(L) prefill transient does not mask
  the decode footprint), plus the bounded "state/KV" MiB. control_window{W}'s
  cost is FLAT (it always processes <=W tokens) = the transformer cost at L=W.

HONESTY (repo culture): absolute accuracy on the current LEAN / un-SFT'd base
will be LOW — that is EXPECTED and FINE. The board is the TARGET and the moat
signal is the DEGRADATION SHAPE (graceful vs cliff) + the COST, not the absolute
number on a weak base. We report a strict exact-match `gen_acc`, a sensitive
`top1(1st-digit)` (gold's first token == the post-prompt argmax), AND a
teacher-forced `tf_exact` so a shape is visible even when exact-match is
floored. Task bodies are SYNTHETIC.

METRIC CAVEAT (fixed 2026-07-01): the SmolLM2 tokenizer digit-splits 4-digit
values into 4 single-digit tokens (e.g. "5440" -> '5','4','4','0'), so
`top1` — which only checks the FIRST post-prompt token — reduces to "is the
first DIGIT the argmax", a ~11% chance floor, NOT full-value retained-recall
evidence. Renamed to `top1(1st-digit)` in the printed table to make this
explicit; kept for continuity with older result files. `tf_exact` is the
stricter, honest companion: teacher-force the GOLD value's FULL token
sequence and score 1 only if EVERY gold token is the argmax at its position
— this is the metric to actually cite as "did the model retain the value".
Applied identically to both the ours arm (via incremental forward_step) and
the transformer arm (via a teacher-forced forward pass).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. HF_HUB_OFFLINE=1 \\
    .venv/bin/python experiments/scoreboard_longctx_cost.py --smoke
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. HF_HUB_OFFLINE=1 \\
    .venv/bin/python experiments/scoreboard_longctx_cost.py \\
      --buckets 512,1024,2048,4096,8192,16384,32768 \\
      --per_bucket 20 --keys_per_task 3
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.flagship_recall_probe import extract_answer
from experiments.flagship_recall_probe_gen import _build_body

MIB = 2 ** 20

# Per-bucket quality accumulator layout, shared by the ours and xf arms:
#   [gen_ok, top1_ok, tf_exact_ok, total, n_failed]
# `total` (the accuracy DENOMINATOR) counts only queries that actually RAN;
# `n_failed` counts queries that raised (OOM / position overflow) and are
# EXCLUDED from `total` — see run_xf_quality.
IDX_GEN, IDX_TOP1, IDX_TF, IDX_TOTAL, IDX_FAIL = range(5)


def prompt_pyout(body: str, key: str) -> str:
    """SHARED best-shot elicitation for BOTH base LMs: a raw-Python
    output-comment continuation. The program (bindings + distractors) is
    followed by `print(vQ)  # ` so the natural next tokens are vQ's printed
    value. Validated (2026-06-30) as best-shot for the lean linearized base
    (the prose "The value of X is" / instruction cue is OOD for it -> reads ~0,
    while this cue recalls cleanly) AND natural for the SmolLM2 code base, so a
    single shared format gives every arm its best shot (equal-opportunity
    mandate). Leak-free: the bound value appears ONLY at the top binding, never
    restated before the query, so a recency copy cannot answer it."""
    return body + f"\nprint({key})  # "


# --------------------------------------------------------------------------- #
# Task generation / loading
# --------------------------------------------------------------------------- #
def gen_tasks(path: str, buckets, per_bucket, n_keys, seed, tokenizer_name):
    """Generate the bucketed leak-free recall set (reuses the flagship
    body-builder). Idempotent: written once, then reused on later runs."""
    import random
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    rng = random.Random(seed)
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for L in buckets:
            for j in range(per_bucket):
                body, keys, answers, ntok = _build_body(rng, n_keys, L, tok)
                f.write(json.dumps({
                    "task_id": f"sb/L{L}/{j}", "bucket": L, "n_keys": n_keys,
                    "body": body, "keys": keys, "answers": answers,
                    "ours_tok_len": ntok}) + "\n")
                n += 1
    print(f"  generated {n} tasks -> {path}")


def load_tasks(path: str):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# --------------------------------------------------------------------------- #
# OURS — quality via the TRUE incremental decode (prefill + forward_step)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _ours_greedy(model, ids, max_gen):
    """Greedy generation through the bounded-state incremental path.
    Returns (generated_token_ids, first_step_logits)."""
    t = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cache, last_logits = model.prefill(t)
        first_logits = last_logits[:, -1, :].float().clone()
        nxt = first_logits.argmax(-1, keepdim=True)  # (1,1)
        out = []
        for _ in range(max_gen):
            out.append(int(nxt))
            logits, cache = model.forward_step(nxt, cache)
            nxt = logits[:, -1:, :].argmax(-1)
    return out, first_logits[0]


@torch.no_grad()
def _ours_tf_exact(model, ids, gold_ids):
    """Stricter companion to top1: TEACHER-FORCE the gold token sequence
    (not the model's own greedy continuation) through the same incremental
    prefill/forward_step path, and require the argmax to match the gold
    token at EVERY position. top1 only checks the first post-prompt token,
    which for a digit-tokenized 4-digit value is just the first digit
    (~11% chance floor) — tf_exact is the real value-level recall check."""
    t = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cache, last_logits = model.prefill(t)
        cur_logits = last_logits[:, -1, :].float()
        ok = True
        for i, g in enumerate(gold_ids):
            if int(cur_logits.argmax(-1).item()) != g:
                ok = False
            if i < len(gold_ids) - 1:
                nxt = torch.tensor([[g]], dtype=torch.long, device="cuda")
                logits, cache = model.forward_step(nxt, cache)
                cur_logits = logits[:, -1, :].float()
    return ok


def run_ours_quality(model, tok, tasks, *, keys_per_task, max_gen,
                     max_problems_per_bucket):
    per_bucket = {}        # L -> [gen_ok, top1_ok, tf_exact_ok, total, n_failed]
    tok_lens = {}
    seen = {}
    model.eval()
    for rec in tasks:
        L = rec["bucket"]
        if max_problems_per_bucket and seen.get(L, 0) >= max_problems_per_bucket:
            continue
        seen[L] = seen.get(L, 0) + 1
        for key in rec["keys"][:keys_per_task]:
            gold = str(rec["answers"][key])
            gold_ids = tok.encode(gold, add_special_tokens=False)
            gold_first = gold_ids[0]
            prompt = prompt_pyout(rec["body"], key)
            ids = tok.encode(prompt, add_special_tokens=False)
            out, first_logits = _ours_greedy(model, ids, max_gen)
            text = tok.decode(out, skip_special_tokens=True)
            pred = extract_answer(text)
            top1 = int(first_logits.argmax().item()) == gold_first
            tf_exact = _ours_tf_exact(model, ids, gold_ids)
            pb = per_bucket.setdefault(L, [0, 0, 0, 0, 0])
            pb[IDX_GEN] += int(pred == gold)
            pb[IDX_TOP1] += int(top1)
            pb[IDX_TF] += int(tf_exact)
            pb[IDX_TOTAL] += 1
            tok_lens.setdefault(L, []).append(len(ids))
    return per_bucket, tok_lens


# --------------------------------------------------------------------------- #
# TRANSFORMER — quality (full context + windowed/truncated control)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_xf_quality(hf, htok, tasks, *, keys_per_task, max_gen, window,
                   max_problems_per_bucket):
    """NOTE on failure handling (fixed 2026-07-01): a query that raises
    (OOM / position overflow past the transformer's trained window) is
    counted in `n_failed` and EXCLUDED from the accuracy denominator
    (`total`), not silently scored as a wrong answer. Blanket-scoring a
    "could not run" as pred=None (0% recall) was auditied as converting an
    infra failure into a fabricated "the transformer is blind" data point."""
    per_bucket = {}
    tok_lens = {}
    n_blind = 0          # queries whose binding fell OUTSIDE a truncated window
    n_failed = 0         # queries that raised -- EXCLUDED from the accuracy denominator
    seen = {}
    eos = htok.eos_token_id
    for rec in tasks:
        L = rec["bucket"]
        if max_problems_per_bucket and seen.get(L, 0) >= max_problems_per_bucket:
            continue
        seen[L] = seen.get(L, 0) + 1
        for key in rec["keys"][:keys_per_task]:
            gold = str(rec["answers"][key])
            gold_ids = htok.encode(gold, add_special_tokens=False)
            gold_first = gold_ids[0]
            prompt = prompt_pyout(rec["body"], key)
            ids = htok.encode(prompt, add_special_tokens=False)
            full_len = len(ids)
            if window is not None and full_len > window:
                ids = ids[-window:]      # fixed-cost agent: keep last W tokens
                n_blind += 1             # binding (at the start) left the window
            tok_lens.setdefault(L, []).append(len(ids))
            t = torch.tensor(ids, device="cuda").unsqueeze(0)
            pb = per_bucket.setdefault(L, [0, 0, 0, 0, 0])
            try:
                gen = hf.generate(
                    t, max_new_tokens=max_gen, do_sample=False,
                    pad_token_id=eos, output_scores=True,
                    return_dict_in_generate=True)
                seq, scores = gen.sequences, gen.scores
                text = htok.decode(seq[0, len(ids):], skip_special_tokens=True)
                pred = extract_answer(text)
                top1 = int(scores[0][0].argmax().item()) == gold_first
                # tf_exact: teacher-force the GOLD tokens (not the model's
                # own greedy continuation) and require argmax-correct at
                # EVERY gold position -- the stricter, honest companion to
                # top1 (which only checks the first, digit-split token).
                tf_ids = torch.tensor(ids + gold_ids, device="cuda").unsqueeze(0)
                tf_logits = hf(tf_ids).logits[0]
                start = len(ids) - 1
                tf_exact = all(
                    int(tf_logits[start + i].argmax(-1).item()) == g
                    for i, g in enumerate(gold_ids))
            except Exception as e:   # noqa: BLE001  (OOM / pos-overflow past window)
                torch.cuda.empty_cache()
                pb[IDX_FAIL] += 1
                n_failed += 1
                if "out of memory" not in str(e).lower():
                    print(f"    [xf gen FAILED L={L} win={window}] "
                          f"{type(e).__name__}: {e}")
                continue      # excluded from the accuracy denominator
            pb[IDX_GEN] += int(pred == gold)
            pb[IDX_TOP1] += int(top1)
            pb[IDX_TF] += int(tf_exact)
            pb[IDX_TOTAL] += 1
    if n_failed:
        print(f"  WARNING: [xf quality win={window}] {n_failed} "
              f"quer{'y' if n_failed == 1 else 'ies'} FAILED to run (OOM / "
              f"position overflow) and were EXCLUDED from the accuracy "
              f"denominator -- see per-bucket n_failed in the table, NOT "
              f"scored as wrong answers.")
    return per_bucket, tok_lens, n_blind


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _acc(pb, L, idx):
    if L not in pb or pb[L][IDX_TOTAL] == 0:
        return None
    return pb[L][idx] / pb[L][IDX_TOTAL]


def _nfail(pb, L):
    """n_failed for bucket L -- 0 if the bucket/arm never fails (e.g. ours,
    or an xf pb dict built before this field existed)."""
    v = pb.get(L)
    return v[IDX_FAIL] if v is not None and len(v) > IDX_FAIL else 0


def _fmt_pct(v):
    return "  -  " if v is None else f"{100 * v:4.0f}%"


def _cost_cell(c, key):
    if c is None:
        return "  -  "
    if c.get("oom"):
        return " OOM "
    return f"{c[key]:.0f}" if key.endswith("mib") or "peak" in key else f"{c[key]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/linearize/linearized_stage3.pt")
    ap.add_argument("--transformer", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--tasks", default="checkpoints/scoreboard/scoreboard_recall.jsonl")
    ap.add_argument("--regen", action="store_true",
                    help="force-regenerate the task file even if it exists")
    ap.add_argument("--buckets", default="512,1024,2048,4096,8192,16384,32768")
    ap.add_argument("--cost_buckets", default="",
                    help="extra cost-only lengths (no quality), e.g. 65536,131072")
    ap.add_argument("--per_bucket", type=int, default=20)
    ap.add_argument("--keys_per_task", type=int, default=3)
    ap.add_argument("--n_keys_gen", type=int, default=6)
    ap.add_argument("--window", type=int, default=2048,
                    help="windowed-transformer control context cap")
    ap.add_argument("--max_gen", type=int, default=8)
    ap.add_argument("--cost_gen", type=int, default=32, help="decode steps for cost")
    ap.add_argument("--cost_warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_cost", action="store_true")
    ap.add_argument("--no_quality", action="store_true")
    ap.add_argument("--out", default="checkpoints/scoreboard/scoreboard_results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.buckets = "512,1024,2048"
        args.cost_buckets = ""
        args.per_bucket = 2
        args.keys_per_task = 2
        args.cost_gen = 4
        args.cost_warmup = 2
        args.out = "checkpoints/scoreboard/scoreboard_results_smoke.json"
        args.tasks = "checkpoints/scoreboard/scoreboard_recall_smoke.jsonl"
        args.regen = True

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    cost_extra = [int(x) for x in args.cost_buckets.split(",") if x.strip()]
    cost_lengths = sorted(set(buckets) | set(cost_extra))

    assert torch.cuda.is_available(), "needs CUDA"
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free={free/MIB:.0f} MiB / {total/MIB:.0f} MiB")
    if free / MIB < 20000:
        print("WARNING: <20 GiB free — per the run policy, skipping. "
              "Free the GPU and re-run.")
        return

    # ---- tasks ----
    tok_name = "HuggingFaceTB/SmolLM2-360M"
    if args.regen or not pathlib.Path(args.tasks).exists():
        print(f"[gen tasks] buckets={buckets} per_bucket={args.per_bucket} "
              f"keys_gen={args.n_keys_gen}")
        gen_tasks(args.tasks, buckets, args.per_bucket, args.n_keys_gen,
                  args.seed, tok_name)
    tasks = load_tasks(args.tasks)
    print(f"[tasks] {len(tasks)} loaded, buckets={sorted({r['bucket'] for r in tasks})}")

    from transformers import AutoTokenizer
    import experiments.decode_bench as db

    results = {"config": {k: getattr(args, k) for k in vars(args)},
               "buckets": buckets, "cost_lengths": cost_lengths,
               "ours": {}, "control_full": {}, "control_window": {},
               "cost_ours": {}, "cost_xf": {}, "window": args.window}

    # =========================== OURS ============================
    print(f"\n[loading OURS] {args.ckpt}")
    ours, cfg = db.load_ours(args.ckpt)
    vocab = int(cfg["vocab_size"])
    ours_params = sum(p.numel() for p in ours.parameters()) / 1e6
    otok = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or tok_name)
    print(f"  OURS: {cfg.get('n_layers')}L x {cfg.get('d_model')}d, "
          f"{ours_params:.1f}M params, arch={cfg.get('arch')}, "
          f"feedback={cfg.get('feedback')}")

    if not args.no_quality:
        t0 = time.time()
        pb, tl = run_ours_quality(
            ours, otok, tasks, keys_per_task=args.keys_per_task,
            max_gen=args.max_gen, max_problems_per_bucket=args.per_bucket)
        results["ours"]["per_bucket"] = {str(k): v for k, v in pb.items()}
        results["ours"]["tok_len_mean"] = {
            str(k): sum(v) / len(v) for k, v in tl.items()}
        print(f"  [ours quality] {time.time()-t0:.0f}s  "
              + "  ".join(f"L{L}:{_fmt_pct(_acc(pb,L,IDX_GEN)).strip()}"
                          f"/{_fmt_pct(_acc(pb,L,IDX_TOP1)).strip()}"
                          f"/{_fmt_pct(_acc(pb,L,IDX_TF)).strip()}"
                          for L in buckets))

    if not args.no_cost:
        for L in cost_lengths:
            r = db.bench_ours(ours, vocab, L, args.cost_gen, args.cost_warmup)
            results["cost_ours"][str(L)] = r
            tag = "OOM" if r.get("oom") else (
                f"{r['ms_per_tok']:.2f}ms  peak={r['peak_decode_mib']:.0f}MiB  "
                f"state={r['cache_mib']:.2f}MiB")
            print(f"  [ours cost] L={L:>6}: {tag}")

    del ours
    import gc
    gc.collect(); torch.cuda.empty_cache()

    # ======================== TRANSFORMER ========================
    print(f"\n[loading TRANSFORMER] {args.transformer}")
    xf = db.load_transformer(args.transformer)
    xf_params = sum(p.numel() for p in xf.parameters()) / 1e6
    xtok = AutoTokenizer.from_pretrained(args.transformer)
    print(f"  {args.transformer}: {xf_params:.1f}M params, "
          f"max_pos={getattr(xf.config,'max_position_embeddings',None)}")

    if not args.no_quality:
        t0 = time.time()
        pb_full, tl_full, _ = run_xf_quality(
            xf, xtok, tasks, keys_per_task=args.keys_per_task,
            max_gen=args.max_gen, window=None,
            max_problems_per_bucket=args.per_bucket)
        results["control_full"]["per_bucket"] = {str(k): v for k, v in pb_full.items()}
        results["control_full"]["n_failed_total"] = sum(
            _nfail(pb_full, L) for L in pb_full)
        print(f"  [control_full quality] {time.time()-t0:.0f}s  "
              + "  ".join(f"L{L}:{_fmt_pct(_acc(pb_full,L,IDX_GEN)).strip()}"
                          f"/{_fmt_pct(_acc(pb_full,L,IDX_TOP1)).strip()}"
                          f"/{_fmt_pct(_acc(pb_full,L,IDX_TF)).strip()}"
                          f"(nF={_nfail(pb_full,L)})" for L in buckets))

        t0 = time.time()
        pb_win, _, n_blind = run_xf_quality(
            xf, xtok, tasks, keys_per_task=args.keys_per_task,
            max_gen=args.max_gen, window=args.window,
            max_problems_per_bucket=args.per_bucket)
        results["control_window"]["per_bucket"] = {str(k): v for k, v in pb_win.items()}
        results["control_window"]["n_blind"] = n_blind
        results["control_window"]["n_failed_total"] = sum(
            _nfail(pb_win, L) for L in pb_win)
        print(f"  [control_window{args.window} quality] {time.time()-t0:.0f}s  "
              f"n_blind={n_blind}  "
              + "  ".join(f"L{L}:{_fmt_pct(_acc(pb_win,L,IDX_GEN)).strip()}"
                          f"/{_fmt_pct(_acc(pb_win,L,IDX_TOP1)).strip()}"
                          f"/{_fmt_pct(_acc(pb_win,L,IDX_TF)).strip()}"
                          f"(nF={_nfail(pb_win,L)})" for L in buckets))

    if not args.no_cost:
        for L in cost_lengths:
            r = db.bench_transformer(xf, vocab, L, args.cost_gen, args.cost_warmup)
            results["cost_xf"][str(L)] = r
            tag = "OOM" if r.get("oom") else (
                f"{r['ms_per_tok']:.2f}ms  peak={r['peak_decode_mib']:.0f}MiB  "
                f"KV={r['cache_mib']:.1f}MiB")
            print(f"  [xf cost] L={L:>6}: {tag}")
        # windowed transformer cost == transformer cost at L=window (flat).
        wl = min(args.window, max(cost_lengths))
        if str(wl) not in results["cost_xf"]:
            r = db.bench_transformer(xf, vocab, wl, args.cost_gen, args.cost_warmup)
            results["cost_xf"][str(wl)] = r
        results["cost_window_ref"] = str(args.window if args.window in cost_lengths else wl)

    del xf
    gc.collect(); torch.cuda.empty_cache()

    # =========================== TABLE ===========================
    _print_table(results, buckets, args.window)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results["ours_params_m"] = ours_params
    results["xf_params_m"] = xf_params
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


def _pct3(v):
    """Compact fixed-width percent for the triple gen/top1/tf_exact cells."""
    return " - " if v is None else f"{100 * v:3.0f}"


def _triple(pb, L):
    return (f"{_pct3(_acc(pb, L, IDX_GEN))}/{_pct3(_acc(pb, L, IDX_TOP1))}"
            f"/{_pct3(_acc(pb, L, IDX_TF))}")


def _print_table(results, buckets, window):
    pb_o = {int(k): v for k, v in results["ours"].get("per_bucket", {}).items()}
    pb_f = {int(k): v for k, v in results["control_full"].get("per_bucket", {}).items()}
    pb_w = {int(k): v for k, v in results["control_window"].get("per_bucket", {}).items()}
    co = {int(k): v for k, v in results["cost_ours"].items()}
    cx = {int(k): v for k, v in results["cost_xf"].items()}
    wref = window if window in cx else None

    n_failed_total = (sum(_nfail(pb_f, L) for L in pb_f)
                       + sum(_nfail(pb_w, L) for L in pb_w))

    print("\n" + "=" * 130)
    print("SCOREBOARD — long-context recall QUALITY (gen_acc/top1(1st-digit)/tf_exact) x decode COST, vs distance L")
    print("  quality: gen_acc = strict first-4-digit exact-match ; "
          "top1(1st-digit) = gold FIRST TOKEN == post-prompt argmax (4-digit "
          "values digit-tokenize, so this is really \"is the first DIGIT "
          "right\" ~11% chance floor, NOT full-value recall) ; "
          "tf_exact = teacher-forced, ALL gold tokens argmax-correct (the "
          "real value-level recall metric)")
    print(f"  cost (single-stream, bf16 eager): ms/tok, decode-peak MiB, bounded state|KV MiB ; "
          f"window control = last {window} tokens (cost flat = xf@L={window})")
    if n_failed_total:
        print(f"  WARNING: {n_failed_total} transformer quer{'y' if n_failed_total == 1 else 'ies'} "
              f"FAILED to run (OOM / position overflow) across control_full + "
              f"control_window and were EXCLUDED from their bucket's accuracy "
              f"denominator, NOT scored as wrong -- see the nF columns below.")
    print("=" * 130)
    hdr = (f"{'L':>7} | {'OURS gen/t1/tf':>16} {'oMs':>6} {'oPeak':>7} {'oState':>7} | "
           f"{'FULL gen/t1/tf':>16} {'fNF':>4} {'fMs':>6} {'fPeak':>7} {'fKV':>7} | "
           f"{'WIN gen/t1/tf':>15} {'wNF':>4} | {'mem x':>6}")
    print(hdr); print("-" * len(hdr))
    for L in buckets:
        fnf, wnf = _nfail(pb_f, L), _nfail(pb_w, L)
        o, x = co.get(L), cx.get(L)
        oms = _cost_cell(o, "ms_per_tok"); opk = _cost_cell(o, "peak_decode_mib")
        ost = "  -  " if o is None or o.get("oom") else f"{o['cache_mib']:.1f}"
        xms = _cost_cell(x, "ms_per_tok"); xpk = _cost_cell(x, "peak_decode_mib")
        xkv = "  -  " if x is None or x.get("oom") else f"{x['cache_mib']:.1f}"
        memx = "  -  "
        if o and x and not o.get("oom") and not x.get("oom"):
            memx = f"{x['peak_decode_mib']/o['peak_decode_mib']:.2f}x"
        elif x and x.get("oom"):
            memx = "xf-OOM"
        print(f"{L:>7} | {_triple(pb_o, L):>16} {oms:>6} {opk:>7} {ost:>7} | "
              f"{_triple(pb_f, L):>16} {fnf:>4} {xms:>6} {xpk:>7} {xkv:>7} | "
              f"{_triple(pb_w, L):>15} {wnf:>4} | {memx:>6}")

    # cost-only extra lengths
    extra = [L for L in results["cost_lengths"] if L not in buckets]
    for L in extra:
        o, x = co.get(L), cx.get(L)
        oms = _cost_cell(o, "ms_per_tok"); opk = _cost_cell(o, "peak_decode_mib")
        ost = "  -  " if o is None or o.get("oom") else f"{o['cache_mib']:.1f}"
        xms = _cost_cell(x, "ms_per_tok"); xpk = _cost_cell(x, "peak_decode_mib")
        xkv = "  -  " if x is None or x.get("oom") else f"{x['cache_mib']:.1f}"
        memx = "  -  "
        if o and x and not o.get("oom") and not x.get("oom"):
            memx = f"{x['peak_decode_mib']/o['peak_decode_mib']:.2f}x"
        elif x and x.get("oom"):
            memx = "xf-OOM"
        print(f"{L:>7} | {'(cost only)':>16} {oms:>6} {opk:>7} {ost:>7} | "
              f"{'(cost only)':>16} {'-':>4} {xms:>6} {xpk:>7} {xkv:>7} | "
              f"{'-':>15} {'-':>4} | {memx:>6}")


if __name__ == "__main__":
    main()
