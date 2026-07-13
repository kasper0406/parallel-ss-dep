"""Exposure-bias diagnostic for the ~6-hop latent horizon
(EXEC_TRACE_LATENT_PLAN.md Stage-B RESULT + ideas_2026_07_13/09_wildcards.md
idea 5).

Question: is the hop-7+ cliff in the Stage-B latent exec-trace program
(per-hop 0.78-0.96 at hops 1-5, ~0.60 at hop 6, <=0.28 at hop 7+ — same
profile at every K incl. lengen) ERROR PROPAGATION through the growing
thread, or a per-slot capacity / training-exposure wall?

There is no token-level teacher forcing to remove here: in the growing
thread the feedback at slot j IS the model's own hidden state at slot j-1
(`grow_latent_thread`). The exposure-bias hypothesis, made concrete: during
TRAINING, deep slots mostly appeared in curriculum stages where the earlier
steps were TEXT (i.e. correct); at eval, slot j's input hidden carries the
errors of slots 1..j-1. So the diagnostic contrast is a STRATIFICATION of
the per-hop reads we already take:

  acc(j | slots 1..j-1 all decoded correctly)   vs
  acc(j | >=1 earlier slot decoded wrong)

computed from per-record per-slot correctness (argmax-decode of each latent
slot through out_norm->lm_head vs the interpreter-truth intermediate, exactly
`eval_exec_trace_latent_trace.latent_perhop_reads` — reused by import; this
probe only re-indexes its output and stratifies).

PRE-REGISTERED READ (thresholds in VERDICT_* constants):
  * acc(j | prefix correct) stays high (>=0.7) at j=7,8 while
    acc(j | prefix wrong) collapses  =>  the cliff is ERROR PROPAGATION
    (exposure-bias-like; fix = DAgger-style training on the model's own
    thread / error-tolerant per-hop supervision).
  * acc(j | prefix correct) ALSO collapses at j=7,8  =>  capacity /
    training-exposure of slot j itself (slots 7-8 get gradient only from
    (K>=7 rung) x (s>=7 stage) draws; fix = depth-weighted stage sampling /
    more deep-stage training).
  Strata can be thin (slot-8-with-correct-prefix is rare at current per-hop
  rates) — Wilson 95% CIs and per-stratum n are always reported; verdicts
  require n >= --min_stratum_n.

Also reported: the unconditional per-hop profile (directly comparable to
runs/stageB_perhop_remeasure.json's `perhop_by_step`).

Usage (ready to fire; heldout K=6,7,8 + lengen K=9,10,12):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/probe_latent_exposure_bias.py \\
      --ckpt checkpoints/stageB_depthfix.pt \\
      --n_per_rung 200 --out runs/probe_latent_exposure_bias.json

  # smoke (20 records, K=6 only, no lengen):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/probe_latent_exposure_bias.py \\
      --ckpt checkpoints/stageB_depthfix.pt --smoke
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

# Reuse (import, don't copy) the validated Stage-B eval machinery.
from experiments.eval_exec_trace_latent_trace import (
    encode_inter_token_ids,
    latent_perhop_reads,
    load_eval_model,
)
from experiments.eval_exec_trace_text import build_prompts, load_rung
from experiments.probe_state_algebra import wilson_ci

DEFAULT_CKPT = "checkpoints/stageB_depthfix.pt"
FALLBACK_CKPT = "checkpoints/stageB_latent_trace.pt"

# Pre-registered verdict thresholds (see module docstring).
VERDICT_SLOTS = (7, 8)          # the cliff slots
VERDICT_PROP_ACC = 0.7          # acc(j|prefix ok) >= this => propagation story
VERDICT_COLLAPSE_ACC = 0.5      # acc(j|prefix ok) < this  => slot-depth story


# --------------------------------------------------------------------------- #
# Indexed per-hop reads. `latent_perhop_reads` returns per-hop correctness
# ONLY for slots whose intermediate encodes to a single token, dropping the
# slot index. Its skip rule is deterministic (append iff j-1 < len(inter)
# and inter[j-1] is not None, in j order), so the slot-indexed view is
# reconstructed EXACTLY from the same call — zero duplicated model code.
# --------------------------------------------------------------------------- #

def reindex_perhop(flat: list[int], inter_tok: list, R: int) -> list:
    """Map latent_perhop_reads' flat output back onto slots 1..R.
    Returns a length-R list: 1/0 for measured slots, None for slots whose
    intermediate was multi-token (skipped by the reader)."""
    it = iter(flat)
    out = []
    for j in range(1, R + 1):
        if (j - 1) < len(inter_tok) and inter_tok[j - 1] is not None:
            out.append(next(it))
        else:
            out.append(None)
    # The reader must have produced exactly the measured slots.
    leftover = list(it)
    assert not leftover, f"flat perhop longer than expected: {leftover}"
    return out


@torch.no_grad()
def indexed_perhop_reads(model, prompt_ids: list[int], R: int,
                         thinking_id: int, inter_tok: list, device) -> list:
    flat = latent_perhop_reads(model, prompt_ids, R, thinking_id, inter_tok,
                               device)
    return reindex_perhop(flat, inter_tok, R)


# --------------------------------------------------------------------------- #
# Stratification (pure — CPU-testable).
# --------------------------------------------------------------------------- #

def stratify(per_record: list[list], K: int) -> dict:
    """Conditional per-slot accuracy from per-record indexed correctness.

    per_record: list of length-K lists with entries 1/0/None (None =
    unmeasurable slot, e.g. multi-token intermediate).

    For slot j (2..K): a record enters a stratum iff slot j AND all slots
    1..j-1 are measurable (records with any None in the prefix are excluded
    and counted in `n_excluded` — prefix correctness would be ambiguous).
      prefix_ok  stratum: slots 1..j-1 all == 1
      prefix_bad stratum: >=1 of slots 1..j-1 == 0

    Slot 1 has no prefix: unconditional only.

    Returns {j: {"n_ok", "k_ok", "n_bad", "k_bad", "n_excluded",
                 "n_uncond", "k_uncond"}} for j = 1..K.
    """
    out = {}
    for j in range(1, K + 1):
        d = {"n_ok": 0, "k_ok": 0, "n_bad": 0, "k_bad": 0,
             "n_excluded": 0, "n_uncond": 0, "k_uncond": 0}
        for rec in per_record:
            assert len(rec) == K, (len(rec), K)
            cur = rec[j - 1]
            if cur is not None:
                d["n_uncond"] += 1
                d["k_uncond"] += cur
            if j == 1:
                continue
            prefix = rec[: j - 1]
            if cur is None or any(x is None for x in prefix):
                d["n_excluded"] += 1
                continue
            if all(x == 1 for x in prefix):
                d["n_ok"] += 1
                d["k_ok"] += cur
            else:
                d["n_bad"] += 1
                d["k_bad"] += cur
        out[j] = d
    return out


def merge_strata(strata_list: list[dict]) -> dict:
    """Pool stratification dicts (e.g. across rungs) by slot index."""
    pooled: dict = {}
    for s in strata_list:
        for j, d in s.items():
            p = pooled.setdefault(j, {k: 0 for k in d})
            for k, v in d.items():
                p[k] += v
    return pooled


def _rate_ci(k: int, n: int) -> dict:
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k": k, "acc": (k / n) if n else None,
            "wilson95": [lo, hi]}


def summarize_strata(strata: dict) -> dict:
    """slot -> {uncond, prefix_ok, prefix_bad, n_excluded} with Wilson CIs."""
    out = {}
    for j in sorted(strata):
        d = strata[j]
        out[j] = {
            "uncond": _rate_ci(d["k_uncond"], d["n_uncond"]),
            "prefix_ok": _rate_ci(d["k_ok"], d["n_ok"]),
            "prefix_bad": _rate_ci(d["k_bad"], d["n_bad"]),
            "n_excluded": d["n_excluded"],
        }
    return out


def compute_verdict(pooled_summary: dict, min_stratum_n: int) -> dict:
    """Mechanically apply the pre-registered read at the cliff slots."""
    per_slot = {}
    for j in VERDICT_SLOTS:
        s = pooled_summary.get(j)
        if s is None or s["prefix_ok"]["n"] < min_stratum_n:
            per_slot[j] = {"verdict": "INSUFFICIENT_N",
                           "n_prefix_ok": (s["prefix_ok"]["n"] if s else 0)}
            continue
        acc_ok = s["prefix_ok"]["acc"]
        acc_bad = (s["prefix_bad"]["acc"]
                   if s["prefix_bad"]["n"] >= min_stratum_n else None)
        if acc_ok >= VERDICT_PROP_ACC:
            v = "ERROR_PROPAGATION (exposure-bias-like; fix = DAgger-style " \
                "training / error-tolerant supervision)"
        elif acc_ok < VERDICT_COLLAPSE_ACC:
            v = "SLOT_DEPTH_COLLAPSE (capacity / exposure-of-slot-j; " \
                "fix = more deep-stage training)"
        else:
            v = "MIXED/INCONCLUSIVE"
        per_slot[j] = {"verdict": v, "acc_prefix_ok": acc_ok,
                       "acc_prefix_bad": acc_bad,
                       "n_prefix_ok": s["prefix_ok"]["n"],
                       "n_prefix_bad": s["prefix_bad"]["n"]}
    return per_slot


# --------------------------------------------------------------------------- #
# Per-rung evaluation.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_rung(model, tok, thinking_id: int, prefix: str, K: int,
              n_per_rung: int, device, tag: str = "") -> dict | None:
    recs = load_rung(prefix, K)[:n_per_rung]
    if not recs:
        print(f"  [{tag}] rung {K}: no data at {prefix}_n{K}.jsonl -- skipped")
        return None
    per_record = []
    for rec in recs:
        with_trace_prompt, _ = build_prompts(rec)
        prompt_ids = tok.encode(with_trace_prompt, add_special_tokens=False)
        inter_tok = encode_inter_token_ids(
            tok, list(rec.get("intermediates", [])))
        per_record.append(indexed_perhop_reads(
            model, prompt_ids, K, thinking_id, inter_tok, device))
    strata = stratify(per_record, K)
    return {"K": K, "n": len(recs), "strata": strata,
            "summary": summarize_strata(strata)}


def print_rung_table(title: str, rung: dict):
    print(f"\n--- {title} (K={rung['K']}, n={rung['n']}) ---")
    hdr = (f"{'slot':>4} | {'uncond':>26} | {'acc|prefix OK':>26} | "
           f"{'acc|prefix WRONG':>26} | {'excl':>4}")
    print(hdr)
    print("-" * len(hdr))
    for j, s in sorted(rung["summary"].items()):
        def _c(d):
            if not d["n"]:
                return f"{'n/a':>26}"
            lo, hi = d["wilson95"]
            return f"{d['acc']:.3f} [{lo:.2f},{hi:.2f}] n={d['n']}".rjust(26)
        print(f"{j:>4} | {_c(s['uncond'])} | {_c(s['prefix_ok'])} | "
              f"{_c(s['prefix_bad'])} | {s['n_excluded']:>4}")


def apply_smoke(args) -> None:
    """--smoke: 20 records, K=6 only, no lengen."""
    args.rungs = "6"
    args.n_per_rung = 20
    args.no_lengen = True


def resolve_ckpt(path: str) -> str:
    if os.path.exists(path):
        return path
    if path == DEFAULT_CKPT and os.path.exists(FALLBACK_CKPT):
        print(f"[warn] {path} missing -> falling back to {FALLBACK_CKPT}")
        return FALLBACK_CKPT
    raise FileNotFoundError(path)


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--heldout_prefix", default="data/exec_trace_heldout")
    ap.add_argument("--lengen_prefix", default="data/exec_trace_lengen_heldout")
    ap.add_argument("--rungs", default="6,7,8")
    ap.add_argument("--lengen_rungs", default="9,10,12")
    ap.add_argument("--no_lengen", action="store_true")
    ap.add_argument("--n_per_rung", type=int, default=200)
    ap.add_argument("--min_stratum_n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/probe_latent_exposure_bias.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        apply_smoke(args)

    assert torch.cuda.is_available() or args.device == "cpu", "needs CUDA"
    t0 = time.time()
    ckpt = resolve_ckpt(args.ckpt)
    model, cfg, thinking_id, tok, eos_id = load_eval_model(ckpt, args.device)
    print(f"[loaded] {ckpt} thinking_id={thinking_id} "
          f"use_latent_feedback_adapter="
          f"{getattr(model, 'use_latent_feedback_adapter', False)}", flush=True)

    jobs = [(args.heldout_prefix, int(k), "heldout")
            for k in args.rungs.split(",") if k.strip()]
    if not args.no_lengen:
        jobs += [(args.lengen_prefix, int(k), "lengen")
                 for k in args.lengen_rungs.split(",") if k.strip()]

    rung_results = []
    for prefix, K, tag in jobs:
        r = eval_rung(model, tok, thinking_id, prefix, K, args.n_per_rung,
                      args.device, tag=tag)
        if r is not None:
            r["tag"] = tag
            rung_results.append(r)
            print_rung_table(f"{tag} {prefix}", r)
            print(f"  [{tag}] rung {K} done ({time.time() - t0:.0f}s elapsed)",
                  flush=True)

    pooled = merge_strata([r["strata"] for r in rung_results])
    pooled_summary = summarize_strata(pooled)
    print_rung_table("POOLED across all rungs",
                     {"K": max((r["K"] for r in rung_results), default=0),
                      "n": sum(r["n"] for r in rung_results),
                      "summary": pooled_summary})

    verdict = compute_verdict(pooled_summary, args.min_stratum_n)
    print("\n=== PRE-REGISTERED READ (pooled, cliff slots"
          f" {list(VERDICT_SLOTS)}) ===")
    print(json.dumps(verdict, indent=2))

    results = {
        "ckpt": ckpt, "n_per_rung": args.n_per_rung,
        "min_stratum_n": args.min_stratum_n,
        "rungs": [{"tag": r["tag"], "K": r["K"], "n": r["n"],
                   "summary": r["summary"]} for r in rung_results],
        "pooled_summary": pooled_summary,
        "verdict": verdict,
        "runtime_s": time.time() - t0,
    }
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[saved] {args.out}")
    print(f"[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
