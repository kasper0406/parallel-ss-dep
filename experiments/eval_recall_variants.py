"""Eval the recall-cotrained ckpt on the held-out GENERALIZATION variants
(experiments/gen_recall_variants.py). OURS-only, bf16, TRUE incremental decode
(prefill + forward_step) — the same path the scoreboard uses. Leak-free /
first-occurrence by construction of the task. Reports per (axis, variant, L)
gen_acc (strict value match) + top1 (gold first-token == post-prompt argmax).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \
    experiments/eval_recall_variants.py --ckpt checkpoints/recall_cotrain/recall_cotrain.pt \
    --tasks data/recall_variants_heldout.jsonl --out checkpoints/recall_cotrain/eval_variants_before.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

_INT4 = re.compile(r"\b\d{4}\b")
_INTANY = re.compile(r"\d+")
_WORD = re.compile(r"[a-z]+")


def extract(text: str, match: str):
    if match == "int4":
        m = _INT4.search(text)
        if m:
            return m.group(0)
        m = _INTANY.search(text)
        return m.group(0) if m else None
    if match == "int_any":
        m = _INTANY.search(text)
        return m.group(0) if m else None
    if match == "string":
        t = text.lstrip().lstrip("'\"")
        m = _WORD.match(t)
        return m.group(0) if m else None
    raise ValueError(match)


def first_gold_token_in_context(tok, prompt_ids, prompt: str, gold: str):
    """The gold's first token encoded STANDALONE (e.g. 'zqfk' -> 'z') is NOT
    what a correctly-completing model actually emits: every prompt here ends
    with a cue that terminates in whitespace (`# `, `== `, `is `, ...), and
    the tokenizer MERGES that trailing space with the gold's leading
    character(s) into one token (e.g. '# zqfk' -> ['#', 'Ġz', 'q', 'f', 'k'],
    the SmolLM2 tokenizer's 'Ġz' = id 1892) -- so comparing against the
    standalone 'z' token scores a correct completion as wrong. Fix: tokenize
    prompt+gold IN CONTEXT and diff against the already-tokenized prompt
    (`prompt_ids`) to find the true first "new" token at the boundary.
    Robust to either whitespace-merging (letters) or already-isolated
    space+token pairs (digits, where the merge doesn't happen and this is a
    no-op vs standalone encoding). Returns None if degenerate (empty gold)."""
    full_ids = tok.encode(prompt + gold, add_special_tokens=False)
    k = 0
    n = min(len(prompt_ids), len(full_ids))
    while k < n and prompt_ids[k] == full_ids[k]:
        k += 1
    return full_ids[k] if k < len(full_ids) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--max_gen", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import experiments.decode_bench as db
    from transformers import AutoTokenizer
    from experiments.scoreboard_longctx_cost import _ours_greedy

    recs = [json.loads(l) for l in open(args.tasks) if l.strip()]
    ours, cfg = db.load_ours(args.ckpt)
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or "HuggingFaceTB/SmolLM2-135M")
    ours.eval()

    agg = {}  # (axis,variant,L) -> [gen_ok, top1_ok, n]
    for r in recs:
        ids = tok.encode(r["prompt"], add_special_tokens=False)
        out, first_logits = _ours_greedy(ours, ids, args.max_gen)
        text = tok.decode(out, skip_special_tokens=True)
        pred = extract(text, r["match"])
        gen_ok = int(pred == r["gold"])
        gold_first = first_gold_token_in_context(tok, ids, r["prompt"], r["gold"])
        top1 = int(gold_first is not None
                    and int(first_logits.argmax().item()) == gold_first)
        k = (r["axis"], r["variant"], r["bucket"])
        a = agg.setdefault(k, [0, 0, 0])
        a[0] += gen_ok; a[1] += top1; a[2] += 1

    # ---- report ----
    out = {"ckpt": args.ckpt, "tasks": args.tasks, "per": {}}
    # order: by axis, variant, then bucket
    def key(k):
        order_axis = {"value": 0, "cue": 1, "distractor": 2}
        return (order_axis.get(k[0], 9), k[1], k[2])
    print(f"\n{'axis':<11}{'variant':<11}{'L':>6} | {'gen_acc':>8} {'top1':>6}  (n)")
    print("-" * 52)
    last_av = None
    for k in sorted(agg, key=key):
        g, t, n = agg[k]
        av = (k[0], k[1])
        if last_av is not None and av != last_av:
            print()
        last_av = av
        ga, t1 = 100 * g / n, 100 * t / n
        print(f"{k[0]:<11}{k[1]:<11}{k[2]:>6} | {ga:7.0f}% {t1:5.0f}%  ({n})")
        out["per"][f"{k[0]}/{k[1]}/{k[2]}"] = {"gen_acc": g / n, "top1": t / n, "n": n}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
