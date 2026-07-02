"""OURS-only recall + HumanEval-CE eval for the recall-cotrain probe.

Two axes, one ckpt:
  RECALL  : scoreboard gen_acc / top1 vs distance L, via the EXACT scoreboard
            machinery (decode_bench.load_ours bf16 + the TRUE incremental decode
            prefill+forward_step + the shared `print(vQ)  # ` elicitation + the
            first-4-digit extractor). Single-binding by default (cleanest test).
  KNOWLEDGE: HumanEval-solution CE, fp32, CE only on the canonical-solution
            tokens — the SAME protocol that produced the linearized base's 0.759.

Run BEFORE on checkpoints/linearize/linearized_stage3.pt and AFTER on the
recall-cotrained ckpt; compare.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \
    experiments/eval_recall_ours.py --ckpt <ckpt> --tag before \
    --buckets 64,512,2048,4096,8192 --n_keys 1 --per_bucket 30 \
    --out checkpoints/recall_cotrain/eval_before.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


def run_recall(ckpt, tasks_path, buckets, n_keys, per_bucket, keys_per_task,
               max_gen, seed):
    import experiments.decode_bench as db
    from transformers import AutoTokenizer
    from experiments.scoreboard_longctx_cost import (
        gen_tasks, load_tasks, run_ours_quality, _acc)

    tok_name = "HuggingFaceTB/SmolLM2-135M"
    if not pathlib.Path(tasks_path).exists():
        gen_tasks(tasks_path, buckets, per_bucket, n_keys, seed, tok_name)
    tasks = load_tasks(tasks_path)

    ours, cfg = db.load_ours(ckpt)
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer") or tok_name)
    pb, _ = run_ours_quality(ours, tok, tasks, keys_per_task=keys_per_task,
                             max_gen=max_gen, max_problems_per_bucket=per_bucket)
    res = {}
    for L in buckets:
        res[str(L)] = {"gen_acc": _acc(pb, L, 0), "top1": _acc(pb, L, 1),
                       "n": pb[L][2] if L in pb else 0}
    del ours
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return res


def run_he_ce(ckpt):
    """fp32 HumanEval-solution CE — identical to /tmp/probe_he_ce.py (the 0.759
    protocol): CE only on the canonical-solution tokens."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from experiments.eval_bracket_structure import build_model_from_ckpt

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model, cfg = build_model_from_ckpt(ckpt)
    model.eval()
    ds = load_dataset("openai_humaneval", split="test")
    tot_ce, tot_tok = 0.0, 0

    @torch.no_grad()
    def ce_on_solution(prompt, sol):
        pids = tok.encode(prompt, add_special_tokens=False)
        fids = tok.encode(prompt + sol, add_special_tokens=False)
        if len(fids) <= len(pids) + 1:
            return None
        x = torch.tensor([fids], device="cuda")
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        lp = logits[0, :-1, :]
        tgt = x[0, 1:]
        start = max(len(pids) - 1, 0)
        ce = F.cross_entropy(lp[start:].float(), tgt[start:], reduction="sum")
        return ce.item(), tgt[start:].numel()

    for ex in ds:
        r = ce_on_solution(ex["prompt"], ex["canonical_solution"])
        if r:
            tot_ce += r[0]; tot_tok += r[1]
    del model
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return tot_ce / tot_tok, tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--buckets", default="64,512,2048,4096,8192")
    ap.add_argument("--n_keys", type=int, default=1)
    ap.add_argument("--per_bucket", type=int, default=30)
    ap.add_argument("--keys_per_task", type=int, default=1)
    ap.add_argument("--max_gen", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--no_recall", action="store_true")
    ap.add_argument("--no_he", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    tasks_path = args.tasks or (
        f"checkpoints/recall_cotrain/tasks_nk{args.n_keys}_pb{args.per_bucket}_"
        f"{'-'.join(str(b) for b in buckets)}.jsonl")

    out = {"ckpt": args.ckpt, "tag": args.tag, "buckets": buckets,
           "n_keys": args.n_keys, "per_bucket": args.per_bucket}

    if not args.no_recall:
        print(f"[recall] {args.ckpt}  n_keys={args.n_keys} buckets={buckets}")
        out["recall"] = run_recall(
            args.ckpt, tasks_path, buckets, args.n_keys, args.per_bucket,
            args.keys_per_task, args.max_gen, args.seed)
        for L in buckets:
            r = out["recall"][str(L)]
            g = "  -  " if r["gen_acc"] is None else f"{100*r['gen_acc']:4.0f}%"
            t = "  -  " if r["top1"] is None else f"{100*r['top1']:4.0f}%"
            print(f"   L={L:>6}  gen_acc={g}  top1={t}  (n={r['n']})")

    if not args.no_he:
        print(f"[humaneval-CE] {args.ckpt}")
        ce, nt = run_he_ce(args.ckpt)
        out["humaneval_solution_ce"] = ce
        out["humaneval_solution_tokens"] = nt
        print(f"   HumanEval-solution CE = {ce:.4f}  ({nt} sol tokens)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {args.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
