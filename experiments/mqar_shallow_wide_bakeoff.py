"""MQAR bake-off: 30L baseline vs 12L shallow-wide variants.

Tests the hypothesis: a shallow (L=12) wider (d=832) DeltaNet stack with
cross-layer feedback can match or beat the current 30L×576 + sparse-FiLM
production stack on saturated MQAR recall.

Five configs at T=512, K=128 (the saturation regime per CLAUDE.md):
  A  30L × 576d  + FiLM(2,28) K=3                — baseline (matches v5-pkm trunk)
  B  12L × 832d, no feedback                     — does shallow-wide alone help?
  C  12L × 832d  + FiLM(2,10) K=3                — shallow-wide + matched sparse FiLM
  D  12L × 832d  + xattn 3-targets × 4-sources [attn form]    — the key test
  E  12L × 832d  + xattn 3-targets × 4-sources [film_sum form]  — alt fusion

Metric: recall_acc on a held-out 1024-batch.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.train_mqar import train_one


CONFIGS = [
    dict(label="A_30L576_FiLM",
         d_model=576, n_layers=30, n_heads=9, d_head=64,
         feedback="film", feedback_pairs="2,28", feedback_self_k=3),
    dict(label="B_12L832_none",
         d_model=832, n_layers=12, n_heads=13, d_head=64,
         feedback="none"),
    dict(label="C_12L832_FiLM",
         d_model=832, n_layers=12, n_heads=13, d_head=64,
         feedback="film", feedback_pairs="2,10", feedback_self_k=3),
    dict(label="D_12L832_xattn",
         d_model=832, n_layers=12, n_heads=13, d_head=64,
         feedback="none",
         feedback_xattn="2:8,9,10,11;4:8,9,10,11;6:8,9,10,11",
         feedback_xattn_form="attn", feedback_xattn_heads=8),
    dict(label="E_12L832_xattn_filmsum",
         d_model=832, n_layers=12, n_heads=13, d_head=64,
         feedback="none",
         feedback_xattn="2:8,9,10,11;4:8,9,10,11;6:8,9,10,11",
         feedback_xattn_form="film_sum"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n_pairs", type=int, default=128)
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--log_every", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", type=str, default="",
                   help="Comma-sep prefix of labels to run (e.g. 'A,D'). "
                        "Empty = run all.")
    args = p.parse_args()

    only = set(args.only.split(",")) if args.only else None
    configs = CONFIGS
    if only:
        configs = [c for c in CONFIGS if c["label"].split("_")[0] in only]
    print(f"Running {len(configs)} configs at T={args.T}, K={args.n_pairs}, vocab={args.vocab}")

    results = []
    for cfg in configs:
        t0 = time.perf_counter()
        r = train_one(
            arch="deltanet",
            T=args.T, n_pairs=args.n_pairs, vocab=args.vocab,
            steps=args.steps, batch_size=args.batch,
            d_model=cfg["d_model"], n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"], d_head=cfg["d_head"],
            lr=args.lr, log_every=args.log_every, seed=args.seed,
            feedback=cfg.get("feedback", "none"),
            feedback_pairs=cfg.get("feedback_pairs", ""),
            feedback_self_k=cfg.get("feedback_self_k", 0),
            feedback_xattn=cfg.get("feedback_xattn", ""),
            feedback_xattn_form=cfg.get("feedback_xattn_form", "attn"),
            feedback_xattn_heads=cfg.get("feedback_xattn_heads", 4),
            label=cfg["label"],
            use_memory=False,
        )
        print(f"  → {cfg['label']}: recall={r.recall_acc:.3f} "
              f"val_loss={r.final_val_loss:.4f} secs={time.perf_counter()-t0:.0f}")
        results.append(r)

    print("\n" + "=" * 90)
    print(f"{'label':<28} {'params':>10} {'recall':>8} {'val_loss':>9} {'secs':>7}")
    print("-" * 90)
    for r in sorted(results, key=lambda r: -r.recall_acc):
        print(f"{r.label:<28} {r.params:>10,} {r.recall_acc:>8.3f} "
              f"{r.final_val_loss:>9.4f} {r.secs:>7.0f}")
    print("=" * 90)
    print(f"\nMQAR T={args.T}  K={args.n_pairs}")


if __name__ == "__main__":
    sys.exit(main())
