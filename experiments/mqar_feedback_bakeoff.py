"""MQAR architectural feedback bake-off — tiny scale, fast turnaround.

Tests the cross-layer-feedback hypothesis on a regime where DeltaNet
ACTUALLY learns (T=256, K=32, vocab=128 — the original 4L MQAR sweet
spot). All configs are iso-shape (4L × 384d) so the only variable is
the feedback mechanism. Reports final recall and the final α magnitudes
(so we can tell whether the gate parameters even moved).

Configs:
  base    4L × 384d, no feedback                            — baseline
  xattn   4L × 384d + xattn(attn form)                       — scalar α gate
  fsig    4L × 384d + xattn(film_sigmoid form)               — per-token σ gates
  film3   4L × 384d + 3 FiLM pairs: (0,2),(1,3),(0,3)        — the fallback
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
    dict(label="base",
         d_model=384, n_layers=4, n_heads=6, d_head=64,
         feedback="none"),
    dict(label="xattn_attn",
         d_model=384, n_layers=4, n_heads=6, d_head=64,
         feedback="none",
         feedback_xattn="0:1,2,3;1:2,3",
         feedback_xattn_form="attn", feedback_xattn_heads=6),
    dict(label="xattn_filmsigmoid",
         d_model=384, n_layers=4, n_heads=6, d_head=64,
         feedback="none",
         feedback_xattn="0:1,2,3;1:2,3",
         feedback_xattn_form="film_sigmoid", feedback_xattn_heads=6),
    dict(label="film_x3",
         d_model=384, n_layers=4, n_heads=6, d_head=64,
         feedback="film", feedback_pairs="0,2;1,3;0,3"),
]


def _alpha_summary(model) -> str:
    """Return a compact dump of every learnable α-like scalar in the model."""
    out = []
    for name, p in model.named_parameters():
        if name.endswith(".alpha") or "alpha" in name.split("."):
            v = p.detach()
            if v.numel() == 1:
                out.append(f"{name}={float(v):+.3f}")
            else:
                out.append(f"{name}|mean|={float(v.abs().mean()):+.3f}")
    return ",".join(out) if out else "(no alpha params)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--n_pairs", type=int, default=32)
    p.add_argument("--vocab", type=int, default=128)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"MQAR T={args.T} K={args.n_pairs} V={args.vocab}  "
          f"steps={args.steps} batch={args.batch} lr={args.lr}")

    results = []
    for cfg in CONFIGS:
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
              f"val_loss={r.final_val_loss:.4f} "
              f"secs={time.perf_counter()-t0:.0f}")
        results.append(r)

    print("\n" + "=" * 78)
    print(f"{'label':<20} {'params':>10} {'recall':>8} {'val_loss':>9} {'secs':>7}")
    print("-" * 78)
    for r in sorted(results, key=lambda r: -r.recall_acc):
        print(f"{r.label:<20} {r.params:>10,} {r.recall_acc:>8.3f} "
              f"{r.final_val_loss:>9.4f} {r.secs:>7.0f}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
