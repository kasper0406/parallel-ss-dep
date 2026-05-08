"""
Aggregate Phase 22 validation results: multi-seed + natural-text.

Usage:
    python experiments/aggregate_validation.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys


# Multi-seed (codeparrot, statement-stratified):
# Seed 0 reference comes from Phase 22 — `bench_stmt_ppl_lsem_b10_uniform.json`.
SEEDS = {
    0: "bench_stmt_ppl_lsem_b10_uniform.json",
    1: "bench_stmt_ppl_lsem_uniform_seed1.json",
    2: "bench_stmt_ppl_lsem_uniform_seed2.json",
}
# Phase 21c K=3-only and plain DN seed=0 references on codeparrot.
SEED0_REFS = [
    ("Plain DN baseline (Phase 22 ref)",
     "bench_stmt_ppl_dn_baseline_v2.json"),
    ("K=3 self-feed (Phase 21c)",
     "bench_stmt_ppl_k3_v2.json"),
]

# Natural-text (TinyStories):
TINYSTORIES = [
    ("Plain DN baseline (TinyStories)",
     "bench_filmed_ppl_217M_tinystories_dn.json"),
    ("K=3 self-feed (TinyStories)",
     "bench_filmed_ppl_217M_tinystories_k3.json"),
    ("K=3 + per-token L_sem β=1 (TinyStories)",
     "bench_filmed_ppl_217M_tinystories_k3_lsem.json"),
]


def load_stmt(path: pathlib.Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    s = d["summary"]
    return {
        "ppl": s["overall"]["ppl"],
        "top10": s.get("top_decile", {}).get("ppl", float("nan")),
        "bot10": s.get("bottom_decile", {}).get("ppl", float("nan")),
    }


def load_filmed(path: pathlib.Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {
        "ppl_train": d["training_protocol"]["ppl"],
        "ppl_2pass": d.get("2pass", {}).get("ppl", float("nan")),
        "ppl_lc": d.get("lagged_cached", {}).get("ppl", float("nan")),
    }


def stat(vals):
    """Mean and standard deviation (sample, ddof=1)."""
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return m, s


def main():
    base = pathlib.Path("/home/knielsen/ml/parallel-ss-dep")
    print("=" * 80)
    print("Multi-seed: K=3 + uniform L_sem β=1.0 on codeparrot (statement-stratified)")
    print("=" * 80)
    print(f"{'Seed':<8} {'PPL':>10} {'Top10':>10} {'Bot10':>10}")
    print("-" * 80)
    rows = []
    for seed, fname in SEEDS.items():
        path = base / fname
        d = load_stmt(path)
        if d is None:
            print(f"seed={seed:<3} {'MISS':>10} {'MISS':>10} {'MISS':>10}  ({fname})")
            continue
        rows.append((seed, d))
        print(f"seed={seed:<3} {d['ppl']:>10.4f} {d['top10']:>10.4f} {d['bot10']:>10.4f}")
    if rows:
        ppls = [d["ppl"] for _, d in rows]
        tops = [d["top10"] for _, d in rows]
        bots = [d["bot10"] for _, d in rows]
        m_ppl, s_ppl = stat(ppls)
        m_top, s_top = stat(tops)
        m_bot, s_bot = stat(bots)
        print("-" * 80)
        print(f"{'mean':<8} {m_ppl:>10.4f} {m_top:>10.4f} {m_bot:>10.4f}")
        print(f"{'σ':<8} {s_ppl:>10.4f} {s_top:>10.4f} {s_bot:>10.4f}")
        if m_ppl > 0:
            print(f"{'σ/mean %':<8} {100*s_ppl/m_ppl:>10.4f} "
                  f"{100*s_top/m_top:>10.4f} {100*s_bot/m_bot:>10.4f}")
    print()
    print("Codeparrot reference (seed=0):")
    for label, fname in SEED0_REFS:
        path = base / fname
        d = load_stmt(path)
        if d is None:
            print(f"  {label}: MISS ({fname})")
            continue
        print(f"  {label}: PPL={d['ppl']:.4f}  Top10={d['top10']:.4f}  Bot10={d['bot10']:.4f}")
    print()

    print("=" * 80)
    print("Natural-text (TinyStories): held-out PPL")
    print("=" * 80)
    print(f"{'Variant':<48} {'PPL_train':>12} {'PPL_2pass':>12} {'PPL_lc':>12}")
    print("-" * 80)
    ts_rows = []
    for label, fname in TINYSTORIES:
        path = base / fname
        d = load_filmed(path)
        if d is None:
            print(f"{label:<48} {'MISS':>12} {'MISS':>12} {'MISS':>12}  ({fname})")
            continue
        ts_rows.append((label, d))
        print(f"{label:<48} {d['ppl_train']:>12.4f} {d['ppl_2pass']:>12.4f} {d['ppl_lc']:>12.4f}")

    # Δ vs DN baseline.
    if len(ts_rows) >= 1:
        dn = ts_rows[0][1]["ppl_train"]
        print()
        print("Δ vs Plain DN baseline (training-protocol PPL):")
        for label, d in ts_rows:
            delta = (d["ppl_train"] - dn) / dn * 100
            print(f"  {label:<48} {delta:+.2f} %")

    return 0


if __name__ == "__main__":
    sys.exit(main())
