"""Multi-seed aggregation: mean +/- std steps-to-threshold per arm.

Convergence on MQAR T256/K32 SATURATES (recall->1.0), so we rank by SPEED:
the step (interpolated) at which val_recall first crosses a threshold. We
report per-(arm,lr) mean+/-std over seeds, and the per-arm BEST-LR result both
iso-step and iso-wallclock.
"""
from __future__ import annotations
import glob
import json
import os
import statistics as st
import sys


def cross_step(hist, thr):
    """Linear-interpolated step at which val_recall first reaches thr."""
    prev = None
    for h in hist:
        if h["val_recall"] >= thr:
            if prev is None:
                return float(h["step"])
            s0, r0 = prev["step"], prev["val_recall"]
            s1, r1 = h["step"], h["val_recall"]
            if r1 == r0:
                return float(s1)
            return s0 + (thr - r0) * (s1 - s0) / (r1 - r0)
        prev = h
    return None


def cross_wall(hist, thr):
    prev = None
    for h in hist:
        if h["val_recall"] >= thr:
            if prev is None:
                return float(h["wall"])
            s0, r0, w0 = prev["step"], prev["val_recall"], prev["wall"]
            s1, r1, w1 = h["step"], h["val_recall"], h["wall"]
            if r1 == r0:
                return float(w1)
            frac = (thr - r0) / (r1 - r0)
            return w0 + frac * (w1 - w0)
        prev = h
    return None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "runs/deltanet_precond_ms"
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.90
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(d, "*.json"))]
    # group by (arm, lr) -> per seed
    cells = {}
    opt_ms = {}
    for r in runs:
        key = (r["arm"], r["lr_mat"])
        cs = cross_step(r["history"], thr)
        cw = cross_wall(r["history"], thr)
        cells.setdefault(key, []).append((r["seed"], cs, cw))
        opt_ms.setdefault(r["arm"], []).append(r["opt_ms_per_step"])

    arms = ["muon", "perhead", "qk_coupled"]
    lrs = sorted({k[1] for k in cells})
    print(f"steps-to-recall>={thr}: mean +/- std over seeds (n per cell shown)\n")
    print(f"{'lr_mat':>8s} | " + " | ".join(f"{a:>16s}" for a in arms))
    print("-" * 70)
    for lr in lrs:
        row = [f"{lr:>8.0e}"]
        for a in arms:
            vals = [c[1] for c in cells.get((a, lr), []) if c[1] is not None]
            if vals:
                m = st.mean(vals)
                s = st.pstdev(vals) if len(vals) > 1 else 0.0
                row.append(f"{m:6.0f}+/-{s:<4.0f}(n{len(vals)})")
            else:
                row.append(f"{'—':>16s}")
        print(" | ".join(f"{x:>16s}" if i else x for i, x in enumerate(row)))

    # Per-arm BEST: for each seed pick that seed's fastest LR (fair per-seed
    # LR selection), then average those best crossing-steps over seeds.
    print(f"\n=== PER-ARM BEST (per-seed fastest LR; mean over seeds) ===")
    print(f"{'arm':12s} {'steps(iso)':>16s} {'wall_s(iso)':>16s} {'opt_ms/step':>12s}")
    print("-" * 60)
    seeds = sorted({c[0] for k in cells for c in cells[k]})
    best = {}
    for a in arms:
        per_seed_step, per_seed_wall = [], []
        for sd in seeds:
            cand = []
            for lr in lrs:
                for (seed, cs, cw) in cells.get((a, lr), []):
                    if seed == sd and cs is not None:
                        cand.append((cs, cw))
            if cand:
                bs, bw = min(cand, key=lambda x: x[0])
                per_seed_step.append(bs)
                per_seed_wall.append(bw)
        ms = st.mean(per_seed_step)
        ss = st.pstdev(per_seed_step) if len(per_seed_step) > 1 else 0.0
        mw = st.mean(per_seed_wall)
        sw = st.pstdev(per_seed_wall) if len(per_seed_wall) > 1 else 0.0
        om = st.mean(opt_ms[a])
        best[a] = (ms, ss, mw, sw, om)
        print(f"{a:12s} {f'{ms:.0f}+/-{ss:.0f}':>16s} "
              f"{f'{mw:.2f}+/-{sw:.2f}':>16s} {om:>12.2f}")

    if "muon" in best:
        bm = best["muon"]
        print("\n=== vs MUON (per-arm best LR per seed) ===")
        for a in ("perhead", "qk_coupled"):
            ba = best[a]
            d_step = ba[0] - bm[0]
            d_wall = ba[2] - bm[2]
            print(f"{a:12s}: iso-step {d_step:+.0f} steps "
                  f"({100*d_step/bm[0]:+.0f}%)   "
                  f"iso-wallclock {d_wall:+.2f}s ({100*d_wall/bm[2]:+.0f}%)")


if __name__ == "__main__":
    main()
