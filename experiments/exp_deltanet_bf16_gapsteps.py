"""Definitive 'does the per-head gap survive bf16' view.

The dense-LM convergence SATURATES (both arms reach the same floor by step 599),
so -- exactly like the MQAR probe -- the discriminating signal is convergence
SPEED, not the converged value. This reports:

  (1) GAP-vs-STEP: gap(step) = muon_CE - perhead_CE at MATCHED matrix-LR
      (positive => per-head ahead), mean +/- std over seeds, at several steps,
      for fp32 and bf16 side by side. The cleanest 'does it survive' view.
  (2) steps-to-CE-threshold at several steep-region thresholds, per arm,
      best-LR-per-seed, mean +/- std, with the per-head step-delta and %.

Per-seed numbers are printed because the gap is small.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys


def ce_at(hist, step):
    out = hist[0]["val_ce"]
    for h in hist:
        if h["step"] <= step:
            out = h["val_ce"]
        else:
            break
    return out


def cross_step(hist, thr):
    prev = None
    for h in hist:
        if h["val_ce"] <= thr:
            if prev is None:
                return float(h["step"])
            s0, c0 = prev["step"], prev["val_ce"]
            s1, c1 = h["step"], h["val_ce"]
            return s1 if c1 == c0 else s0 + (thr - c0) * (s1 - s0) / (c1 - c0)
        prev = h
    return None


def msd(v):
    if not v:
        return float("nan"), float("nan")
    return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/deltanet_bf16/grid"
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(run_dir, "*.json"))]
    by = {(r["n_heads"], r["regime"], r["arm"], r["lr_mat"], r["seed"]): r for r in runs}
    nhs = sorted({r["n_heads"] for r in runs})
    regimes = [x for x in ("fp32", "bf16") if any(r["regime"] == x for r in runs)]
    lrs = sorted({r["lr_mat"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})

    STEPS = [40, 80, 120, 200, 300, 400, 599]
    THRS = [5.5, 5.2, 5.0, 4.8, 4.7]

    for nh in nhs:
        print("\n" + "=" * 80)
        print(f"CONFIG n_heads={nh}  (d_model=256, 4L)   seeds={seeds}  lrs={lrs}")
        print("=" * 80)

        # (1) GAP-vs-STEP, matched LR (per seed pick lr minimizing mean(m,p) CE
        #     AT THAT STEP), mean+/-std over seeds, fp32 vs bf16.
        print("\n(1) GAP = muon_CE - perhead_CE at matched-LR  (+ => per-head ahead)")
        hdr = f"{'step':>5} | " + " | ".join(f"{rg:>22s}" for rg in regimes)
        print(hdr); print("-" * len(hdr))
        for stp in STEPS:
            cells = []
            for rg in regimes:
                gaps = []
                for sd in seeds:
                    cand = []
                    for lr in lrs:
                        km = (nh, rg, "muon", lr, sd); kp = (nh, rg, "perhead", lr, sd)
                        if km in by and kp in by:
                            cm, cp = ce_at(by[km]["history"], stp), ce_at(by[kp]["history"], stp)
                            cand.append(((cm + cp) / 2, cm - cp))
                    if cand:
                        gaps.append(min(cand, key=lambda x: x[0])[1])
                m, s = msd(gaps)
                ps = ",".join(f"{g:+.3f}" for g in gaps)
                cells.append(f"{m:+.4f}+/-{s:.4f} [{ps}]")
            print(f"{stp:>5} | " + " | ".join(f"{c:>22s}" for c in cells))

        # (2) steps-to-threshold, per arm best-LR-per-seed, fp32 vs bf16.
        print("\n(2) steps-to-(val CE <= thr): per arm best-LR-per-seed, mean+/-std")
        for thr in THRS:
            print(f"  thr={thr}:")
            for rg in regimes:
                arm_m = {}
                parts = [f"    [{rg}]"]
                for a in ("muon", "perhead"):
                    per_seed = []
                    for sd in seeds:
                        cand = [cross_step(by[(nh, rg, a, lr, sd)]["history"], thr)
                                for lr in lrs if (nh, rg, a, lr, sd) in by]
                        cand = [c for c in cand if c is not None]
                        if cand:
                            per_seed.append(min(cand))
                    m, s = msd(per_seed)
                    arm_m[a] = (m, per_seed)
                    parts.append(f"{a} {m:.0f}+/-{s:.0f}")
                if arm_m.get("muon") and arm_m.get("perhead") and arm_m["muon"][0] == arm_m["muon"][0]:
                    d = arm_m["perhead"][0] - arm_m["muon"][0]
                    if arm_m["muon"][0]:
                        parts.append(f"=> per-head {d:+.1f} ({100*d/arm_m['muon'][0]:+.1f}%)")
                print("  ".join(parts))


if __name__ == "__main__":
    main()
