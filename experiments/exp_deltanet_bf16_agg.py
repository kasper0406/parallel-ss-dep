"""Aggregate the bf16-regime dense-LM probe: is the per-head-NS - Muon gap in
bf16 ~= the fp32 gap, smaller, or gone?

Primary metric = val CE at a FIXED step (iso-step), per seed -> mean +/- std.
Gap(arm) = muon_CE - perhead_CE  (POSITIVE => per-head is better).
Reported (a) at MATCHED matrix-LR (isolates the optimizer; the cleanest gap)
and (b) per-arm-BEST-LR-per-seed (each optimizer at its best). Corroborated
with steps-to-CE-threshold (interpolated), like the MQAR probe.

Usage: exp_deltanet_bf16_agg.py [run_dir] [--step N|--final] [--thr X]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def ce_at_step(hist, step):
    """val CE at the eval whose 'step' == step (or the nearest <= step)."""
    best = None
    for h in hist:
        if h["step"] <= step:
            best = h
        else:
            break
    return best["val_ce"] if best else hist[0]["val_ce"]


def ce_final(hist):
    return hist[-1]["val_ce"]


def cross_step(hist, thr):
    """Interpolated step at which val_ce first drops to <= thr (descending)."""
    prev = None
    for h in hist:
        if h["val_ce"] <= thr:
            if prev is None:
                return float(h["step"])
            s0, c0 = prev["step"], prev["val_ce"]
            s1, c1 = h["step"], h["val_ce"]
            if c1 == c0:
                return float(s1)
            return s0 + (thr - c0) * (s1 - s0) / (c1 - c0)
        prev = h
    return None


def msd(vals):
    if not vals:
        return (float("nan"), float("nan"))
    m = st.mean(vals)
    s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="runs/deltanet_bf16/grid")
    ap.add_argument("--step", type=int, default=None,
                    help="iso-step for the CE metric (default: final)")
    ap.add_argument("--thr", type=float, default=None,
                    help="CE threshold for steps-to-threshold (default: auto)")
    args = ap.parse_args()

    runs = [json.load(open(f)) for f in glob.glob(os.path.join(args.run_dir, "*.json"))]
    if not runs:
        print("no runs found in", args.run_dir); return

    # index: (nh, regime, arm, lr, seed) -> run
    by = {}
    for r in runs:
        key = (r["n_heads"], r["regime"], r["arm"], r["lr_mat"], r["seed"])
        by[key] = r
    nhs = sorted({r["n_heads"] for r in runs})
    regimes = [x for x in ("fp32", "bf16") if any(r["regime"] == x for r in runs)]
    arms = ["muon", "perhead"]
    lrs = sorted({r["lr_mat"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})

    metric_step = args.step
    metric_name = f"val CE @ step {metric_step}" if metric_step is not None else "val CE @ final"

    def get_ce(r):
        return ce_final(r["history"]) if metric_step is None else ce_at_step(r["history"], metric_step)

    # auto threshold: worst (max) final CE across all runs + small margin so every
    # run crosses it (steps-to-thr defined for all), but not at the convergence floor.
    if args.thr is None:
        finals = [ce_final(r["history"]) for r in runs]
        thr = round(max(finals) + 0.05, 2)
    else:
        thr = args.thr

    for nh in nhs:
        print("\n" + "=" * 78)
        print(f"CONFIG: n_heads={nh}   (d_model=256, 4 layers)   metric = {metric_name}")
        print("=" * 78)

        # ---- per-(regime,arm,lr) mean+/-std table ----
        print(f"\n{metric_name}: mean +/- std over {len(seeds)} seeds")
        print(f"{'lr_mat':>8s} | " +
              " | ".join(f"{rg+'/'+a:>16s}" for rg in regimes for a in arms))
        print("-" * 90)
        for lr in lrs:
            row = [f"{lr:>8.0e}"]
            for rg in regimes:
                for a in arms:
                    vals = [get_ce(by[(nh, rg, a, lr, sd)])
                            for sd in seeds if (nh, rg, a, lr, sd) in by]
                    m, s = msd(vals)
                    row.append(f"{m:6.3f}+/-{s:<5.3f}")
            print(" | ".join(f"{x:>16s}" if i else x for i, x in enumerate(row)))

        # ---- gap analysis per regime ----
        print(f"\nGAP = muon - perhead  (POSITIVE => per-head better), per regime:")
        regime_gap_matched, regime_gap_best = {}, {}
        for rg in regimes:
            # (a) matched-LR gap: per seed, for each lr compute muon-perhead; pick
            #     the lr where the PAIR mean is best (lowest perhead CE), report gap.
            # Simpler & robust: per seed, choose the matched lr minimizing
            # mean(muon,perhead) CE, take the gap there.
            matched_gaps = []
            for sd in seeds:
                cand = []
                for lr in lrs:
                    km = (nh, rg, "muon", lr, sd); kp = (nh, rg, "perhead", lr, sd)
                    if km in by and kp in by:
                        cm, cp = get_ce(by[km]), get_ce(by[kp])
                        cand.append(((cm + cp) / 2, cm - cp, lr))
                if cand:
                    _, gap, lr_sel = min(cand, key=lambda x: x[0])
                    matched_gaps.append(gap)
            # (b) best-vs-best gap: per seed, muon's best-LR CE - perhead's best-LR CE
            best_gaps, muon_best, perhead_best = [], [], []
            for sd in seeds:
                cm = [get_ce(by[(nh, rg, "muon", lr, sd)]) for lr in lrs
                      if (nh, rg, "muon", lr, sd) in by]
                cp = [get_ce(by[(nh, rg, "perhead", lr, sd)]) for lr in lrs
                      if (nh, rg, "perhead", lr, sd) in by]
                if cm and cp:
                    bm, bp = min(cm), min(cp)
                    best_gaps.append(bm - bp); muon_best.append(bm); perhead_best.append(bp)
            mm, ms = msd(matched_gaps)
            bm_, bs = msd(best_gaps)
            regime_gap_matched[rg] = (mm, ms, matched_gaps)
            regime_gap_best[rg] = (bm_, bs, best_gaps)
            print(f"  [{rg}] matched-LR gap: {mm:+.4f} +/- {ms:.4f}  "
                  f"per-seed {['%+.4f' % g for g in matched_gaps]}")
            print(f"  [{rg}] best-vs-best  : {bm_:+.4f} +/- {bs:.4f}  "
                  f"per-seed {['%+.4f' % g for g in best_gaps]}  "
                  f"(muon {msd(muon_best)[0]:.3f} | perhead {msd(perhead_best)[0]:.3f})")

        # ---- side by side ----
        if "fp32" in regimes and "bf16" in regimes:
            print(f"\n  >>> fp32 vs bf16 (matched-LR gap):  "
                  f"fp32 {regime_gap_matched['fp32'][0]:+.4f}  "
                  f"bf16 {regime_gap_matched['bf16'][0]:+.4f}  "
                  f"(bf16/fp32 = {regime_gap_matched['bf16'][0]/regime_gap_matched['fp32'][0]:.2f}x)"
                  if regime_gap_matched['fp32'][0] != 0 else "")
            print(f"  >>> fp32 vs bf16 (best-vs-best gap): "
                  f"fp32 {regime_gap_best['fp32'][0]:+.4f}  "
                  f"bf16 {regime_gap_best['bf16'][0]:+.4f}")

        # ---- steps-to-threshold corroboration ----
        print(f"\nsteps-to-(val CE <= {thr}): per-arm best-LR-per-seed, mean +/- std")
        for rg in regimes:
            line = [f"  [{rg}]"]
            arm_means = {}
            for a in arms:
                per_seed = []
                for sd in seeds:
                    cand = []
                    for lr in lrs:
                        k = (nh, rg, a, lr, sd)
                        if k in by:
                            cs = cross_step(by[k]["history"], thr)
                            if cs is not None:
                                cand.append(cs)
                    if cand:
                        per_seed.append(min(cand))
                m, s = msd(per_seed)
                arm_means[a] = m
                line.append(f"{a} {m:.0f}+/-{s:.0f} (n{len(per_seed)})")
            if "muon" in arm_means and "perhead" in arm_means and arm_means["muon"]:
                d = arm_means["perhead"] - arm_means["muon"]
                line.append(f"=> per-head {d:+.0f} steps ({100*d/arm_means['muon']:+.1f}%)")
            print("   ".join(line))

        # ---- optimizer ms/step (wall, exclusive-GPU caveat applies) ----
        print(f"\nopt ms/step (mean over all runs of this config):")
        for rg in regimes:
            for a in arms:
                v = [by[(nh, rg, a, lr, sd)]["opt_ms_per_step"]
                     for lr in lrs for sd in seeds if (nh, rg, a, lr, sd) in by]
                if v:
                    print(f"  {rg}/{a:8s}: {st.mean(v):.2f} ms")


if __name__ == "__main__":
    main()
