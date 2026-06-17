"""Analyze bench_optimizer.py JSON outputs → fair comparison table.

Reports, per arm: final val, ms/step (pure train), opt_ms/step, peak mem.
Picks best-LR arm per optimizer (lowest final val). For the best Muon vs
best SOAP, computes steps-to-target and SECONDS-to-target (= steps ×
pure-train ms/step, eval excluded) for several target val-loss levels.
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load(out_dir):
    arms = []
    for fp in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        with open(fp) as f:
            arms.append(json.load(f))
    return arms


def val_curve(arm):
    """(step, val_loss) pairs, sorted by step."""
    pts = [(h["step"], h["val_loss"]) for h in arm["history"]
           if h.get("val_loss") is not None]
    return sorted(pts)


def steps_to_target(arm, target):
    """First step whose val_loss <= target (linear interp between evals)."""
    pts = val_curve(arm)
    prev = None
    for step, vl in pts:
        if vl <= target:
            if prev is None:
                return step
            ps, pv = prev
            if pv == vl:
                return step
            frac = (pv - target) / (pv - vl)
            return ps + frac * (step - ps)
        prev = (step, vl)
    return None


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/bench_optim"
    arms = [a for a in load(out_dir)
            if a["tag"].startswith(("muon_", "soap_", "adamw_"))]
    if not arms:
        print(f"no arm JSONs in {out_dir}")
        return

    print(f"\n{'tag':18s} {'opt':6s} {'lr_mat':>9s} {'final_val':>9s} "
          f"{'ms/step':>8s} {'opt_ms':>7s} {'mem_GiB':>8s}")
    print("-" * 78)
    for a in sorted(arms, key=lambda a: (a["optimizer"], a["lr_muon"])):
        print(f"{a['tag']:18s} {a['optimizer']:6s} {a['lr_muon']:9.1e} "
              f"{a['final_val_loss']:9.4f} {a['ms_per_step']:8.1f} "
              f"{a['opt_ms_per_step']:7.2f} {a['peak_mem_gib']:8.2f}")

    # Best LR per optimizer (lowest final val).
    best = {}
    for a in arms:
        o = a["optimizer"]
        if o not in best or a["final_val_loss"] < best[o]["final_val_loss"]:
            best[o] = a
    print("\nBest-LR arm per optimizer (by final val):")
    for o, a in best.items():
        print(f"  {o:6s}  {a['tag']}  lr_mat={a['lr_muon']:.1e}  "
              f"final_val={a['final_val_loss']:.4f}")

    if "muon" in best and "soap" in best:
        m, s = best["muon"], best["soap"]
        print(f"\nSteps / SECONDS to target val-loss (pure train time; "
              f"sec = steps × ms/step):")
        print(f"  muon ms/step={m['ms_per_step']:.1f}  "
              f"soap ms/step={s['ms_per_step']:.1f}")
        print(f"\n  {'target':>7s} | {'muon_steps':>10s} {'muon_sec':>9s} | "
              f"{'soap_steps':>10s} {'soap_sec':>9s} | {'step_x':>7s} {'wall_x':>7s}")
        print("  " + "-" * 78)
        # Data-driven targets in the meaningful low-loss region: from just
        # above the worst of the two BEST finals, spanning up to the easy zone.
        worst_final = max(m["final_val_loss"], s["final_val_loss"])
        targets = sorted({round(worst_final + d, 3) for d in
                          [0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.5, 0.75,
                           1.1, 1.6, 2.4]}, reverse=True)
        for t in targets:
            ms_ = steps_to_target(m, t)
            ss_ = steps_to_target(s, t)
            msec = ms_ * m["ms_per_step"] / 1e3 if ms_ is not None else None
            ssec = ss_ * s["ms_per_step"] / 1e3 if ss_ is not None else None
            step_x = (ms_ / ss_) if (ms_ and ss_) else None
            wall_x = (msec / ssec) if (msec and ssec) else None
            def f(v, n=1):
                return f"{v:.{n}f}" if v is not None else "  --  "
            print(f"  {t:7.2f} | {f(ms_,0):>10s} {f(msec):>9s} | "
                  f"{f(ss_,0):>10s} {f(ssec):>9s} | "
                  f"{f(step_x,2):>7s} {f(wall_x,2):>7s}")
        print("\n  step_x = muon_steps/soap_steps (>1 → SOAP fewer steps)")
        print("  wall_x = muon_sec/soap_sec   (>1 → SOAP less wall-clock)")


if __name__ == "__main__":
    main()
