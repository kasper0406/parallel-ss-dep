"""Aggregate the DeltaNet-precond sweep: per-arm BEST by convergence speed.

Discriminating metric on MQAR T256/K32: since the task SATURATES to recall=1.0
within the step budget, we rank by SPEED -- steps (and wall-clock) to reach a
recall threshold -- not final recall. Reports, per arm, the LR that reaches the
threshold fastest (iso-step) AND the wall-clock to threshold (iso-wallclock,
which folds in the optimizer's per-step cost).
"""
from __future__ import annotations
import glob
import json
import os
import sys


def steps_to(hist, thr):
    """First (step, wall) at which val_recall >= thr; (None, None) if never."""
    for h in hist:
        if h["val_recall"] >= thr:
            return h["step"], h["wall"]
    return None, None


def recall_at(hist, step):
    best = None
    for h in hist:
        if h["step"] <= step:
            best = h["val_recall"]
    return best


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "runs/deltanet_precond"
    seed = sys.argv[2] if len(sys.argv) > 2 else "s0"
    files = sorted(glob.glob(os.path.join(d, f"*_{seed}.json")))
    runs = [json.load(open(f)) for f in files]
    if not runs:
        print("no runs found in", d, "for", seed); return
    thr = 0.90
    by_arm = {}
    print(f"{'tag':28s} {'arm':11s} {'lr_mat':>8s} {'fin_rec':>7s} "
          f"{f'st>={thr}':>9s} {'wall_s':>7s} {'opt_ms':>7s} {'ms/st':>6s}")
    print("-" * 92)
    for r in sorted(runs, key=lambda r: (r["arm"], r["lr_mat"])):
        st, wl = steps_to(r["history"], thr)
        rec = {"tag": r["tag"], "arm": r["arm"], "lr_mat": r["lr_mat"],
               "fin": r["final_val_recall"], "best": r["best_val_recall"],
               "step_thr": st, "wall_thr": wl,
               "opt_ms": r["opt_ms_per_step"], "ms": r["ms_per_step"],
               "hist": r["history"]}
        by_arm.setdefault(r["arm"], []).append(rec)
        print(f"{r['tag']:28s} {r['arm']:11s} {r['lr_mat']:>8.0e} "
              f"{r['final_val_recall']:>7.3f} "
              f"{(str(st) if st is not None else '—'):>9s} "
              f"{(f'{wl:.1f}' if wl is not None else '—'):>7s} "
              f"{r['opt_ms_per_step']:>7.2f} {r['ms_per_step']:>6.1f}")

    print("\n=== PER-ARM BEST (fastest LR to reach recall>=%.2f) ===" % thr)
    print(f"{'arm':11s} {'best_lr':>8s} {'steps':>6s} {'wall_s':>7s} "
          f"{'opt_ms':>7s} {'rec@500':>8s} {'rec@1000':>9s} {'rec@1500':>9s}")
    print("-" * 80)
    summary = {}
    for arm, recs in by_arm.items():
        # rank by steps-to-threshold (fewer is better); never-reached sorts last
        reached = [x for x in recs if x["step_thr"] is not None]
        if reached:
            best = min(reached, key=lambda x: x["step_thr"])
        else:
            best = max(recs, key=lambda x: x["best"])
        h = best["hist"]
        summary[arm] = best
        print(f"{arm:11s} {best['lr_mat']:>8.0e} "
              f"{(str(best['step_thr']) if best['step_thr'] else '—'):>6s} "
              f"{(f'{best['wall_thr']:.1f}' if best['wall_thr'] else '—'):>7s} "
              f"{best['opt_ms']:>7.2f} "
              f"{(f'{recall_at(h,500):.3f}' if recall_at(h,500) is not None else '—'):>8s} "
              f"{(f'{recall_at(h,1000):.3f}' if recall_at(h,1000) is not None else '—'):>9s} "
              f"{(f'{recall_at(h,1500):.3f}' if recall_at(h,1500) is not None else '—'):>9s}")

    if "muon" in summary:
        mu = summary["muon"]
        print("\n=== vs MUON baseline ===")
        for arm in ("perhead", "qk_coupled"):
            if arm not in summary:
                continue
            a = summary[arm]
            ds = (mu["step_thr"] - a["step_thr"]) if (mu["step_thr"] and a["step_thr"]) else None
            dw = (mu["wall_thr"] - a["wall_thr"]) if (mu["wall_thr"] and a["wall_thr"]) else None
            print(f"{arm:11s}: iso-step Δ(steps-to-{thr}) = "
                  f"{(f'{ds:+d} ({100*ds/mu['step_thr']:+.0f}%)' if ds is not None else '—')}"
                  f"   iso-wallclock Δ = "
                  f"{(f'{dw:+.1f}s ({100*dw/mu['wall_thr']:+.0f}%)' if dw is not None else '—')}")


if __name__ == "__main__":
    main()
