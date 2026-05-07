"""Side-by-side comparison of the KL+CE run vs the CE-only baseline.

Reads the JSON metrics dumped by distill_pilot.py and prints the
comparison table the report needs.

Run:
  /home/knielsen/ml/parallel-ss-dep/.venv/bin/python \
    experiments/distill_pilot_compare.py \
    --kl_ce logs/distill_pilot/metrics/kl_ce.json \
    --ce    logs/distill_pilot/metrics/ce.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kl_ce", type=str, required=True)
    p.add_argument("--ce", type=str, required=True)
    p.add_argument("--out_md", type=str, default=None,
                   help="If set, write a markdown table for the report.")
    args = p.parse_args()

    d_kl = json.loads(pathlib.Path(args.kl_ce).read_text())
    d_ce = json.loads(pathlib.Path(args.ce).read_text())

    fv_kl = d_kl["final_val"]
    fv_ce = d_ce["final_val"]

    print("=" * 64)
    print("Distill pilot validation @ 1K steps — KL+CE vs CE-only")
    print("=" * 64)
    print(f"  Student:      plain DN, {d_kl['params_M']:.1f} M params, "
          f"d={d_kl['d_model']} h={d_kl['n_heads']} dh={d_kl['d_head']} L={d_kl['n_layers']}")
    print(f"  Teacher:      {d_kl['teacher']}")
    print(f"  Dataset:      {d_kl['dataset']}")
    print(f"  Seed:         {d_kl['seed']} (both runs)")
    print(f"  Steps:        {d_kl['steps']}, batch={d_kl['batch']}, T={d_kl['T']}")
    print()
    print(f"  {'Run':<24}{'val_ce':>10}{'val_ppl':>10}{'val_kl':>10}{'wallclock':>12}")
    print(f"  {'---':<24}{'------':>10}{'-------':>10}{'------':>10}{'---------':>12}")
    print(f"  {'KL+CE (alpha=' + str(d_kl['alpha']) + ')':<24}"
          f"{fv_kl['val_ce']:>10.4f}{fv_kl['val_ppl']:>10.2f}"
          f"{fv_kl['val_kl']:>10.4f}{d_kl['wallclock_s']:>10.0f}s")
    print(f"  {'CE-only baseline':<24}"
          f"{fv_ce['val_ce']:>10.4f}{fv_ce['val_ppl']:>10.2f}"
          f"{fv_ce['val_kl']:>10.4f}{d_ce['wallclock_s']:>10.0f}s")

    # Val trajectory.
    print(f"\n  {'step':>6}  {'kl_ce val_ppl':>14}  {'ce val_ppl':>12}")
    steps_kl = [v["step"] for v in d_kl["val_history"]]
    by_step_kl = {v["step"]: v for v in d_kl["val_history"]}
    by_step_ce = {v["step"]: v for v in d_ce["val_history"]}
    for s in sorted(set(steps_kl) | set(by_step_ce.keys())):
        kl_p = by_step_kl.get(s, {}).get("val_ppl", float("nan"))
        ce_p = by_step_ce.get(s, {}).get("val_ppl", float("nan"))
        print(f"  {s:>6}  {kl_p:>14.2f}  {ce_p:>12.2f}")

    # Headline: relative PPL.
    rel_ppl = (fv_kl['val_ppl'] - fv_ce['val_ppl']) / fv_ce['val_ppl'] * 100
    sign = "" if rel_ppl >= 0 else "+"
    direction = "KL+CE WORSE than CE-only" if rel_ppl > 0 else \
                ("KL+CE BETTER than CE-only" if rel_ppl < 0 else "tied")
    print()
    print(f"  KL+CE vs CE-only: {sign}{rel_ppl:.1f} % PPL  ({direction})")

    # Verdict.
    print()
    if rel_ppl <= -0.5:
        verdict = "READY TO SCALE"
        message = ("KL+CE matches or beats CE-only — recipe is sound. "
                   "Phase 15's collapse averted by teacher-aligned data.")
    elif rel_ppl <= 10.0:
        verdict = "PROBABLY READY (within 10% tolerance)"
        message = ("KL+CE is not catastrophically worse. Some mild signal "
                   "loss but within the validation success criterion.")
    else:
        verdict = "NEEDS CORPUS FIX"
        message = ("KL+CE is materially worse than CE-only by >10% PPL. "
                   "Likely teacher-data misalignment (Phase 15 redux). "
                   "Do NOT scale up; investigate corpus / loss weights / "
                   "top-K coverage.")
    print(f"  Verdict: {verdict}")
    print(f"    {message}")

    if args.out_md:
        out = pathlib.Path(args.out_md)
        with out.open("w") as f:
            f.write("| Run | Final val CE | Final val PPL | Final val KL | Wallclock |\n")
            f.write("| --- | ---: | ---: | ---: | ---: |\n")
            f.write(f"| KL+CE (`alpha = {d_kl['alpha']}`, top-K = {d_kl['top_k']}) | "
                    f"{fv_kl['val_ce']:.4f} | {fv_kl['val_ppl']:.2f} | "
                    f"{fv_kl['val_kl']:.4f} | {d_kl['wallclock_s']:.0f}s |\n")
            f.write(f"| CE-only baseline | {fv_ce['val_ce']:.4f} | "
                    f"{fv_ce['val_ppl']:.2f} | {fv_ce['val_kl']:.4f} | "
                    f"{d_ce['wallclock_s']:.0f}s |\n\n")
            f.write(f"**KL+CE vs CE-only: {sign}{rel_ppl:.1f} % PPL** "
                    f"— {direction.lower()}.\n\n")
            f.write(f"**Verdict: {verdict}.** {message}\n")
        print(f"\n  markdown table written to {out}")


if __name__ == "__main__":
    main()
