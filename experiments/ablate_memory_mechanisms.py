"""Ablate WM + PKM individually and re-eval HumanEval to measure
contribution to pass@1.

Four conditions per ckpt:
  - baseline       : as-trained
  - wm_off         : zero memory.W_proj.weight in state_dict → injection 0
  - pkm_off        : zero pkm_layer.out_alpha in state_dict → contribution 0
  - both_off       : both disabled

Delta from baseline tells us whether each mechanism is load-bearing.
If wm_off pass@1 ≈ baseline, WM isn't doing useful work for this task.
Same for PKM. If BOTH are decorative, the architecture's bet didn't pay
off on coding tasks and we should pivot to scaling data instead.

Implementation note: rather than monkey-patching a live model, we write
a modified ckpt to /tmp per condition and call eval_humaneval.evaluate()
on the tmp path. This keeps the eval path identical to production runs.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/ablate_memory_mechanisms.py \\
      --ckpt checkpoints/sft_v7_pkm_film_combined.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def _write_ablated_ckpt(src_path: str, dst_path: str,
                          disable_wm: bool, disable_pkm: bool) -> None:
    """Load src ckpt, zero the relevant tensors in state_dict, write
    to dst_path. The full state_dict + cfg are preserved otherwise so
    build_model_from_ckpt reconstructs the same architecture."""
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    if disable_wm and "memory.W_proj.weight" in sd:
        sd["memory.W_proj.weight"] = torch.zeros_like(sd["memory.W_proj.weight"])
    if disable_pkm and "pkm_layer.out_alpha" in sd:
        sd["pkm_layer.out_alpha"] = torch.zeros_like(sd["pkm_layer.out_alpha"])
    torch.save(ckpt, dst_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--max_problems", type=int, default=None,
                   help="Cap problems; default = full 164.")
    p.add_argument("--max_gen", type=int, default=800)
    p.add_argument("--total_think_budget", type=int, default=400)
    p.add_argument("--out_log", type=str, default=None,
                   help="Optional JSON summary path.")
    p.add_argument("--conditions", type=str,
                   default="baseline,wm_off,pkm_off,both_off",
                   help="Comma-separated subset to run.")
    p.add_argument("--generator", type=str, default="standard",
                   choices=["standard", "retrieval_as_input"],
                   help="Generation mode for the eval. Use "
                        "'retrieval_as_input' for ckpts trained with "
                        "--retrieval_as_input_thinking (v5+).")
    args = p.parse_args()

    from experiments.eval_humaneval import evaluate

    all_conditions = {
        "baseline": (False, False),
        "wm_off":   (True,  False),
        "pkm_off":  (False, True),
        "both_off": (True,  True),
    }
    requested = [c.strip() for c in args.conditions.split(",")]

    print(f"Ablating: {args.ckpt}")
    print(f"Conditions: {requested}")
    print()

    results = {}
    with tempfile.TemporaryDirectory() as td:
        for label in requested:
            if label not in all_conditions:
                print(f"skipping unknown condition: {label}")
                continue
            disable_wm, disable_pkm = all_conditions[label]
            print(f"\n{'='*70}\n{label}: disable_wm={disable_wm} "
                  f"disable_pkm={disable_pkm}\n{'='*70}")
            # Write ablated ckpt to tmp.
            if label == "baseline":
                ablated_path = args.ckpt
            else:
                ablated_path = str(pathlib.Path(td) / f"{label}.pt")
                _write_ablated_ckpt(args.ckpt, ablated_path,
                                      disable_wm, disable_pkm)
            result = evaluate(
                ablated_path, n_samples=1, temperature=0.0,
                max_gen=args.max_gen,
                max_problems=args.max_problems,
                use_thinking=True,
                max_think_per_step=8,
                total_think_budget=args.total_think_budget,
                emit_threshold=0.5,
                min_emit_before_eos=30,
                gate_floor=0.0,
                prompt_style="sft_comment",
                extract_code_block=True,
                generator=args.generator,
            )
            results[label] = result
            print(f"  pass@1={result['pass_rate']:.3f}  "
                  f"({result['n_passed']}/{result['n_total']})  "
                  f"think_rate={result.get('think_rate', 0):.3f}")

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'condition':<12} {'pass@1':>8} {'(passed/total)':>16} {'think_rate':>11}  Δ_baseline")
    base_pass = results.get("baseline", {}).get("n_passed", 0)
    for label, r in results.items():
        delta = r["n_passed"] - base_pass
        print(f"  {label:<10} {r['pass_rate']*100:>6.1f}% "
              f"{r['n_passed']:>5}/{r['n_total']:<5}  "
              f"{r.get('think_rate', 0):>9.3f}  "
              f"{delta:>+5d}")

    print("""
Interpretation:
  - large negative delta = mechanism IS load-bearing
  - delta ≈ 0           = mechanism is decorative / pure capacity
  - positive delta      = mechanism HURTS the metric (rare)
""")

    if args.out_log:
        with open(args.out_log, "w") as f:
            json.dump({label: {"pass_rate": r["pass_rate"],
                                "n_passed": r["n_passed"],
                                "n_total": r["n_total"],
                                "think_rate": r.get("think_rate", 0)}
                       for label, r in results.items()}, f, indent=2)
        print(f"summary → {args.out_log}")


if __name__ == "__main__":
    sys.exit(main())
