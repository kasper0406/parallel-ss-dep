"""Difficulty probe for the synth_reasoning train + held-out splits.

The stochastic-gate discovery RL experiment (THINKING_PLAN pivot) uses
`data/synth_reasoning_train.jsonl` as the RL pool, with
`data/synth_reasoning_heldout.jsonl` as the generalisation eval. For
that experiment to be informative, the BASELINE pass-rate (no thinking,
greedy, our historical-best SFT ckpt) must be in a "challenging but not
impossible" band — high enough to give RL gradient signal, low enough
that there's headroom for the thinking gate to help.

Target band: **5 – 20 % pass-rate**.
- < 5 %: tasks are too hard; the model has no foothold, RL will see
  near-uniform zero reward and won't learn.
- > 30 %: tasks are too easy; the model solves them WITHOUT thinking, so
  the gate has no incentive to fire and the experiment is structurally
  uninformative.

This probe:
  1. Loads `checkpoints/sft_phase_c_combined.pt` (our historical-best
     SFT, HumanEval 10/164 — the canonical SFT base used for the
     grader-RL runs).
  2. Generates greedy completions (temperature=0.0, use_thinking=False)
     for `--n_probe` problems from each of the train and held-out splits.
  3. Reports per-split pass-rate, per-family pass-rate, and a few
     example failures + successes for eyeballing.

DO NOT run this co-resident with another training process on the same
GPU — load_humaneval-style ckpt loading takes ~1.5 GiB just to hold the
model in memory, and `_run_test_in_subprocess` forks Python again per
problem (cheap but spiky).

Usage (when the GPU is free):
  PYTHONPATH=. .venv/bin/python experiments/probe_synth_reasoning_difficulty.py \\
      --ckpt checkpoints/sft_phase_c_combined.pt \\
      --n_probe 50 \\
      --max_gen 192 \\
      [--gpu 0]

Outputs `probe_synth_reasoning_difficulty_<ckpt-basename>.json` next to
the CWD with the full per-problem record.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str,
                    default="checkpoints/sft_phase_c_combined.pt",
                    help="Historical-best SFT ckpt to probe.")
    ap.add_argument("--n_probe", type=int, default=50,
                    help="Number of problems sampled from EACH split.")
    ap.add_argument("--max_gen", type=int, default=192,
                    help="Max emit tokens per problem.")
    ap.add_argument("--prompt_style", type=str, default="sft_comment",
                    choices=["sft_comment", "raw"],
                    help="Prompt wrapping. sft_phase_c_combined was "
                         "trained with the '# Complete the following "
                         "Python function.\\n' prefix; use it here.")
    ap.add_argument("--extract_code_block", action="store_true",
                    default=True,
                    help="Extract ```python ... ``` from the model "
                         "output before grading (matches the SFT format).")
    ap.add_argument("--no_extract_code_block", dest="extract_code_block",
                    action="store_false")
    ap.add_argument("--gpu", type=int, default=None,
                    help="Pin to CUDA_VISIBLE_DEVICES=<gpu> before "
                         "importing torch. Leave unset to use whatever "
                         "is currently visible.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Random seed for sampling the n_probe subset.")
    ap.add_argument("--out_json", type=str,
                    default="probe_synth_reasoning_difficulty.json")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Heavy imports AFTER CUDA_VISIBLE_DEVICES is fixed.
    import random
    import torch
    from transformers import AutoTokenizer

    from experiments.code_grader import (
        LOADERS, grade,
    )
    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.eval_humaneval import (
        _extract_code_block, generate,
    )

    rng = random.Random(args.seed)

    print(f"[probe] loading ckpt: {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model.eval()
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M")
    )
    eos_id = tok.eos_token_id
    print(f"[probe] model loaded. feedback={cfg.get('feedback_mode')} "
          f"n_layers={cfg.get('n_layers')} max_T={cfg.get('max_T')}")

    splits = {
        "train":   LOADERS["synth_reasoning_train"](),
        "heldout": LOADERS["synth_reasoning_heldout"](),
    }
    per_split_summary: dict[str, dict] = {}
    per_problem_records: list[dict] = []

    for split_name, problems in splits.items():
        # Sample a deterministic subset of size n_probe.
        idxs = list(range(len(problems)))
        rng_split = random.Random((args.seed, split_name))
        rng_split.shuffle(idxs)
        sub = [problems[i] for i in idxs[:args.n_probe]]
        n_total = 0
        n_passed = 0
        per_family: dict[str, list[int]] = collections.defaultdict(
            lambda: [0, 0]  # [n_total, n_passed]
        )
        t_split0 = time.perf_counter()
        for prob in sub:
            family = prob.task_id.split("/")[1]
            # Build prompt (sft_comment style matches Phase C SFT
            # training data).
            if args.prompt_style == "sft_comment":
                wrapped = ("# Complete the following Python function.\n"
                           + prob.prompt)
            else:
                wrapped = prob.prompt
            prompt_ids = tok.encode(wrapped, add_special_tokens=False)
            eff_max_T = cfg["max_T"] if cfg["max_T"] > 0 else 2048
            if len(prompt_ids) + args.max_gen > eff_max_T:
                prompt_ids = prompt_ids[-(eff_max_T - args.max_gen):]
            prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                     device="cuda").unsqueeze(0)
            gen, _diag = generate(
                model, prompt_t,
                max_gen=args.max_gen,
                temperature=0.0,
                eos_token_id=eos_id,
                use_thinking=False,  # <- THE KEY: no thinking.
            )
            gen_only = gen[0, len(prompt_ids):].tolist()
            gen_text = tok.decode(gen_only, skip_special_tokens=True)
            if args.extract_code_block:
                code = _extract_code_block(gen_text) or gen_text
                completion = code
            else:
                completion = gen_text
            res = grade(prob, completion)
            n_total += 1
            n_passed += int(res.passed)
            per_family[family][0] += 1
            per_family[family][1] += int(res.passed)
            per_problem_records.append({
                "split": split_name,
                "task_id": prob.task_id,
                "family": family,
                "passed": res.passed,
                "tier": res.tier,
                "gen_preview": gen_text[:240],
            })
            if n_total % 10 == 0:
                print(f"  [{split_name}] {n_total:>3d}/{len(sub)}  "
                      f"pass={n_passed} "
                      f"({n_passed/n_total:.3f})  "
                      f"elapsed={time.perf_counter()-t_split0:.1f}s")
        rate = n_passed / max(1, n_total)
        per_split_summary[split_name] = {
            "n_total": n_total,
            "n_passed": n_passed,
            "pass_rate": rate,
            "per_family": {
                fam: {"n_total": tot, "n_passed": pas,
                      "pass_rate": pas / max(1, tot)}
                for fam, (tot, pas) in per_family.items()
            },
        }
        print(f"[probe] {split_name} done: pass_rate={rate:.3f} "
              f"({n_passed}/{n_total})")

    # Verdict against the 5 – 20 % target band.
    train_rate = per_split_summary["train"]["pass_rate"]
    held_rate = per_split_summary["heldout"]["pass_rate"]
    verdict_lines: list[str] = []
    for name, rate in [("train", train_rate), ("heldout", held_rate)]:
        if rate > 0.30:
            verdict_lines.append(
                f"  {name}: pass_rate {rate:.3f} > 30% — TOO EASY, "
                f"curriculum is wrong (thinking has no room to help)."
            )
        elif rate < 0.05:
            verdict_lines.append(
                f"  {name}: pass_rate {rate:.3f} < 5% — TOO HARD, "
                f"baseline has no foothold for RL gradient."
            )
        elif rate <= 0.20:
            verdict_lines.append(
                f"  {name}: pass_rate {rate:.3f} in 5-20% band — GOOD."
            )
        else:
            verdict_lines.append(
                f"  {name}: pass_rate {rate:.3f} in 20-30% — borderline; "
                f"acceptable but on the easy side."
            )
    print("\n[probe] verdict (target band 5 – 20 %):")
    print("\n".join(verdict_lines))

    out = {
        "ckpt": args.ckpt,
        "n_probe": args.n_probe,
        "max_gen": args.max_gen,
        "prompt_style": args.prompt_style,
        "extract_code_block": args.extract_code_block,
        "per_split": per_split_summary,
        "per_problem": per_problem_records,
        "verdict": verdict_lines,
    }
    out_path = pathlib.Path(args.out_json)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
