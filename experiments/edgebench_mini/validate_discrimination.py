"""EdgeBench-mini ACCEPTANCE GATE — discriminative-power validation.

>>> UNTESTED until a GPU frees. Model-in-the-loop validation is DEFERRED (both
    RTX 5090s were busy at build time). The harness/scoring/tasks are validated
    on CPU with the scripted ReplayAgent (`experiments/test_edgebench_mini.py`);
    this driver is the piece that needs real checkpoints + CUDA and has NOT been
    run. Treat every number it prints as unverified until this script has been
    executed on a free GPU. <<<

The gate (from the ideation brief, `ideas_2026_07_13/08_agentic_native.md`
idea 2): the harness must ORDER known-different checkpoints monotonically with
NON-OVERLAPPING confidence intervals where greedy HumanEval-164 cannot. If it
cannot separate a base < SFT < RL ladder, the harness is dead.

Usage (once a GPU is free):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python \\
      experiments/edgebench_mini/validate_discrimination.py \\
      --ckpt base=checkpoints/feature_pilot_A.pt \\
      --ckpt sft=checkpoints/stageA_executor.pt \\
      --ckpt rl=checkpoints/<rl_ckpt>.pt \\
      --expected_order rl,sft,base \\
      --out runs/edgebench_mini_discrimination.json

`--expected_order` is best->worst; PASS = observed ranking matches AND every
adjacent expected pair has disjoint CIs on the headline metric (default
auc_tokens). A GPU-free smoke of the plumbing (no real model) is available via
`--dry_run`, which drives the ReplayAgent gold solutions at graded budgets so
you can see the table/verdict shape before spending GPU.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.edgebench_mini import scoring
from experiments.edgebench_mini.harness import Budgets, run_suite
from experiments.edgebench_mini.tasks import build_suite


def _parse_ckpt_args(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--ckpt expects name=path, got {it!r}")
        name, path = it.split("=", 1)
        out[name] = path
    return out


def _dry_run_agents(tasks):
    """Plumbing smoke WITHOUT a model: three 'checkpoints' of decreasing skill,
    built from the ReplayAgent gold solutions truncated to different budgets.
    Lets you inspect the comparison table shape before spending GPU. This is a
    fixture, NOT a validation of any real checkpoint."""
    from experiments.edgebench_mini.harness import ReplayAgent

    def make(cap):
        def factory(task):
            turns = task.scripted_solution_turns()
            return ReplayAgent(turns[:cap] if cap is not None else turns)
        return factory
    # rl solves fully, sft partially, base barely.
    return {"rl": make(None), "sft": make(3), "base": make(1)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="name=path (repeatable), e.g. base=checkpoints/x.pt")
    ap.add_argument("--expected_order", default="rl,sft,base",
                    help="best->worst comma-separated checkpoint names")
    ap.add_argument("--headline_metric", default="auc_tokens")
    ap.add_argument("--max_iters", type=int, default=14)
    ap.add_argument("--max_tool_calls", type=int, default=28)
    ap.add_argument("--max_gen_tokens", type=int, default=6000)
    ap.add_argument("--max_ckpt_gen", type=int, default=384,
                    help="per-turn generated-token cap for CkptAgent")
    ap.add_argument("--n_boot", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="")
    ap.add_argument("--dry_run", action="store_true",
                    help="drive scripted agents (no model / no GPU) to smoke "
                         "the plumbing + table shape")
    args = ap.parse_args()

    tasks = build_suite()
    budgets = Budgets(max_iters=args.max_iters,
                      max_tool_calls=args.max_tool_calls,
                      max_gen_tokens=args.max_gen_tokens)
    expected = [x for x in args.expected_order.split(",") if x]

    named_summaries: dict[str, list[dict]] = {}
    t0 = time.time()

    if args.dry_run:
        print("[dry_run] driving scripted agents (NO model, NO GPU) — fixture "
              "only, not a real-checkpoint validation")
        factories = _dry_run_agents(tasks)
        for name in expected:
            factory = factories[name]
            summaries = []
            for task in tasks:
                from experiments.edgebench_mini.harness import run_episode
                res = run_episode(task, factory(task), budgets,
                                  keep_transcript=False)
                summaries.append(scoring.task_summary(res.to_dict()))
            named_summaries[name] = summaries
            print(f"  [{name}] mean best="
                  f"{sum(s['best_score'] for s in summaries)/len(summaries):.3f}")
    else:
        ckpts = _parse_ckpt_args(args.ckpt)
        if not ckpts:
            raise SystemExit("pass --ckpt name=path (or use --dry_run)")
        from experiments.edgebench_mini.harness import CkptAgent
        for name, path in ckpts.items():
            print(f"[ckpt] loading {name} = {path}", flush=True)
            agent = CkptAgent(path, max_gen=args.max_ckpt_gen)
            results = run_suite(tasks, agent, budgets, verbose=True)
            named_summaries[name] = [scoring.task_summary(r.to_dict())
                                     for r in results]
            del agent

    comparison = scoring.compare_checkpoints(
        named_summaries, headline_metric=args.headline_metric,
        n_boot=args.n_boot, alpha=args.alpha)
    verdict = scoring.check_monotonic_separation(comparison, expected)

    print("\n" + scoring.render_comparison_table(comparison))
    print("\n=== DISCRIMINATION VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nPASS/FAIL: {'PASS' if verdict['PASS'] else 'FAIL'}")
    print(f"[done] {time.time() - t0:.1f}s")

    if args.out:
        outp = pathlib.Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump({
                "comparison": comparison, "verdict": verdict,
                "per_ckpt_summaries": named_summaries,
                "budgets": vars(budgets), "dry_run": args.dry_run,
            }, f, indent=2, default=str)
        print(f"[saved] {args.out}")
    return 0 if verdict["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
