"""PRIORITY-1 go/no-go diagnostic — execution-grounded "does thinking help".

Per THINKING_AUDIT_2026_05_28.md flaw C/D and PLAN_FLAW_C §4 / PLAN_FLAW_D §1.

This is the HONEST replacement for the per-token-logp aux-loss diagnostics
(`probe_process_reward.py`, `probe_gate_calibration.py`), which the audit found
to be self-fulfilling (they measure Δlogp on the candidate set the loss itself
moves, and Δlogp rewards sharpening surface tokens regardless of whether the
emitted *function passes tests*).

What this probe does instead:
  - For each problem, generate a completion TWICE using the ACTUAL deploy
    generator `generate_with_retrieval_as_input` (additive-α, iterative
    re-decode — NOT the K-at-once forced forward of process_reward.py):
      * `with_think`  : the gate's learned think policy (gate free to fire).
      * `no_think`    : gate forced never-think via `emit_threshold = 1.0`
                        (identical model, identical decode, zero think budget).
  - Grade BOTH with `code_grader.grade(...)` (the dense tier ladder, score
    in [0,1]).
  - Report mean Δ(grader score) = with_think − without_think, plus a per-tier
    breakdown and how many problems improved / regressed / unchanged.

The audit predicts Δ ≤ 0 on the current ckpts. That number is the go/no-go
gate for whether the execution-grounded RL run (launch_rl_stochastic_gate.sh)
is worth launching: RL is what *fixes* a ≤0 gate, but the terminal Δ(reward)
is the metric that must be tracked throughout — never Δlogp.

CLI:
    PYTHONPATH=. .venv/bin/python experiments/probe_think_grader_reward.py \\
        --ckpt checkpoints/sft_phase_c_combined.pt \\
        --dataset humaneval --max_problems 40 \\
        --prompt_style sft_comment --extract_code_block
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments import code_grader
from experiments.eval_humaneval import (
    _extract_code_block,
    _truncate_at_stop,
    generate_with_retrieval_as_input,
)


def _build_completion_for_grading(
    problem: code_grader.Problem,
    gen_text: str,
    *,
    extract_code_block: bool,
) -> tuple[code_grader.Problem, str]:
    """Mirror experiments.eval_humaneval.evaluate's grading-prep exactly.

    Returns a (Problem, completion) pair to feed `code_grader.grade`. When
    `extract_code_block` is set we pull the ```python``` fence out and grade
    the extracted code as a standalone program (prompt_is_code=False,
    empty prompt) — identical to evaluate()'s `full_code = code` path.
    """
    if extract_code_block:
        code = _extract_code_block(gen_text)
        full_code = code if code is not None else gen_text
        # Grade the extracted block as a standalone program: exec the
        # completion alone (prompt_is_code=False) so we don't double-prepend
        # the prompt. This matches `_run_test_in_subprocess(full_code, ...)`.
        graded_problem = dataclasses.replace(
            problem, prompt="", prompt_is_code=False)
        return graded_problem, full_code
    # Non-extract path: prompt+truncated-gen, exactly as code_grader.grade
    # does internally for prompt_is_code problems. Pass the raw gen_text and
    # let grade() do the truncate+concat.
    return problem, gen_text


def _gen_once(
    model, tok, problem: code_grader.Problem, *,
    prompt_style: str,
    use_thinking: bool,
    max_gen: int,
    total_think_budget: int,
    max_think_per_step: int,
    emit_threshold: float,
    gate_floor: float,
    min_emit_before_eos: int,
    thinking_token_id: int,
    additive: bool,
    max_T: int,
) -> tuple[str, dict]:
    """One generation through the deploy generator. `use_thinking=False`
    forces the gate to never fire by setting emit_threshold=1.0 (so
    `gate_val >= emit_threshold` is True at every step → no think)."""
    device = next(model.parameters()).device
    raw_prompt = problem.prompt
    if prompt_style == "sft_comment":
        prompt = "# Complete the following Python function.\n" + raw_prompt
    else:
        prompt = raw_prompt
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    thinks_reserve = total_think_budget if use_thinking else 0
    room = max_gen + thinks_reserve
    if len(prompt_ids) + room > max_T:
        prompt_ids = prompt_ids[-(max_T - room):]
    prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                            device=device).unsqueeze(0)

    eff_emit_threshold = emit_threshold if use_thinking else 1.0
    eff_budget = int(total_think_budget) if use_thinking else 0
    gen, diag = generate_with_retrieval_as_input(
        model, prompt_t, max_gen=max_gen,
        temperature=0.0, eos_token_id=tok.eos_token_id,
        thinking_token_id=thinking_token_id,
        max_think_per_step=max_think_per_step,
        total_think_budget=eff_budget,
        emit_threshold=eff_emit_threshold,
        min_emit_before_eos=min_emit_before_eos,
        gate_floor=gate_floor,
        additive=additive,
    )
    gen_only_full = gen[0, len(prompt_ids):].tolist()
    gen_only = [t for t in gen_only_full if t != int(thinking_token_id)]
    gen_text = tok.decode(gen_only, skip_special_tokens=True)
    return gen_text, diag


def run_probe(
    model, tok, problems: list[code_grader.Problem], *,
    prompt_style: str = "sft_comment",
    extract_code_block: bool = True,
    max_gen: int = 256,
    total_think_budget: int = 120,
    max_think_per_step: int = 8,
    emit_threshold: float = 0.5,
    gate_floor: float = 0.0,
    min_emit_before_eos: int = 30,
    thinking_token_id: Optional[int] = None,
    additive: bool = True,
    timeout_s: int = 7,
    max_T: int = 2048,
    log_fn=print,
) -> dict:
    if thinking_token_id is None:
        thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        raise ValueError("thinking_token_id could not be resolved")

    was_training = model.training
    model.eval()
    t0 = time.perf_counter()
    per_problem: list[dict] = []
    try:
        with torch.no_grad():
            for i, problem in enumerate(problems):
                nt_text, nt_diag = _gen_once(
                    model, tok, problem, prompt_style=prompt_style,
                    use_thinking=False, max_gen=max_gen,
                    total_think_budget=total_think_budget,
                    max_think_per_step=max_think_per_step,
                    emit_threshold=emit_threshold, gate_floor=gate_floor,
                    min_emit_before_eos=min_emit_before_eos,
                    thinking_token_id=thinking_token_id, additive=additive,
                    max_T=max_T)
                wt_text, wt_diag = _gen_once(
                    model, tok, problem, prompt_style=prompt_style,
                    use_thinking=True, max_gen=max_gen,
                    total_think_budget=total_think_budget,
                    max_think_per_step=max_think_per_step,
                    emit_threshold=emit_threshold, gate_floor=gate_floor,
                    min_emit_before_eos=min_emit_before_eos,
                    thinking_token_id=thinking_token_id, additive=additive,
                    max_T=max_T)

                nt_prob, nt_comp = _build_completion_for_grading(
                    problem, nt_text, extract_code_block=extract_code_block)
                wt_prob, wt_comp = _build_completion_for_grading(
                    problem, wt_text, extract_code_block=extract_code_block)
                nt_res = code_grader.grade(nt_prob, nt_comp,
                                           timeout_s=timeout_s)
                wt_res = code_grader.grade(wt_prob, wt_comp,
                                           timeout_s=timeout_s)
                per_problem.append({
                    "task_id": problem.task_id,
                    "no_think": {"score": nt_res.score, "tier": nt_res.tier,
                                 "passed": nt_res.passed,
                                 "think_total": int(nt_diag.get("think_total", 0))},
                    "with_think": {"score": wt_res.score, "tier": wt_res.tier,
                                   "passed": wt_res.passed,
                                   "think_total": int(wt_diag.get("think_total", 0))},
                    "delta": wt_res.score - nt_res.score,
                })
                if (i + 1) % 5 == 0 or (i + 1) == len(problems):
                    d_run = sum(r["delta"] for r in per_problem) / len(per_problem)
                    log_fn(f"  [{i+1}/{len(problems)}] running mean "
                           f"Δscore={d_run:+.4f}")
    finally:
        if was_training:
            model.train()

    n = len(per_problem)
    deltas = [r["delta"] for r in per_problem]
    mean_delta = sum(deltas) / max(1, n)
    n_improved = sum(1 for d in deltas if d > 1e-9)
    n_regressed = sum(1 for d in deltas if d < -1e-9)
    n_unchanged = n - n_improved - n_regressed
    nt_mean = sum(r["no_think"]["score"] for r in per_problem) / max(1, n)
    wt_mean = sum(r["with_think"]["score"] for r in per_problem) / max(1, n)
    nt_pass = sum(1 for r in per_problem if r["no_think"]["passed"])
    wt_pass = sum(1 for r in per_problem if r["with_think"]["passed"])

    def _tier_hist(field):
        h: dict[str, int] = {}
        for r in per_problem:
            h[r[field]["tier"]] = h.get(r[field]["tier"], 0) + 1
        return dict(sorted(h.items()))

    mean_think = (sum(r["with_think"]["think_total"] for r in per_problem)
                  / max(1, n))
    return {
        "n_total": n,
        "mean_delta_score": mean_delta,
        "n_improved": n_improved,
        "n_regressed": n_regressed,
        "n_unchanged": n_unchanged,
        "no_think": {"mean_score": nt_mean, "n_passed": nt_pass,
                     "tier_hist": _tier_hist("no_think")},
        "with_think": {"mean_score": wt_mean, "n_passed": wt_pass,
                       "tier_hist": _tier_hist("with_think"),
                       "mean_think_tokens": mean_think,
                       "think_budget": int(total_think_budget)},
        "per_problem": per_problem,
        "elapsed_s": time.perf_counter() - t0,
    }


def _load_problems(dataset: str, max_problems: Optional[int]
                   ) -> list[code_grader.Problem]:
    loaders = {
        "humaneval": code_grader.load_humaneval,
        "mbpp": code_grader.load_mbpp,
        "mbpp_all": code_grader.load_mbpp_all,
        "mbpp_plus": code_grader.load_mbpp_plus,
        "mbpp_combined": code_grader.load_mbpp_combined,
    }
    if dataset not in loaders:
        raise ValueError(f"unknown dataset {dataset!r}; "
                         f"choose from {sorted(loaders)}")
    probs = loaders[dataset]()
    if max_problems is not None:
        probs = probs[:max_problems]
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, default="humaneval")
    p.add_argument("--max_problems", type=int, default=40)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--total_think_budget", type=int, default=120)
    p.add_argument("--max_think_per_step", type=int, default=8)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--prompt_style", type=str, default="sft_comment",
                   choices=["humaneval", "sft_comment"])
    p.add_argument("--extract_code_block", action="store_true")
    p.add_argument("--timeout_s", type=int, default=7)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer

    print(f"Loading checkpoint: {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")
    additive = bool(cfg.get("retrieval_input_additive", True))
    max_T = cfg["max_T"] if cfg.get("max_T", 0) and cfg["max_T"] > 0 else 2048
    print(f"  retrieval_input_additive={additive}  "
          f"thinking_token_id={thinking_token_id}  max_T={max_T}")

    problems = _load_problems(args.dataset, args.max_problems)
    print(f"  dataset={args.dataset}  n_problems={len(problems)}")

    res = run_probe(
        model, tok, problems,
        prompt_style=args.prompt_style,
        extract_code_block=args.extract_code_block,
        max_gen=args.max_gen,
        total_think_budget=args.total_think_budget,
        max_think_per_step=args.max_think_per_step,
        emit_threshold=args.emit_threshold,
        gate_floor=args.gate_floor,
        min_emit_before_eos=args.min_emit_before_eos,
        thinking_token_id=thinking_token_id,
        additive=additive,
        timeout_s=args.timeout_s,
        max_T=max_T,
    )

    print("\n==================== PROBE: think-grader-reward Δ "
          "====================")
    print(f"  ckpt: {args.ckpt}")
    print(f"  dataset={args.dataset}  n={res['n_total']}  "
          f"elapsed={res['elapsed_s']:.1f}s")
    print(f"  HEADLINE  mean Δ(grader score) = "
          f"{res['mean_delta_score']:+.4f}   "
          f"(with_think − no_think)")
    print(f"            improved={res['n_improved']}  "
          f"regressed={res['n_regressed']}  "
          f"unchanged={res['n_unchanged']}")
    nt = res["no_think"]; wt = res["with_think"]
    print(f"  no_think   : mean_score={nt['mean_score']:.4f}  "
          f"pass={nt['n_passed']}/{res['n_total']}  tiers={nt['tier_hist']}")
    print(f"  with_think : mean_score={wt['mean_score']:.4f}  "
          f"pass={wt['n_passed']}/{res['n_total']}  tiers={wt['tier_hist']}")
    print(f"               mean_think_tokens={wt['mean_think_tokens']:.1f}  "
          f"(budget={wt['think_budget']})")
    go = res["mean_delta_score"] > -1e-3
    verdict = ("NOT significantly negative → RL launch is justified "
               "(per PLAN_FLAW_C §4)") if go else (
        "significantly NEGATIVE → confirms the audit; RL is still the fix "
        "but thinking currently HURTS the task")
    print(f"  GO/NO-GO  : {verdict}")
    print("====================================================="
          "===============")

    out_path = args.out_json
    if out_path:
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in res.items() if k != "per_problem"}
                      | {"per_problem": res["per_problem"],
                         "ckpt": args.ckpt, "dataset": args.dataset}, f,
                      indent=2)
        print(f"  full result -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
