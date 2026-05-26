from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_humaneval import (
    _run_test_in_subprocess,
    _truncate_at_stop,
    generate,
)


def _load_probe(probe_path: str, n_problems: Optional[int]) -> list[dict]:
    rows: list[dict] = []
    with open(probe_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if n_problems is not None and len(rows) >= n_problems:
                break
            rows.append(json.loads(line))
    return rows


def _max_T_from_model(model) -> int:
    cfg = getattr(model, "config", None)
    if isinstance(cfg, dict) and "max_T" in cfg:
        v = int(cfg["max_T"])
        if v > 0:
            return v
    for attr in ("max_T", "max_seq_len", "max_position_embeddings"):
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 2048


def _run_one(model, tokenizer, problem, *,
             use_thinking: bool, max_gen: int, total_think_budget: int,
             thinking_token_id: Optional[int],
             max_think_per_step: int,
             emit_threshold: float, gate_floor: float,
             min_emit_before_eos: int, timeout_s: int) -> dict:
    device = next(model.parameters()).device
    max_T = _max_T_from_model(model)
    prompt = problem["prompt"]
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    thinks_reserve = total_think_budget if use_thinking else 0
    room = max_gen + thinks_reserve
    if len(ids) + room > max_T:
        ids = ids[-(max_T - room):]
    prompt_t = torch.tensor(ids, dtype=torch.long,
                            device=device).unsqueeze(0)
    eos = getattr(tokenizer, "eos_token_id", None)

    gen, diag = generate(
        model, prompt_t, max_gen=max_gen,
        temperature=0.0, eos_token_id=eos,
        use_thinking=use_thinking,
        thinking_token_id=thinking_token_id,
        max_think_per_step=max_think_per_step,
        total_think_budget=total_think_budget,
        emit_threshold=emit_threshold,
        gate_floor=gate_floor,
        min_emit_before_eos=min_emit_before_eos,
    )
    gen_only_full = gen[0, len(ids):].tolist()
    if use_thinking and thinking_token_id is not None:
        gen_only = [t for t in gen_only_full
                    if t != int(thinking_token_id)]
    else:
        gen_only = gen_only_full
    text = tokenizer.decode(gen_only, skip_special_tokens=True)
    text = _truncate_at_stop(text)
    full_code = prompt + text
    passed = _run_test_in_subprocess(
        full_code, problem["test"], problem["entry_point"],
        timeout_s=timeout_s)
    return {
        "task_id": problem["task_id"],
        "passed": bool(passed),
        "emit_tokens": len(gen_only),
        "think_tokens": int(diag.get("think_total", 0)),
        "preview": text[:120],
    }


def run_probe(model, tokenizer, *,
              probe_path: str = "data/probe_humaneval_50.jsonl",
              n_problems: Optional[int] = None,
              max_gen: int = 256,
              think_budget: int = 120,
              thinking_token_id: Optional[int] = None,
              max_think_per_step: int = 8,
              emit_threshold: float = 0.5,
              gate_floor: float = 0.0,
              min_emit_before_eos: int = 30,
              timeout_s: int = 5,
              ) -> dict:
    rows = _load_probe(probe_path, n_problems=n_problems)
    if not rows:
        return {
            "n_total": 0,
            "no_think": {"n_passed": 0, "mean_emit": 0.0, "mean_think": 0.0},
            "with_think": {"n_passed": 0, "mean_emit": 0.0, "mean_think": 0.0},
            "only_with_think": [], "only_no_think": [],
            "both": [], "neither": [], "per_problem": [],
        }

    if thinking_token_id is None:
        thinking_token_id = getattr(model, "thinking_token_id", None)
        if thinking_token_id is None:
            cfg = getattr(model, "config", None)
            if isinstance(cfg, dict):
                thinking_token_id = cfg.get("thinking_token_id")

    was_training = model.training
    model.eval()
    t0 = time.perf_counter()

    per_problem = []
    try:
        with torch.no_grad():
            for problem in rows:
                no_think_res = _run_one(
                    model, tokenizer, problem,
                    use_thinking=False, max_gen=max_gen,
                    total_think_budget=0,
                    thinking_token_id=thinking_token_id,
                    max_think_per_step=max_think_per_step,
                    emit_threshold=emit_threshold, gate_floor=gate_floor,
                    min_emit_before_eos=min_emit_before_eos,
                    timeout_s=timeout_s,
                )
                if think_budget > 0:
                    with_think_res = _run_one(
                        model, tokenizer, problem,
                        use_thinking=True, max_gen=max_gen,
                        total_think_budget=int(think_budget),
                        thinking_token_id=thinking_token_id,
                        max_think_per_step=max_think_per_step,
                        emit_threshold=emit_threshold, gate_floor=gate_floor,
                        min_emit_before_eos=min_emit_before_eos,
                        timeout_s=timeout_s,
                    )
                else:
                    with_think_res = dict(no_think_res)
                per_problem.append({
                    "task_id": problem["task_id"],
                    "no_think": no_think_res,
                    "with_think": with_think_res,
                })
    finally:
        if was_training:
            model.train()

    n_total = len(per_problem)

    def _avg(field, key):
        vs = [r[field][key] for r in per_problem]
        return sum(vs) / max(1, len(vs))

    no_think_passed = [r["task_id"] for r in per_problem
                       if r["no_think"]["passed"]]
    with_think_passed = [r["task_id"] for r in per_problem
                         if r["with_think"]["passed"]]
    no_set = set(no_think_passed)
    with_set = set(with_think_passed)
    only_with = sorted(with_set - no_set)
    only_no = sorted(no_set - with_set)
    both = sorted(no_set & with_set)
    neither = sorted(
        r["task_id"] for r in per_problem
        if not r["no_think"]["passed"] and not r["with_think"]["passed"])

    return {
        "n_total": n_total,
        "no_think": {
            "n_passed": len(no_think_passed),
            "pass_rate": len(no_think_passed) / max(1, n_total),
            "mean_emit": _avg("no_think", "emit_tokens"),
            "mean_think": _avg("no_think", "think_tokens"),
        },
        "with_think": {
            "n_passed": len(with_think_passed),
            "pass_rate": len(with_think_passed) / max(1, n_total),
            "mean_emit": _avg("with_think", "emit_tokens"),
            "mean_think": _avg("with_think", "think_tokens"),
            "think_budget": int(think_budget),
        },
        "only_with_think": only_with,
        "only_no_think": only_no,
        "both": both,
        "neither": neither,
        "per_problem": per_problem,
        "elapsed_s": time.perf_counter() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_jsonl", type=str,
                   default="data/probe_humaneval_50.jsonl")
    p.add_argument("--n_problems", type=int, default=None)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--think_budget", type=int, default=120)
    p.add_argument("--max_think_per_step", type=int, default=8)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--timeout_s", type=int, default=5)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer

    model, cfg = build_model_from_ckpt(args.ckpt)
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")

    res = run_probe(
        model, tok,
        probe_path=args.data_jsonl,
        n_problems=args.n_problems,
        max_gen=args.max_gen,
        think_budget=args.think_budget,
        thinking_token_id=thinking_token_id,
        max_think_per_step=args.max_think_per_step,
        emit_threshold=args.emit_threshold,
        gate_floor=args.gate_floor,
        min_emit_before_eos=args.min_emit_before_eos,
        timeout_s=args.timeout_s,
    )

    print(f"\nProbe 2 — counterfactual think_budget=0 vs {args.think_budget}")
    print(f"  ckpt: {args.ckpt}  n_problems={res['n_total']}")
    nt = res["no_think"]; wt = res["with_think"]
    print(f"  no_think:    n_passed={nt['n_passed']}/{res['n_total']}  "
          f"pass_rate={nt['pass_rate']:.3f}  "
          f"mean_emit={nt['mean_emit']:.1f}  "
          f"mean_think={nt['mean_think']:.1f}")
    print(f"  with_think:  n_passed={wt['n_passed']}/{res['n_total']}  "
          f"pass_rate={wt['pass_rate']:.3f}  "
          f"mean_emit={wt['mean_emit']:.1f}  "
          f"mean_think={wt['mean_think']:.1f}  "
          f"(budget={wt['think_budget']})")
    print(f"  only_with_think_passed: {len(res['only_with_think'])} "
          f"{res['only_with_think']}")
    print(f"  only_no_think_passed:   {len(res['only_no_think'])} "
          f"{res['only_no_think']}")
    print(f"  both: {len(res['both'])}    neither: {len(res['neither'])}")

    out_path = args.out_json or (
        str(pathlib.Path(args.ckpt).with_suffix("")) +
        ".thinking_counterfactual.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  full result -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
