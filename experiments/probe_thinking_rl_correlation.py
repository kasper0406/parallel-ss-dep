from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_humaneval import generate


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    def _ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    denx = math.sqrt(sum((r - mean_rx) ** 2 for r in rx))
    deny = math.sqrt(sum((r - mean_ry) ** 2 for r in ry))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def _bucket_depth(n_think: int) -> str:
    if n_think == 0:
        return "0"
    if n_think <= 30:
        return "1-30"
    if n_think <= 60:
        return "31-60"
    if n_think <= 90:
        return "61-90"
    return "91-120+"


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


def _grade_completion(problem, completion: str, *, grade_fn,
                      thinking_token_id: Optional[int] = None,
                      timeout_s: int = 7) -> float:
    res = grade_fn(problem, completion, timeout_s=timeout_s)
    return float(res.score)


def _think_positions_from_ids(full_ids: list[int],
                              prompt_len: int,
                              thinking_token_id: int) -> list[int]:
    """Return think-position offsets relative to start of generation
    (i.e. position in [0, gen_len))."""
    out = []
    for i, tok in enumerate(full_ids[prompt_len:]):
        if int(tok) == int(thinking_token_id):
            out.append(i)
    return out


def run_probe(model, tokenizer, problems, *, grade_fn,
              n_problems: int = 50,
              n_rollouts_per_problem: int = 8,
              temperature: float = 0.7,
              max_gen: int = 192,
              total_think_budget: int = 120,
              max_think_per_step: int = 8,
              thinking_token_id: Optional[int] = None,
              emit_threshold: float = 0.5,
              gate_floor: float = 0.0,
              min_emit_before_eos: int = 30,
              timeout_s: int = 7,
              ) -> dict:
    if thinking_token_id is None:
        thinking_token_id = getattr(model, "thinking_token_id", None)
        if thinking_token_id is None:
            cfg = getattr(model, "config", None)
            if isinstance(cfg, dict):
                thinking_token_id = cfg.get("thinking_token_id")
    if thinking_token_id is None:
        raise ValueError("thinking_token_id is required for probe 3")
    thinking_token_id = int(thinking_token_id)

    probs = problems[:int(n_problems)]
    if not probs:
        return {
            "n_rollouts": 0, "spearman_think_vs_reward": 0.0,
            "buckets": {}, "per_rollout": [],
        }

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    max_T = _max_T_from_model(model)
    eos = getattr(tokenizer, "eos_token_id", None)

    per_rollout = []
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            for problem in probs:
                ids = tokenizer.encode(problem.prompt, add_special_tokens=False)
                room = max_gen + total_think_budget
                if len(ids) + room > max_T:
                    ids = ids[-(max_T - room):]
                prompt_t = torch.tensor(ids, dtype=torch.long,
                                        device=device).unsqueeze(0)
                prompt_len = len(ids)
                for _ in range(int(n_rollouts_per_problem)):
                    gen, diag = generate(
                        model, prompt_t, max_gen=max_gen,
                        temperature=temperature, eos_token_id=eos,
                        use_thinking=True,
                        thinking_token_id=thinking_token_id,
                        max_think_per_step=max_think_per_step,
                        total_think_budget=total_think_budget,
                        emit_threshold=emit_threshold,
                        gate_floor=gate_floor,
                        min_emit_before_eos=min_emit_before_eos,
                    )
                    full_ids = gen[0].tolist()
                    think_positions = _think_positions_from_ids(
                        full_ids, prompt_len, thinking_token_id)
                    gen_only = [t for t in full_ids[prompt_len:]
                                if t != thinking_token_id]
                    completion = tokenizer.decode(
                        gen_only, skip_special_tokens=True)
                    reward = _grade_completion(
                        problem, completion, grade_fn=grade_fn,
                        thinking_token_id=thinking_token_id,
                        timeout_s=timeout_s)
                    per_rollout.append({
                        "task_id": problem.task_id,
                        "n_think_tokens": int(diag.get("think_total", 0)),
                        "n_emit_tokens": int(diag.get("emit_count", 0)),
                        "reward": float(reward),
                        "think_positions": think_positions,
                        "bucket": _bucket_depth(int(diag.get("think_total", 0))),
                    })
    finally:
        if was_training:
            model.train()

    xs = [r["n_think_tokens"] for r in per_rollout]
    ys = [r["reward"] for r in per_rollout]
    rho = _spearman([float(x) for x in xs], ys)

    buckets: dict[str, list[float]] = {
        "0": [], "1-30": [], "31-60": [], "61-90": [], "91-120+": []
    }
    for r in per_rollout:
        buckets[r["bucket"]].append(r["reward"])
    bucket_stats = {
        k: {"n": len(v), "mean_reward": (sum(v) / len(v)) if v else 0.0}
        for k, v in buckets.items()
    }

    return {
        "n_rollouts": len(per_rollout),
        "n_problems": len(probs),
        "n_rollouts_per_problem": int(n_rollouts_per_problem),
        "temperature": float(temperature),
        "spearman_think_vs_reward": rho,
        "buckets": bucket_stats,
        "per_rollout": per_rollout,
        "elapsed_s": time.perf_counter() - t0,
    }


def _load_problems(dataset: str, n: int):
    from experiments import code_grader as cg
    loader_name = f"load_{dataset}"
    fn = getattr(cg, loader_name, None)
    if fn is None:
        raise ValueError(f"unknown dataset '{dataset}' (no {loader_name})")
    return fn()[:int(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mbpp_combined")
    p.add_argument("--n_problems", type=int, default=50)
    p.add_argument("--n_rollouts_per_problem", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_gen", type=int, default=192)
    p.add_argument("--total_think_budget", type=int, default=120)
    p.add_argument("--max_think_per_step", type=int, default=8)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--timeout_s", type=int, default=7)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.code_grader import grade
    from transformers import AutoTokenizer

    model, cfg = build_model_from_ckpt(args.ckpt)
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")

    problems = _load_problems(args.dataset, args.n_problems)

    res = run_probe(
        model, tok, problems, grade_fn=grade,
        n_problems=args.n_problems,
        n_rollouts_per_problem=args.n_rollouts_per_problem,
        temperature=args.temperature,
        max_gen=args.max_gen,
        total_think_budget=args.total_think_budget,
        max_think_per_step=args.max_think_per_step,
        thinking_token_id=thinking_token_id,
        emit_threshold=args.emit_threshold,
        gate_floor=args.gate_floor,
        min_emit_before_eos=args.min_emit_before_eos,
        timeout_s=args.timeout_s,
    )

    print(f"\nProbe 3 — think-count vs reward correlation across rollouts")
    print(f"  ckpt: {args.ckpt}  dataset={args.dataset}  "
          f"n_problems={res['n_problems']}  "
          f"rollouts_per_problem={res['n_rollouts_per_problem']}  "
          f"τ={res['temperature']}")
    print(f"  n_rollouts: {res['n_rollouts']}")
    print(f"  Spearman ρ(n_think_tokens, reward): "
          f"{res['spearman_think_vs_reward']:+.4f}")
    print(f"  bucketed mean reward:")
    for k in ["0", "1-30", "31-60", "61-90", "91-120+"]:
        b = res["buckets"][k]
        print(f"    depth {k:<8}  n={b['n']:>4}  "
              f"mean_reward={b['mean_reward']:.4f}")

    out_path = args.out_json or (
        str(pathlib.Path(args.ckpt).with_suffix("")) +
        ".thinking_rl_correlation.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  full result -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
