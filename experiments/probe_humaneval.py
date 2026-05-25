"""
In-process HumanEval probe for the pretrain trainer.

Tiny (default 50 problems), runs on the same `model` object (no subprocess
loads a separate model), candidate code is graded in a *subprocess* via
`_run_test_in_subprocess` from `eval_humaneval.py` — cheap, isolates
faulty/looping generated code.

Public entry: ``run_humaneval_probe(model, tokenizer, ...) -> dict``.
"""
from __future__ import annotations

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
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if n_problems is not None and i >= n_problems:
                break
            rows.append(json.loads(line))
    return rows


def _max_T_from_model(model) -> int:
    cfg = getattr(model, "config", None)
    if isinstance(cfg, dict) and "max_T" in cfg:
        return int(cfg["max_T"])
    for attr in ("max_T", "max_seq_len", "max_position_embeddings"):
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 2048


def run_humaneval_probe(
    model,
    tokenizer,
    *,
    probe_path: str = "data/probe_humaneval_50.jsonl",
    max_gen: int = 192,
    n_problems: Optional[int] = None,
    timeout_s: int = 5,
    temperature: float = 0.0,
    verbose: bool = False,
) -> dict:
    """Run the small HumanEval-style probe.

    Returns ``{"pass_rate", "n_passed", "n_total", "mean_emit_tokens",
    "elapsed_s"}``. Restores ``model.training`` to its prior state.
    """
    rows = _load_probe(probe_path, n_problems)
    if not rows:
        return {
            "pass_rate": 0.0, "n_passed": 0, "n_total": 0,
            "mean_emit_tokens": 0.0, "elapsed_s": 0.0,
        }

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    max_T = _max_T_from_model(model)

    eos = getattr(tokenizer, "eos_token_id", None)
    n_passed = 0
    emit_tokens: list[int] = []
    t0 = time.perf_counter()

    try:
        with torch.no_grad():
            for problem in rows:
                prompt = problem["prompt"]
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(prompt_ids) + max_gen > max_T:
                    prompt_ids = prompt_ids[-(max_T - max_gen):]
                prompt_t = torch.tensor(
                    prompt_ids, dtype=torch.long, device=device
                ).unsqueeze(0)

                gen = generate(
                    model, prompt_t, max_gen=max_gen,
                    temperature=temperature, eos_token_id=eos,
                )
                gen_only = gen[0, len(prompt_ids):].tolist()
                emit_tokens.append(len(gen_only))
                gen_text = tokenizer.decode(gen_only, skip_special_tokens=True)
                gen_text = _truncate_at_stop(gen_text)
                full_code = prompt + gen_text
                passed = _run_test_in_subprocess(
                    full_code, problem["test"], problem["entry_point"],
                    timeout_s=timeout_s,
                )
                if passed:
                    n_passed += 1
                if verbose:
                    print(f"  {problem['task_id']:<14} "
                          f"{'PASS' if passed else 'fail':<5}  "
                          f"emit={len(gen_only):>4}")
    finally:
        if was_training:
            model.train()

    n_total = len(rows)
    elapsed = time.perf_counter() - t0
    return {
        "pass_rate": n_passed / max(1, n_total),
        "n_passed": n_passed,
        "n_total": n_total,
        "mean_emit_tokens": sum(emit_tokens) / max(1, len(emit_tokens)),
        "elapsed_s": elapsed,
    }
