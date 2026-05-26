"""Probe: does inserting K think tokens reduce next-token error?

Mirrors the Phase A training signal in evaluation. On a held-out batch
of code-like prompts, at each position where the gate already wants to
fire (σ > --emit_threshold), we measure:

    Δlogp(t) = log p_after(y_{t+1}) - log p_before(y_{t+1})

where `y_{t+1}` is the true next token in the prompt:
  - `p_before` = main forward at position t (no thinks inserted)
  - `p_after`  = forward over [prompt[:t+1], K * THINK_ID], read at the
                 last position (K thinks then predict y_{t+1})

Output:
  - histogram of Δlogp
  - mean Δlogp
  - fraction of positions with Δlogp > 0 (think helps)

A model that has been trained with --process_reward_weight>0 should
have a noticeably-positive mean Δlogp; an untrained model
(`checkpoints/sft_phase_c_combined.pt`) is the canonical "thinking
mostly noise" baseline.

Usage:
  PYTHONPATH=. python experiments/probe_process_reward.py \\
      --ckpt checkpoints/sft_phase_c_combined.pt \\
      --K 4 --n_positions 200
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


def _load_probe(probe_path: str, n_problems: Optional[int] = None) -> list[dict]:
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


def _forward_logits_and_gate(model, ids: torch.Tensor):
    out = model(ids)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    gate = getattr(model, "_last_gate", None)
    return logits, gate


def _logp_at(logits: torch.Tensor, target_id: int, pos: int) -> float:
    row = logits[0, pos].float()
    logp = F.log_softmax(row, dim=-1)
    return float(logp[int(target_id)].item())


def _bucketise(values: list[float], n_buckets: int = 11) -> list[dict]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [{"lo": lo, "hi": hi, "count": len(values)}]
    # Use symmetric bins around 0 when possible.
    bound = max(abs(lo), abs(hi))
    lo, hi = -bound, +bound
    width = (hi - lo) / n_buckets
    buckets = [
        {"lo": lo + i * width, "hi": lo + (i + 1) * width, "count": 0}
        for i in range(n_buckets)
    ]
    for v in values:
        idx = min(int((v - lo) / width), n_buckets - 1)
        buckets[idx]["count"] += 1
    return buckets


def run_probe(
    model,
    tokenizer,
    *,
    probe_path: str,
    n_positions: int = 200,
    K: int = 4,
    emit_threshold: float = 0.5,
    thinking_token_id: Optional[int] = None,
    n_problems: Optional[int] = None,
    max_prompt_tokens: int = 1024,
) -> dict:
    rows = _load_probe(probe_path, n_problems=n_problems)
    if thinking_token_id is None:
        thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        cfg = getattr(model, "config", None)
        if isinstance(cfg, dict):
            thinking_token_id = cfg.get("thinking_token_id")
    if thinking_token_id is None:
        raise ValueError(
            "probe_process_reward requires a thinking_token_id "
            "(model.thinking_token_id, model.config['thinking_token_id'], "
            "or the --thinking_token_id CLI flag).")
    thinking_token_id = int(thinking_token_id)

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    max_T = _max_T_from_model(model)

    deltas: list[float] = []
    logp_before_list: list[float] = []
    logp_after_list: list[float] = []
    n_gate_fires_seen = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for problem in rows:
            if len(deltas) >= n_positions:
                break
            prompt_text = problem.get("prompt") or problem.get("text") or ""
            if not prompt_text:
                continue
            ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            if len(ids) > max_prompt_tokens:
                ids = ids[:max_prompt_tokens]
            # Leave headroom for K think tokens at the end.
            room = max_T - K - 2
            if len(ids) > room:
                ids = ids[:room]
            prompt_t = torch.tensor(
                ids, dtype=torch.long, device=device).unsqueeze(0)

            base_logits, gate = _forward_logits_and_gate(model, prompt_t)
            if gate is None:
                continue
            gate_row = gate[0].float()
            T = prompt_t.shape[1]
            fire_positions = [
                int(t) for t in range(T - 1)
                if float(gate_row[t].item()) > emit_threshold
            ]
            n_gate_fires_seen += len(fire_positions)
            if not fire_positions:
                continue

            for t in fire_positions:
                if len(deltas) >= n_positions:
                    break
                target_id = int(ids[t + 1])
                logp_before = _logp_at(base_logits, target_id, t)

                # Construct prefix[:t+1] + K * [THINK] and forward.
                prefix = prompt_t[:, : t + 1]
                think_block = torch.full(
                    (1, K), thinking_token_id,
                    dtype=prefix.dtype, device=device)
                augmented = torch.cat([prefix, think_block], dim=1)
                logits_aug, _ = _forward_logits_and_gate(model, augmented)
                # Last position holds the K-thinks prediction of y_{t+1}.
                logp_after = _logp_at(
                    logits_aug, target_id, augmented.shape[1] - 1)

                deltas.append(logp_after - logp_before)
                logp_before_list.append(logp_before)
                logp_after_list.append(logp_after)

    if was_training:
        model.train()

    n = len(deltas)
    elapsed = time.perf_counter() - t0
    if n == 0:
        return {
            "n_positions_probed": 0,
            "n_positions_requested": int(n_positions),
            "n_gate_fires_seen": int(n_gate_fires_seen),
            "mean_delta_logp": 0.0,
            "median_delta_logp": 0.0,
            "frac_positions_delta_positive": 0.0,
            "delta_histogram": [],
            "mean_logp_before": 0.0,
            "mean_logp_after": 0.0,
            "K": int(K),
            "emit_threshold": float(emit_threshold),
            "elapsed_s": float(elapsed),
        }

    sorted_deltas = sorted(deltas)
    median = sorted_deltas[n // 2]
    mean_delta = sum(deltas) / n
    frac_positive = sum(1 for d in deltas if d > 0) / n
    hist = _bucketise(deltas, n_buckets=11)

    return {
        "n_positions_probed": int(n),
        "n_positions_requested": int(n_positions),
        "n_gate_fires_seen": int(n_gate_fires_seen),
        "mean_delta_logp": float(mean_delta),
        "median_delta_logp": float(median),
        "frac_positions_delta_positive": float(frac_positive),
        "delta_histogram": hist,
        "mean_logp_before": float(sum(logp_before_list) / n),
        "mean_logp_after": float(sum(logp_after_list) / n),
        "K": int(K),
        "emit_threshold": float(emit_threshold),
        "elapsed_s": float(elapsed),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_jsonl", default="data/probe_humaneval_50.jsonl")
    p.add_argument("--n_positions", type=int, default=200)
    p.add_argument("--K", type=int, default=4,
                   help="Number of think tokens inserted before the "
                        "target position for the 'after' forward.")
    p.add_argument("--emit_threshold", type=float, default=0.5,
                   help="Only probe positions where σ(gate) > this "
                        "(positions the gate already wants to think at).")
    p.add_argument("--thinking_token_id", type=int, default=None)
    p.add_argument("--n_problems", type=int, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=1024)
    p.add_argument("--out_json", type=str, default=None)
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from transformers import AutoTokenizer

    print(f"loading: {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    thinking_token_id = args.thinking_token_id
    if thinking_token_id is None:
        thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")

    res = run_probe(
        model, tok,
        probe_path=args.data_jsonl,
        n_positions=args.n_positions,
        K=args.K,
        emit_threshold=args.emit_threshold,
        thinking_token_id=thinking_token_id,
        n_problems=args.n_problems,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    print()
    print(f"Process-reward probe — does K={res['K']} thinks reduce next-token error?")
    print(f"  ckpt: {args.ckpt}")
    print(f"  positions_probed: {res['n_positions_probed']} "
          f"(requested {res['n_positions_requested']}, "
          f"gate_fires_seen {res['n_gate_fires_seen']})")
    print(f"  mean Δlogp (after - before): {res['mean_delta_logp']:+.4f}")
    print(f"  median Δlogp: {res['median_delta_logp']:+.4f}")
    print(f"  frac positions Δlogp > 0 (think helps): "
          f"{res['frac_positions_delta_positive']:.3f}")
    print(f"  mean logp_before (no think): {res['mean_logp_before']:.4f}")
    print(f"  mean logp_after (K thinks):  {res['mean_logp_after']:.4f}")
    print(f"  Δlogp histogram (symmetric):")
    for b in res["delta_histogram"]:
        bar = "#" * min(60, b["count"])
        print(f"    [{b['lo']:+.3f}, {b['hi']:+.3f}]  n={b['count']:>4}  {bar}")
    print(f"  elapsed: {res['elapsed_s']:.1f}s")

    out_path = args.out_json or (
        str(pathlib.Path(args.ckpt).with_suffix("")) +
        ".process_reward_probe.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  full result -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
