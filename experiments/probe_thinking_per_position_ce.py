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
        for i, line in enumerate(f):
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


def _pos_ce_at(logits: torch.Tensor, target_id: int, pos: int) -> float:
    row = logits[0, pos].float()
    logp = F.log_softmax(row, dim=-1)
    return float(-logp[int(target_id)].item())


def _bucketise(values: list[float], n_buckets: int = 10) -> list[dict]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [{"lo": lo, "hi": hi, "count": len(values)}]
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
    probe_path: str = "data/probe_humaneval_50.jsonl",
    n_positions: int = 200,
    emit_threshold: float = 0.5,
    gate_floor: float = 0.0,
    thinking_token_id: Optional[int] = None,
    n_problems: Optional[int] = None,
    max_prompt_tokens: int = 1024,
) -> dict:
    rows = _load_probe(probe_path, n_problems=n_problems)
    if not rows:
        return {
            "n_positions_probed": 0,
            "n_positions_requested": int(n_positions),
            "mean_delta_ce": 0.0,
            "frac_positions_delta_positive": 0.0,
            "delta_histogram": [],
            "mean_ce_without_think": 0.0,
            "mean_ce_with_think": 0.0,
            "n_gate_fires_seen": 0,
        }

    if thinking_token_id is None:
        thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        cfg = getattr(model, "config", None)
        if isinstance(cfg, dict):
            thinking_token_id = cfg.get("thinking_token_id")
    if thinking_token_id is None:
        raise ValueError(
            "probe_thinking_per_position_ce requires a thinking_token_id "
            "(model.thinking_token_id, model.config['thinking_token_id'], "
            "or the --thinking_token_id CLI flag).")
    thinking_token_id = int(thinking_token_id)

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    max_T = _max_T_from_model(model)

    deltas: list[float] = []
    ce_without_list: list[float] = []
    ce_with_list: list[float] = []
    n_gate_fires_seen = 0

    t0 = time.perf_counter()
    try:
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
                room = max_T - 2
                if len(ids) > room:
                    ids = ids[:room]
                prompt_t = torch.tensor(
                    ids, dtype=torch.long, device=device).unsqueeze(0)

                base_logits, gate = _forward_logits_and_gate(model, prompt_t)
                if gate is None:
                    # No gate head — nothing to probe.
                    continue
                gate_row = gate[0].float()
                if gate_floor > 0.0:
                    gate_row = gate_row.clamp_min(gate_floor)
                # "Would think" = effective gate < emit_threshold.
                # Skip the final position (no target token).
                T = prompt_t.shape[1]
                fire_positions = [
                    int(t) for t in range(T - 1)
                    if float(gate_row[t].item()) < emit_threshold
                ]
                n_gate_fires_seen += len(fire_positions)
                if not fire_positions:
                    continue

                for t in fire_positions:
                    if len(deltas) >= n_positions:
                        break
                    target_id = int(ids[t + 1])
                    ce_no_think = _pos_ce_at(base_logits, target_id, t)

                    # Construct prefix[:t+1] + [THINK] and forward.
                    prefix = prompt_t[:, : t + 1]
                    think_tok = torch.full(
                        (1, 1), thinking_token_id,
                        dtype=prefix.dtype, device=device)
                    augmented = torch.cat([prefix, think_tok], dim=1)
                    logits_aug, _ = _forward_logits_and_gate(model, augmented)
                    # Position t+1 in `augmented` is the THINK token; its
                    # logits predict the next non-think token (the same
                    # target_id as before).
                    ce_with_think = _pos_ce_at(
                        logits_aug, target_id, t + 1)

                    deltas.append(ce_no_think - ce_with_think)
                    ce_without_list.append(ce_no_think)
                    ce_with_list.append(ce_with_think)
    finally:
        if was_training:
            model.train()

    n = len(deltas)
    if n == 0:
        return {
            "n_positions_probed": 0,
            "n_positions_requested": int(n_positions),
            "n_gate_fires_seen": int(n_gate_fires_seen),
            "mean_delta_ce": 0.0,
            "frac_positions_delta_positive": 0.0,
            "delta_histogram": [],
            "mean_ce_without_think": 0.0,
            "mean_ce_with_think": 0.0,
            "elapsed_s": time.perf_counter() - t0,
        }

    mean_delta = sum(deltas) / n
    frac_positive = sum(1 for d in deltas if d > 0) / n
    hist = _bucketise(deltas, n_buckets=10)
    return {
        "n_positions_probed": n,
        "n_positions_requested": int(n_positions),
        "n_gate_fires_seen": int(n_gate_fires_seen),
        "mean_delta_ce": mean_delta,
        "frac_positions_delta_positive": frac_positive,
        "mean_ce_without_think": sum(ce_without_list) / n,
        "mean_ce_with_think": sum(ce_with_list) / n,
        "delta_histogram": hist,
        "deltas": deltas,
        "ce_without_think": ce_without_list,
        "ce_with_think": ce_with_list,
        "thinking_token_id": int(thinking_token_id),
        "emit_threshold": float(emit_threshold),
        "elapsed_s": time.perf_counter() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_jsonl", type=str,
                   default="data/probe_humaneval_50.jsonl")
    p.add_argument("--n_positions", type=int, default=200)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--n_problems", type=int, default=None)
    p.add_argument("--max_prompt_tokens", type=int, default=1024)
    p.add_argument("--out_json", type=str, default=None,
                   help="Where to dump the full JSON result. Default: "
                        "<ckpt_stem>.thinking_perpos_ce.json beside the ckpt.")
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
        n_positions=args.n_positions,
        emit_threshold=args.emit_threshold,
        gate_floor=args.gate_floor,
        thinking_token_id=thinking_token_id,
        n_problems=args.n_problems,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    print(f"\nProbe 1 — per-position CE delta")
    print(f"  ckpt: {args.ckpt}")
    print(f"  positions_probed: {res['n_positions_probed']} "
          f"(requested {res['n_positions_requested']}, "
          f"gate_fires_seen {res['n_gate_fires_seen']})")
    print(f"  mean Δce (ce_without - ce_with): {res['mean_delta_ce']:+.4f}")
    print(f"  frac positions Δce > 0 (think helps): "
          f"{res['frac_positions_delta_positive']:.3f}")
    print(f"  mean ce_without_think: {res['mean_ce_without_think']:.4f}")
    print(f"  mean ce_with_think:    {res['mean_ce_with_think']:.4f}")
    print(f"  Δce histogram (10 buckets):")
    for b in res["delta_histogram"]:
        print(f"    [{b['lo']:+.3f}, {b['hi']:+.3f}]  n={b['count']}")

    out_path = args.out_json or (
        str(pathlib.Path(args.ckpt).with_suffix("")) +
        ".thinking_perpos_ce.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  full result -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
