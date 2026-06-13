"""Stage A kill-gate: does trained WM addressing help long-context recall?

The WM×latent COOPERATION kill-gate (M1). Runs the held-out long-context recall
eval twice on a cooperation ckpt (mem_alpha + DKV addressing):
  - WM-on  : wm_ablate='none'      (the trained cooperation channel active)
  - WM-off : wm_ablate='coop_off'  (mem_alpha→0 → latent step is adapter-only)
and reports recall(WM-on) − recall(WM-off), overall and on the kill-gate band
(distance ≥ --min_distance). M1 passes if the band delta > +15pp.

The coop_off arm is the ONLY correct WM-off ablation for a cooperation ckpt: the
latent step's WM channel is `adapter(h)+mem_alpha·WM_inj`, so zeroing mem_alpha
disables exactly the retrieval contribution without touching the adapter, trunk,
or gate. (The read_alpha 'mean' ablation does NOT disable cooperation — the
cooperation read uses _last_injection, stashed before read_alpha scaling.)

Usage:
  # a trained Stage A ckpt (has mem_alpha + memory.W_k in its state_dict)
  PYTHONPATH=. .venv/bin/python experiments/eval_stage_a_killgate.py \
      --ckpt checkpoints/wm_cotrain_stage_a.pt --n 150 --min_distance 384

  # the UNTRAINED baseline (attach fresh DKV + mem_alpha to a legacy WM ckpt)
  PYTHONPATH=. .venv/bin/python experiments/eval_stage_a_killgate.py \
      --ckpt checkpoints/sft_baked_pure.pt --fresh --n 150 --min_distance 384
"""
from __future__ import annotations

import argparse
import json
import tempfile

import torch


def _distance(rec) -> int:
    return int(rec.get("approx_distance_tokens", 0))


def _write_filtered(path: str, min_distance: int, n: int | None) -> tuple[str, int]:
    """Write a temp JSONL of tasks with distance >= min_distance (cap n)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if _distance(rec) >= min_distance:
                rows.append(line)
            if n is not None and len(rows) >= n:
                break
    tmp = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, prefix="killgate_")
    tmp.write("\n".join(rows))
    tmp.close()
    return tmp.name, len(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tasks", default="data/longctx_recall_heldout.jsonl")
    p.add_argument("--fresh", action="store_true",
                   help="attach a FRESH DKV + mem_alpha to a legacy WM ckpt "
                        "(the untrained-baseline arm); else auto-detect from "
                        "the ckpt state_dict (a trained Stage A ckpt).")
    p.add_argument("--n", type=int, default=150,
                   help="cap tasks per arm (full + band each capped).")
    p.add_argument("--min_distance", type=int, default=384,
                   help="kill-gate band: distance >= this (M1 is on this band).")
    p.add_argument("--max_gen", type=int, default=24)
    p.add_argument("--total_think_budget", type=int, default=64)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from experiments.eval_longctx_recall import eval_longctx_recall
    from transformers import AutoTokenizer

    if args.fresh:
        from experiments.thinking import load_latent_model
        model, cfg, tid, tok, eos = load_latent_model(
            args.ckpt, args.device, train=False, wm_on=True, dkv=True)
        print(f"[killgate] FRESH cooperation attached to {args.ckpt} "
              f"(mem_alpha={float(model.mem_alpha.detach()):.3f}, untrained DKV)")
    else:
        from experiments.eval_bracket_structure import build_model_from_ckpt
        model, cfg = build_model_from_ckpt(args.ckpt, force_state_readonly=True)
        tok = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
        ma = getattr(model, "mem_alpha", None)
        if ma is None:
            raise SystemExit(
                f"{args.ckpt} has no mem_alpha — not a cooperation ckpt. Use "
                "--fresh to attach a fresh channel for the untrained baseline.")
        print(f"[killgate] loaded cooperation ckpt {args.ckpt} "
              f"(mem_alpha={float(ma.detach()):.3f})")
    model.eval()

    def _run(path, n):
        on = eval_longctx_recall(
            model, tok, path=path, n=n, generator="latent_think",
            wm_ablate="none", max_gen=args.max_gen,
            total_think_budget=args.total_think_budget,
            emit_threshold=args.emit_threshold, gate_floor=args.gate_floor,
            device=args.device)
        off = eval_longctx_recall(
            model, tok, path=path, n=n, generator="latent_think",
            wm_ablate="coop_off", max_gen=args.max_gen,
            total_think_budget=args.total_think_budget,
            emit_threshold=args.emit_threshold, gate_floor=args.gate_floor,
            device=args.device)
        return on, off

    print("\n=== FULL held-out set ===")
    full_on, full_off = _run(args.tasks, args.n)
    print(f"  WM-on  recall={full_on['recall']:.3f} "
          f"(n={int(full_on['n_total'])}, think_rate={full_on['think_rate']:.3f})")
    print(f"  WM-off recall={full_off['recall']:.3f} "
          f"(n={int(full_off['n_total'])}, think_rate={full_off['think_rate']:.3f})")
    print(f"  Δ(on-off) = {100*(full_on['recall']-full_off['recall']):+.1f} pp")

    band_path, n_band = _write_filtered(args.tasks, args.min_distance, args.n)
    print(f"\n=== KILL-GATE band (distance ≥ {args.min_distance}, {n_band} tasks) ===")
    band_on, band_off = _run(band_path, args.n)
    d_band = 100 * (band_on["recall"] - band_off["recall"])
    print(f"  WM-on  recall={band_on['recall']:.3f} "
          f"(n={int(band_on['n_total'])}, think_rate={band_on['think_rate']:.3f})")
    print(f"  WM-off recall={band_off['recall']:.3f} "
          f"(n={int(band_off['n_total'])}, think_rate={band_off['think_rate']:.3f})")
    print(f"  Δ(on-off) = {d_band:+.1f} pp")

    verdict = "PASS" if d_band > 15.0 else "FAIL"
    print(f"\nKILL-GATE M1 (band Δ > +15pp): {verdict}  ({d_band:+.1f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
