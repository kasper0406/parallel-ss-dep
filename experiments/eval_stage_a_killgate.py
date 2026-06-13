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
    p.add_argument("--max_gen", type=int, default=12)
    p.add_argument("--total_think_budget", type=int, default=64)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--force_prefix_think", type=int, default=2,
                   help="force N latent think steps before the answer (B2: the "
                        "frozen gate emits at the answer position → without this "
                        "neither WM channel fires and the kill-gate is trivially "
                        "0). Set =R used in training.")
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

    def _eval(path, n, ablate):
        return eval_longctx_recall(
            model, tok, path=path, n=n, generator="latent_think",
            wm_ablate=ablate, max_gen=args.max_gen,
            total_think_budget=args.total_think_budget,
            emit_threshold=args.emit_threshold, gate_floor=args.gate_floor,
            force_prefix_think=args.force_prefix_think, device=args.device)

    def _report(label, path, n):
        # WM-on (cooperation active) vs WM-full-off (use_memory=False → BOTH the
        # direct read_alpha channel AND the mem_alpha cooperation channel off:
        # the only honest WM-off arm — M0 review B1). coop_off is the secondary
        # cooperation-channel-only probe.
        on = _eval(path, n, "none")
        full = _eval(path, n, "full_off")
        coop = _eval(path, n, "coop_off")
        d_full = 100 * (on["recall"] - full["recall"])
        d_coop = 100 * (on["recall"] - coop["recall"])
        print(f"\n=== {label} (n={int(on['n_total'])}, "
              f"think_rate={on['think_rate']:.2f}) ===")
        print(f"  WM-on      recall={on['recall']:.3f}")
        print(f"  WM-full-off recall={full['recall']:.3f}   "
              f"Δ(on − full_off) = {d_full:+.1f} pp   <- M1 metric")
        print(f"  coop-off   recall={coop['recall']:.3f}   "
              f"Δ(on − coop_off) = {d_coop:+.1f} pp   (cooperation-channel only)")
        if on["think_rate"] <= 0.0:
            print("  WARNING: think_rate=0 — thinking never fired; kill-gate "
                  "is uninformative (raise --force_prefix_think).")
        return d_full

    print(f"[killgate] force_prefix_think={args.force_prefix_think}")
    _report("FULL held-out set", args.tasks, args.n)

    band_path, n_band = _write_filtered(args.tasks, args.min_distance, args.n)
    d_band = _report(f"KILL-GATE band (distance ≥ {args.min_distance}, "
                     f"{n_band} tasks)", band_path, args.n)

    verdict = "PASS" if d_band > 15.0 else "FAIL"
    print(f"\nKILL-GATE M1 (band Δ(on − full_off) > +15pp): {verdict}  "
          f"({d_band:+.1f} pp)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
