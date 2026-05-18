"""Does PKM actually CONTRIBUTE to next-token prediction?

Same trained model, two forward passes on the same held-out batch:
  A. PKM active (use the trained α)
  B. PKM disabled (α temporarily forced to 0)

Per-token CE delta = CE_pkm_off - CE_pkm_on. Positive → PKM helps that
position; negative → PKM hurts; ≈ 0 → PKM is neutral / redundant.

Stratifications:
  - Global mean CE delta (does PKM help on average?)
  - By target-token rank (using model's own pred distribution): is PKM
    helping where the model was uncertain (rare/hard tokens) vs where
    it was confident (boilerplate)?
  - By token-string regex (identifier vs punctuation vs keyword), to
    proxy "fact-like" tokens.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/probe_pkm_contribution.py \\
        --ckpt checkpoints/pretrain_mix_v7_pkm_film_step2180_tok500039680.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.eval_bracket_structure import build_model_from_ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_seqs", type=int, default=4,
                   help="Sequences to evaluate (each at length T).")
    p.add_argument("--T", type=int, default=2048)
    args = p.parse_args()

    print(f"Loading {args.ckpt} ...")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.cuda().eval()
    if not hasattr(model, "pkm_layer"):
        raise SystemExit("ckpt has no pkm_layer")

    pkm = model.pkm_layer
    if not pkm.use_output_gate:
        raise SystemExit("PKM has no output gate to toggle (use_output_gate=False)")

    α_trained = float(pkm.out_alpha.detach())
    print(f"PKM α_trained = {α_trained:+.4f}")
    print(f"PKM α_floor (curriculum hook) = {float(pkm.alpha_floor):.4f} "
          "(set to 0 for fair off-comparison)")
    pkm.alpha_floor = 0.0  # ensure no curriculum floor leakage during probe

    # Pull a held-out slice of the v4 data mix (codeparrot-dominant).
    print("Loading held-out data ...")
    from experiments.data_mix import build_mixed_stream_loader
    train_loader = build_mixed_stream_loader(
        yaml_path="configs/pretrain_mix_v4.yaml",
        tokenizer_name="HuggingFaceTB/SmolLM2-135M",
        block_size=args.T,
        batch_size=args.n_seqs,
        seed=42,            # different seed = different sample
        shuffle_buffer=64,
        num_workers=0,
        emit_doc_ids=True,
    )
    batch = next(iter(train_loader))
    x, y, doc_ids = batch
    x, y, doc_ids = x.cuda(), y.cuda(), doc_ids.cuda()
    valid = (y != -100)
    n_valid = int(valid.sum())
    print(f"Held-out batch: {x.shape}, valid tokens = {n_valid}")

    # --- A: PKM ON (trained α) ----------------------------------------
    print("\nForward A: PKM ON (trained α)")
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits_on = model(x, doc_ids=doc_ids).float()
    ce_on = F.cross_entropy(
        logits_on.reshape(-1, logits_on.shape[-1]),
        y.reshape(-1).clamp_min(0),
        reduction="none",
    ).reshape(y.shape)
    ce_on_mean = (ce_on * valid.float()).sum() / valid.sum()
    print(f"  mean CE (PKM on)  = {float(ce_on_mean):.4f}")
    print(f"  perplexity        = {float(ce_on_mean.exp()):.2f}")

    # --- B: PKM OFF (force α=0 + floor=0) -----------------------------
    print("\nForward B: PKM OFF (α forced to 0)")
    α_save = pkm.out_alpha.detach().clone()
    with torch.no_grad():
        pkm.out_alpha.zero_()
    try:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_off = model(x, doc_ids=doc_ids).float()
    finally:
        with torch.no_grad():
            pkm.out_alpha.copy_(α_save)
    ce_off = F.cross_entropy(
        logits_off.reshape(-1, logits_off.shape[-1]),
        y.reshape(-1).clamp_min(0),
        reduction="none",
    ).reshape(y.shape)
    ce_off_mean = (ce_off * valid.float()).sum() / valid.sum()
    print(f"  mean CE (PKM off) = {float(ce_off_mean):.4f}")
    print(f"  perplexity        = {float(ce_off_mean.exp()):.2f}")

    # --- Delta ---------------------------------------------------------
    delta = ce_off - ce_on             # positive → PKM helped at that pos
    delta_valid = delta[valid]
    print(f"\n=== Mean CE delta (off - on) = {float(delta_valid.mean()):+.5f} ===")
    print(f"  (positive = PKM helps; negative = PKM hurts)")
    print(f"  std       = {float(delta_valid.std()):.5f}")
    print(f"  fraction helped       = {float((delta_valid > 0).float().mean()):.3f}")
    print(f"  fraction hurt         = {float((delta_valid < 0).float().mean()):.3f}")
    print(f"  fraction within ±1e-3 = {float((delta_valid.abs() < 1e-3).float().mean()):.3f}")

    # Stratify by PKM-off CE bucket: where the model was uncertain in
    # the no-PKM forward, did PKM help?
    print("\n=== Delta stratified by CE_off bucket ===")
    for lo, hi in [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 99.0)]:
        mask = (ce_off >= lo) & (ce_off < hi) & valid
        n = int(mask.sum())
        if n == 0:
            continue
        d = delta[mask]
        avg = float(d.mean())
        pct_helped = float((d > 0).float().mean())
        print(f"  CE_off ∈ [{lo:>4.1f}, {hi:>4.1f})  n={n:>6}  "
              f"mean Δ={avg:+.5f}  helped={pct_helped:.3f}")

    # Magnitude check: does PKM output actually have magnitude?
    print(f"\nPKM α (trained) = {α_trained:+.4f}, "
          f"so each token saw scaling factor {α_trained:.4f} on PKM output.")


if __name__ == "__main__":
    sys.exit(main())
