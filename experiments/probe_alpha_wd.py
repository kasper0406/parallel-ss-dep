"""One-shot probe: is the FiLM α saturated, in WD-equilibrium, or wanting
more?

At a converged-ish parameter, the AdamW update rule for an unbounded scalar
is roughly:
    new = old − lr · (m_first / (sqrt(m_second) + eps) + WD · old)

If the model has *truly saturated* (loss is flat in the α direction), the
first-moment gradient ≈ 0, so the only force on α is `−lr · WD · old`, and
α slowly shrinks. If instead the loss-gradient and the WD-pull-down are
matched, you get a steady-state equilibrium where neither force wins.

This probe loads the checkpoint, runs a small averaged forward+backward,
and reports |∂L/∂α| against WD · α.

Usage:
    PYTHONPATH=. python experiments/probe_alpha_wd.py \\
        --ckpt checkpoints/pretrain_mix_v1_step61036_tok500006912.pt \\
        --config configs/pretrain_mix_v1.yaml \\
        --n_batches 8 --batch 4

Reports per-batch and averaged grad-on-α; classifies regime.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.data_mix import MixedSourceStream, load_sources_from_yaml
from experiments.eval_bracket_structure import build_model_from_ckpt


def find_alpha_params(model) -> dict[str, torch.nn.Parameter]:
    """Return a name→param dict of all FiLM α parameters reachable from
    `model.sparse_feedback['<L>'].alpha`."""
    out: dict[str, torch.nn.Parameter] = {}
    sf = getattr(model, "sparse_feedback", None)
    if sf is None:
        return out
    for key in sf.keys():
        proj = sf[key]
        if hasattr(proj, "alpha") and isinstance(proj.alpha, torch.nn.Parameter):
            out[f"sparse_feedback['{key}'].alpha"] = proj.alpha
    return out


def classify(grad_mag: float, wd_term: float) -> str:
    if wd_term == 0:
        return "N/A (WD=0)"
    ratio = grad_mag / wd_term
    if ratio < 0.5:
        return ("TRUE SATURATION — loss is flat in α direction; α would "
                "decay under WD even with no other force")
    if ratio > 2.0:
        return ("WD UNDER-WEIGHTING — loss strongly wants higher α; WD is "
                "the brake, not a saturated mechanism")
    return ("WD EQUILIBRIUM — gradient pull-up ~ WD pull-down; α is at the "
            "balance point. Removing WD or re-parameterising would let α "
            "climb further.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True,
                   help="data mix YAML to source representative batches from")
    p.add_argument("--n_batches", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--weight_decay", type=float, default=0.1,
                   help="The Muon WD that's applied to α during training. "
                        "Default matches the launcher's value.")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model.train()                       # need grads
    alpha_params = find_alpha_params(model)
    if not alpha_params:
        raise SystemExit("No sparse_feedback α parameters found in model.")
    print(f"Found {len(alpha_params)} α parameters:")
    for name, p in alpha_params.items():
        val = p.item() if p.numel() == 1 else float(p.mean().item())
        print(f"  {name}: shape={tuple(p.shape)}  value={val:+.6f}")

    # Build streamer matching the original training data.
    from transformers import AutoTokenizer
    tok_name = cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M")
    tok = AutoTokenizer.from_pretrained(tok_name)
    sources = load_sources_from_yaml(args.config)
    thinking_id = int(cfg.get("thinking_token_id", tok.vocab_size))
    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.T,
        thinking_token_id=thinking_id,
        # No bursts — we want a clean signal, not the burst-injection
        # version of the training data.
        think_burst_prob=0.0,
        base_seed=12345,
    )
    loader = DataLoader(ds, batch_size=args.batch, num_workers=1)
    it = iter(loader)

    # Accumulate gradients on α across N batches (do not zero_grad between
    # them — pytorch sums; we'll divide by N at the end). This averages out
    # batch-level noise.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    per_batch: list[dict[str, float]] = []
    for b in range(args.n_batches):
        # Need per-batch grad to also report variability. So zero grads here
        # and read after each backward.
        for p in alpha_params.values():
            if p.grad is not None:
                p.grad.zero_()
        x, y = next(it)
        x, y = x.to("cuda"), y.to("cuda")
        logits = model(x)
        V = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                              reduction="none")
        valid = (y != -100).float().reshape(-1)
        loss = (ce * valid).sum() / valid.sum().clamp(min=1.0)
        loss.backward()
        rec = {name: (p.grad.item() if p.grad.numel() == 1
                       else float(p.grad.mean().item()))
                for name, p in alpha_params.items()}
        per_batch.append(rec)
        # Avoid grad accumulation across batches blowing up memory.
        alpha_ids = {id(p) for p in alpha_params.values()}
        for p in model.parameters():
            if p.grad is not None and id(p) not in alpha_ids:
                p.grad = None

    print(f"\nPer-batch gradients on α (over {args.n_batches} batches):")
    print(f"  {'param':<30} {'value':>10} {'mean(g)':>10} {'std(g)':>10} "
          f"{'|mean|/(WD·α)':>16} {'verdict'}")
    for name, p in alpha_params.items():
        val = p.item() if p.numel() == 1 else float(p.mean().item())
        grads = [rec[name] for rec in per_batch]
        gmean = sum(grads) / len(grads)
        gstd = (sum((g - gmean) ** 2 for g in grads) / max(1, len(grads) - 1)
                ) ** 0.5
        wd_term = args.weight_decay * abs(val)
        ratio = abs(gmean) / (wd_term + 1e-12)
        verdict = classify(abs(gmean), wd_term)
        print(f"  {name:<30} {val:>+10.4f} {gmean:>+10.5f} "
              f"{gstd:>10.5f} {ratio:>16.3f}")
        print(f"    → {verdict}")
        print(f"    (WD={args.weight_decay} · |α|={abs(val):.4f} = "
              f"{wd_term:.5f}; per-batch grads: "
              f"{[f'{g:+.5f}' for g in grads]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
