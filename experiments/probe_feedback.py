"""
Diagnostic probe of trained feedback models.

Answers: how much is the architecture *actually doing* with the feedback
machinery? Specifically:

1. **alpha distribution** — final α per layer per distance. Already
   logged during training; reload from checkpoint for clean snapshot.

2. **Feedback contribution magnitude** — at inference on real data,
   measure ||α · scale · h|| / ||h|| for each layer. Tells us how much
   the modulation actually moves the residual stream as a fraction.

3. **Pass-1 vs pass-2 output divergence** — for each layer, compute
   ||h_pass2 - h_pass1|| / ||h_pass1||. If small (<5%), pass 2 is
   nearly a no-op; if large, feedback is making real changes.

4. **Feedback gradient flow** — magnitude of gradients on W_scale,
   W_shift compared to W_q/W_k/W_v gradients. Tells us if optimizer
   is putting any work into feedback weights vs main attention.

Usage:
  python experiments/probe_feedback.py --ckpt /path/to/film_multi.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.model import _shift_right_by_1


@torch.no_grad()
def probe(ckpt_path: str, n_batches: int = 4, batch: int = 4, T: int = 512):
    print(f"\n{'=' * 70}\nProbing: {ckpt_path}\n{'=' * 70}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    print(f"  feedback={cfg.get('feedback_mode')}  n_layers={cfg['n_layers']}  "
          f"d_model={cfg['d_model']}  distances={cfg.get('feedback_distances', (1,))}")

    if cfg.get("feedback_mode") == "none":
        print("  (no feedback — baseline checkpoint)")
        return

    # 1. Alpha distribution from saved weights.
    alphas = model.feedback_alphas()
    print("\n--- α per layer ---")
    if isinstance(alphas[0], list):
        # Multi-scale: one row per layer, K cols.
        distances = cfg["feedback_distances"]
        header = "  L   " + "  ".join(f"d={d:>3d}" for d in distances)
        print(header)
        for L, row in enumerate(alphas):
            print(f"  {L:>3d}   " + "  ".join(f"{a:+.3f}" for a in row))
    else:
        for L, a in enumerate(alphas):
            print(f"  L={L:>3d}: {a:+.4f}")

    # 2. Feedback contribution magnitude. To measure this we need pass-1
    # outputs, then for each layer compute ||film(h, state_above)||/||h||.
    print("\n--- Feedback contribution magnitudes (at inference) ---")
    print("  Per layer, mean of ||(film(h, state_above) - h) / ||h|||| over batches.")

    # Load some real val data
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    val_stream = load_dataset("codeparrot/codeparrot-clean", split="train",
                               streaming=True).shuffle(seed=42).skip(20_000)
    buf: list[int] = []
    eos = tok.eos_token_id or 0
    samples = []
    for example in val_stream:
        text = example["content"]
        ids = tok.encode(text, add_special_tokens=False)
        buf.extend(ids); buf.append(eos)
        while len(buf) >= batch * T:
            chunk = buf[: batch * T]
            buf = buf[batch * T:]
            samples.append(torch.tensor(chunk, dtype=torch.long).view(batch, T))
            if len(samples) >= n_batches:
                break
        if len(samples) >= n_batches:
            break

    # Hook to capture per-layer h (input to each block) and feedback modulation
    # We do this by manually running the 2-pass forward and recording at each
    # layer step.
    contribs = [[] for _ in model.feedback]
    pass2_diff = [[] for _ in model.feedback]

    for x in samples:
        x = x.to("cuda")
        h = model.embed(x)
        # Pass 1: vanilla
        pass1 = []
        h1 = h
        for blk in model.blocks:
            h1 = blk(h1)
            pass1.append(h1)
        # Pass 2 with feedback, tracking contributions
        h2 = h
        N = len(model.blocks)
        for L, blk in enumerate(model.blocks):
            states_above = []
            for d in model.feedback_distances:
                src = L + d
                if src < N:
                    states_above.append(_shift_right_by_1(pass1[src]))
                else:
                    states_above.append(None)
            h_modulated = model.feedback[L](h2, states_above)
            # Contribution: how much did feedback move h?
            diff = (h_modulated - h2).norm(dim=-1)
            base = h2.norm(dim=-1).clamp_min(1e-8)
            contribs[L].append((diff / base).mean().item())
            h2 = blk(h_modulated)
            # Pass-2 vs pass-1 divergence for this layer
            d12 = (h2 - pass1[L]).norm(dim=-1)
            b1 = pass1[L].norm(dim=-1).clamp_min(1e-8)
            pass2_diff[L].append((d12 / b1).mean().item())

    print(f"  {'L':>3}  {'fb contrib':>12}  {'pass1_vs_pass2':>14}")
    for L in range(len(contribs)):
        c = sum(contribs[L]) / len(contribs[L])
        d = sum(pass2_diff[L]) / len(pass2_diff[L])
        print(f"  {L:>3}  {c*100:>11.2f}%  {d*100:>13.2f}%")

    # Summary stats
    mean_contrib = sum(sum(c) / len(c) for c in contribs) / len(contribs)
    mean_diff = sum(sum(d) / len(d) for d in pass2_diff) / len(pass2_diff)
    print(f"\n  Average feedback contribution: {mean_contrib*100:.2f}% of ||h||")
    print(f"  Average pass-2 vs pass-1 layer-output divergence: {mean_diff*100:.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, action="append")
    p.add_argument("--n_batches", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--T", type=int, default=512)
    args = p.parse_args()

    for ckpt in args.ckpt:
        probe(ckpt, n_batches=args.n_batches, batch=args.batch, T=args.T)
    return 0


if __name__ == "__main__":
    sys.exit(main())
