"""
Phase 22 ablations: build a random-initialised 217M plain-DN checkpoint
that has the SAME architecture as the trained baseline encoder
(`dn_baseline_30L_217M_for_oracle.pt`) but with un-trained weights.

Used by Ablation 1 to test whether the encoder's training matters.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                    default="checkpoints/dn_random_30L_217M.pt")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--vocab_size", type=int, default=49152)
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--T", type=int, default=512)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    print(f"Constructing random plain-DN model with seed={args.seed} ...")
    model = TinyLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_head=args.d_head,
        attention_cls=DeltaNetAttention,
        max_T=0,
        feedback_mode="none",
        feedback_pairs=(),
        feedback_self_k=0,
        feedback_alpha_mode="scalar",
        semantic_loss_dim=0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params/1e6:.1f}M")

    ckpt = {
        "state_dict": model.state_dict(),
        "config": {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_head": args.d_head,
            "n_layers": args.n_layers,
            "max_T": args.T,
            "feedback_mode": "none",
            "feedback_distances": (1,),
            "feedback_pairs": (),
            "feedback_xattn_pairs": (),
            "feedback_xattn_heads": 4,
            "feedback_xattn_form": "attn",
            "feedback_self_k": 0,
            "feedback_alpha_mode": "scalar",
            "arch": "deltanet",
            "layers_spec": None,
            "n_symbols": 512,
            "tokenizer": "HuggingFaceTB/SmolLM2-135M",
        },
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(f"Saved random encoder to {out}")


if __name__ == "__main__":
    sys.exit(main())
