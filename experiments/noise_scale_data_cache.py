"""Cache a fixed pool of pretrain sequences for the gradient-noise-scale study.

GPU-FREE. Streams `configs/pretrain_mix_v4.yaml` (the canonical pretrain mix)
with think-burst insertion DISABLED (we measure the plain LM cross-entropy
gradient), packs into (N, T) tensors of (inputs, targets, doc_ids), and saves
them to a .pt file. The SAME cached pool is reused across every checkpoint so
the across-checkpoint B_simple growth reflects the MODEL changing, not the data
draw.

The pool is split into `n_draws` independent draws, each `accum` microbatches of
`micro_b` sequences. B_big = accum*micro_b sequences == the production step
(batch 4 * grad_accum 32 = 128 seqs = 262,144 tokens). B_small = micro_b seqs.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/noise_scale_data_cache.py \
        --out runs/noise_scale/pool_v4.pt --n_draws 25 --accum 32 --micro_b 4
"""
from __future__ import annotations

import argparse
import pathlib
import time

import torch

from experiments.data_mix import MixedSourceStream, load_sources_from_yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pretrain_mix_v4.yaml")
    p.add_argument("--out", default="runs/noise_scale/pool_v4.pt")
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--n_draws", type=int, default=25)
    p.add_argument("--accum", type=int, default=32)
    p.add_argument("--micro_b", type=int, default=4)
    p.add_argument("--base_seed", type=int, default=1234)
    p.add_argument("--mask_eos_in_targets", action="store_true", default=True)
    args = p.parse_args()

    n_seq = args.n_draws * args.accum * args.micro_b
    print(f"Caching {n_seq} sequences (T={args.T}) from {args.config}")
    print(f"  layout: n_draws={args.n_draws} x accum={args.accum} "
          f"x micro_b={args.micro_b}")
    print(f"  B_big  = {args.accum * args.micro_b} seqs "
          f"= {args.accum * args.micro_b * args.T} tokens (production step)")
    print(f"  B_small= {args.micro_b} seqs "
          f"= {args.micro_b * args.T} tokens (one microbatch)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    base_vocab = tok.vocab_size
    thinking_token_id = base_vocab  # matches train_lm / data_mix convention

    sources = load_sources_from_yaml(args.config)
    ds = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.T,
        thinking_token_id=thinking_token_id,
        think_burst_prob=0.0,            # NO think bursts: plain LM gradient
        base_seed=args.base_seed,
        mask_eos_in_targets=bool(args.mask_eos_in_targets),
        emit_doc_ids=True,
    )
    it = iter(ds)

    inputs = torch.empty((n_seq, args.T), dtype=torch.long)
    targets = torch.empty((n_seq, args.T), dtype=torch.long)
    doc_ids = torch.empty((n_seq, args.T), dtype=torch.long)
    t0 = time.time()
    valid_counts = []
    for i in range(n_seq):
        x, y, d = next(it)
        inputs[i] = x
        targets[i] = y
        doc_ids[i] = d
        valid_counts.append(int((y != -100).sum()))
        if (i + 1) % 256 == 0:
            dt = time.time() - t0
            print(f"  {i+1}/{n_seq}  ({(i+1)*args.T/dt/1e3:.1f}k tok/s, "
                  f"{dt:.0f}s)", flush=True)

    vc = torch.tensor(valid_counts, dtype=torch.float)
    print(f"valid-token count per seq: mean={vc.mean():.1f} "
          f"min={vc.min():.0f} max={vc.max():.0f} "
          f"(of T={args.T}); EOS+last masked)")

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "inputs": inputs, "targets": targets, "doc_ids": doc_ids,
        "config": args.config, "T": args.T, "n_draws": args.n_draws,
        "accum": args.accum, "micro_b": args.micro_b,
        "thinking_token_id": thinking_token_id,
        "tokenizer": args.tokenizer,
    }, outp)
    print(f"Saved pool -> {outp}  ({outp.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
