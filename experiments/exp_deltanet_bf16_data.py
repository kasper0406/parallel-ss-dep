"""Pre-tokenize a FIXED codeparrot token pool for the bf16-regime dense-LM probe.

Caches a (train_pool, val_pool) of packed T+1 sequences to disk so that EVERY
arm / regime / seed run trains on byte-identical data. Streamed from
codeparrot-clean (shuffle seed 42) and tokenized with the repo's SmolLM2
tokenizer (the same one `eval_bracket_structure._make_val_iter` uses). Val is
the first `n_val` packed sequences, train the next `n_train` — disjoint sequence
sets (aside from at most one document straddling the boundary, which is symmetric
across all arms/regimes and so cannot bias the muon-vs-perhead gap).

Run once; the harness loads the cached tensors.
"""
from __future__ import annotations

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=512, help="seq len (stores T+1 for shift)")
    ap.add_argument("--n_val", type=int, default=1024)
    ap.add_argument("--n_train", type=int, default=16384)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--dataset", default="codeparrot/codeparrot-clean")
    ap.add_argument("--text_field", default="content")
    ap.add_argument("--shuffle_seed", type=int, default=42)
    ap.add_argument("--out", default="runs/deltanet_bf16/pool.pt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id or tok.bos_token_id or 0
    vocab = tok.vocab_size
    L = args.T + 1
    need = (args.n_val + args.n_train)

    stream = load_dataset(args.dataset, split="train", streaming=True
                          ).shuffle(seed=args.shuffle_seed)

    seqs: list[list[int]] = []
    buf: list[int] = []
    n_docs = 0
    for ex in stream:
        ids = tok.encode(ex[args.text_field], add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos)
        n_docs += 1
        while len(buf) >= L:
            seqs.append(buf[:L])
            buf = buf[L:]
        if len(seqs) >= need:
            break
        if n_docs % 500 == 0:
            print(f"  {n_docs} docs -> {len(seqs)}/{need} seqs")

    seqs = seqs[:need]
    arr = torch.tensor(seqs, dtype=torch.long)            # (need, L)
    val_pool = arr[: args.n_val].contiguous()
    train_pool = arr[args.n_val:].contiguous()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"train_pool": train_pool, "val_pool": val_pool,
                "T": args.T, "vocab": int(vocab), "tokenizer": args.tokenizer,
                "dataset": args.dataset, "shuffle_seed": args.shuffle_seed},
               args.out)
    print(f"\nSaved {args.out}")
    print(f"  train_pool {tuple(train_pool.shape)}  val_pool {tuple(val_pool.shape)}  "
          f"vocab {vocab}  docs_used {n_docs}")


if __name__ == "__main__":
    main()
