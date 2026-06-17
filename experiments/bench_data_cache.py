"""Cache a FIXED slice of real code+reasoning text → a single .pt tensor.

So every optimizer arm in `bench_optimizer.py` trains on byte-identical
data (same tokens, same order, same val split). Source is the project's
own cleaned Phase-C SFT corpus (`data/sft_phasec_clean.jsonl`: real
(problem, CoT, code) triples). Fully offline + reproducible — avoids the
slow HF-streaming startup of the full pretrain mix while staying on the
real code-model data distribution.

Run once; arms just `torch.load` the cache.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/sft_phasec_clean.jsonl")
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--train_seqs", type=int, default=9000)
    ap.add_argument("--val_seqs", type=int, default=512)
    ap.add_argument("--max_lines", type=int, default=40000)
    ap.add_argument("--out", default="runs/bench_optim_data.pt")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    model_vocab_size = ((int(tok.vocab_size) + 1 + 63) // 64) * 64
    print(f"vocab base={tok.vocab_size} model={model_vocab_size} eos={eos}")

    need_tokens = (args.train_seqs + args.val_seqs) * (args.T + 1) + args.T
    t0 = time.time()
    ids: list[int] = []
    n_docs = 0
    with open(args.jsonl) as f:
        for i, line in enumerate(f):
            if i >= args.max_lines or len(ids) >= need_tokens:
                break
            d = json.loads(line)
            parts = [d.get("problem_prompt", ""), d.get("qwen_completion", ""),
                     d.get("extracted_code", "")]
            text = "\n".join(p for p in parts if p)
            if not text.strip():
                continue
            enc = tok(text, add_special_tokens=False)["input_ids"]
            ids.extend(enc)
            ids.append(eos)
            n_docs += 1
    dt = time.time() - t0
    print(f"tokenized {n_docs} docs → {len(ids)} tokens in {dt:.1f}s "
          f"({len(ids)/dt/1e3:.0f}k tok/s)")
    if len(ids) < need_tokens:
        print(f"WARNING: only {len(ids)} tokens < needed {need_tokens}; "
              f"reduce train_seqs or raise max_lines.")

    arr = torch.tensor(ids, dtype=torch.long)

    # Pack into (n_seqs, T+1) windows: x = [:T], y = [1:T+1]. Non-overlapping.
    stride = args.T + 1
    n_total = len(arr) // stride
    arr = arr[: n_total * stride].view(n_total, stride)
    X = arr[:, : args.T].to(torch.int32).contiguous()
    Y = arr[:, 1: args.T + 1].clone().contiguous()  # int64 targets

    n_tr = min(args.train_seqs, n_total - args.val_seqs)
    Xtr, Ytr = X[:n_tr], Y[:n_tr]
    Xva, Yva = X[n_tr: n_tr + args.val_seqs], Y[n_tr: n_tr + args.val_seqs]

    blob = {
        "Xtr": Xtr, "Ytr": Ytr, "Xva": Xva, "Yva": Yva,
        "T": args.T, "model_vocab_size": model_vocab_size,
        "tokenizer": args.tokenizer, "jsonl": args.jsonl,
    }
    torch.save(blob, args.out)
    print(f"train {tuple(Xtr.shape)}  val {tuple(Xva.shape)} → saved {args.out}")


if __name__ == "__main__":
    main()
