"""
Compute bits-per-byte for the existing plain-DN-708M (SmolLM2) checkpoint
on the standard codeparrot val tail. The PPL is already known (35.38);
this adds the tokenizer-invariant BPB so it can be compared to the
DN-4B distilled student which uses the Qwen tokenizer.

This script reads the SAME byte slice the eval_filmed_ppl_708m.py
script processes for the SmolLM2 student, but tracks the raw byte
count alongside, so we can compute:

    BPB = (total nat-loss / ln 2) / total_bytes

which is invariant to tokenizer choice.

Usage:
  CUDA_VISIBLE_DEVICES=0 \\
    /home/knielsen/ml/parallel-ss-dep/.venv/bin/python -u \\
    experiments/eval_708m_bpb.py \\
    --ckpt checkpoints/dn_36L_708M_muon.pt \\
    --T 512 --n_tokens 32768 --batch 2 \\
    --out logs/distill_pilot_full/dn_708M_bpb.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.eval_bracket_structure import build_model_from_ckpt


def make_chunks_with_bytes(tokenizer, T: int, n_tokens: int,
                            dataset: str = "codeparrot/codeparrot-clean",
                            text_field: str = "content",
                            skip: int = 10_000):
    """Build (N, T+1) chunks from the codeparrot val tail, AND track
    raw UTF-8 byte counts per chunk so we can compute BPB.

    Approach: walk the stream like the existing eval, but accumulate
    byte_per_token = len(text.encode('utf-8')) / len(token_ids) for
    each example, accumulate per-chunk byte counts.
    """
    from datasets import load_dataset
    val_stream = (
        load_dataset(dataset, split="train", streaming=True)
        .shuffle(seed=42)
        .skip(skip)
    )
    eos = tokenizer.eos_token_id or 0
    target = max(1, math.ceil(n_tokens / T))
    chunks: list[list[int]] = []
    chunk_bytes: list[float] = []
    cur_ids: list[int] = []
    cur_bytes = 0.0
    for example in val_stream:
        text = example[text_field]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        text_bytes = len(text.encode("utf-8"))
        per_tok_bytes = text_bytes / len(ids)
        for tok_id in ids + [eos]:
            cur_ids.append(int(tok_id))
            cur_bytes += per_tok_bytes
            if len(cur_ids) >= T + 1:
                chunks.append(cur_ids[: T + 1])
                # The chunk's predicted tokens are positions 1..T (T positions).
                # The first position has no target; bytes assigned to it
                # do not contribute to loss. Adjust by T/(T+1).
                chunk_bytes.append(cur_bytes * T / (T + 1))
                cur_ids = cur_ids[T + 1:]
                cur_bytes = 0.0
                if len(chunks) >= target:
                    break
        if len(chunks) >= target:
            break
    out = torch.tensor(chunks, dtype=torch.long)
    return out, chunk_bytes


@torch.no_grad()
def eval_ppl_and_bpb(model, chunks, total_bytes_per_chunk, batch=2,
                     amp_dtype=torch.bfloat16):
    device = next(model.parameters()).device
    losses_sum = 0.0
    n_tokens = 0
    N = chunks.shape[0]
    t_start = time.perf_counter()
    last_print = t_start
    for i in range(0, N, batch):
        batch_chunks = chunks[i : i + batch].to(device)
        x = batch_chunks[:, :-1]
        y = batch_chunks[:, 1:]
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            y.reshape(-1), reduction="sum",
        )
        losses_sum += float(loss.item())
        n_tokens += y.numel()
        now = time.perf_counter()
        if now - last_print > 30:
            partial_ce = losses_sum / max(1, n_tokens)
            print(f"  [{i+batch}/{N}]  partial CE={partial_ce:.4f}  "
                  f"PPL={math.exp(partial_ce):.2f}", flush=True)
            last_print = now
    mean_ce = losses_sum / max(1, n_tokens)
    ppl = math.exp(mean_ce)
    total_bytes = sum(total_bytes_per_chunk)
    bpb = (losses_sum / math.log(2)) / max(1, total_bytes)
    return mean_ce, n_tokens, ppl, bpb, total_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n_tokens", type=int, default=32768)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model, cfg = build_model_from_ckpt(args.ckpt)
    print(f"  config: d_model={cfg['d_model']} n_layers={cfg['n_layers']}",
          flush=True)

    from transformers import AutoTokenizer
    tok_name = cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M")
    tok = AutoTokenizer.from_pretrained(tok_name)
    print(f"  tokenizer: {tok_name}", flush=True)

    print(f"\nBuilding {args.n_tokens // args.T} chunks at T={args.T} ...",
          flush=True)
    chunks, chunk_bytes = make_chunks_with_bytes(
        tok, T=args.T, n_tokens=args.n_tokens, skip=10_000,
    )
    n_chunks = chunks.shape[0]
    print(f"  {n_chunks} chunks, {sum(chunk_bytes)/1e6:.2f} MB raw text",
          flush=True)

    ce, n_tok, ppl, bpb, total_bytes = eval_ppl_and_bpb(
        model, chunks, chunk_bytes, batch=args.batch,
    )
    print(f"\n  CE   = {ce:.4f}", flush=True)
    print(f"  PPL  = {ppl:.4f}", flush=True)
    print(f"  BPB  = {bpb:.4f}  (total bytes={total_bytes:,.0f})", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt,
        "tokenizer": tok_name,
        "T": args.T,
        "n_chunks": n_chunks,
        "n_tokens": n_tok,
        "total_bytes": total_bytes,
        "ce": ce,
        "ppl": ppl,
        "bpb": bpb,
        "config": {k: cfg.get(k) for k in (
            "d_model", "n_layers", "n_heads", "d_head", "vocab_size",
            "feedback_mode", "tie_embeddings",
        )},
    }, indent=2))
    print(f"Results written to {out}", flush=True)


if __name__ == "__main__":
    main()
