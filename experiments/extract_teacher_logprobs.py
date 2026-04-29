"""
Pre-extract Qwen3.6-35B-A3B-AWQ logprobs over codeparrot for KL distillation.

For each T-token chunk, saves (token_ids, top_K_token_ids, top_K_logprobs)
in NPZ shards. The student trainer (train_distill.py) reads these
offline and trains against them.

Run from the dedicated vLLM venv:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    ~/ml/parallel-ss-dep/.venv-vllm/bin/python \\
    experiments/extract_teacher_logprobs.py \\
    --total_tokens 10_000_000 --out /home/knielsen/ml/parallel-ss-dep/data/distill_10M
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean")
    p.add_argument("--text_field", type=str, default="content")
    p.add_argument("--total_tokens", type=int, default=10_000_000,
                   help="Approximate total token budget across all shards.")
    p.add_argument("--T", type=int, default=512,
                   help="Chunk length per sample.")
    p.add_argument("--batch", type=int, default=8,
                   help="Number of T-chunks per vLLM batch.")
    p.add_argument("--top_k", type=int, default=20,
                   help="Top-K logprobs per position (vLLM caps at 20).")
    p.add_argument("--shard_chunks", type=int, default=2048,
                   help="Number of T-chunks per output shard (~1M tokens at T=512).")
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--gpu_mem_fraction", type=float, default=0.95)
    p.add_argument("--out", type=str, required=True,
                   help="Output directory for shards.")
    p.add_argument("--seed", type=int, default=42,
                   help="Stream-shuffle seed for reproducibility.")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing shards to {out_dir}")

    print(f"Loading {args.model} ...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model, quantization="awq",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_fraction,
        trust_remote_code=True, dtype="float16",
        enforce_eager=True, max_num_seqs=args.batch,
    )
    tok = llm.get_tokenizer()
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  vocab size: {tok.vocab_size}")

    sampling = SamplingParams(
        max_tokens=1, temperature=0.0,
        prompt_logprobs=args.top_k,
    )

    # Stream-tokenize codeparrot into T-chunks.
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train", streaming=True
                      ).shuffle(seed=args.seed)
    eos = tok.eos_token_id or tok.bos_token_id or 0

    def chunk_iter():
        buf: list[int] = []
        for ex in ds:
            text = ex[args.text_field]
            ids = tok.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= args.T:
                yield np.asarray(buf[:args.T], dtype=np.int32)
                buf = buf[args.T:]

    # Loop: gather batch of B chunks → vLLM scoring → top-K extract → shard write.
    target_chunks = (args.total_tokens + args.T - 1) // args.T
    print(f"  will extract ~{target_chunks} chunks of {args.T} tokens "
          f"= {target_chunks * args.T:,} tokens")

    chunk_gen = chunk_iter()
    total_done = 0
    shard_idx = 0
    shard_token_ids: list[np.ndarray] = []
    shard_topk_ids: list[np.ndarray] = []
    shard_topk_lps: list[np.ndarray] = []
    t_start = time.time()
    last_log = t_start

    while total_done < target_chunks:
        # Collect batch.
        batch_ids: list[np.ndarray] = []
        try:
            for _ in range(args.batch):
                batch_ids.append(next(chunk_gen))
        except StopIteration:
            if not batch_ids:
                break
        if not batch_ids:
            break

        # vLLM accepts `prompt_token_ids` directly; use that to skip text round-trip.
        prompts = [{"prompt_token_ids": ids.tolist()} for ids in batch_ids]
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)

        for ids, out in zip(batch_ids, outputs):
            T = len(ids)
            top_ids = np.zeros((T, args.top_k), dtype=np.int32)
            top_lps = np.full((T, args.top_k), -1e9, dtype=np.float16)
            pl = out.prompt_logprobs
            assert len(pl) == T, f"len mismatch {len(pl)} vs {T}"
            for pos in range(1, T):
                slot = pl[pos]
                if slot is None:
                    continue
                # Sort by rank (rank=1 is highest logprob).
                ordered = sorted(slot.items(), key=lambda kv: kv[1].rank)
                for k, (tid, lp) in enumerate(ordered[:args.top_k]):
                    top_ids[pos, k] = tid
                    top_lps[pos, k] = lp.logprob
            shard_token_ids.append(ids)
            shard_topk_ids.append(top_ids)
            shard_topk_lps.append(top_lps)
            total_done += 1

        if len(shard_token_ids) >= args.shard_chunks:
            shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
            np.savez_compressed(
                shard_path,
                token_ids=np.stack(shard_token_ids).astype(np.int32),
                top_k_ids=np.stack(shard_topk_ids).astype(np.int32),
                top_k_logprobs=np.stack(shard_topk_lps).astype(np.float16),
            )
            print(f"  wrote {shard_path}  "
                  f"({len(shard_token_ids)} chunks, "
                  f"{shard_path.stat().st_size / 1e6:.1f} MB)")
            shard_token_ids.clear()
            shard_topk_ids.clear()
            shard_topk_lps.clear()
            shard_idx += 1

        now = time.time()
        if now - last_log > 30 or total_done >= target_chunks:
            elapsed = now - t_start
            tps = total_done * args.T / elapsed
            eta_s = (target_chunks - total_done) * args.T / max(tps, 1)
            print(f"  [{total_done}/{target_chunks}] {total_done * args.T:,} tokens, "
                  f"{tps:.0f} tok/s, eta {eta_s/60:.1f} min")
            last_log = now

    # Flush final shard.
    if shard_token_ids:
        shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
        np.savez_compressed(
            shard_path,
            token_ids=np.stack(shard_token_ids).astype(np.int32),
            top_k_ids=np.stack(shard_topk_ids).astype(np.int32),
            top_k_logprobs=np.stack(shard_topk_lps).astype(np.float16),
        )
        print(f"  wrote final {shard_path}  "
              f"({len(shard_token_ids)} chunks, "
              f"{shard_path.stat().st_size / 1e6:.1f} MB)")
        shard_idx += 1

    # Manifest.
    manifest = out_dir / "manifest.npz"
    np.savez(
        manifest,
        n_shards=shard_idx,
        n_chunks=total_done,
        T=args.T,
        top_k=args.top_k,
        vocab_size=tok.vocab_size,
        model=args.model,
        dataset=args.dataset,
        text_field=args.text_field,
        seed=args.seed,
    )
    elapsed = time.time() - t_start
    total_tokens = total_done * args.T
    print(f"\nDone: {shard_idx} shards, {total_done} chunks, {total_tokens:,} tokens "
          f"in {elapsed/60:.1f} min ({total_tokens/elapsed:.0f} tok/s)")


if __name__ == "__main__":
    main()
