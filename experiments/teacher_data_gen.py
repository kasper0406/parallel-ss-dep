"""
Teacher-aligned data generation for the distillation pilot.

The lesson from Phase 15: training KL+CE on raw codeparrot HURT student PPL by
36% — Qwen3.6 was trained on agent/RLHF instruction-style data, codeparrot is
unstyled GitHub Python. Teacher-data misalignment.

This script generates teacher-ALIGNED training data: Qwen3.6 itself produces
the completions, AND we capture the per-step teacher distribution as part of
the SAME generation pass (via vLLM `logprobs=K`). One pass, not two — much
faster than the previous extract-then-score flow.

Pipeline per chunk:
  1. Build a synthetic short code prompt ("def foo(args): docstring ...").
  2. vLLM samples a continuation at temperature=0.7 and records the top-K
     distribution at every emitted position.
  3. Concatenate (prompt_token_ids + sampled_token_ids), pack multiple
     completions into one T-token chunk separated by `eos`, and store
     teacher distributions ONLY at the sampled positions (prompt positions
     are masked as -inf so the student's KL loss skips them — masking is
     handled in the trainer's compute_loss).

Run from the dedicated vLLM venv:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    /home/knielsen/ml/parallel-ss-dep/.venv-vllm/bin/python \\
    experiments/teacher_data_gen.py \\
    --total_tokens 1_000_000 \\
    --out /home/knielsen/ml/parallel-ss-dep-distill/data/distill_pilot_1M
"""
from __future__ import annotations

import argparse
import os
import pathlib
import random
import time

import numpy as np


# Short, varied prompts seeding the teacher generation. These are designed
# to land Qwen in code-completion mode similar to its training distribution.
_DEFAULT_PROMPT_TEMPLATES = [
    "def {name}({args}):\n    \"\"\"{doc}.\"\"\"\n    ",
    "class {cls}:\n    \"\"\"{doc}.\"\"\"\n\n    def __init__(self{args2}):\n        ",
    "import {mod}\n\n\ndef {name}({args}):\n    ",
    "# Implement a function that {task}.\n\ndef {name}({args}):\n    ",
    "from typing import List\n\n\ndef {name}({args}) -> {ret}:\n    \"\"\"{doc}.\"\"\"\n    ",
]

_FUNC_NAMES = [
    "compute", "process", "transform", "build", "find", "filter", "merge",
    "validate", "parse", "format", "encode", "decode", "load", "save",
    "extract", "count", "sort", "search", "convert", "normalise",
    "tokenize", "chunk", "flatten", "group", "rotate", "sample", "score",
    "train", "predict", "evaluate", "rank",
]
_ARG_LISTS = [
    "data", "x", "items, key=None", "a, b", "n: int", "s: str",
    "values: list", "obj, default=None", "lo, hi", "x, y, z=0",
    "tokens, vocab", "graph, src", "matrix, k=3", "model, batch",
]
_DOCS = [
    "return the result of the operation",
    "transform the input and return a new value",
    "process all items and return aggregated output",
    "compute a derived quantity",
    "convert between two representations",
    "scan the input and emit any matching elements",
    "produce a normalised view",
    "validate the inputs and raise on error",
]
_TASKS = [
    "computes the running mean of a list",
    "merges two sorted sequences in O(n+m)",
    "counts the number of unique elements",
    "tokenises a string by whitespace and punctuation",
    "computes the longest common prefix of strings",
    "checks whether a number is prime",
    "finds the kth smallest element using quickselect",
    "groups items by an arbitrary key function",
    "implements a simple LRU cache as a class",
    "computes the SHA-256 digest of a file",
    "validates that all values fall within a given range",
    "performs binary search on a sorted list",
    "parses an ISO-8601 timestamp string",
    "rotates a 2D matrix in place by 90 degrees",
    "computes the Levenshtein edit distance",
]
_CLASSES = ["Cache", "Buffer", "Worker", "Tokeniser", "Parser", "Stream",
            "Pipeline", "Index", "Heap", "Queue", "Tree", "Graph"]
_MODULES = ["math", "json", "re", "os", "collections", "itertools",
            "functools", "heapq", "bisect", "hashlib"]
_RETURNS = ["int", "float", "str", "list", "dict", "bool", "tuple"]


def make_prompts(n: int, rng: random.Random) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        t = rng.choice(_DEFAULT_PROMPT_TEMPLATES)
        out.append(t.format(
            name=rng.choice(_FUNC_NAMES),
            args=rng.choice(_ARG_LISTS),
            args2=", " + rng.choice(_ARG_LISTS),
            doc=rng.choice(_DOCS),
            task=rng.choice(_TASKS),
            cls=rng.choice(_CLASSES),
            mod=rng.choice(_MODULES),
            ret=rng.choice(_RETURNS),
        ))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--total_tokens", type=int, default=1_000_000)
    p.add_argument("--T", type=int, default=512,
                   help="Chunk length per stored sample.")
    p.add_argument("--gen_batch", type=int, default=32,
                   help="Number of prompts per vLLM batch.")
    p.add_argument("--max_completion", type=int, default=256,
                   help="Max tokens per completion. Smaller => more diverse "
                        "sequences, faster per-batch turnaround.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k_logprobs", type=int, default=20,
                   help="Top-K logprobs per generated position.")
    p.add_argument("--shard_chunks", type=int, default=256)
    p.add_argument("--max_model_len", type=int, default=512)
    p.add_argument("--gpu_mem_fraction", type=float, default=0.92)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing shards to {out_dir}")

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print(f"Loading {args.model} via vLLM ...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model, quantization="awq",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_fraction,
        trust_remote_code=True, dtype="float16",
        enforce_eager=True, max_num_seqs=args.gen_batch,
    )
    tok = llm.get_tokenizer()
    print(f"  loaded in {time.time() - t0:.1f}s; vocab={tok.vocab_size}")
    eos = tok.eos_token_id or tok.bos_token_id or 0

    target_chunks = (args.total_tokens + args.T - 1) // args.T
    print(f"  will produce {target_chunks} chunks of {args.T} tokens "
          f"= {target_chunks * args.T:,} tokens")
    print(f"  gen_batch={args.gen_batch}  max_completion={args.max_completion}  "
          f"top_k={args.top_k_logprobs}  T={args.T}  max_model_len={args.max_model_len}")

    chunks_done = 0
    shard_idx = 0
    shard_token_ids: list[np.ndarray] = []
    shard_topk_ids: list[np.ndarray] = []
    shard_topk_lps: list[np.ndarray] = []
    t_start = time.time()
    last_log = t_start

    # Single-pass generation with logprobs:
    #   - vLLM samples up to max_completion tokens per prompt at temp/top_p.
    #   - For each sampled token, vLLM returns the top-K distribution
    #     EVALUATED AT THAT POSITION (i.e., the teacher's distribution
    #     over what the next token COULD have been, given the prefix).
    #   - We DON'T request prompt_logprobs (saves a second forward pass);
    #     the prompt positions are masked as -inf in the stored array,
    #     and the trainer's compute_loss skips invalid positions.
    gen_sampling = SamplingParams(
        max_tokens=args.max_completion,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        logprobs=args.top_k_logprobs,
    )

    # Token+logprob packing buffers. We accumulate (tok_id, top_K_ids,
    # top_K_lps) tuples per generated position, plus prompt tokens with
    # their logprob slots blanked (-inf), and stitch into T-chunks.
    pack_tok: list[int] = []
    pack_topk_ids: list[np.ndarray] = []  # (K,) int32 per position
    pack_topk_lps: list[np.ndarray] = []  # (K,) float16 per position

    K = args.top_k_logprobs
    blank_ids = np.zeros(K, dtype=np.int32)
    blank_lps = np.full(K, -np.inf, dtype=np.float16)

    def emit_chunk():
        """Slice off the head T of pack_* and append to shard."""
        T = args.T
        ids_arr = np.asarray(pack_tok[:T], dtype=np.int32)
        topk_ids_arr = np.stack(pack_topk_ids[:T])  # (T, K) int32
        topk_lps_arr = np.stack(pack_topk_lps[:T])  # (T, K) fp16
        shard_token_ids.append(ids_arr)
        shard_topk_ids.append(topk_ids_arr)
        shard_topk_lps.append(topk_lps_arr)
        del pack_tok[:T]
        del pack_topk_ids[:T]
        del pack_topk_lps[:T]

    while chunks_done < target_chunks:
        n_to_gen = min(args.gen_batch, max(1, 2 * (target_chunks - chunks_done)))
        prompts = make_prompts(n_to_gen, rng)
        gens = llm.generate(prompts, sampling_params=gen_sampling, use_tqdm=False)
        print(f"  [batch] returned {len(gens)} gens")

        for g in gens:
            prompt_ids = list(g.prompt_token_ids)
            out_obj = g.outputs[0]
            out_ids = list(out_obj.token_ids)
            # Per-step logprobs: list[ Optional[ dict[int, Logprob] ] ],
            # one entry per output token. Top-K dict at that step.
            step_lps = out_obj.logprobs

            # Push prompt tokens with blank teacher slots (we did not score
            # them; the trainer mask will skip these).
            for tid in prompt_ids:
                pack_tok.append(int(tid))
                pack_topk_ids.append(blank_ids.copy())
                pack_topk_lps.append(blank_lps.copy())

            # Push generated tokens with the teacher's top-K at each step.
            for tid, slot in zip(out_ids, step_lps or []):
                pack_tok.append(int(tid))
                if slot is None:
                    pack_topk_ids.append(blank_ids.copy())
                    pack_topk_lps.append(blank_lps.copy())
                else:
                    ordered = sorted(slot.items(), key=lambda kv: kv[1].rank)
                    top_ids = np.zeros(K, dtype=np.int32)
                    top_lps = np.full(K, -np.inf, dtype=np.float16)
                    for k, (t_id, lp) in enumerate(ordered[:K]):
                        top_ids[k] = t_id
                        top_lps[k] = lp.logprob
                    pack_topk_ids.append(top_ids)
                    pack_topk_lps.append(top_lps)

            # Boundary marker between completions.
            pack_tok.append(eos)
            pack_topk_ids.append(blank_ids.copy())
            pack_topk_lps.append(blank_lps.copy())

        # Slice T-chunks off the buffer.
        while len(pack_tok) >= args.T and chunks_done < target_chunks:
            emit_chunk()
            chunks_done += 1

        # Flush shard if full.
        if len(shard_token_ids) >= args.shard_chunks or chunks_done >= target_chunks:
            shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
            np.savez_compressed(
                shard_path,
                token_ids=np.stack(shard_token_ids).astype(np.int32),
                top_k_ids=np.stack(shard_topk_ids).astype(np.int32),
                top_k_logprobs=np.stack(shard_topk_lps).astype(np.float16),
            )
            print(f"  wrote {shard_path}  ({len(shard_token_ids)} chunks, "
                  f"{shard_path.stat().st_size / 1e6:.1f} MB)")
            shard_token_ids.clear()
            shard_topk_ids.clear()
            shard_topk_lps.clear()
            shard_idx += 1

        now = time.time()
        if now - last_log > 30 or chunks_done >= target_chunks:
            elapsed = now - t_start
            tps = chunks_done * args.T / max(elapsed, 1e-3)
            eta_s = (target_chunks - chunks_done) * args.T / max(tps, 1)
            print(f"  [{chunks_done}/{target_chunks}] "
                  f"{chunks_done * args.T:,} tokens, "
                  f"{tps:.0f} tok/s, eta {eta_s/60:.1f} min")
            last_log = now

    # Manifest.
    manifest = out_dir / "manifest.npz"
    np.savez(
        manifest,
        n_shards=shard_idx,
        n_chunks=chunks_done,
        T=args.T,
        top_k=K,
        vocab_size=tok.vocab_size,
        model=args.model,
        # Mark this corpus as teacher-self-generated.
        dataset="qwen_self_generated_code",
        text_field="(synthetic prompt + Qwen continuation, single-pass logprobs)",
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    elapsed = time.time() - t_start
    total_tokens = chunks_done * args.T
    print(f"\nDone: {shard_idx} shards, {chunks_done} chunks, "
          f"{total_tokens:,} tokens in {elapsed/60:.1f} min "
          f"({total_tokens/elapsed:.0f} tok/s)")


if __name__ == "__main__":
    main()
