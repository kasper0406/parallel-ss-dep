"""
Realistic batched-prompt throughput test for the distillation
extraction pipeline. Sends N prompts of T tokens each in a single
batch, requests top-K prompt_logprobs, measures aggregate throughput.

This is what determines feasibility of pre-extraction at scale:
  100 M tokens / X tok/s = total wall-time

Run:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    ~/ml/parallel-ss-dep/.venv-vllm/bin/python \\
    experiments/test_qwen_throughput.py
"""
from __future__ import annotations

import argparse
import time

# Sample ~1 KB of representative Python code, padded/repeated to fill T.
SAMPLE = """import math
import numpy as np
from typing import Optional, List, Dict


def fibonacci(n: int) -> int:
    \"\"\"Return the nth Fibonacci number, recursively.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right


def inorder(root: Optional[TreeNode]) -> List[int]:
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


def euclid(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def primes(limit: int) -> List[int]:
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--gpu_mem_fraction", type=float, default=0.95)
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--prompt_T", type=int, default=512)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_num_seqs", type=int, default=16)
    args = p.parse_args()

    print(f"Loading {args.model} ...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model, quantization="awq",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_fraction,
        trust_remote_code=True, dtype="float16",
        enforce_eager=True, max_num_seqs=args.max_num_seqs,
    )
    tok = llm.get_tokenizer()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Build N prompts of length ~prompt_T tokens each by tiling sample text.
    sample_ids = tok.encode(SAMPLE, add_special_tokens=False)
    print(f"  sample text encodes to {len(sample_ids)} tokens")
    while len(sample_ids) < args.prompt_T:
        sample_ids = sample_ids + sample_ids
    sample_ids = sample_ids[:args.prompt_T]
    prompt_text = tok.decode(sample_ids, skip_special_tokens=True)
    prompts = [prompt_text] * args.n_prompts
    total_tokens = args.n_prompts * args.prompt_T
    print(f"  batch: {args.n_prompts} prompts × {args.prompt_T} tokens = {total_tokens} total")

    sampling = SamplingParams(
        max_tokens=1, temperature=0.0,
        prompt_logprobs=args.top_k,
    )
    print("\nWarmup ...")
    _ = llm.generate(prompts[:1], sampling_params=SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=args.top_k))

    print("Scoring batch ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling)
    dt = time.time() - t0
    print(f"  {total_tokens} tokens in {dt:.2f}s = {total_tokens/dt:.0f} tok/s")
    print(f"  per-prompt: {args.prompt_T/(dt/args.n_prompts):.0f} tok/s")
    n_in = sum(len(o.prompt_token_ids) for o in outputs)
    print(f"  (actual total {n_in} tokens after tokenization variance)")

    # Estimate total time for N tokens of code
    for tgt_M in [10, 100, 1000]:
        tgt = tgt_M * 1_000_000
        hours = tgt / (total_tokens / dt) / 3600
        print(f"  → {tgt_M:>4d} M tokens ≈ {hours:>6.2f} h at this rate")


if __name__ == "__main__":
    main()
