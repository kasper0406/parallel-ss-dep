"""
vLLM-based smoke test for Qwen3.6-35B-A3B-AWQ teacher inference.

Run with the dedicated vLLM venv:
  CUDA_VISIBLE_DEVICES=0 ~/ml/parallel-ss-dep/.venv-vllm/bin/python \\
    experiments/test_qwen_vllm.py

Loads the AWQ-quantized teacher on a single 5090, runs a short
code completion, and reports throughput.
"""
from __future__ import annotations

import argparse
import time

PROMPT = """def fibonacci(n):
    \"\"\"Return the nth Fibonacci number.\"\"\"
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--max_model_len", type=int, default=2048,
                   help="Limit context to keep KV-cache small.")
    p.add_argument("--gpu_mem_fraction", type=float, default=0.92)
    p.add_argument("--enforce_eager", action="store_true",
                   help="Disable CUDA graphs (saves ~1 GB at cost of perf).")
    p.add_argument("--prompt", type=str, default=PROMPT)
    args = p.parse_args()

    print(f"Loading {args.model} via vLLM ...")
    t0 = time.time()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        quantization="awq",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_fraction,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=args.enforce_eager,
        max_num_seqs=8,        # small KV-cache budget
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        max_tokens=args.max_new,
        temperature=0.0,
    )

    print("\nWarmup ...")
    _ = llm.generate([args.prompt], sampling_params=SamplingParams(
        max_tokens=8, temperature=0.0))

    print("Generating ...")
    t0 = time.time()
    outputs = llm.generate([args.prompt], sampling_params=sampling)
    dt = time.time() - t0
    text = outputs[0].outputs[0].text
    n_out = len(outputs[0].outputs[0].token_ids)
    print(f"  {n_out} new tokens in {dt:.2f}s = {n_out/dt:.1f} tok/s\n")
    print("=" * 70)
    print(args.prompt + text)
    print("=" * 70)


if __name__ == "__main__":
    main()
