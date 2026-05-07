"""Quick smoke test: Qwen3.6 generates a code completion.

Run with the vLLM venv:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/knielsen/ml/parallel-ss-dep/.venv-vllm/bin/python \
    experiments/test_qwen_gen.py
"""
from __future__ import annotations

import time

PROMPT = """def fibonacci(n):
    \"\"\"Return the nth Fibonacci number.\"\"\"
    """


def main():
    print("Loading Qwen3.6-35B-A3B-AWQ ...")
    t0 = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="QuantTrio/Qwen3.6-35B-A3B-AWQ", quantization="awq",
        max_model_len=512, gpu_memory_utilization=0.92,
        trust_remote_code=True, dtype="float16",
        enforce_eager=True, max_num_seqs=4,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    sp = SamplingParams(max_tokens=120, temperature=0.7, top_p=0.95, seed=42)
    out = llm.generate([PROMPT, PROMPT, PROMPT, PROMPT],
                       sampling_params=sp, use_tqdm=False)
    for i, g in enumerate(out):
        text = g.outputs[0].text
        n = len(g.outputs[0].token_ids)
        print(f"\n--- generation {i} ({n} tokens) ---")
        print(repr(text[:300]))


if __name__ == "__main__":
    main()
