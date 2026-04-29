"""
Verify Qwen3.6-35B-A3B-AWQ produces top-K prompt logprobs (per-position
distributions over the input). This is the data we need to train a
student via KL distillation: at every position t, the teacher's next-
token distribution over its full vocabulary, top-K-truncated.

Run with:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    ~/ml/parallel-ss-dep/.venv-vllm/bin/python \\
    experiments/test_qwen_logprobs.py
"""
from __future__ import annotations

import argparse
import time

PROMPT = """def fibonacci(n):
    \"\"\"Return the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--gpu_mem_fraction", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20,
                   help="K for top-K prompt logprobs (vLLM caps at 20).")
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
        enforce_eager=True,
        max_num_seqs=8,
    )
    tok = llm.get_tokenizer()
    print(f"  loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        max_tokens=1,                # don't generate, just score
        temperature=0.0,
        prompt_logprobs=args.top_k,  # top-K logprobs per input position
    )

    print("\nScoring prompt ...")
    t0 = time.time()
    outputs = llm.generate([PROMPT], sampling_params=sampling)
    dt = time.time() - t0
    out = outputs[0]
    n_in = len(out.prompt_token_ids)
    print(f"  {n_in} tokens in {dt:.2f}s = {n_in/dt:.1f} tok/s")

    pl = out.prompt_logprobs  # list[Optional[dict[int, Logprob]]]
    print(f"  prompt_logprobs has {len(pl)} entries (matches token count)")
    print(f"  first slot is {pl[0]} (always None — no prediction at position 0)")
    print(f"  slot at position 5 has top-{len(pl[5])} entries:")
    items = sorted(pl[5].items(), key=lambda kv: kv[1].rank)[:5]
    for tok_id, lp in items:
        token_str = tok.decode([tok_id]).replace("\n", "\\n")
        print(f"    rank={lp.rank}  logprob={lp.logprob:+.3f}  "
              f"token_id={tok_id}  token={token_str!r}")

    print("\nThe full distribution over top-K tokens at each prompt position is "
          "what we'll consume as KL distillation targets for the student.")


if __name__ == "__main__":
    main()
