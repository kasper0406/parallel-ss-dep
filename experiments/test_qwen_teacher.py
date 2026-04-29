"""
Smoke test for Qwen3.6-35B-A3B-AWQ teacher inference via autoawq.

Loads the AWQ-quantized teacher on a single 5090 and times a short
code-completion. Establishes the inference pipeline for distillation.

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/test_qwen_teacher.py
"""
from __future__ import annotations

import argparse
import time
import warnings

# autoawq is deprecated but still works; suppress noise
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from transformers import AutoTokenizer

PROMPT = """def fibonacci(n):
    \"\"\"Return the nth Fibonacci number.\"\"\"
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--prompt", type=str, default=PROMPT)
    p.add_argument("--backend", choices=["autoawq", "transformers"],
                   default="autoawq")
    args = p.parse_args()

    print(f"Loading {args.model} via {args.backend} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.backend == "autoawq":
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            args.model, fuse_layers=False, trust_remote_code=True,
            safetensors=True, device_map="auto",
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True,
            dtype=torch.float16, device_map="auto",
        )
    print(f"  loaded in {time.time() - t0:.1f}s")

    free, total = torch.cuda.mem_get_info()
    print(f"  GPU mem: {(total-free)/1e9:.1f} / {total/1e9:.1f} GB used")

    # Prep inputs.
    inputs = tok(args.prompt, return_tensors="pt").to("cuda")
    n_in = inputs.input_ids.shape[1]
    print(f"  prompt: {n_in} tokens")

    # Warm-up + time.
    print("\nWarmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=8,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    print("Generating...")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_out = out.shape[1] - n_in
    print(f"  {n_out} new tokens in {dt:.2f}s = {n_out/dt:.1f} tok/s\n")
    print("=" * 70)
    print(tok.decode(out[0], skip_special_tokens=True))
    print("=" * 70)


if __name__ == "__main__":
    main()
