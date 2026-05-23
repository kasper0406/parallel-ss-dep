"""Generate (problem → reasoning → solution) traces from Qwen 3.6 AWQ
for student distillation.

Pipeline:
  1. Load a problem corpus (any LOADERS key in code_grader.py).
  2. For each problem, ask Qwen for N solutions via vLLM batched
     inference at temperature τ.
  3. Extract code from ```python ... ``` fenced blocks in Qwen's output.
  4. For problems WITH executable tests: rejection-sample (keep only
     completions that pass grade(); discard the rest). For problems
     WITHOUT tests (Magicoder, CodeFeedback): keep all completions
     (we trust the teacher).
  5. Append JSONL rows to --out so the run is resumable / streamable.

The student SFT stage (sft_code.py --distilled_jsonl) consumes the
output and trains the v7.1 model to imitate Qwen, optionally with
thinking-token injection around CoT segments.

Run from the dedicated vLLM venv:
  CUDA_VISIBLE_DEVICES=0 ~/ml/parallel-ss-dep/.venv-vllm/bin/python \\
      experiments/distill_solutions.py \\
      --dataset mbpp_combined \\
      --n_samples 4 \\
      --out data/distill_v7_solutions.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import IO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.code_grader import LOADERS, Problem, grade


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_USER_PROMPT_NL = """\
Solve the following Python coding problem. First think step by step about \
the approach, then write the final solution as a Python code block wrapped in \
```python ... ``` fences.

Problem:
{problem}

Remember to wrap your final code in ```python ... ``` fences.
"""

_USER_PROMPT_CODE = """\
Complete the following Python function. First think step by step about the \
approach, then provide the full function implementation as a Python code block \
wrapped in ```python ... ``` fences. Include the original function signature \
in your code block.

```python
{problem}
```

Remember to wrap your final implementation (including the function signature) \
in ```python ... ``` fences.
"""


def build_user_prompt(problem: Problem) -> str:
    """Render the user-side message for Qwen. Branches on
    prompt_is_code so HumanEval-style code-completion gets a different
    instruction than MBPP-style natural-language problems."""
    if problem.prompt_is_code:
        return _USER_PROMPT_CODE.format(problem=problem.prompt)
    return _USER_PROMPT_NL.format(problem=problem.prompt)


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------

# Match ```python ... ``` first (preferred), then any ``` ... ``` fence.
_FENCE_PY = re.compile(r"```python\s*\n(.*?)\n?```", re.DOTALL)
_FENCE_ANY = re.compile(r"```[a-zA-Z]*\s*\n(.*?)\n?```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    """Pull the FIRST python fenced block from `text`. Falls back to
    the first unmarked fenced block. Returns None if no fence found —
    caller decides whether to skip the row or grade the raw text.
    """
    m = _FENCE_PY.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_ANY.search(text)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Grade-or-passthrough
# ---------------------------------------------------------------------------

def grade_or_passthrough(problem: Problem, completion: str) -> tuple[str, float]:
    """Run grade() if the problem has tests, otherwise return a
    passthrough sentinel. Distillation-only sources (magicoder,
    codefeedback) lack executable tests, so we accept every Qwen
    output by trust.
    """
    if not problem.tests:
        return "no_tests", 1.0
    r = grade(problem, completion)
    return r.tier, r.score


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------

def write_jsonl_record(f: IO[str], *,
                       task_id: str,
                       problem_prompt: str,
                       qwen_completion: str,
                       extracted_code: str | None,
                       has_tests: bool,
                       tier: str,
                       score: float,
                       sample_idx: int) -> None:
    """Append one JSONL row. We keep BOTH the full qwen_completion
    (includes CoT) and the extracted code block separately — the SFT
    stage chooses which to train on (full text trains the thinking
    behaviour; code-only is the cleanest target)."""
    row = {
        "task_id": task_id,
        "sample_idx": sample_idx,
        "problem_prompt": problem_prompt,
        "qwen_completion": qwen_completion,
        "extracted_code": extracted_code,
        "has_tests": has_tests,
        "tier": tier,
        "score": score,
    }
    f.write(json.dumps(row) + "\n")
    f.flush()


# ---------------------------------------------------------------------------
# Main (vLLM-driven)
# ---------------------------------------------------------------------------

def _build_chat_template_prompt(tokenizer, user_text: str) -> str:
    """Apply Qwen's chat template so the model receives the message
    in the format it was trained on."""
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mbpp_combined",
                   help="LOADERS key from code_grader.py.")
    p.add_argument("--max_problems", type=int, default=None,
                   help="Cap on problems sampled.")
    p.add_argument("--max_magicoder", type=int, default=20000,
                   help="Only used if --dataset distill_corpus.")
    p.add_argument("--max_codefeedback", type=int, default=20000,
                   help="Only used if --dataset distill_corpus.")
    p.add_argument("--out", type=str, required=True,
                   help="Output JSONL path. Appends, so resumable.")
    p.add_argument("--model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Number of problems per vLLM batch.")
    p.add_argument("--gpu_mem_fraction", type=float, default=0.92)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--keep_only_passing", action="store_true",
                   help="For problems WITH tests, drop completions that "
                        "don't pass grade() (rejection sampling).")
    p.add_argument("--smoke", action="store_true",
                   help="2 problems × 2 samples × max_new=64.")
    args = p.parse_args()

    if args.smoke:
        args.max_problems = 2
        args.n_samples = 2
        args.max_new_tokens = 64
        args.batch_size = 4

    # ---- Load problems
    if args.dataset not in LOADERS:
        raise SystemExit(
            f"unknown --dataset {args.dataset!r}; choose from {list(LOADERS)}")
    if args.dataset == "distill_corpus":
        from experiments.code_grader import load_distill_corpus
        problems = load_distill_corpus(
            max_magicoder=args.max_magicoder,
            max_codefeedback=args.max_codefeedback,
        )
    else:
        problems = LOADERS[args.dataset]()
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    print(f"[distill] loaded {len(problems)} problems from --dataset "
          f"{args.dataset}")

    # ---- Load vLLM teacher
    print(f"[distill] loading {args.model} via vLLM ...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    t0 = time.time()
    llm = LLM(
        model=args.model,
        quantization="awq",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_fraction,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=args.enforce_eager,
        max_num_seqs=max(8, args.batch_size),
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=0.95,
    )

    # ---- Stream out as we go (resumable)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "a")
    print(f"[distill] writing to {out_path}")

    # ---- Main batched loop
    n_kept = 0
    n_total_samples = 0
    n_passed = 0
    n_batches = (len(problems) + args.batch_size - 1) // args.batch_size
    t_start = time.time()
    for b_i in range(n_batches):
        batch = problems[b_i*args.batch_size:(b_i+1)*args.batch_size]
        prompts = [_build_chat_template_prompt(tok, build_user_prompt(p))
                   for p in batch]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling)
        gen_dt = time.time() - t0
        for prob, out in zip(batch, outputs):
            for s_i, sample in enumerate(out.outputs):
                completion = sample.text
                code = extract_code_block(completion)
                n_total_samples += 1
                # Grade if we have tests, else passthrough
                grade_target = code if code is not None else completion
                tier, score = grade_or_passthrough(prob, grade_target)
                # Rejection-sample if requested
                if args.keep_only_passing and prob.tests and tier != "pass":
                    continue
                # Always-skip if no code at all extractable AND we required tests
                if code is None and prob.tests:
                    continue
                if tier == "pass":
                    n_passed += 1
                write_jsonl_record(out_f, task_id=prob.task_id,
                                   problem_prompt=prob.prompt,
                                   qwen_completion=completion,
                                   extracted_code=code,
                                   has_tests=bool(prob.tests),
                                   tier=tier, score=score,
                                   sample_idx=s_i)
                n_kept += 1
        elapsed = time.time() - t_start
        n_done = (b_i + 1) * args.batch_size
        print(f"[distill] batch {b_i+1}/{n_batches} "
              f"({n_done:>5}/{len(problems)} problems)  "
              f"kept={n_kept}  passed={n_passed}/{n_total_samples}  "
              f"gen_dt={gen_dt:.1f}s  elapsed={elapsed/60:.1f}m")

    out_f.close()
    print(f"\n[distill] done. kept={n_kept} samples, passed={n_passed}, "
          f"output={out_path}")


if __name__ == "__main__":
    sys.exit(main())
