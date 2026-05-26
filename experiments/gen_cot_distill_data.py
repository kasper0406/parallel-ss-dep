"""Generate (problem, chain-of-thought, solution) triples from Qwen 3.6
AWQ for thinking distillation (THINKING_PLAN.md Phase 4).

The existing distill_solutions.py captures (problem, qwen_completion,
extracted_code) where qwen_completion is free-form prose+code. For
Phase 4 we want a STRUCTURED separation: the CoT prose alone, and the
final code alone — so a downstream conversion can materialise each CoT
step into our [THINKING]-token format for SFT (build_cot_sft_data.py).

The prompt forces a strict THINKING: / SOLUTION: layout so the parser
can reliably extract the two halves. A validator drops outputs that
don't have at least 2 numbered thinking steps AND extractable,
AST-parseable code.

Pipeline:
  1. Load problems from any code_grader.LOADERS key.
  2. For each problem, ask Qwen for N rollouts via vLLM at temperature τ.
  3. Parse THINKING: / SOLUTION: ```python``` blocks out of each
     completion.
  4. Validate (>=2 thinking steps, AST-parseable code).
  5. Append JSONL rows to --out: {task_id, problem_text, cot_text,
     solution_code, source}.

Run from the dedicated vLLM venv (mirrors distill_solutions.py):
  CUDA_VISIBLE_DEVICES=0 ~/ml/parallel-ss-dep/.venv-vllm/bin/python \\
      experiments/gen_cot_distill_data.py \\
      --dataset mbpp_combined \\
      --n_per_problem 4 \\
      --out data/cot_distill.jsonl

Smoke (5 problems × 5 rollouts = 25 generations, GPU required):
  ... --smoke
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
import time
from typing import IO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.code_grader import LOADERS, Problem


# ---------------------------------------------------------------------------
# Prompt template — strict THINKING:/SOLUTION: layout for reliable parsing.
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You will solve a Python programming problem. Think step-by-step, then \
write the solution.

Problem:
{problem_prompt}

First, think through the problem in plain prose — what's the algorithm, \
what edge cases, what helper functions. Be concrete. 4-8 short steps \
is ideal.

Then write the complete Python solution in a ```python``` block. The \
solution must work as-is (no `# TODO`, no `pass`).

Format your response as:
THINKING:
1. <first step>
2. <second step>
...

SOLUTION:
```python
<complete working code>
```
"""


def build_prompt(problem: Problem) -> str:
    """Render the user-side prompt for one problem."""
    if problem.prompt_is_code:
        body = f"```python\n{problem.prompt}\n```"
    else:
        body = problem.prompt
    return _PROMPT_TEMPLATE.format(problem_prompt=body)


# ---------------------------------------------------------------------------
# Parser: split THINKING from SOLUTION, extract code block.
# ---------------------------------------------------------------------------

# Header tolerance: optional leading markdown (#, **), optional colon, then EOL.
_RE_THINKING = re.compile(
    r"(?im)^\s*(?:#+\s*|\*+\s*)?THINKING\s*[:\-]?\s*$"
)
_RE_SOLUTION = re.compile(
    r"(?im)^\s*(?:#+\s*|\*+\s*)?SOLUTION\s*[:\-]?\s*$"
)
_FENCE_PY = re.compile(r"```python\s*\n(.*?)\n?```", re.DOTALL)
_FENCE_ANY = re.compile(r"```[a-zA-Z]*\s*\n(.*?)\n?```", re.DOTALL)
# Numbered-step line — "1." / "1)" / "(1)" at start of line.
_RE_STEP = re.compile(r"(?m)^\s*(?:\(?\d+\)|\d+\.)\s+\S")


def extract_code_block(text: str) -> str | None:
    """Pull the FIRST python fenced block from `text`. Falls back to
    any unmarked fenced block. Returns None if no fence found."""
    m = _FENCE_PY.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_ANY.search(text)
    if m:
        return m.group(1).strip()
    return None


def parse_cot_response(text: str) -> tuple[str | None, str | None]:
    """Split a Qwen completion into (cot_text, solution_code).

    Returns (None, None) if the THINKING or SOLUTION header isn't
    parseable. The two halves are returned independently so the caller
    can decide how to validate each.
    """
    m_think = _RE_THINKING.search(text)
    m_sol = _RE_SOLUTION.search(text)
    if m_think is None or m_sol is None:
        return None, None
    if m_sol.start() <= m_think.end():
        return None, None
    cot_text = text[m_think.end():m_sol.start()].strip()
    after_sol = text[m_sol.end():]
    code = extract_code_block(after_sol)
    if code is None:
        # Sometimes the model puts the fenced code BEFORE the next
        # heading; search the whole tail.
        code = extract_code_block(text[m_sol.end():])
    return (cot_text or None), code


def count_thinking_steps(cot_text: str) -> int:
    """How many `1. ...` numbered steps does the CoT contain?"""
    return len(_RE_STEP.findall(cot_text))


def validate_triple(cot_text: str | None, solution_code: str | None) -> bool:
    """Reject outputs missing thinking structure or with unparseable code.

    - cot_text must exist and have >= 2 numbered steps (Qwen sometimes
      collapses to one paragraph; we drop those because they don't
      provide a sequence of think positions for materialisation).
    - solution_code must exist and AST-parse (we don't run it; grader
      lives downstream).
    """
    if not cot_text or not solution_code:
        return False
    if count_thinking_steps(cot_text) < 2:
        return False
    try:
        ast.parse(solution_code)
    except SyntaxError:
        return False
    return True


# ---------------------------------------------------------------------------
# JSONL writer.
# ---------------------------------------------------------------------------

def write_jsonl_record(f: IO[str], *,
                       task_id: str,
                       problem_text: str,
                       cot_text: str,
                       solution_code: str,
                       source: str,
                       sample_idx: int) -> None:
    """Append one JSONL row. Schema mirrors the Phase-4 spec: the
    structured (problem, cot, solution) triple ready for materialisation
    into think-token SFT data via build_cot_sft_data.py."""
    row = {
        "task_id": task_id,
        "sample_idx": sample_idx,
        "problem_text": problem_text,
        "cot_text": cot_text,
        "solution_code": solution_code,
        "source": source,
    }
    f.write(json.dumps(row) + "\n")
    f.flush()


# ---------------------------------------------------------------------------
# Chat-template prompt (vLLM expects the apply_chat_template output).
# ---------------------------------------------------------------------------

def _build_chat_template_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Main (vLLM-driven).
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen_model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ")
    p.add_argument("--dataset", type=str, default="mbpp_combined",
                   help="LOADERS key from code_grader.py.")
    p.add_argument("--n_per_problem", type=int, default=4,
                   help="Independent rollouts per problem.")
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--out", type=str, required=True,
                   help="Output JSONL path. Appends, so resumable.")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gpu_mem_fraction", type=float, default=0.92)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="5 problems × 5 rollouts (25 generations).")
    args = p.parse_args()

    if args.smoke:
        args.max_problems = 5
        args.n_per_problem = 5
        args.max_new_tokens = 1024
        args.batch_size = 8

    # ---- Load problems
    if args.dataset not in LOADERS:
        raise SystemExit(
            f"unknown --dataset {args.dataset!r}; choose from {list(LOADERS)}")
    problems = LOADERS[args.dataset]()
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    if args.num_shards > 1:
        problems = [pb for i, pb in enumerate(problems)
                    if i % args.num_shards == args.shard_id]
    if not problems:
        raise SystemExit(f"no problems after shard {args.shard_id}/"
                         f"{args.num_shards}")
    print(f"[cot] loaded {len(problems)} problems from {args.dataset} "
          f"× {args.n_per_problem} rollouts = "
          f"{len(problems) * args.n_per_problem} planned generations")

    # ---- Load vLLM teacher
    print(f"[cot] loading {args.qwen_model} via vLLM ...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    t0 = time.time()
    llm = LLM(
        model=args.qwen_model,
        quantization="awq",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_fraction,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=args.enforce_eager,
        max_num_seqs=max(8, args.batch_size),
    )
    tok = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
    print(f"  loaded in {time.time() - t0:.1f}s")

    sampling = SamplingParams(
        n=args.n_per_problem,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=0.95,
    )

    # ---- Stream out as we go
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "a")
    print(f"[cot] writing to {out_path}")

    n_kept = 0
    n_total = 0
    n_batches = (len(problems) + args.batch_size - 1) // args.batch_size
    t_start = time.time()
    for b_i in range(n_batches):
        batch = problems[b_i*args.batch_size:(b_i+1)*args.batch_size]
        prompts = [_build_chat_template_prompt(tok, build_prompt(pb))
                   for pb in batch]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling)
        gen_dt = time.time() - t0
        for pb, out in zip(batch, outputs):
            for s_i, sample in enumerate(out.outputs):
                n_total += 1
                cot_text, sol_code = parse_cot_response(sample.text)
                if not validate_triple(cot_text, sol_code):
                    continue
                write_jsonl_record(out_f,
                                   task_id=pb.task_id,
                                   problem_text=pb.prompt,
                                   cot_text=cot_text,
                                   solution_code=sol_code,
                                   source=args.dataset,
                                   sample_idx=s_i)
                n_kept += 1
        elapsed = time.time() - t_start
        print(f"[cot] batch {b_i+1}/{n_batches}  "
              f"kept={n_kept}/{n_total} "
              f"({n_kept/max(1,n_total)*100:.1f}%)  "
              f"gen_dt={gen_dt:.1f}s  elapsed={elapsed/60:.1f}m")

    out_f.close()
    print(f"\n[cot] done. kept={n_kept}/{n_total}, output={out_path}")


if __name__ == "__main__":
    sys.exit(main())
