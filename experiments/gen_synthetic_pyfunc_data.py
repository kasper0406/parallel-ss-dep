"""Generate Phi-1-style synthetic (signature+docstring → body) Python
function exercises via Qwen, ready to fold into pretrain as plain LM
text.

Phi-1 hit 50% HumanEval at 1.3B params by aggressively pretraining on
synthetic "textbook quality" docstring → code completion examples.
Our current pretrain mix has 0% of this format — only raw code +
instruction-following code + CVE-fix. The model never sees
`def f(x): \"\"\"docstring\"\"\"\\n    {fill in body}` during pretrain,
which IS HumanEval's exact format. This script closes that gap.

Pipeline:
  1. Take a list of ~50 algorithmic topics.
  2. For each (topic, difficulty) draw, prompt Qwen for ONE
     HumanEval-style function exercise (signature + docstring +
     implementation), batched via vLLM.
  3. Pull the first ```python ... ``` fence out of each completion.
  4. Append a JSONL row `{task_id, text, source}` where `text` is the
     full function — that is what pretrain consumes (plain LM loss).

Run from the dedicated vLLM venv (mirrors distill_solutions.py):
  CUDA_VISIBLE_DEVICES=0 ~/ml/parallel-ss-dep/.venv-vllm/bin/python \\
      experiments/gen_synthetic_pyfunc_data.py \\
      --n_per_topic 200 \\
      --out data/synthetic_pyfunc.jsonl

Smoke (5 examples × 5 topics, 25 total — validate the pipeline):
  ... --smoke

The output JSONL plugs into `data_mix.py` via a new `jsonl_path` field
on a YAML source (see `configs/pretrain_mix_v5_synth.yaml`).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time
from typing import IO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Topic catalog (~50 algorithmic categories).
# ---------------------------------------------------------------------------

DEFAULT_TOPICS: list[str] = [
    # String manipulation
    "string manipulation: case conversion, splitting, joining",
    "string searching and pattern matching",
    "string reversal and palindrome checks",
    "string parsing and tokenization",
    "string formatting and templating",
    # List operations
    "list filtering and selection",
    "list mapping and transformation",
    "list aggregation: sum, product, mean",
    "list sorting with custom comparators",
    "list deduplication and unique elements",
    "list flattening and nested structures",
    "list partitioning and chunking",
    "list rotation and shifting",
    "list intersection, union, and difference",
    # Dictionary operations
    "dictionary lookup and default values",
    "dictionary merging and updating",
    "dictionary inversion (key/value swap)",
    "counting occurrences with dictionaries",
    "group-by operations with dictionaries",
    # Math
    "basic arithmetic and number properties",
    "prime numbers and factorization",
    "fibonacci-style sequences and recurrences",
    "greatest common divisor and least common multiple",
    "modular arithmetic problems",
    "combinatorics: permutations and combinations",
    "number base conversion (binary, hex, decimal)",
    "geometry: distance, area, perimeter",
    "statistics: median, mode, variance",
    # Recursion
    "recursive functions over integers",
    "recursive functions over lists",
    "tree traversal (preorder, inorder, postorder)",
    # Iteration patterns
    "two-pointer and sliding-window algorithms",
    "running totals and cumulative sums",
    "monotonic-sequence detection",
    # Search and sort
    "binary search variants",
    "insertion / selection / bubble sort",
    "finding minimum, maximum, kth element",
    # Date / time
    "date arithmetic and formatting",
    "time-of-day parsing and conversion",
    # Validation / parsing
    "input validation and sanitization",
    "URL / email / phone-number parsing",
    "CSV-like row parsing",
    # Numeric arrays
    "matrix operations: transpose, multiplication",
    "2D-grid traversal and neighbours",
    # Bitwise
    "bitwise operations and bit-counting",
    # Misc
    "set operations and membership tests",
    "tuple unpacking and packing patterns",
    "generator and iterator patterns",
    "boolean logic and short-circuit evaluation",
    "type-coercion and safe casting",
]


# ---------------------------------------------------------------------------
# Prompt template.
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
Generate a Python function exercise in the same style as HumanEval problems:
- A function signature with type hints
- A clear docstring describing what the function should do, including at \
least one example (using >>> notation)
- A clean implementation

Topic: {topic}
Difficulty: {difficulty}

Output ONLY the function (signature, docstring, body) in a ```python``` block.
"""


def build_prompt(topic: str, difficulty: str) -> str:
    """Render the user-side prompt for a (topic, difficulty) draw."""
    return _PROMPT_TEMPLATE.format(topic=topic, difficulty=difficulty)


# ---------------------------------------------------------------------------
# Code-fence extraction (mirrors distill_solutions.extract_code_block).
# ---------------------------------------------------------------------------

_FENCE_PY = re.compile(r"```python\s*\n(.*?)\n?```", re.DOTALL)
_FENCE_ANY = re.compile(r"```[a-zA-Z]*\s*\n(.*?)\n?```", re.DOTALL)


def extract_code_block(text: str) -> str | None:
    """Pull the FIRST python fenced block from `text`. Falls back to
    the first unmarked fenced block. Returns None if no fence found.
    """
    m = _FENCE_PY.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_ANY.search(text)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# JSONL writer.
# ---------------------------------------------------------------------------

def write_jsonl_record(f: IO[str], *,
                       task_id: str,
                       text: str,
                       source: str = "synthetic_pyfunc") -> None:
    """Append one JSONL row. The `text` field is the whole function
    body that pretrain will LM-train on."""
    row = {"task_id": task_id, "text": text, "source": source}
    f.write(json.dumps(row) + "\n")
    f.flush()


# ---------------------------------------------------------------------------
# Validation: keep only outputs that parse + look like a function.
# ---------------------------------------------------------------------------

def looks_like_function(code: str) -> bool:
    """Cheap filter: must parse as Python AND contain at least one
    top-level `def` with a docstring. We don't try to grade the body
    here — Phi-1 lesson is that even noisy synthetic exercises help."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node):
                return True
    return False


# ---------------------------------------------------------------------------
# Chat-template prompt (vLLM expects the apply_chat_template output).
# ---------------------------------------------------------------------------

def _build_chat_template_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def _topic_slug(topic: str) -> str:
    """Make a filesystem-safe slug from a topic string."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", topic.lower()).strip("_")
    return s[:40]


def _draw_difficulty(rng: random.Random) -> str:
    """Bias toward easy — Phi-1 corpus is overwhelmingly textbook-easy."""
    return "easy" if rng.random() < 0.75 else "medium"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen_model", type=str,
                   default="QuantTrio/Qwen3.6-35B-A3B-AWQ",
                   help="vLLM model name; defaults to the model used by "
                        "distill_solutions.py")
    p.add_argument("--n_per_topic", type=int, default=200)
    p.add_argument("--topics_file", type=str, default=None,
                   help="Optional path to a newline-delimited topics file. "
                        "Defaults to the hardcoded DEFAULT_TOPICS.")
    p.add_argument("--out", type=str, required=True,
                   help="Output JSONL path. Appends, so resumable.")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu_mem_fraction", type=float, default=0.92)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="5 examples × 5 topics (25 total). "
                        "Validates the pipeline without burning hours.")
    args = p.parse_args()

    # Load topics.
    if args.topics_file:
        with open(args.topics_file) as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = list(DEFAULT_TOPICS)
    if not topics:
        raise SystemExit("no topics loaded")

    if args.smoke:
        topics = topics[:5]
        args.n_per_topic = 5
        args.max_new_tokens = 256
        args.batch_size = 8

    # Shard across machines (round-robin over topics).
    if args.num_shards > 1:
        topics = [t for i, t in enumerate(topics)
                  if i % args.num_shards == args.shard_id]
    if not topics:
        raise SystemExit(f"no topics remain after shard {args.shard_id}/"
                         f"{args.num_shards}")
    print(f"[synth] {len(topics)} topics × {args.n_per_topic} examples "
          f"= {len(topics)*args.n_per_topic} planned generations")

    # Build (topic, difficulty) draw list.
    rng = random.Random(args.seed + 7919 * args.shard_id)
    draws: list[tuple[str, str]] = []
    for topic in topics:
        for _ in range(args.n_per_topic):
            draws.append((topic, _draw_difficulty(rng)))
    rng.shuffle(draws)

    # Load vLLM teacher.
    print(f"[synth] loading {args.qwen_model} via vLLM ...")
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
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=0.95,
    )

    # Stream out as we go (resumable).
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(out_path, "a")
    print(f"[synth] writing to {out_path}")

    # Per-topic monotonically increasing index (for task_id).
    topic_idx: dict[str, int] = {t: 0 for t in topics}

    n_kept = 0
    n_total = 0
    n_batches = (len(draws) + args.batch_size - 1) // args.batch_size
    t_start = time.time()
    for b_i in range(n_batches):
        batch = draws[b_i*args.batch_size:(b_i+1)*args.batch_size]
        prompts = [
            _build_chat_template_prompt(tok, build_prompt(topic, diff))
            for (topic, diff) in batch
        ]
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling)
        gen_dt = time.time() - t0
        for (topic, diff), out in zip(batch, outputs):
            n_total += 1
            sample = out.outputs[0]
            completion = sample.text
            code = extract_code_block(completion)
            if code is None:
                continue
            if not looks_like_function(code):
                continue
            i = topic_idx[topic]
            topic_idx[topic] = i + 1
            task_id = f"synth_pyfunc/{_topic_slug(topic)}/{i:05d}"
            write_jsonl_record(out_f, task_id=task_id, text=code,
                               source="synthetic_pyfunc")
            n_kept += 1
        elapsed = time.time() - t_start
        print(f"[synth] batch {b_i+1}/{n_batches}  "
              f"kept={n_kept}/{n_total} ({n_kept/max(1,n_total)*100:.1f}%)  "
              f"gen_dt={gen_dt:.1f}s  elapsed={elapsed/60:.1f}m")

    out_f.close()
    print(f"\n[synth] done. kept={n_kept}/{n_total} samples, output={out_path}")


if __name__ == "__main__":
    sys.exit(main())
