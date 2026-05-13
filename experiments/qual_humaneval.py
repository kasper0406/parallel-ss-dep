"""Qualitative investigation: pick easy HumanEval problems, show full
prompt + generation + grading result for the v2 mid-eval ckpt.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/qual_humaneval.py \\
      --ckpt checkpoints/pretrain_mix_v2_step34878_tok500011008.pt \\
      --n_problems 8 --max_gen 192
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate
from experiments.code_grader import load_humaneval, grade, truncate_at_stop


_BAR = "=" * 80
_DASH = "-" * 80


def _describe_failure(error: str | None, completion: str) -> str:
    """Heuristic one-line classification of a failed completion."""
    if error is None:
        return "passed"
    if "timeout" in error:
        return "TIMEOUT — generation likely entered infinite loop / nonterminating body"
    if "SyntaxError" in error:
        return "SYNTAX ERROR — model produced non-parsable Python"
    if "IndentationError" in error:
        return "INDENT — model lost track of block structure"
    if "NameError" in error:
        return f"NAME ERROR — references undefined symbol ({error.splitlines()[-1].strip()[:80]})"
    if "TypeError" in error:
        return f"TYPE ERROR — wrong call signature ({error.splitlines()[-1].strip()[:80]})"
    if "AttributeError" in error:
        return f"ATTR ERROR — bad method/attr ({error.splitlines()[-1].strip()[:80]})"
    if "AssertionError" in error or "Test" in error:
        return "LOGIC ERROR — code ran but produced wrong answer"
    if "ImportError" in error or "ModuleNotFound" in error:
        return "IMPORT — model tried to use a missing module"
    if not completion.strip():
        return "EMPTY — model produced no body"
    if len(completion.strip().splitlines()) == 1 and "..." in completion:
        return "ELLIPSIS — model output a placeholder"
    return f"OTHER — {error.splitlines()[-1].strip()[:120]}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_problems", type=int, default=8,
                   help="How many problems (from index 0) to inspect.")
    p.add_argument("--max_gen", type=int, default=192)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--max_think_per_step", type=int, default=8)
    p.add_argument("--min_emit_before_eos", type=int, default=0,
                   help="Suppress eos_token_id for the first N emitted tokens.")
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M")
    args = p.parse_args()

    print(_BAR)
    print(f"Loading ckpt: {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    has_gate = hasattr(model, "gate_head")
    has_memory = getattr(model, "use_memory", False)
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")
    print(f"  feedback={cfg.get('feedback_mode')} "
          f"n_layers={cfg['n_layers']} d_model={cfg['d_model']}")
    print(f"  output_gate={has_gate}  memory={has_memory}  "
          f"think_tok_id={thinking_token_id}")
    use_thinking = has_gate and thinking_token_id is not None
    print(f"  use_thinking={use_thinking}  emit_threshold={args.emit_threshold}")
    print(_BAR)

    print("\nLoading tokenizer:", args.tokenizer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print("\nLoading HumanEval ...")
    problems = load_humaneval()[: args.n_problems]
    print(f"  {len(problems)} problems")

    classifications = []

    for idx, problem in enumerate(problems):
        print(f"\n{_BAR}\nProblem {idx}: {problem.task_id}\n{_DASH}")
        # Prompt
        print("PROMPT:")
        print(textwrap.indent(problem.prompt.rstrip(), "  "))
        # Encode + generate
        prompt_ids = tok(problem.prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out, diag = generate(
                model, prompt_ids,
                max_gen=args.max_gen,
                use_thinking=use_thinking,
                thinking_token_id=thinking_token_id,
                max_think_per_step=args.max_think_per_step,
                emit_threshold=args.emit_threshold,
                eos_token_id=tok.eos_token_id,
                min_emit_before_eos=args.min_emit_before_eos,
            )
        # Decode (strip think tokens from output)
        gen_ids = out[0, prompt_ids.shape[1]:].tolist()
        if thinking_token_id is not None:
            gen_ids = [t for t in gen_ids if t != int(thinking_token_id)]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        # Show full generation
        print(f"\nGENERATION (emit_count={diag['emit_count']}, "
              f"think_total={diag['think_total']}, "
              f"think_rate={diag['think_rate']:.3f}, "
              f"mean_gate@emit={(sum(diag['gate_emit_values'])/max(1,len(diag['gate_emit_values']))):.3f}):")
        truncated = truncate_at_stop(gen_text)
        print(textwrap.indent(gen_text, "  "))
        print(f"\n(truncated for grading at first '\\ndef/class/#/print(/if __name__':")
        print(textwrap.indent(truncated, "  "))
        # Grade
        result = grade(problem, gen_text, timeout_s=7)
        classification = _describe_failure(result.error, truncated)
        print(f"\nGRADE: passed={result.passed}  elapsed={result.elapsed_s:.2f}s")
        if result.error:
            err_compact = result.error.strip().splitlines()
            err_compact = err_compact[-3:] if len(err_compact) > 3 else err_compact
            print(f"  error tail: {' | '.join(e.strip() for e in err_compact)}")
        print(f"  FAILURE PATTERN: {classification}")
        classifications.append((problem.task_id, result.passed, classification))

    print(f"\n{_BAR}\nSUMMARY")
    print(_DASH)
    n_pass = sum(1 for _, ok, _ in classifications if ok)
    print(f"Pass: {n_pass}/{len(classifications)}")
    # Failure-pattern breakdown
    from collections import Counter
    patterns = Counter(c.split(" — ")[0] for _, ok, c in classifications if not ok)
    for pat, cnt in patterns.most_common():
        print(f"  {pat:<24} ×{cnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
