"""Diagnose WHY v7.1-mbpp-combined RL scores 0/164 on HumanEval.

Three diagnostics on the SAME 20-problem HumanEval slice:

  H1. Failure-mode tier distribution.
      How many of the 164 fail with syntax_error vs exec_error vs
      runtime_error vs partial? Different tiers → different diagnoses:
        - all syntax_error  → can't produce valid Python
        - all exec_error    → produces Python but doesn't define entry_point
        - all partial 0/N   → produces a runnable function but wrong logic
        - all partial M/N   → close to right, generalisation just imperfect

  H2. Per-token CE on the GOLD solution.
      Force-feed the gold completion through the model and compute mean
      next-token CE. Low CE → model "knows" the answer, generation just
      can't surface it (sampling/decoding issue). High CE → model has
      never seen this pattern, no amount of decoding finesse will help.

  H3. Prompt-format mismatch test.
      RL trained on MBPP (natural-language prompt: "Write a function to
      ..."). HumanEval is code-completion (function header + docstring,
      model continues the body). Re-render each HumanEval prompt as a
      natural-language instruction and re-score. If pass rate moves above
      0 → format mismatch was the bottleneck. If still 0 → it's not.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/diag_humaneval_failure.py \\
        --ckpt checkpoints/rl_grader_v7_pkm_film_combined.pt \\
        --n_problems 20
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.code_grader import (
    Problem, grade, load_humaneval, truncate_at_stop, _STOP_SEQUENCES,
)
from experiments.eval_bracket_structure import build_model_from_ckpt


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _strip_docstring_quotes(s: str) -> str:
    """Strip surrounding triple-quoted docstring markers."""
    s = s.strip()
    for q in ('"""', "'''"):
        if s.startswith(q) and s.endswith(q) and len(s) > 2 * len(q):
            return s[len(q):-len(q)].strip()
    return s


def _parse_humaneval_prompt(prompt: str) -> tuple[str, str, str]:
    """Return (imports_and_helpers, def_header, docstring). Best-effort."""
    # The prompt format for HumanEval is roughly:
    #   <optional from typing import ...>
    #   <optional helper defs>
    #   def fn_name(args) -> ret:
    #       """<docstring>"""
    # We split on the LAST `def ` to get the target function.
    m = re.search(r"def\s+\w+\s*\(", prompt)
    if not m:
        return prompt, "", ""
    before = prompt[:m.start()]
    after = prompt[m.start():]
    # Find the docstring inside `after`. Look for triple-quoted block.
    doc_match = re.search(r'("""|\'\'\')(.*?)\1', after, flags=re.DOTALL)
    if doc_match:
        docstring = doc_match.group(2).strip()
    else:
        docstring = ""
    # def_header = up to (and including) the closing `:` of the signature
    sig_match = re.search(r"def\s+\w+\s*\([^)]*\)\s*(->\s*[^:]+)?\s*:", after)
    def_header = sig_match.group(0) if sig_match else after.split("\n")[0]
    return before, def_header, docstring


def _natural_lang_prompt(problem: Problem) -> str:
    """Re-render a HumanEval problem as a natural-language MBPP-style prompt.

    Output is a single string with no function header — the model must
    generate the whole function from scratch (matching the MBPP RL distribution).
    """
    imports, def_header, docstring = _parse_humaneval_prompt(problem.prompt)
    entry = problem.entry_point or "function"
    return (
        f"Write a Python function named `{entry}` that does the following:\n"
        f"{docstring}\n"
        f"The function should be importable and callable.\n"
    )


# -----------------------------------------------------------------------------
# Greedy generation (single-batch, simple)
# -----------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(model, tok, prompt: str, max_gen: int = 256,
                     use_thinking: bool = True,
                     thinking_token_id: int | None = None,
                     emit_threshold: float = 0.5,
                     min_emit_before_eos: int = 30) -> tuple[str, dict]:
    """Greedy decode with optional gate-driven thinking. Returns
    (emit_text, diag_info) where diag_info has think_count, emit_count.
    """
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    prompt_len = ids.shape[1]
    emit_tokens = []
    n_think = 0
    eos_id = tok.eos_token_id
    finished = False
    for _ in range(max_gen + 200):  # extra slack for think tokens
        out = model(ids, return_gate=use_thinking, return_aux=False)
        if use_thinking:
            logits, gate_logits = out[0], out[2] if len(out) >= 3 else None
            if isinstance(out, tuple) and len(out) >= 3:
                logits = out[0]; gate_logits = out[2]
            else:
                logits = out; gate_logits = None
        else:
            logits = out
        if isinstance(logits, tuple):
            logits = logits[0]
        last_logits = logits[0, -1]  # (V,)
        # Gate decision
        if use_thinking and gate_logits is not None and thinking_token_id is not None:
            g = torch.sigmoid(gate_logits[0, -1].float()).item()
            if g < emit_threshold and n_think < max_gen:
                # think
                ids = torch.cat([ids, torch.tensor([[thinking_token_id]], device=ids.device)], dim=1)
                n_think += 1
                continue
        # Emit
        nxt = int(last_logits.argmax().item())
        # EOS guard
        if eos_id is not None and nxt == eos_id and len(emit_tokens) < min_emit_before_eos:
            last_logits[eos_id] = -1e9
            nxt = int(last_logits.argmax().item())
        if eos_id is not None and nxt == eos_id:
            break
        emit_tokens.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
        if len(emit_tokens) >= max_gen:
            break
    text = tok.decode(emit_tokens, skip_special_tokens=True)
    return text, {"n_think": n_think, "n_emit": len(emit_tokens)}


@torch.no_grad()
def gold_ce(model, tok, problem: Problem) -> float:
    """Per-token CE of the GOLD completion under the model, conditioned
    on the prompt. Low → model "knows" the answer."""
    if not problem.gold_solution:
        return float("nan")
    full = problem.prompt + problem.gold_solution
    ids = tok(full, return_tensors="pt").input_ids.cuda()
    prompt_len = tok(problem.prompt, return_tensors="pt").input_ids.shape[1]
    if ids.shape[1] <= prompt_len:
        return float("nan")
    logits = model(ids)
    if isinstance(logits, tuple):
        logits = logits[0]
    # CE only on the completion tokens (shifted)
    shift_logits = logits[0, prompt_len-1:-1].float()  # predicts comp tokens
    shift_targets = ids[0, prompt_len:]
    if shift_logits.shape[0] != shift_targets.shape[0]:
        n = min(shift_logits.shape[0], shift_targets.shape[0])
        shift_logits = shift_logits[:n]
        shift_targets = shift_targets[:n]
    ce = F.cross_entropy(shift_logits, shift_targets, reduction="mean").item()
    return ce


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_problems", type=int, default=20)
    p.add_argument("--max_gen", type=int, default=192)
    args = p.parse_args()

    print(f"Loading {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.cuda().eval()
    thinking_token_id = int(cfg.get("thinking_token_id"))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer",
                                                 "HuggingFaceTB/SmolLM2-135M"))

    print(f"Loading HumanEval ({args.n_problems} problems)")
    problems = load_humaneval()[:args.n_problems]

    # ---------------------------------------------------------------------
    # H3 prep: natural-language re-rendered versions
    # ---------------------------------------------------------------------
    nl_problems: list[Problem] = []
    for p_orig in problems:
        nl_prompt = _natural_lang_prompt(p_orig)
        nl_problems.append(Problem(
            task_id=p_orig.task_id + ".nl",
            prompt=nl_prompt,
            tests=p_orig.tests,
            entry_point=p_orig.entry_point,
            prompt_is_code=False,  # MBPP-style: prompt is NL, completion is whole code
            gold_solution=p_orig.gold_solution,
        ))

    # ---------------------------------------------------------------------
    # Generate + grade both ORIGINAL and NL versions
    # ---------------------------------------------------------------------
    results_orig = []
    results_nl = []

    print(f"\n{'='*78}\nH1+H3: Generate + grade ORIGINAL HumanEval prompts\n{'='*78}")
    for i, prob in enumerate(problems):
        text, diag = greedy_generate(model, tok, prob.prompt, max_gen=args.max_gen,
                                      use_thinking=True,
                                      thinking_token_id=thinking_token_id)
        g = grade(prob, text)
        results_orig.append((prob, text, g, diag))
        print(f"  [{i:>2}] {prob.task_id:<25} tier={g.tier:<14} score={g.score:.2f}  "
              f"think={diag['n_think']:>3}  emit={diag['n_emit']:>3}")

    print(f"\n{'='*78}\nH3: Re-rendered NATURAL-LANGUAGE prompts (MBPP-style)\n{'='*78}")
    for i, prob in enumerate(nl_problems):
        text, diag = greedy_generate(model, tok, prob.prompt, max_gen=args.max_gen,
                                      use_thinking=True,
                                      thinking_token_id=thinking_token_id)
        g = grade(prob, text)
        results_nl.append((prob, text, g, diag))
        print(f"  [{i:>2}] {prob.task_id:<25} tier={g.tier:<14} score={g.score:.2f}  "
              f"think={diag['n_think']:>3}  emit={diag['n_emit']:>3}")

    # ---------------------------------------------------------------------
    # H1: tier histogram
    # ---------------------------------------------------------------------
    print(f"\n{'='*78}\nH1: Failure-mode tier histogram\n{'='*78}")
    for label, results in [("HumanEval orig (code-completion)", results_orig),
                            ("HumanEval as NL (MBPP-style)", results_nl)]:
        tier_hist = Counter(g.tier for _, _, g, _ in results)
        score_mean = sum(g.score for _, _, g, _ in results) / len(results)
        pass_n = sum(1 for _, _, g, _ in results if g.tier == "pass")
        print(f"\n{label}:")
        print(f"  tier dist: {dict(tier_hist)}")
        print(f"  mean score: {score_mean:.3f}  (pass: {pass_n}/{len(results)})")

    # ---------------------------------------------------------------------
    # H2: per-token CE on gold
    # ---------------------------------------------------------------------
    print(f"\n{'='*78}\nH2: Per-token CE on GOLD solution (low → model 'knows')\n{'='*78}")
    ces = []
    for prob in problems:
        ce = gold_ce(model, tok, prob)
        ces.append(ce)
        print(f"  {prob.task_id:<25} CE={ce:.3f}  ppl={2.718**ce:.2f}")
    valid_ces = [c for c in ces if c == c]  # filter NaN
    if valid_ces:
        print(f"\n  mean CE on gold: {sum(valid_ces)/len(valid_ces):.3f}  "
              f"(ppl ≈ {2.718**(sum(valid_ces)/len(valid_ces)):.2f})")
        # Interpretation
        mean = sum(valid_ces)/len(valid_ces)
        if mean < 1.0:
            verdict = "VERY LOW — model strongly knows the answer (decode bug?)"
        elif mean < 2.5:
            verdict = "MODERATE — model has some signal but isn't confident"
        else:
            verdict = "HIGH — model fundamentally hasn't learned this distribution"
        print(f"  verdict: {verdict}")

    # ---------------------------------------------------------------------
    # Sample outputs side-by-side
    # ---------------------------------------------------------------------
    print(f"\n{'='*78}\nSample outputs (first 3 problems, both formats)\n{'='*78}")
    for i in range(min(3, len(problems))):
        orig_prob, orig_text, orig_g, _ = results_orig[i]
        nl_prob, nl_text, nl_g, _ = results_nl[i]
        print(f"\n--- Problem {orig_prob.task_id} ---")
        print(f"Original prompt (last 200 chars):\n  ...{orig_prob.prompt[-200:]}")
        print(f"Model EMIT (original, tier={orig_g.tier}):\n  {orig_text[:300]!r}")
        print(f"NL prompt:\n  {nl_prob.prompt[:300]}")
        print(f"Model EMIT (NL, tier={nl_g.tier}):\n  {nl_text[:300]!r}")
        print(f"Gold solution (first 200 chars):\n  {(orig_prob.gold_solution or '')[:200]!r}")


if __name__ == "__main__":
    sys.exit(main())
