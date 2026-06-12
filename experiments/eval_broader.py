"""Evaluate a ckpt on any code_grader LOADER (MBPP, MBPP+, LeetCode, etc.).

Companion to eval_humaneval.py — uses the same generate() but grades
via code_grader.grade() so any dataset registered in
code_grader.LOADERS works uniformly.

Usage:
    PYTHONPATH=. .venv/bin/python -u experiments/eval_broader.py \
        --ckpt checkpoints/foo.pt \
        --dataset mbpp_plus \
        --use_thinking --extract_code_block \
        --prompt_style sft_comment \
        --max_problems 200
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.code_grader import LOADERS, grade
from experiments.distill_solutions import extract_code_block as _extract_code
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import (generate, generate_with_retrieval_as_input,
                                         generate_latent_think)
from experiments.thinking import clean_latent_thread


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True,
                   choices=sorted(LOADERS.keys()))
    p.add_argument("--max_problems", type=int, default=None,
                   help="Truncate dataset to first N problems (for fast probes).")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--use_thinking", action="store_true")
    p.add_argument("--max_think_per_step", type=int, default=8)
    p.add_argument("--total_think_budget", type=int, default=400)
    p.add_argument("--emit_threshold", type=float, default=0.5)
    p.add_argument("--gate_floor", type=float, default=0.0)
    p.add_argument("--min_emit_before_eos", type=int, default=30)
    p.add_argument("--prompt_style", type=str, default="sft_comment",
                   choices=["raw", "sft_comment"])
    p.add_argument("--extract_code_block", action="store_true", default=True)
    p.add_argument("--no_extract_code_block", dest="extract_code_block",
                   action="store_false")
    p.add_argument("--generator", type=str, default="standard",
                   choices=["standard", "retrieval_as_input", "latent_think"])
    p.add_argument("--allow_legacy_thinking", action="store_true",
                   help="opt-in for the DEPRECATED thinking mechanisms "
                        "(--generator retrieval_as_input, or --generator "
                        "standard with --use_thinking = discrete-token "
                        "thinking). The validated mechanism is "
                        "--generator latent_think. Mirrors eval_humaneval.")
    p.add_argument("--grader_timeout_s", type=int, default=5)
    p.add_argument("--report_every", type=int, default=20)
    args = p.parse_args()
    _legacy_think = (args.generator == "retrieval_as_input"
                     or (args.generator == "standard" and args.use_thinking))
    if _legacy_think and not args.allow_legacy_thinking:
        p.error("Refusing a DEPRECATED thinking mechanism: "
                f"--generator {args.generator}"
                + (" with --use_thinking (discrete-token thinking)"
                   if args.generator == "standard" else "")
                + ". The validated mechanism is `--generator latent_think`. "
                "To deliberately reproduce a legacy result, pass "
                "--allow_legacy_thinking.")

    print(f"[eval-broader] dataset={args.dataset}  ckpt={args.ckpt}")
    problems = LOADERS[args.dataset]()
    if args.max_problems is not None:
        problems = problems[:args.max_problems]
    print(f"  loaded {len(problems)} problems")

    print(f"[eval-broader] loading {args.ckpt}")
    # Mirror eval_humaneval: the latent generator needs DeltaNet β=0 at think
    # positions (state-readonly), or the latent burst corrupts recurrent state.
    model, cfg = build_model_from_ckpt(
        args.ckpt,
        force_state_readonly=True if args.generator == "latent_think" else None)
    model = model.to("cuda").eval()
    has_gate = hasattr(model, "gate_head")
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")
    thinking_active = args.use_thinking or args.generator in (
        "latent_think", "retrieval_as_input")
    if thinking_active and (not has_gate or thinking_token_id is None):
        raise RuntimeError("thinking generators require gate + thinking_token_id")

    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"),
        trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    n_total = 0
    n_passed = 0
    tier_hist: dict[str, int] = {}
    t0 = time.time()
    for i, prob in enumerate(problems):
        raw_prompt = prob.prompt
        if args.prompt_style == "sft_comment":
            # Match the trained SFT format EXACTLY (eval_humaneval, sft_code,
            # latent_rl, latent_code_cotrain all use this prefix + the raw
            # prompt). The previous f"# {raw_prompt}" single-line comment was an
            # OOD prompt format that systematically deflated scores.
            prompt = "# Complete the following Python function.\n" + raw_prompt
        else:
            prompt = raw_prompt
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        eff_max_T = cfg["max_T"] if cfg["max_T"] > 0 else 2048
        thinks_reserve = (args.total_think_budget if thinking_active else 0)
        room = args.max_gen + thinks_reserve
        if len(prompt_ids) + room > eff_max_T:
            prompt_ids = prompt_ids[-(eff_max_T - room):]
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                 device="cuda").unsqueeze(0)

        gen_kw = dict(
            max_gen=args.max_gen, temperature=args.temperature,
            eos_token_id=tok.eos_token_id,
            use_thinking=args.use_thinking,
            thinking_token_id=thinking_token_id,
            max_think_per_step=args.max_think_per_step,
            total_think_budget=args.total_think_budget,
            emit_threshold=args.emit_threshold,
            min_emit_before_eos=args.min_emit_before_eos,
            gate_floor=args.gate_floor,
        )
        if args.generator == "retrieval_as_input":
            gen_kw.pop("use_thinking")
            gen_kw["additive"] = cfg.get("retrieval_input_additive", False)
            gen, _ = generate_with_retrieval_as_input(model, prompt_t, **gen_kw)
        elif args.generator == "latent_think":
            gen_kw.pop("use_thinking")
            with clean_latent_thread(model):
                gen, _ = generate_latent_think(model, prompt_t, **gen_kw)
        else:
            gen, _ = generate(model, prompt_t, **gen_kw)
        gen_only = gen[0, len(prompt_ids):].tolist()
        if thinking_token_id is not None:
            gen_only = [t for t in gen_only if t != int(thinking_token_id)]
        gen_text = tok.decode(gen_only, skip_special_tokens=True)
        if args.extract_code_block:
            code = _extract_code(gen_text)
            full_code = code if code is not None else gen_text
        else:
            full_code = prompt + gen_text

        g_res = grade(prob, full_code, timeout_s=args.grader_timeout_s)
        tier_hist[g_res.tier] = tier_hist.get(g_res.tier, 0) + 1
        n_total += 1
        if g_res.tier == "pass":
            n_passed += 1
        if (i + 1) % args.report_every == 0 or i == len(problems) - 1:
            elapsed = time.time() - t0
            tiers = "  ".join(f"{k}={v}" for k, v in sorted(tier_hist.items()))
            print(f"  {i + 1:>4}/{len(problems)}  "
                  f"pass={n_passed}/{n_total}={n_passed/n_total:.3f}  "
                  f"{tiers}  ({elapsed:.0f}s)")

    rate = n_passed / max(1, n_total)
    print(f"\nFINAL  dataset={args.dataset}  pass@1={rate:.3f}  "
          f"({n_passed}/{n_total})")
    tiers = "  ".join(f"{k}={v}" for k, v in sorted(tier_hist.items()))
    print(f"  tiers: {tiers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
