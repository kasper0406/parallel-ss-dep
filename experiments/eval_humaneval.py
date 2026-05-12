"""
HumanEval pass@1 / pass@k evaluation.

Phase 2 of post-PPL evals. Tests if architectural differences hide
behind PPL parity show up as actual code-generation differences.

Caveat: 135M / 5K-step models will likely score very low on HumanEval
in absolute terms. The point is relative comparison: does film @30L
beat DN @30L on actual code generation, even if PPL is tied?

Usage:
  python experiments/eval_humaneval.py \\
      --ckpt /path/to/model.pt \\
      --n_samples 1 --temperature 0.0 --max_gen 256
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import re
import signal
import sys
from contextlib import contextmanager

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt


# Stop sequences typical of function-end at column 0.
_STOP_SEQUENCES = ["\nclass ", "\ndef ", "\nif __name__", "\n#", "\nprint("]


@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _run_test_in_subprocess(code: str, test: str, entry_point: str,
                             timeout_s: int = 5) -> bool:
    """Execute code+test in a subprocess; return True if check passes."""
    def target(code, test, entry_point, q):
        try:
            ns = {}
            with time_limit(timeout_s):
                exec(code, ns)
                exec(test, ns)
                ns["check"](ns[entry_point])
            q.put(True)
        except Exception:
            q.put(False)

    q = mp.Queue()
    p = mp.Process(target=target, args=(code, test, entry_point, q))
    p.start()
    p.join(timeout=timeout_s + 2)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    try:
        return q.get_nowait()
    except Exception:
        return False


def _truncate_at_stop(text: str) -> str:
    """Trim generated text at first natural function boundary."""
    earliest = len(text)
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx >= 0 and idx < earliest:
            earliest = idx
    return text[:earliest]


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_gen: int = 256,
             temperature: float = 0.0, eos_token_id: int | None = None,
             use_thinking: bool = False,
             thinking_token_id: int | None = None,
             max_think_per_step: int = 8,
             total_think_budget: int | None = None,
             emit_threshold: float = 0.5,
             ) -> tuple[torch.Tensor, dict]:
    """Token-by-token generation.

    When `use_thinking=True`, after each forward pass we consult the output
    gate at the final position. The convention (matching `thinking.py`
    rollout) is: g = σ(gate_head(h)), where g = P(Emit). If g < emit_threshold
    we append `thinking_token_id` and run forward again, up to
    `max_think_per_step` consecutive thinks per emit step; we always force an
    emit when the budget is exhausted. `total_think_budget` (default 2×max_gen)
    caps the lifetime sum of think tokens.

    Returns (out_ids_including_thinks, diagnostics_dict). The caller is
    responsible for stripping `thinking_token_id` from `out` before grading.
    """
    if use_thinking:
        assert thinking_token_id is not None, \
            "use_thinking=True requires thinking_token_id"
        if total_think_budget is None:
            total_think_budget = 2 * max_gen
    out = prompt_ids.clone()
    emit_count = 0
    think_total = 0
    think_steps_used = []      # length = emit_count; thinks before each emit
    gate_emit_values = []      # gate values at each emit step
    while emit_count < max_gen:
        # Inner think loop: try to coax the gate above emit_threshold.
        thinks_this_step = 0
        while True:
            logits = model(out)
            next_logits = logits[:, -1, :]
            if not use_thinking:
                gate_val = 1.0  # force emit
                break
            # σ(gate_head(h)) at last position; >threshold ⇒ emit.
            gate_t = getattr(model, "_last_gate", None)
            if gate_t is None:
                gate_val = 1.0
            else:
                gate_val = float(gate_t[0, -1].item())
            force_emit = (
                thinks_this_step >= max_think_per_step
                or think_total >= total_think_budget
            )
            if gate_val >= emit_threshold or force_emit:
                break
            # Else: append a think token and loop.
            think_tok = torch.full((out.shape[0], 1), int(thinking_token_id),
                                    dtype=out.dtype, device=out.device)
            out = torch.cat([out, think_tok], dim=1)
            thinks_this_step += 1
            think_total += 1
        # Emit step: mask thinking_token_id from sampled output.
        if use_thinking and thinking_token_id is not None:
            next_logits = next_logits.clone()
            next_logits[..., int(thinking_token_id)] = -float("inf")
        if temperature == 0.0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_tok], dim=1)
        emit_count += 1
        if use_thinking:
            think_steps_used.append(thinks_this_step)
            gate_emit_values.append(gate_val)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    diag = {
        "emit_count": emit_count,
        "think_total": think_total,
        "think_steps_used": think_steps_used,
        "gate_emit_values": gate_emit_values,
        "think_rate": (think_total / max(1, think_total + emit_count))
                       if use_thinking else 0.0,
    }
    return out, diag


def evaluate(ckpt_path: str, n_samples: int = 1, temperature: float = 0.0,
             max_gen: int = 256, max_problems: int | None = None,
             use_thinking: bool = False,
             max_think_per_step: int = 8,
             total_think_budget: int | None = None,
             emit_threshold: float = 0.5):
    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    print(f"  feedback={cfg.get('feedback_mode')}  n_layers={cfg['n_layers']}")

    # Resolve thinking_token_id. We need it whenever use_thinking=True OR the
    # model has a working-memory module (the inference loop should know which
    # token to strip even if thinking is disabled, in case the model emits it).
    has_gate = hasattr(model, "gate_head")
    has_memory = getattr(model, "use_memory", False)
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")
    if use_thinking:
        if not has_gate:
            raise RuntimeError(
                "use_thinking=True but model has no output gate; this ckpt "
                "wasn't trained with --output_gate.")
        if thinking_token_id is None:
            raise RuntimeError(
                "use_thinking=True but ckpt has no thinking_token_id in cfg "
                "and model.thinking_token_id is None.")
        print(f"  THINKING ON: max_think_per_step={max_think_per_step} "
              f"emit_threshold={emit_threshold} "
              f"total_think_budget={total_think_budget or 2*max_gen} "
              f"thinking_token_id={thinking_token_id} memory={has_memory}")
    else:
        print(f"  THINKING OFF  (gate={has_gate} memory={has_memory} "
              f"thinking_token_id={thinking_token_id})")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    ds = load_dataset("openai_humaneval", split="test")

    n_passed = 0
    n_total = 0
    failures: list[dict] = []
    agg_think_total = 0
    agg_emit_total = 0
    agg_gate_values: list[float] = []
    n_problems_with_any_think = 0
    for i, problem in enumerate(ds):
        if max_problems is not None and i >= max_problems:
            break
        prompt = problem["prompt"]
        # Tokenise; check fits in model's max_T.
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        # max_T==0 means "no positional embedding limit" — the model has no
        # built-in T cap. Use a generous cap of 2048 to keep generation
        # bounded; don't truncate when the prompt is short.
        eff_max_T = cfg["max_T"] if cfg["max_T"] > 0 else 2048
        # Generation may produce up to max_gen emits + total_think_budget
        # think tokens, so reserve room for both when budget-truncating.
        thinks_reserve = (total_think_budget if total_think_budget is not None
                          else 2 * max_gen) if use_thinking else 0
        room_needed = max_gen + thinks_reserve
        if len(prompt_ids) + room_needed > eff_max_T:
            prompt_ids = prompt_ids[-(eff_max_T - room_needed):]
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)

        any_passed = False
        last_diag: dict = {}
        for _ in range(n_samples):
            gen, diag = generate(
                model, prompt_t, max_gen=max_gen,
                temperature=temperature, eos_token_id=tok.eos_token_id,
                use_thinking=use_thinking,
                thinking_token_id=thinking_token_id,
                max_think_per_step=max_think_per_step,
                total_think_budget=total_think_budget,
                emit_threshold=emit_threshold,
            )
            last_diag = diag
            gen_only_full = gen[0, len(prompt_ids):].tolist()
            # Strip think tokens before decoding (they're internal control
            # symbols, not part of the visible output).
            if thinking_token_id is not None:
                gen_only = [t for t in gen_only_full
                            if t != int(thinking_token_id)]
            else:
                gen_only = gen_only_full
            gen_text = tok.decode(gen_only, skip_special_tokens=True)
            gen_text = _truncate_at_stop(gen_text)
            full_code = prompt + gen_text
            if _run_test_in_subprocess(full_code, problem["test"],
                                        problem["entry_point"]):
                any_passed = True
                break
        n_total += 1
        if any_passed:
            n_passed += 1
        else:
            failures.append({
                "task_id": problem["task_id"],
                "gen_preview": gen_text[:120],
            })
        if use_thinking and last_diag:
            agg_think_total += last_diag["think_total"]
            agg_emit_total += last_diag["emit_count"]
            agg_gate_values.extend(last_diag["gate_emit_values"])
            if last_diag["think_total"] > 0:
                n_problems_with_any_think += 1

        if (i + 1) % 20 == 0:
            msg = f"  {i + 1} problems: pass@{n_samples}={n_passed/n_total:.3f}"
            if use_thinking:
                tr = agg_think_total / max(1, agg_think_total + agg_emit_total)
                msg += (f"  think_rate={tr:.3f}  "
                        f"problems_using_think={n_problems_with_any_think}/{n_total}")
            print(msg)

    rate = n_passed / max(1, n_total)
    print(f"\npass@{n_samples} = {rate:.3f}  ({n_passed}/{n_total})")
    result = {"pass_rate": rate, "n_passed": n_passed, "n_total": n_total,
              "n_samples_per_problem": n_samples, "temperature": temperature,
              "failures_first_5": failures[:5]}
    if use_thinking:
        import statistics as _stats
        mean_gate = (_stats.fmean(agg_gate_values) if agg_gate_values else 0.0)
        result.update({
            "use_thinking": True,
            "think_total": agg_think_total,
            "emit_total": agg_emit_total,
            "think_rate": agg_think_total / max(1, agg_think_total + agg_emit_total),
            "problems_using_think": n_problems_with_any_think,
            "mean_gate_at_emit": mean_gate,
        })
        print(f"  think_total={agg_think_total}  emit_total={agg_emit_total}  "
              f"think_rate={result['think_rate']:.3f}  "
              f"mean_gate_at_emit={mean_gate:.3f}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, action="append")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--use_thinking", action="store_true",
                   help="At inference time, consult the output gate and "
                        "append thinking tokens when σ(gate) < emit_threshold.")
    p.add_argument("--max_think_per_step", type=int, default=8,
                   help="Max consecutive think tokens between two emits.")
    p.add_argument("--total_think_budget", type=int, default=None,
                   help="Lifetime cap on think tokens per problem. "
                        "Default = 2 × max_gen.")
    p.add_argument("--emit_threshold", type=float, default=0.5,
                   help="σ(gate) threshold above which we emit (else think).")
    args = p.parse_args()

    all_results = {}
    for ckpt in args.ckpt:
        print(f"\n{'=' * 70}\nEvaluating: {ckpt}\n{'=' * 70}")
        all_results[ckpt] = evaluate(
            ckpt, n_samples=args.n_samples, temperature=args.temperature,
            max_gen=args.max_gen, max_problems=args.max_problems,
            use_thinking=args.use_thinking,
            max_think_per_step=args.max_think_per_step,
            total_think_budget=args.total_think_budget,
            emit_threshold=args.emit_threshold,
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ckpt':<60} {'pass@k':>10} {'(passed/total)':>16}")
    for ckpt, r in all_results.items():
        name = pathlib.Path(ckpt).stem
        print(f"{name:<60} {r['pass_rate']*100:>9.1f}% "
              f"{r['n_passed']:>5}/{r['n_total']:<5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
