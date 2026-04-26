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
             temperature: float = 0.0, eos_token_id: int | None = None
             ) -> torch.Tensor:
    out = prompt_ids.clone()
    for _ in range(max_gen):
        logits = model(out)
        next_logits = logits[:, -1, :]
        if temperature == 0.0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_tok], dim=1)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    return out


def evaluate(ckpt_path: str, n_samples: int = 1, temperature: float = 0.0,
             max_gen: int = 256, max_problems: int | None = None):
    print(f"Loading checkpoint: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    print(f"  feedback={cfg.get('feedback_mode')}  n_layers={cfg['n_layers']}")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    ds = load_dataset("openai_humaneval", split="test")

    n_passed = 0
    n_total = 0
    failures: list[dict] = []
    for i, problem in enumerate(ds):
        if max_problems is not None and i >= max_problems:
            break
        prompt = problem["prompt"]
        # Tokenise; check fits in model's max_T.
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) + max_gen > cfg["max_T"]:
            # Truncate from start (keep tail).
            prompt_ids = prompt_ids[-(cfg["max_T"] - max_gen):]
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)

        any_passed = False
        for _ in range(n_samples):
            gen = generate(model, prompt_t, max_gen=max_gen,
                           temperature=temperature, eos_token_id=tok.eos_token_id)
            gen_only = gen[0, len(prompt_ids):].tolist()
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

        if (i + 1) % 20 == 0:
            print(f"  {i + 1} problems: pass@{n_samples}={n_passed/n_total:.3f}")

    rate = n_passed / max(1, n_total)
    print(f"\npass@{n_samples} = {rate:.3f}  ({n_passed}/{n_total})")
    return {"pass_rate": rate, "n_passed": n_passed, "n_total": n_total,
            "n_samples_per_problem": n_samples, "temperature": temperature,
            "failures_first_5": failures[:5]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, action="append")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--max_problems", type=int, default=None)
    args = p.parse_args()

    all_results = {}
    for ckpt in args.ckpt:
        print(f"\n{'=' * 70}\nEvaluating: {ckpt}\n{'=' * 70}")
        all_results[ckpt] = evaluate(
            ckpt, n_samples=args.n_samples, temperature=args.temperature,
            max_gen=args.max_gen, max_problems=args.max_problems,
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
