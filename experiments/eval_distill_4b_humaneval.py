"""
HumanEval pass@1 for the DN-4B distilled student.

The distilled student uses the Qwen3.6 tokenizer (vocab 248k), so we
cannot reuse `eval_humaneval.py` directly (which assumes SmolLM2). This
mirrors the same logic but loads the Qwen tokenizer.

Usage:
  CUDA_VISIBLE_DEVICES=0 \\
    /home/knielsen/ml/parallel-ss-dep/.venv/bin/python -u \\
    experiments/eval_distill_4b_humaneval.py \\
    --ckpt checkpoints/dn_4B_distilled_qwen3p6.pt \\
    --max_gen 256 --max_problems 164 \\
    --out logs/distill_pilot_full/dn_4B_humaneval.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import signal
import sys
import time
from contextlib import contextmanager

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_distill_4b_ppl import load_distilled_ckpt


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
    earliest = len(text)
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx >= 0 and idx < earliest:
            earliest = idx
    return text[:earliest]


@torch.no_grad()
def generate_greedy(model, prompt_ids: torch.Tensor, max_gen: int = 256,
                     eos_token_id: int | None = None) -> torch.Tensor:
    """Greedy generation by re-running the full forward pass each step.
    Slow but simple — sufficient for a 164-problem benchmark."""
    out = prompt_ids.clone()
    for _ in range(max_gen):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(out)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--max_T", type=int, default=2048,
                   help="Hard ceiling on prompt+gen tokens. The student is "
                        "trained at T=512 but inference is unrolled.")
    p.add_argument("--out", type=str,
                   default="logs/distill_pilot_full/dn_4B_humaneval.json")
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Loading checkpoint: {args.ckpt}", flush=True)
    model, cfg = load_distilled_ckpt(args.ckpt)
    print(f"  vocab={cfg['vocab_size']}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "QuantTrio/Qwen3.6-35B-A3B-AWQ"),
        trust_remote_code=True,
    )
    print(f"  tokenizer: {cfg.get('tokenizer')}", flush=True)
    eos = tok.eos_token_id

    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    print(f"  {len(ds)} HumanEval problems", flush=True)

    n_passed = 0
    n_total = 0
    failures: list[dict] = []
    per_problem: list[dict] = []
    t_start = time.time()

    for i, problem in enumerate(ds):
        if args.max_problems is not None and i >= args.max_problems:
            break
        prompt = problem["prompt"]
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        # Cap input length so we leave room for max_gen.
        max_prompt = args.max_T - args.max_gen
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                device="cuda").unsqueeze(0)

        gen = generate_greedy(model, prompt_t, max_gen=args.max_gen,
                               eos_token_id=eos)
        gen_only = gen[0, len(prompt_ids):].tolist()
        gen_text = tok.decode(gen_only, skip_special_tokens=True)
        gen_text = _truncate_at_stop(gen_text)
        full_code = prompt + gen_text

        passed = _run_test_in_subprocess(full_code, problem["test"],
                                          problem["entry_point"])
        n_total += 1
        if passed:
            n_passed += 1
        else:
            failures.append({
                "task_id": problem["task_id"],
                "gen_preview": gen_text[:200],
            })
        per_problem.append({
            "task_id": problem["task_id"],
            "passed": passed,
            "gen_chars": len(gen_text),
        })

        if (i + 1) % 10 == 0 or i + 1 == len(ds):
            rate = n_passed / max(1, n_total)
            elapsed = time.time() - t_start
            print(f"  {i + 1:>3} problems: pass@1={rate*100:.1f}% "
                  f"({n_passed}/{n_total})  elapsed={elapsed:.0f}s",
                  flush=True)

    rate = n_passed / max(1, n_total)
    print(f"\npass@1 = {rate:.4f}  ({n_passed}/{n_total})", flush=True)

    out = {
        "ckpt": args.ckpt,
        "pass_rate": rate,
        "n_passed": n_passed,
        "n_total": n_total,
        "max_gen": args.max_gen,
        "max_T": args.max_T,
        "config": {k: cfg.get(k) for k in (
            "d_model", "n_layers", "vocab_size", "params_M", "alpha",
            "kl_weight", "ce_weight", "steps",
        )},
        "failures_first_5": failures[:5],
        "per_problem": per_problem,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
