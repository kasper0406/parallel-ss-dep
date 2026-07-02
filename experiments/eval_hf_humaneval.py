"""Fair HumanEval pass@1 for a HuggingFace causal LM (e.g. SmolLM2 base),
graded with the SAME execution harness as experiments/eval_humaneval.py.

Base completion models get the NATIVE HumanEval setup: prompt = the function
signature + docstring verbatim, model greedily completes the body, completion
is truncated at the first natural function boundary (our _STOP_SEQUENCES), then
full_code = prompt + completion is run against the problem's hidden test via the
identical _run_test_in_subprocess we use for our own checkpoints. This is the
apples-to-apples "best deployment of each model" comparison.

SAMPLING PARITY: at temperature>0, `--top_p` defaults to 1.0 (off, samples the
full distribution) to match `experiments/eval_humaneval.py::generate`, which
does not nucleus-filter. Runs from before this flag existed used a hardcoded
top_p=0.95 — cross-model temp pass@k comparisons against those runs are NOT
arm-matched; re-run with the new default (or pass --top_p 0.95 to reproduce
the old behaviour) before comparing.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/eval_hf_humaneval.py \
      --model HuggingFaceTB/SmolLM2-360M --max_new_tokens 256 [--diagnose_json out.json]
"""
import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.eval_humaneval import _run_test_in_subprocess, _truncate_at_stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--min_new_tokens", type=int, default=30,
                    help="suppress EOS for the first N generated tokens — the "
                         "fair analog of our eval's --min_emit_before_eos, "
                         "needed because base completion models greedily emit "
                         "EOS right after the prompt's closing docstring.")
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="nucleus-sampling cutoff at temperature>0 (default 1.0 "
                         "= off, sampling the full distribution, matching "
                         "experiments/eval_humaneval.py::generate so cross-model "
                         "temp pass@k is arm-matched. Runs from before this flag "
                         "existed used top_p=0.95.")
    ap.add_argument("--diagnose_json", type=str, default=None)
    ap.add_argument("--instruct", action="store_true",
                    help="apply the chat template + a 'complete this function' "
                         "instruction and extract the ```python``` block — the "
                         "fair packaged-deployment eval for *-Instruct models.")
    args = ap.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).cuda().eval()

    ds = load_dataset("openai_humaneval", split="test")
    greedy = (args.temperature == 0.0 and args.n_samples == 1)

    n_pass = 0
    n_total = 0
    per_problem = []
    from experiments.eval_humaneval import _extract_code_block
    for i, problem in enumerate(ds):
        prompt = problem["prompt"]
        if args.instruct:
            instr = ("Complete the following Python function. Return the "
                     "complete function in a ```python code block.\n\n" + prompt)
            text = tok.apply_chat_template(
                [{"role": "user", "content": instr}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to("cuda")
        else:
            ids = tok(prompt, return_tensors="pt").to("cuda")
        any_passed = False
        last_completion = ""
        for _ in range(args.n_samples):
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=(0 if args.instruct else args.min_new_tokens),
                    do_sample=not greedy,
                    temperature=(args.temperature if not greedy else None),
                    top_p=(args.top_p if not greedy else None),
                    pad_token_id=tok.pad_token_id,
                )
            gen_ids = out[0, ids["input_ids"].shape[1]:]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
            if args.instruct:
                code = _extract_code_block(gen_text)
                # instruct models emit the WHOLE function (with signature) in the
                # fence; use it directly. Fall back to prompt+raw if no fence.
                full_code = code if code is not None else (prompt + gen_text)
                last_completion = (code or gen_text)
            else:
                completion = _truncate_at_stop(gen_text)
                last_completion = completion
                full_code = prompt + completion
            if _run_test_in_subprocess(full_code, problem["test"],
                                       problem["entry_point"]):
                any_passed = True
                break
        n_total += 1
        n_pass += int(any_passed)
        per_problem.append({"task_id": problem["task_id"],
                            "passed": bool(any_passed),
                            "completion": last_completion[:600]})
        if (i + 1) % 20 == 0:
            print(f"  {i+1} problems: pass@{args.n_samples}={n_pass/n_total:.3f}",
                  flush=True)

    rate = n_pass / max(1, n_total)
    print(f"\n{args.model}: pass@{args.n_samples} = {rate:.3f}  "
          f"({n_pass}/{n_total})  [temp={args.temperature}]", flush=True)
    if args.diagnose_json:
        with open(args.diagnose_json, "w") as f:
            json.dump({"model": args.model, "n_passed": n_pass,
                       "n_total": n_total, "pass_rate": rate,
                       "per_problem": per_problem}, f)
        print(f"[diagnose] dumped to {args.diagnose_json}", flush=True)


if __name__ == "__main__":
    main()
