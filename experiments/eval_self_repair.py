"""Real agentic measurement: does ONE execution-feedback repair turn rescue
failures the model can't solve one-shot?

This is the agentic workflow (write -> run tests -> read error -> fix), not an
artificial probe. It also tests the knowledge-bound finding: error feedback that
reveals INPUT->EXPECTED pairs (the failing assert) is genuinely new information
the model didn't have one-shot — it can rescue formula/recall errors and some
"wrong output shape" errors even when the model is otherwise knowledge-limited.

Per problem:
  turn 0: generate one-shot, grade.
  turn 1 (only if turn 0 failed): build_repair_prompt(desc, failed_code,
          error_text), generate, grade.
Report: pass@turn0, pass@(turn0 or turn1), and which problems repair RESCUED.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/eval_self_repair.py checkpoints/latent_code_adapteronly.pt 80
"""
import re
import sys

import torch

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate
from experiments.iterative_repair import build_repair_prompt
import experiments.code_grader as CG


def _clean(code, entry_point):
    lines = code.split("\n"); st = 0
    for k, l in enumerate(lines):
        if re.match(r"^(def |import |from |class |@)", l):
            st = k; break
    body = "\n".join(lines[st:])
    names = re.findall(r"^def (\w+)", body, re.M)
    if names and entry_point not in names:
        body = body + f"\n{entry_point} = {names[0]}\n"
    return body


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latent_code_adapteronly.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    m, cfg = build_model_from_ckpt(ckpt, force_state_readonly=True)
    m = m.to(device).eval()
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id
    probs = CG.LOADERS["mbpp_combined"]()[:n]
    comment = "# Complete the following Python function.\n"
    print(f"[self-repair] ckpt={ckpt} n={len(probs)}", flush=True)

    def gen_code(prompt_text, entry_point):
        ids = tok.encode(prompt_text, add_special_tokens=False)
        with torch.no_grad():
            g, _ = generate(m, torch.tensor([ids], device=device), max_gen=200,
                            temperature=0.0, eos_token_id=eos, min_emit_before_eos=10)
        out = [t for t in g[0, len(ids):].tolist() if t != tid]
        return _clean(tok.decode(out, skip_special_tokens=True), entry_point)

    p0 = p1 = 0
    rescued, broke = [], []
    for i, prob in enumerate(probs):
        body0 = gen_code(comment + prob.prompt, prob.entry_point)
        r0 = CG.grade(prob, body0)
        if r0.passed:
            p0 += 1; p1 += 1
            continue
        # repair turn with execution feedback
        rp = build_repair_prompt(prob.prompt, body0, r0.error_text or r0.error or "")
        body1 = gen_code(rp, prob.entry_point)
        r1 = CG.grade(prob, body1)
        if r1.passed:
            p1 += 1; rescued.append(prob.task_id)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(probs)}] pass@0={p0} pass@repair={p1} "
                  f"rescued={len(rescued)}", flush=True)

    print(f"\n=== Self-repair (1 feedback turn) on MBPP (n={len(probs)}) ===")
    print(f"  pass @ turn0 (one-shot)      = {p0}")
    print(f"  pass @ turn0|turn1 (repair)  = {p1}   (+{p1 - p0})")
    print(f"  RESCUED by repair: {rescued}")
    print(f"\nVERDICT: +N rescued = real agentic gain from execution feedback "
          f"(feedback supplies INPUT->EXPECTED the model lacked one-shot). ~0 "
          f"rescued = even feedback can't cross the knowledge gap -> only "
          f"knowledge injection (distill/scale) moves the headline.")


if __name__ == "__main__":
    main()
