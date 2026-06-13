"""WHY don't PKM/WM/latent thinking help on MBPP? Stratify the FAILURES by
bottleneck, because a mechanism can only help the failures it ADDRESSES:

  - latent thinking adds COMPUTE DEPTH  -> helps "right approach, wrong arithmetic/index"
  - WM adds long-range RECALL           -> helps "forgot an earlier binding"
  - PKM adds parametric KNOWLEDGE        -> helps "didn't know the approach/API"

If MBPP failures are dominated by "doesn't produce a valid/plausible approach"
(a base-competence/knowledge gap), then the addressable surface for any
inference-time mechanism is tiny — THAT is why the benefit is marginal.

This generates no-think greedy on N problems, grades with the tier ladder, and
for the failures reports:
  - tier histogram (syntax_error / exec_error / runtime_error / partial / pass)
  - "close" = partial (runs, passes >=1 test, fails some) — the surface thinking
    or memory could plausibly flip
  - "broken" = syntax/exec/0-tests-pass — needs the model to KNOW more, not think

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_failure_bottleneck.py checkpoints/latent_code_adapteronly.pt 120
"""
import sys

import torch

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate
from experiments.latent_rl import grade_clean
import experiments.code_grader as CG


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latent_code_adapteronly.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    m, cfg = build_model_from_ckpt(ckpt, force_state_readonly=True)
    m = m.to(device).eval()
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id
    probs = CG.LOADERS["mbpp_combined"]()[off:off + n]
    comment = "# Complete the following Python function.\n"
    print(f"[bottleneck] ckpt={ckpt} n={len(probs)}", flush=True)

    tiers = {}
    n_pass = n_partial = n_broken = 0
    partial_frac = []     # n_passed/n_tests for partial failures
    for i, prob in enumerate(probs):
        cids = tok.encode(comment + prob.prompt, add_special_tokens=False)
        pt = torch.tensor([cids], device=device)
        with torch.no_grad():
            gen, _ = generate(m, pt, max_gen=200, temperature=0.0,
                              eos_token_id=eos, min_emit_before_eos=10)
        out = gen[0, len(cids):].tolist()
        code = tok.decode([t for t in out if t != tid], skip_special_tokens=True)
        # grade_clean strips prose + aliases the first def; reuse the canonical
        # grader so the tier reflects the real submission.
        lines = code.split("\n")
        import re
        st = 0
        for k, l in enumerate(lines):
            if re.match(r"^(def |import |from |class |@)", l):
                st = k
                break
        body = "\n".join(lines[st:])
        names = re.findall(r"^def (\w+)", body, re.M)
        if names and prob.entry_point not in names:
            body = body + f"\n{prob.entry_point} = {names[0]}\n"
        res = CG.grade(prob, body)
        tiers[res.tier] = tiers.get(res.tier, 0) + 1
        if res.tier == "pass":
            n_pass += 1
        elif res.tier == "partial":
            n_partial += 1
            if res.n_tests:
                partial_frac.append(res.n_passed / res.n_tests)
        else:
            n_broken += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(probs)}] pass={n_pass} partial={n_partial} "
                  f"broken={n_broken}", flush=True)

    nf = len(probs) - n_pass
    print(f"\n=== MBPP no-think tier distribution (n={len(probs)}) ===")
    for t in ("pass", "partial", "runtime_error", "exec_error",
              "syntax_error", "timeout", "grader_error"):
        if t in tiers:
            print(f"  {t:<14} {tiers[t]:>4}")
    print(f"\n=== Failure bottleneck (failures={nf}) ===")
    print(f"  CLOSE   (partial, runs+passes>=1 test) = {n_partial}  "
          f"({n_partial/max(1,nf):.1%} of failures)  <- addressable by think/memory")
    print(f"  BROKEN  (syntax/exec/runtime/0-pass)   = {nf - n_partial}  "
          f"({(nf-n_partial)/max(1,nf):.1%})  <- needs KNOWLEDGE, not thinking")
    if partial_frac:
        import statistics
        print(f"  partial pass-fraction: mean={statistics.fmean(partial_frac):.2f} "
              f"(how close the 'close' ones are)")
    print(f"\nINTERPRETATION: if BROKEN >> CLOSE, MBPP failures are knowledge/"
          f"competence-bound — inference-time thinking/memory has almost no "
          f"surface to help, which is WHY the benefit is marginal. The mechanisms "
          f"help where the bottleneck MATCHES (recall task / arithmetic), not here.")


if __name__ == "__main__":
    main()
