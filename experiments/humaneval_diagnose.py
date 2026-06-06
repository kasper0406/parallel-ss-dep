"""Pre-RL diagnostics on the routed thinking model:
  (1) FAILURE MODES — per-problem grader tier (syntax/exec/runtime/partial/pass)
      + sample diagnoses, for thinking-ON. "Obvious bugs or just hard?"
  (2) THINKING QUANTIFICATION — latent steps/problem, think_rate, correlation with
      pass/fail; plus WM status and PKM α (how much each memory is used).

Grades the EXTRACTED full code (eval_humaneval's working path) against an
empty-prompt Problem so code_grader's prompt+completion concat doesn't duplicate
the def (the bug that zeroed the MBPP run).
"""
import argparse, sys, statistics
import torch
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate, generate_latent_think, _extract_code_block
import experiments.code_grader as CG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/route_emit_code.pt")
    ap.add_argument("--emit_threshold", type=float, default=0.3)
    ap.add_argument("--max_gen", type=int, default=256)
    ap.add_argument("--max_problems", type=int, default=164)
    ap.add_argument("--mode", choices=["think", "nothink"], default="think")
    args = ap.parse_args()

    model, cfg = build_model_from_ckpt(args.ckpt, force_state_readonly=True)
    model = model.cuda().eval(); model._film_bypass = True
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    eos = tok.eos_token_id
    saved_um = getattr(model, "use_memory", False)
    pkm_alpha = None
    for n, p in model.named_parameters():
        if "pkm" in n and "alpha" in n.lower():
            pkm_alpha = float(p.detach().float().mean())
    probs = CG.load_humaneval()[: args.max_problems]

    tiers = {}; thinks = []; thinks_pass = []; thinks_fail = []; examples = {}
    npass = 0
    for i, prob in enumerate(probs):
        prompt = "# Complete the following Python function.\n" + prob.prompt
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        plen = ids.shape[1]
        if args.mode == "think":
            model.use_memory = False
            try:
                out, diag = generate_latent_think(model, ids, max_gen=args.max_gen, temperature=0.0,
                    eos_token_id=eos, thinking_token_id=tid, emit_threshold=args.emit_threshold,
                    min_emit_before_eos=10)
            finally:
                model.use_memory = saved_um
            nthink = diag.get("think_total", 0)
        else:
            out, _ = generate(model, ids, max_gen=args.max_gen, temperature=0.0,
                              eos_token_id=eos, min_emit_before_eos=10)
            nthink = 0
        toks = [t for t in out[0, plen:].tolist() if t != tid]
        txt = tok.decode(toks, skip_special_tokens=True)
        code = _extract_code_block(txt)
        full = code if code is not None else txt
        gp = CG.Problem(task_id=prob.task_id, prompt="", tests=prob.tests, entry_point=prob.entry_point)
        res = CG.grade(gp, full)
        tiers[res.tier] = tiers.get(res.tier, 0) + 1
        thinks.append(nthink)
        (thinks_pass if res.passed else thinks_fail).append(nthink)
        npass += int(res.passed)
        if not res.passed and res.tier not in examples:
            examples[res.tier] = (prob.task_id, (res.error_text or "")[:200], full[:160])
        if (i + 1) % 40 == 0:
            print(f"  {i+1}: pass={npass} tiers={tiers}", flush=True)

    n = len(probs)
    print(f"\n===== DIAGNOSE mode={args.mode} ckpt={args.ckpt} thr={args.emit_threshold} =====", flush=True)
    print(f"pass = {npass}/{n}", flush=True)
    print(f"FAILURE-MODE TIERS: {dict(sorted(tiers.items(), key=lambda x:-x[1]))}", flush=True)
    if thinks:
        print(f"THINKING steps/problem: mean={statistics.mean(thinks):.2f} "
              f"median={statistics.median(thinks)} max={max(thinks)} total={sum(thinks)}", flush=True)
    if thinks_pass and thinks_fail:
        print(f"  steps on PASS: mean={statistics.mean(thinks_pass):.2f} | "
              f"on FAIL: mean={statistics.mean(thinks_fail):.2f}", flush=True)
    print(f"WM during thinking: {'ON' if (args.mode=='think' and saved_um and False) else 'OFF (use_memory toggled off in latent_think)' if args.mode=='think' else ('ON' if saved_um else 'OFF')}", flush=True)
    print(f"PKM mean alpha (how much PKM contributes): {pkm_alpha}", flush=True)
    print("\nEXAMPLE FAILURES per tier (task, diagnosis, code[:160]):", flush=True)
    for t, (tid_, err, cd) in examples.items():
        print(f"  [{t}] {tid_}: {err!r}\n      code: {cd!r}", flush=True)


if __name__ == "__main__":
    main()
