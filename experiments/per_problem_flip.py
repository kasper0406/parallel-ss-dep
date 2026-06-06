"""Per-problem flip analysis: is the HumanEval thinking lift real or noise?

For each HumanEval problem, generate BOTH no-think (standard) and selective
thinking-ON (latent_think, WM-off, low emit_threshold) on the SAME ckpt, grade
both, and report exactly which problems thinking FLIPS:
  gained = thinking passes, no-think fails  (thinking solved it)
  lost   = thinking fails, no-think passes  (thinking broke it)
net = gained - lost. If gained >> lost on identifiable problems, the lift is real.
"""
import argparse, sys
import torch
sys.path.insert(0, ".")
from datasets import load_dataset
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import (generate, generate_latent_think,
                                        _extract_code_block, _run_test_in_subprocess)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/route_emit_code.pt")
    ap.add_argument("--emit_threshold", type=float, default=0.3)
    ap.add_argument("--max_gen", type=int, default=256)
    ap.add_argument("--max_problems", type=int, default=164)
    args = ap.parse_args()

    model, cfg = build_model_from_ckpt(args.ckpt, force_state_readonly=True)
    model = model.cuda().eval(); model._film_bypass = True
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    eos = tok.eos_token_id
    saved_um = getattr(model, "use_memory", False)
    ds = load_dataset("openai_humaneval", split="test")

    def code_of(out, plen):
        toks = [t for t in out[0, plen:].tolist() if t != tid]
        txt = tok.decode(toks, skip_special_tokens=True)
        c = _extract_code_block(txt)
        return c if c is not None else txt

    gained, lost, both, neither = [], [], [], 0
    nt_pass = th_pass = 0
    for i, p in enumerate(ds):
        if i >= args.max_problems:
            break
        prompt = "# Complete the following Python function.\n" + p["prompt"]
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        plen = ids.shape[1]
        # no-think
        o1, _ = generate(model, ids, max_gen=args.max_gen, temperature=0.0, eos_token_id=eos,
                         min_emit_before_eos=10)
        nt_ok = _run_test_in_subprocess(code_of(o1, plen), p["test"], p["entry_point"])
        # thinking-ON (WM off)
        model.use_memory = False
        try:
            o2, d = generate_latent_think(model, ids, max_gen=args.max_gen, temperature=0.0,
                                          eos_token_id=eos, thinking_token_id=tid,
                                          emit_threshold=args.emit_threshold,
                                          min_emit_before_eos=10)
        finally:
            model.use_memory = saved_um
        th_ok = _run_test_in_subprocess(code_of(o2, plen), p["test"], p["entry_point"])
        nt_pass += int(nt_ok); th_pass += int(th_ok)
        if th_ok and not nt_ok: gained.append(p["task_id"])
        elif nt_ok and not th_ok: lost.append(p["task_id"])
        elif nt_ok and th_ok: both.append(p["task_id"])
        else: neither += 1
        if (i + 1) % 40 == 0:
            print(f"  {i+1}: nothink={nt_pass} think={th_pass} "
                  f"gained={len(gained)} lost={len(lost)}", flush=True)
    print("\n===== PER-PROBLEM FLIP =====", flush=True)
    print(f"no-think pass: {nt_pass}/{min(args.max_problems,len(ds))}", flush=True)
    print(f"thinking pass: {th_pass}/{min(args.max_problems,len(ds))}", flush=True)
    print(f"both pass: {len(both)} | neither: {neither}", flush=True)
    print(f"GAINED (think solved, no-think failed) [{len(gained)}]: {gained}", flush=True)
    print(f"LOST   (think broke, no-think passed)  [{len(lost)}]: {lost}", flush=True)
    print(f"NET = {len(gained)-len(lost):+d}  -> lift is {'REAL' if len(gained)>len(lost) else 'noise/negative'}", flush=True)


if __name__ == "__main__":
    main()
