"""Gate-routing probe (2026-06-02): does the gate THINK on reasoning and EMIT on code?

The mechanistic probe showed forced thinking on code/text collapses to a harmful
fixed point (dlogp ~ -4). That only matters at deployment if the GATE mis-fires
on code. This measures the autonomous gate's think-rate on two prompt classes:
  - code prompts (HumanEval-style)   -> gate should EMIT (think_total ~ 0)
  - reasoning prompts (pointer-chase) -> gate should THINK (~ the depth)
If routing is clean, the model is a working code+thinking system: it codes
(gate emits, no harmful collapse) AND solves reasoning (gate thinks).

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/probe_gate_routing.py \
      --ckpt checkpoints/latent_dnh.pt --n 30
"""
from __future__ import annotations
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate_latent_think
from experiments.sft_code import load_distilled_jsonl

CODE_COMMENT = "# Complete the following Python function.\n"


@torch.no_grad()
def think_rate(model, prompts, thinking_id, eos_id, device, emit_threshold, max_gen=24, max_think=12):
    tot_think, tot_emit, n = 0, 0, 0
    per = []
    for ids in prompts:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        _out, diag = generate_latent_think(
            model, x, max_gen=max_gen, temperature=0.0, eos_token_id=eos_id,
            thinking_token_id=thinking_id, force_prefix_think=0,
            emit_threshold=emit_threshold, max_think_per_step=max_think,
            total_think_budget=max_think + max_gen)
        tot_think += diag.get("think_total", 0)
        tot_emit += diag.get("emit_count", 0)
        per.append(diag.get("think_total", 0))
        n += 1
    return dict(n=n, avg_think=tot_think / max(1, n), avg_emit=tot_emit / max(1, n),
                think_per_first_emit=sum(min(p, 1) for p in per) / max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/latent_dnh.pt")
    ap.add_argument("--code_jsonl", default="data/sft_phase_c_combined.jsonl")
    ap.add_argument("--reason_heldout", default="data/ptrdict_heldout_n5.jsonl")
    ap.add_argument("--emit_threshold", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    device = "cuda"
    model, cfg = build_model_from_ckpt(args.ckpt, force_state_readonly=True,
                                       force_use_latent_feedback_adapter=True)
    model = model.to(device).eval(); model._film_bypass = True
    if getattr(model, "use_memory", False):
        model.use_memory = False
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id

    pairs = load_distilled_jsonl(args.code_jsonl, prefer_full_completion=False,
                                 require_extracted_code=True)[:args.n]
    code_prompts = [tok.encode(CODE_COMMENT + p, add_special_tokens=False) for p, _ in pairs]
    rsn = [json.loads(l) for l in open(args.reason_heldout) if l.strip()][:args.n]
    rsn_prompts = [tok.encode(r["prompt"] + "\ndef solve():\n    return ",
                              add_special_tokens=False) for r in rsn]

    print(f"# ckpt={args.ckpt} emit_threshold={args.emit_threshold}")
    cr = think_rate(model, code_prompts, thinking_id, eos, device, args.emit_threshold)
    rr = think_rate(model, rsn_prompts, thinking_id, eos, device, args.emit_threshold)
    print(f"  CODE      (want ~0 thinks): avg_think={cr['avg_think']:.2f}  "
          f"frac_prompts_that_thought={cr['think_per_first_emit']:.2f}")
    print(f"  REASONING (want > 0 thinks): avg_think={rr['avg_think']:.2f}  "
          f"frac_prompts_that_thought={rr['think_per_first_emit']:.2f}")
    verdict = "ROUTES CLEANLY" if (cr['avg_think'] < 1.0 and rr['avg_think'] > 2.0) \
        else "MIS-ROUTES (gate fires on code or not on reasoning)"
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
