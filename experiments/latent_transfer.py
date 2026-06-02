"""Transfer latent autonomous thinking onto a CODE model (2026-06-02).

The toy program is done: latent thinking works on the real model, the gate
allocates compute autonomously, and it halts on a recognised (implicit) condition
(see project memory / latent_arith_real.py). This script carries that recipe to a
code-capable base, co-training TWO streams JOINTLY (same batches — sequential
training catastrophically forgets the other skill):

  - REASONING-as-code (pointer-chase etc., depth-labelled): R = depth, autonomous
    gate-HALT supervision. The gate learns to FIRE thinking and stop at the right
    step (`autonomous_halt_loss`).
  - REAL CODE (distill pairs): R=0, gate-EMIT supervision + answer-span CE over the
    solution. The gate learns to NOT think and just write code.

So the single gate learns to ROUTE: think on reasoning prompts, emit on code.

De-risking question this first run answers:
  (1) does code ability survive joint training?  -> HumanEval on the saved ckpt
  (2) does thinking still help on reasoning?      -> autonomous eval here
  (3) does the gate route correctly?              -> HumanEval thinking-on == off

WM is OFF (clean latent feedback, matches the validated toy; HumanEval is short so
WM-recall is ~decorative). Keeping WM for long-context recall needs the pre-memory
feedback fix — a follow-on.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/latent_transfer.py \
      --base checkpoints/sft_v8_combined.pt \
      --reason_prefix data/ptr_train --reason_heldout data/ptr_heldout \
      --rungs 0,1,2,3,4,5,6,7,8 \
      --code_jsonl data/sft_phase_c_combined.jsonl --code_frac 0.5 \
      --steps 4000 --save checkpoints/latent_transfer_v1.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoTokenizer

import experiments.latent_arith_real as lar
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.latent_arith_real import (autonomous_halt_loss, _load_rung,
                                           _run_heldout_eval, _run_autonomous_eval)
from experiments.sft_code import load_distilled_jsonl

CODE_COMMENT = "# Complete the following Python function.\n"


def _tokenize_code(pairs, tok, max_len):
    data = []
    for prob, sol in pairs:
        c = tok.encode(CODE_COMMENT + prob, add_special_tokens=False)
        s = tok.encode(sol, add_special_tokens=False)
        if 4 < len(c) and 0 < len(s) and len(c) + len(s) + 2 <= max_len:
            data.append((c, s))
    return data


def train(args):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = build_model_from_ckpt(
        args.base, force_state_readonly=True,
        force_use_latent_feedback_adapter=True)
    model = model.to(device).train()
    model._film_bypass = True
    if getattr(model, "use_memory", False):
        model.use_memory = False
        print("  [no_memory] WM disabled (clean latent feedback)", flush=True)
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    lar.model_tok = tok                          # the eval helpers read this global
    eos_id = tok.eos_token_id
    print(f"[latent-transfer] base={args.base} thinking_id={thinking_id} "
          f"params={model.num_params():,}", flush=True)

    band = [int(x) for x in args.rungs.split(",") if x.strip()]
    reason_data = {}
    for n in list(band):
        path = pathlib.Path(f"{args.reason_prefix}_n{n}.jsonl")
        if not path.exists():
            print(f"  reason rung {n}: SKIP (missing {path})", flush=True)
            band.remove(n)
            continue
        reason_data[n] = _load_rung(args.reason_prefix, n, tok, args.max_len)
        print(f"  reason rung {n}: {len(reason_data[n])} ex", flush=True)
    print(f"  loading code pairs from {args.code_jsonl} ...", flush=True)
    pairs = load_distilled_jsonl(args.code_jsonl, prefer_full_completion=True,
                                 require_extracted_code=True)
    if args.code_max_pairs:
        pairs = pairs[:args.code_max_pairs]
    code_data = _tokenize_code(pairs, tok, args.code_max_len)
    print(f"  code examples usable: {len(code_data)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed)
    rptr = {n: 0 for n in band}
    rperm = {n: torch.randperm(len(reason_data[n]), generator=g).tolist() for n in band}
    cptr = 0
    cperm = torch.randperm(len(code_data), generator=g).tolist()

    def next_reason(n):
        if rptr[n] >= len(rperm[n]):
            rperm[n] = torch.randperm(len(reason_data[n]), generator=g).tolist(); rptr[n] = 0
        ex = reason_data[n][rperm[n][rptr[n]]]; rptr[n] += 1
        return ex

    def next_code():
        nonlocal cptr, cperm
        if cptr >= len(cperm):
            cperm = torch.randperm(len(code_data), generator=g).tolist(); cptr = 0
        ex = code_data[cperm[cptr]]; cptr += 1
        return ex

    t0 = time.time()
    run_loss = run_code = run_reason = 0.0
    n_code = 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        n = band[int(torch.randint(0, len(band), (1,), generator=g).item())]
        loss_accum = 0.0
        for _ in range(args.accum):
            is_code = (torch.rand(1, generator=g).item() < args.code_frac)
            if is_code:
                c, s = next_code()
                loss = autonomous_halt_loss(model, c, s, eos_id, 0, thinking_id,
                                            device, gate_weight=args.gate_weight)
                run_code += loss.item(); n_code += 1
            else:
                c, s, _a, _i = next_reason(n)
                loss = autonomous_halt_loss(model, c, s, eos_id, n, thinking_id,
                                            device, gate_weight=args.gate_weight)
                run_reason += loss.item()
            (loss / args.accum).backward()
            loss_accum += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        run_loss += loss_accum
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:>5}  loss {run_loss/min(step,args.log_every):.4f}  "
                  f"code% {n_code/(step*args.accum):.2f}  ({time.time()-t0:.0f}s)",
                  flush=True)
            run_loss = 0.0
        if args.eval_every and step % args.eval_every == 0:
            _run_autonomous_eval(model, tok, args, thinking_id, eos_id, device,
                                 band, tag=f" @step{step}")
            if args.save:
                cfg["state_readonly_at_think"] = True
                cfg["use_latent_feedback_adapter"] = True
                cfg["use_memory"] = bool(getattr(model, "use_memory", False))
                torch.save({"state_dict": model.state_dict(), "config": cfg,
                            "step": step}, args.save)

    if args.save:
        cfg["state_readonly_at_think"] = True
        cfg["use_latent_feedback_adapter"] = True
        cfg["use_memory"] = bool(getattr(model, "use_memory", False))
        torch.save({"state_dict": model.state_dict(), "config": cfg, "step": args.steps},
                   args.save)
        print(f"[saved] {args.save}", flush=True)
    _run_heldout_eval(model, tok, args, thinking_id, eos_id, device, band, tag=" FINAL")
    _run_autonomous_eval(model, tok, args, thinking_id, eos_id, device, band, tag=" FINAL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft_v8_combined.pt")
    ap.add_argument("--reason_prefix", default="data/ptr_train")
    ap.add_argument("--reason_heldout", default="data/ptr_heldout")
    ap.add_argument("--heldout_prefix", default="")   # set from reason_heldout
    ap.add_argument("--rungs", default="0,1,2,3,4,5,6,7,8")
    ap.add_argument("--eval_rungs", default="")
    ap.add_argument("--code_jsonl", default="data/sft_phase_c_combined.jsonl")
    ap.add_argument("--code_frac", type=float, default=0.5)
    ap.add_argument("--code_max_pairs", type=int, default=30000)
    ap.add_argument("--code_max_len", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--gate_weight", type=float, default=1.0)
    ap.add_argument("--emit_threshold", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_max_problems", type=int, default=60)
    ap.add_argument("--eval_max_gen", type=int, default=8)
    ap.add_argument("--autonomous_halt", action="store_true", default=True)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="checkpoints/latent_transfer_v1.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.heldout_prefix = args.reason_heldout
    if args.smoke:
        args.steps, args.accum, args.code_max_pairs = 6, 2, 200
        args.eval_every, args.log_every, args.save = 0, 2, ""
        args.eval_max_problems = 12
    train(args)


if __name__ == "__main__":
    main()
