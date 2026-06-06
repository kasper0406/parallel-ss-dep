"""
Latent-thinking SFT on the real 287M code model (2026-05-28).

Co-trains the validated latent-ponder mechanism (`THINKING_LATENT_2026_05_28.md`)
INTO the pretrained code model, so it becomes load-bearing rather than inert
(the documented failure mode for anything bolted on post-pretrain).

Layout per example:  [comment][THINK x R][solution][eos]
  - The R THINK slots run state-readonly (DeltaNet β=0); their input
    embedding is the model's OWN hidden fed back (Coconut-style), built by a
    sequential R-step loop (identical to `generate_latent_think`).
  - The last THINK slot predicts the first solution token, so the solution
    generation is conditioned on the final refined latent.
  - CE loss on the solution span (+eos) only.
  - Depth curriculum: ramp R 1->max_R over 60% of steps, then a consolidation
    phase sampling R uniformly (so it retains shallow as well as deep budgets).

batch=1 + grad-accum keeps the sequential think-burst alignment trivially
correct (no mid-sequence padding). `_film_bypass` is set for speed (single-
forward FiLM, the documented deployment approximation).

Usage:
  PYTHONPATH=. .venv/bin/python experiments/latent_sft.py \
    --base checkpoints/sft_phase_c_combined.pt \
    --data data/sft_phase_c_combined.jsonl \
    --steps 1500 --max_R 4 --save checkpoints/latent_sft_v1.pt
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.sft_code import load_distilled_jsonl


def latent_sft_loss(model, comment_ids, sol_ids, eos_id, R, thinking_id, device):
    """One example. Returns scalar CE loss on the solution span after an
    R-step latent ponder burst. Mirrors generate_latent_think's feedback."""
    base_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long, device=device)
    P = len(comment_ids)
    hybrid = getattr(model, "_hybrid_mem", False) and hasattr(model, "memory")
    for _ in range(R):
        _logits, h = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        z = h[:, -1:, :].to(cur_emb.dtype)          # hidden carries the thread
        # Learned input adapter: map the fed-back out_norm hidden into the
        # input-embedding manifold (identity when absent/untrained → matches the
        # prior behaviour). Mirrors thinking.py + eval_humaneval (adapter
        # precedes any WM augmentation).
        z = model.apply_latent_feedback_adapter(z).to(cur_emb.dtype)
        if hybrid:
            # Unified mechanism: augment the hidden-feedback thread with a
            # learned-α WM retrieval (the model pulls in new info as it thinks).
            inj = model.memory._last_injection_grad[:, -1:, :].to(cur_emb.dtype)
            z = z + model.mem_alpha * inj
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)    # think slot input
    sol_t = torch.tensor([sol_ids + [eos_id]], dtype=torch.long, device=device)
    full_ids = torch.cat([cur_ids, sol_t], dim=1)
    full_emb = torch.cat([cur_emb, model.embed(sol_t)], dim=1)
    out = model(full_ids, inputs_embeds=full_emb)
    logits = out[0] if isinstance(out, tuple) else out
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:].clone()
    # loss only from the last-think-slot position onward (predicts sol[0]..eos)
    start = P + R - 1
    shift_labels[:, :start] = -100
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), ignore_index=-100)


def train(args):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, cfg = build_model_from_ckpt(args.base, force_state_readonly=True)
    model = model.to(device).train()
    model._film_bypass = True                  # single-forward FiLM (D11: also
                                               # avoids the FiLM×WM multipass bug)
    if args.hybrid_mem and getattr(model, "use_memory", False):
        # Unified thinking-reads-memory: learned, no-WD retrieval-mix α (D8).
        model._hybrid_mem = True
        model.mem_alpha = torch.nn.Parameter(
            torch.tensor(float(args.alpha_init), device=device))
        print(f"  HYBRID memory thinking ON (α init {args.alpha_init})")
    thinking_id = cfg.get("thinking_token_id")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos_id = tok.eos_token_id
    print(f"[latent-sft] base={args.base} thinking_id={thinking_id} "
          f"film={cfg.get('feedback_mode')} max_R={args.max_R} "
          f"params={model.num_params():,}")

    # prefer_full_completion=True keeps the CoT-prose + ```python``` fence
    # format the baseline ckpt emits, so HumanEval grading (--extract_code_block)
    # is matched and the comparison isn't confounded by an output-format shift.
    pairs = load_distilled_jsonl(args.data,
                                 prefer_full_completion=not args.code_only,
                                 require_extracted_code=True,
                                 keep_only_passing=args.keep_only_passing)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    comment_line = "# Complete the following Python function.\n"
    # Pre-tokenize (truncate long ones to keep the latent loop cheap).
    print(f"  tokenizing {len(pairs)} pairs...", flush=True)
    data = []
    for k, (prob, sol) in enumerate(pairs):
        c = tok.encode(comment_line + prob, add_special_tokens=False)
        s = tok.encode(sol, add_special_tokens=False)
        if len(c) + len(s) + args.max_R + 2 <= args.max_len:
            data.append((c, s))
        if (k + 1) % 5000 == 0:
            print(f"    tokenized {k+1}/{len(pairs)}  (kept {len(data)})",
                  flush=True)
    print(f"  usable examples: {len(data)}", flush=True)

    if getattr(args, "freeze_trunk", False):
        # CLEAN INVARIANT TEST: freeze everything except the latent thinking
        # params (adapter + gate + retrieval-α). Then the R=0/no-think path is
        # BIT-IDENTICAL to the base ckpt (adapter unused at emit positions), so
        # no-think competence + output style are preserved by construction, and
        # thinking-ON isolates whether the code-trained latent op helps.
        keep = ("latent_feedback_adapter", "gate_head", "retrieval_input_alpha")
        n_tr = 0
        for name, p in model.named_parameters():
            train_it = any(k in name for k in keep)
            p.requires_grad = train_it
            n_tr += p.numel() if train_it else 0
        print(f"  [freeze_trunk] training ONLY latent params: {n_tr:,} "
              f"({100*n_tr/sum(p.numel() for p in model.parameters()):.2f}%)",
              flush=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = list(model.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(data), generator=g).tolist()
    ptr = 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    running = 0.0
    for step in range(1, args.steps + 1):
        # A fraction of steps train with NO think burst (R=0) so the model
        # retains direct-solving competence; the rest use the latent burst so
        # thinking adds on top (fixes the "no-think collapse to 0/164").
        if torch.rand(1, generator=g).item() < args.no_think_frac:
            R = 0
        elif step < 0.6 * args.steps:
            frac = step / (0.6 * args.steps)
            R = max(1, min(args.max_R, int(round(1 + frac * (args.max_R - 1)))))
        else:
            R = int(torch.randint(1, args.max_R + 1, (1,), generator=g).item())
        loss_accum = 0.0
        for _ in range(args.accum):
            if ptr >= len(data):
                perm = torch.randperm(len(data), generator=g).tolist()
                ptr = 0
            c, s = data[perm[ptr]]
            ptr += 1
            loss = latent_sft_loss(model, c, s, eos_id, R, thinking_id, device)
            (loss / args.accum).backward()
            loss_accum += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        running += loss_accum
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:>5}  loss {running/min(step,args.log_every):.4f}  "
                  f"R {R}  ({time.time()-t0:.0f}s)")
            running = 0.0
    if args.save:
        cfg["state_readonly_at_think"] = True   # so reloads auto-enable the hook
        torch.save({"state_dict": model.state_dict(), "config": cfg,
                    "step": args.steps}, args.save)
        print(f"[saved] {args.save}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft_phase_c_combined.pt")
    ap.add_argument("--data", default="data/sft_phase_c_combined.jsonl")
    ap.add_argument("--hybrid_mem", action="store_true",
                    help="unified thinking-reads-memory: think input = hidden "
                         "+ learned-α WM retrieval (D8/D11)")
    ap.add_argument("--alpha_init", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_R", type=int, default=4)
    ap.add_argument("--no_think_frac", type=float, default=0.5,
                    help="fraction of steps trained with R=0 (no think burst) "
                         "so direct-solving competence is retained")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--code_only", action="store_true",
                    help="train on extracted_code (no CoT/fence) instead of "
                         "the full completion; changes output format")
    ap.add_argument("--max_pairs", type=int, default=20000)
    ap.add_argument("--freeze_trunk", action="store_true",
                    help="train ONLY the latent params (adapter+gate+α); freeze "
                         "the trunk so no-think == base by construction (clean "
                         "invariant test, no competence/format degradation)")
    ap.add_argument("--keep_only_passing", action="store_true")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="checkpoints/latent_sft_v1.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.accum, args.max_pairs, args.log_every = 20, 2, 200, 5
        args.save = ""
    train(args)


if __name__ == "__main__":
    main()
