"""Answer-stability fine-tune: make the latent op CONVERGE so over-thinking is safe.

Diagnosis (probe_overstep.py): the latent op is a faithful but NON-convergent
iterated map. At step j it emits f^j(s); at exactly R=n it has the answer (1.00),
but one or two extra steps emit f^{n+1}, f^{n+2}... -> WRONG answer -> a CLIFF to
at/below the no-think baseline. The autonomous gate runs ~22 steps -> ~chance.

Fix tested here: train the answer as a FIXED POINT. Slots 1..R keep their per-hop
targets f^1..f^R (preserve the chain), and slots R+1..R+k are ALL supervised to
gold (=f^R). Once the map reaches depth n it should STOP advancing -> over-stepping
becomes a harmless no-op -> the gate's halt precision no longer matters, and
thinking-on can never fall below no-think (the user's invariant) by construction.

Eval: re-run the R-sweep (does gold now hold for R>n?) + autonomous (acc + steps).
"""
import argparse
import json
import random
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.eval_bracket_structure import build_model_from_ckpt
import experiments.latent_arith_real as L


def stability_loss(model, comment_ids, inter_ids, gold_tok, R, k_extra,
                   thinking_id, device, gate_weight=0.5, emit_after_R=False):
    """Per-hop chain (slots 1..R -> f^1..f^R) + stability (slots R+1..R+k -> gold)
    + gate halt BCE (think for first R decisions, emit at R). When emit_after_R,
    ALSO supervise the gate to EMIT at every decision position beyond R (P+R ..
    P+R+k-1) so it actually HALTS at R and stays halted (bounds avg_steps)."""
    base_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    P = len(comment_ids)
    T = R + k_extra
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long, device=device)
    for _ in range(T):
        out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        h = out[1]
        z = model.apply_latent_feedback_adapter(h[:, -1:, :]).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
    logits = out[0] if isinstance(out, tuple) else out
    # slot j (1-indexed) is at position P+j-1; targets for j=1..T:
    #   j<=R: f^j = inter[j-1]; j>R: gold (fixed point)
    tgt = []
    for j in range(1, T + 1):
        tgt.append(inter_ids[j - 1] if j <= R else gold_tok)
    tgt_t = torch.tensor([tgt], dtype=torch.long, device=device)        # (1, T)
    slot_logits = logits[:, P:P + T, :]                                 # (1, T, V)
    ans_loss = F.cross_entropy(slot_logits.reshape(-1, slot_logits.shape[-1]),
                               tgt_t.reshape(-1))
    # gate halt schedule at R+1 decision positions P-1..P+R-1: THINK x R, EMIT last
    gate_logits = model._last_gate_logits
    dec = list(range(P - 1, P + R))
    gl = gate_logits[0, dec]
    gt = torch.zeros(len(dec), device=device, dtype=gl.dtype)
    gt[-1] = 1.0
    gate_loss = F.binary_cross_entropy_with_logits(gl, gt)
    if emit_after_R:
        # decisions beyond R (positions P+R .. P+R+k-1) must EMIT -> halt at R.
        post = list(range(P + R, P + T))
        if post:
            gpl = gate_logits[0, post]
            gate_loss = gate_loss + F.binary_cross_entropy_with_logits(
                gpl, torch.ones(len(post), device=device, dtype=gpl.dtype))
    return ans_loss + gate_weight * gate_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/latent_trace_lineselect.pt")
    ap.add_argument("--train_prefix", default="data/trace_train")
    ap.add_argument("--heldout_prefix", default="data/trace_heldout")
    ap.add_argument("--save", default="checkpoints/latent_trace_stable.pt")
    ap.add_argument("--rungs", default="2,3,4,5,6")
    ap.add_argument("--k_extra", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--max_problems", type=int, default=150)
    ap.add_argument("--emit_after_R", action="store_true",
                    help="supervise gate to EMIT past R -> actually halt at R (v2)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

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
        print("  [no_memory] WM disabled — clean latent feedback", flush=True)
    model.activation_checkpointing = False     # short threads; avoid Blackwell recompute crash
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    L.model_tok = tok
    eos_id = tok.eos_token_id
    print(f"[stability] base={args.base} thinking_id={thinking_id} "
          f"params={model.num_params():,} k_extra={args.k_extra}", flush=True)

    band = [int(x) for x in args.rungs.split(",") if x.strip()]
    data = {}
    for n in band:
        recs = L._load_rung(args.train_prefix, n, tok, args.max_len)
        data[n] = [r for r in recs if r[3] and len(r[3]) >= n]   # need n intermediates
        print(f"  rung n={n}: {len(data[n])} train recs", flush=True)

    def run_eval(tag):
        print(f"\n##### EVAL {tag} #####", flush=True)
        args.eval_rungs = ""
        args.eval_max_gen = 12
        args.eval_max_problems = args.max_problems
        args.emit_threshold = 0.5
        L._run_heldout_eval(model, tok, args, thinking_id, eos_id, device, band)
        L._run_autonomous_eval(model, tok, args, thinking_id, eos_id, device, band)
        # over-step sweep at fixed n=3,5
        for n in (3, 5):
            recs = L._load_rung(args.heldout_prefix, n, tok, args.max_len)[:args.max_problems]
            if not recs:
                continue
            print(f"  over-step n={n}:", end="", flush=True)
            for R in (n, n + 2, n + 5, n + 10):
                res = L._eval_rung(model, recs, R, thinking_id, eos_id, device,
                                   max_gen=12, max_problems=args.max_problems)
                print(f"  R{R}={res['r']:.3f}", end="", flush=True)
            print(f"  (none={res['none']:.3f})", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    nsteps = 5 if args.smoke else args.steps
    rng = random.Random(0)
    if not args.smoke:
        run_eval("BEFORE (sanity = same cliff as probe_overstep)")
    model.train()
    opt.zero_grad()
    for step in range(1, nsteps + 1):
        loss_acc = 0.0
        for _ in range(args.accum):
            n = rng.choice(band)
            c, s, gold, inter = rng.choice(data[n])
            loss = stability_loss(model, c, inter, s[0], n, args.k_extra,
                                  thinking_id, device, gate_weight=0.5,
                                  emit_after_R=args.emit_after_R) / args.accum
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if step % 50 == 0 or step <= 5:
            print(f"step {step:>4}/{nsteps}  loss={loss_acc:.4f}", flush=True)
        if not args.smoke and step % args.eval_every == 0:
            run_eval(f"step {step}")
            model.train()
    if args.smoke:
        print("SMOKE OK", flush=True)
        return
    run_eval("FINAL")
    torch.save({"state_dict": model.state_dict(), "step": nsteps, "config": cfg}, args.save)
    print(f"saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
