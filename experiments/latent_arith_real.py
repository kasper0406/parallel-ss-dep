"""Depth-matched latent-thinking co-train on the REAL 600M code model (2026-06-01).

The decisive experiment for "does latent thinking actually HELP the real model".
The converged diagnosis (THINKING root-cause investigation): latent thinking was
co-trained + measured on GENERAL text where next-token isn't depth-bound, so the
ponder loop correctly learned "I'm not needed" (Δlogp -0.68). On depth-bound text
it was already 2x less harmful (-0.29). The fix: train + measure on a corpus where
the answer is genuinely uncomputable in a single forward, with R = problem depth.

This ports the VALIDATED `latent_arith.py` recipe (depth-matched R, answer-anchored
CE, depth curriculum, state-readonly) onto the real pretrained 600M model:
  - data: `data/arith_pm_train_n{n}.jsonl` (+/- chains, bounded single-int answer,
    SOLE difficulty axis = sequential depth n).
  - loss: `latent_sft.latent_sft_loss` (Coconut hidden-feedback through the learned
    LatentFeedbackAdapter + CE on the answer span). CLEAN feedback (no WM injection
    — hybrid_mem OFF) so the fed-back vector is pure out_norm(h), addressing the
    contamination concern.
  - R = the example's rung (depth-matched). The no-think (R=0) path is NEVER trained,
    so it stays insufficient at deep rungs — that gap IS the demonstration.
  - curriculum: grow the depth frontier over 60% of steps (sampling uniformly up to
    the frontier so shallow rungs aren't forgotten), then full-uniform consolidation.

Success criterion (printed at the end, on a DISJOINT heldout split):
  R=depth accuracy clearly > none accuracy, and the gap grows with depth.

Usage:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/latent_arith_real.py \
      --base checkpoints/pretrain_v9_step25432_tok5000134656.pt \
      --rungs 2,3,4,5,6,7,8 --steps 2000 --accum 8 --lr 3e-5 \
      --save checkpoints/latent_arith_real_v1.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.latent_sft import latent_sft_loss
from experiments.eval_humaneval import generate_latent_think


def latent_perhop_loss(model, comment_ids, inter_ids, R, thinking_id, device):
    """Per-hop-supervised latent loss: force latent step j to emit f^j(s).

    Builds the latent thread exactly like `latent_sft_loss` (Coconut feedback of
    out_norm(h) through the adapter, state-readonly think slots), then does ONE
    forward over [comment, think_1..think_R] and supervises the logits at EACH
    think slot j (position P+j-1) to predict the single-token intermediate
    `inter_ids[j-1]` = f^j(s). The last slot (j=R) predicts the answer, which is
    exactly the position `generate_latent_think` emits from — so eval matches.

    This makes the latent thread provably load-bearing: it can only minimise the
    loss by carrying the running pointer through every step (the failure mode on
    the real model was the thread degrading after ~1-2 hops). `inter_ids` must be
    single tokens (use the m<=10 corpus so each value 0-9 is one token).
    """
    base_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    P = len(comment_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long, device=device)
    for _ in range(R):
        _logits, h = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        z = h[:, -1:, :].to(cur_emb.dtype)
        z = model.apply_latent_feedback_adapter(z).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
    logits = out[0] if isinstance(out, tuple) else out
    # think slot j (1-indexed) is at position P+j-1; its logits predict f^j(s).
    tgt = torch.tensor([inter_ids[:R]], dtype=torch.long, device=device)  # (1, R)
    slot_logits = logits[:, P:P + R, :]                                   # (1, R, V)
    return F.cross_entropy(slot_logits.reshape(-1, slot_logits.shape[-1]),
                           tgt.reshape(-1))


def autonomous_halt_loss(model, comment_ids, sol_ids, eos_id, R, thinking_id,
                         device, gate_weight=1.0):
    """Answer-span CE + gate halt-schedule BCE: teach the model to think R times
    then HALT on its own (handles multi-token answers).

    Gate (P(emit)) is consulted at decision positions P-1 (prompt end), P, ...,
    P+R-1 (R+1 of them); the causal trunk makes the partial-sequence gate at a
    position identical to the full-sequence gate, so we read them all from one
    final forward. Targets: THINK (0) for the first R, EMIT (1) at the last.
    Answer span (sol+eos) is CE-supervised from the emit position onward, exactly
    like `latent_sft_loss`. R is the problem's stated depth, so at inference the
    model reads it and allocates the right number of latent steps with no label."""
    base_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    P = len(comment_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long, device=device)
    for _ in range(R):
        _logits, h = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        z = model.apply_latent_feedback_adapter(h[:, -1:, :]).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    sol_t = torch.tensor([sol_ids + [eos_id]], dtype=torch.long, device=device)
    full_ids = torch.cat([cur_ids, sol_t], dim=1)
    full_emb = torch.cat([cur_emb, model.embed(sol_t)], dim=1)
    out = model(full_ids, inputs_embeds=full_emb, return_hidden=True)
    logits = out[0] if isinstance(out, tuple) else out
    gate_logits = model._last_gate_logits            # (1, T) pre-sigmoid P(emit)
    # Answer-span CE: predict sol[0]..eos from the last think slot (P+R-1) onward.
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:].clone()
    start = P + R - 1
    shift_labels[:, :start] = -100
    ans_loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.shape[-1]),
                               shift_labels.reshape(-1), ignore_index=-100)
    # Gate halt schedule at the R+1 decision positions.
    dec = list(range(P - 1, P + R))
    gl = gate_logits[0, dec]
    tgt = torch.zeros(len(dec), device=device, dtype=gl.dtype)
    tgt[-1] = 1.0                                     # EMIT at the last decision
    gate_loss = F.binary_cross_entropy_with_logits(gl, tgt)
    return ans_loss + gate_weight * gate_loss


def _load_rung(prefix: str, n: int, tok, max_len: int) -> list[tuple]:
    path = f"{prefix}_n{n}.jsonl"
    recs = [json.loads(l) for l in open(path) if l.strip()]
    out = []
    for r in recs:
        pfx = r["prompt"] + "\ndef solve():\n    return "
        c = tok.encode(pfx, add_special_tokens=False)
        s = tok.encode(str(r["answer"]), add_special_tokens=False)
        # First token of each intermediate f^j(s) (single-token for m<=10).
        inter_ids = [tok.encode(str(v), add_special_tokens=False)[0]
                     for v in r.get("intermediates", [])]
        if len(c) + len(s) + n + 2 <= max_len:
            out.append((c, s, int(r["answer"]), inter_ids))
    return out


@torch.no_grad()
def _eval_rung(model, recs, R_eval, thinking_id, eos_id, device,
               max_gen=12, max_problems=150):
    """Greedy none vs R=R_eval exact-match accuracy on heldout records."""
    model.eval()
    n_none = n_r = total = 0
    for c, _s, gold, _inter in recs[:max_problems]:
        prompt_ids = torch.tensor([c], dtype=torch.long, device=device)
        plen = prompt_ids.shape[1]
        total += 1
        for R, is_none in ((0, True), (R_eval, False)):
            # TRUE no-think baseline: emit_threshold=0.0 forces the gate to emit
            # immediately (0 latent steps) regardless of how it was trained. With
            # force_prefix_think=0 alone, a TRAINED gate would think here and the
            # "none" column would no longer be a no-think baseline.
            out, _ = generate_latent_think(
                model, prompt_ids, max_gen=max_gen, temperature=0.0,
                eos_token_id=eos_id, thinking_token_id=thinking_id,
                force_prefix_think=R, emit_threshold=(0.0 if is_none else 0.5))
            new = [t for t in out[0, plen:].tolist() if t != thinking_id]
            text = model_tok.decode(new, skip_special_tokens=True)
            m = re.search(r"-?\d+", text)
            pred = int(m.group()) if m else None
            ok = (pred is not None and pred == gold)
            if is_none and ok:
                n_none += 1
            elif (not is_none) and ok:
                n_r += 1
    model.train()
    if total == 0:
        return dict(total=0, none=0.0, r=0.0)
    return dict(total=total, none=n_none / total, r=n_r / total)


@torch.no_grad()
def _eval_autonomous(model, recs, thinking_id, eos_id, device,
                     emit_threshold=0.5, max_think=16, max_gen=6, max_problems=80):
    """Gate-controlled (AUTONOMOUS) decode: the model decides how many latent
    steps to take. Returns accuracy AND the average #think-steps it chose, so we
    can check it allocates ~the true depth (not too few/many)."""
    model.eval()
    n_correct = total = steps_sum = 0
    for c, _s, gold, _inter in recs[:max_problems]:
        prompt_ids = torch.tensor([c], dtype=torch.long, device=device)
        plen = prompt_ids.shape[1]
        out, diag = generate_latent_think(
            model, prompt_ids, max_gen=max_gen, temperature=0.0,
            eos_token_id=eos_id, thinking_token_id=thinking_id,
            force_prefix_think=0, emit_threshold=emit_threshold,
            max_think_per_step=max_think, total_think_budget=max_think + max_gen)
        new = [t for t in out[0, plen:].tolist() if t != thinking_id]
        text = model_tok.decode(new, skip_special_tokens=True)
        m = re.search(r"-?\d+", text)
        pred = int(m.group()) if m else None
        n_correct += int(pred is not None and pred == gold)
        steps_sum += diag.get("think_total", 0)
        total += 1
    model.train()
    if total == 0:
        return dict(total=0, acc=0.0, avg_steps=0.0)
    return dict(total=total, acc=n_correct / total, avg_steps=steps_sum / total)


model_tok = None  # set in train()


def _run_heldout_eval(model, tok, args, thinking_id, eos_id, device, band, tag=""):
    he_band = [int(x) for x in args.eval_rungs.split(",") if x.strip()] or band
    print(f"\n=== HELDOUT EVAL (none vs R=depth){tag} ===", flush=True)
    print(f"{'n':>3} {'total':>6} {'none':>7} {'R=n':>7} {'lift':>8}", flush=True)
    for n in he_band:
        recs = _load_rung(args.heldout_prefix, n, tok, args.max_len)
        if not recs:
            print(f"{n:>3}  (no heldout)", flush=True)
            continue
        res = _eval_rung(model, recs, n, thinking_id, eos_id, device,
                         max_gen=args.eval_max_gen, max_problems=args.eval_max_problems)
        print(f"{n:>3} {res['total']:>6} {res['none']:>7.3f} {res['r']:>7.3f} "
              f"{res['r']-res['none']:>+8.3f}", flush=True)


def _run_autonomous_eval(model, tok, args, thinking_id, eos_id, device, band, tag=""):
    """Gate decides #steps. Reports accuracy + avg steps chosen vs the true depth."""
    he_band = [int(x) for x in args.eval_rungs.split(",") if x.strip()] or band
    print(f"\n=== AUTONOMOUS EVAL (gate decides #hops){tag} ===", flush=True)
    print(f"{'n':>3} {'total':>6} {'auto_acc':>9} {'avg_steps':>10} "
          f"{'(target n)':>11}", flush=True)
    for n in he_band:
        recs = _load_rung(args.heldout_prefix, n, tok, args.max_len)
        if not recs:
            print(f"{n:>3}  (no heldout)", flush=True)
            continue
        res = _eval_autonomous(model, recs, thinking_id, eos_id, device,
                               emit_threshold=args.emit_threshold,
                               max_think=max(band) + 4,
                               max_gen=args.eval_max_gen,
                               max_problems=args.eval_max_problems)
        print(f"{n:>3} {res['total']:>6} {res['acc']:>9.3f} {res['avg_steps']:>10.2f} "
              f"{n:>11}", flush=True)


def train(args):
    global model_tok
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, cfg = build_model_from_ckpt(
        args.base, force_state_readonly=True,
        force_use_latent_feedback_adapter=True)
    model = model.to(device).train()
    model._film_bypass = True                      # single-forward FiLM (speed + WM bug)
    if args.no_memory and getattr(model, "use_memory", False):
        # CLEAN latent feedback: the model returns `out_norm(h)` AFTER
        # _apply_memory, so with use_memory=True the fed-back latent is
        # contaminated by a WorkingMemory retrieval at EVERY think position —
        # corrupting the sequential thread latent thinking needs (the validated
        # synthetic ran use_memory=False). Disable WM so the latent thread is the
        # pure out_norm(h_raw) the synthetic proved works. WM is a RECALL channel;
        # these short in-context arith problems never need it.
        model.use_memory = False
        print("  [no_memory] WorkingMemory DISABLED — clean latent feedback", flush=True)
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    model_tok = tok
    eos_id = tok.eos_token_id
    has_ad = bool(getattr(model, "use_latent_feedback_adapter", False))
    print(f"[latent-arith-real] base={args.base} thinking_id={thinking_id} "
          f"adapter={has_ad} film={cfg.get('feedback_mode')} "
          f"params={model.num_params():,}", flush=True)

    band = [int(x) for x in args.rungs.split(",") if x.strip()]
    if args.eval_only:
        _run_heldout_eval(model, tok, args, thinking_id, eos_id, device, band,
                          tag=" EVAL-ONLY (none = TRUE no-think)")
        if args.autonomous_halt:
            _run_autonomous_eval(model, tok, args, thinking_id, eos_id, device,
                                 band, tag=" EVAL-ONLY")
        return
    train_data = {n: _load_rung(args.train_prefix, n, tok, args.max_len) for n in band}
    for n in band:
        print(f"  rung {n}: {len(train_data[n])} train examples", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed)
    ptr = {n: 0 for n in band}
    perm = {n: torch.randperm(len(train_data[n]), generator=g).tolist() for n in band}

    def next_example(n):
        if ptr[n] >= len(perm[n]):
            perm[n] = torch.randperm(len(train_data[n]), generator=g).tolist()
            ptr[n] = 0
        ex = train_data[n][perm[n][ptr[n]]]
        ptr[n] += 1
        return ex

    def pick_rung(step):
        ramp_end = 0.0 if args.no_ramp else 0.6 * args.steps
        if step < ramp_end:
            frac = step / ramp_end
            frontier = max(1, int(round(1 + frac * (len(band) - 1))))
            choices = band[:frontier]
        else:
            choices = band
        return choices[int(torch.randint(0, len(choices), (1,), generator=g).item())]

    t0 = time.time()
    running = 0.0
    n_no_think = 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        n = pick_rung(step)
        # A fraction of steps train the R=0 (direct-solve) path so the no-think
        # baseline is WELL-FORMED and FAIRLY TRAINED. Then a residual none-vs-R
        # gap at depth is a genuine reasoning-depth failure of the single forward,
        # not an under-trained-format artifact. The R=n latent path gets the rest.
        if args.autonomous_halt:
            R = n                                   # gate-halt schedule needs full n
            no_think = (n == 0)                     # n=0 rung IS the don't-think case
        else:
            no_think = torch.rand(1, generator=g).item() < args.no_think_frac
            R = 0 if no_think else n                # depth-matched latent
        n_no_think += int(R == 0)
        loss_accum = 0.0
        for _ in range(args.accum):
            c, s, _ans, inter = next_example(n)
            if args.autonomous_halt:
                loss = autonomous_halt_loss(model, c, s, eos_id, R, thinking_id,
                                            device, gate_weight=args.gate_weight)
            elif args.per_hop and R > 0 and len(inter) >= R:
                loss = latent_perhop_loss(model, c, inter, R, thinking_id, device)
            else:
                loss = latent_sft_loss(model, c, s, eos_id, R, thinking_id, device)
            (loss / args.accum).backward()
            loss_accum += loss.item() / args.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        running += loss_accum
        if step % args.log_every == 0 or step == 1:
            ad = model.latent_feedback_adapter
            an = float(ad.alpha.detach().item()) if hasattr(ad, "alpha") else float("nan")
            pn = float(ad.proj.weight.detach().norm().item()) if hasattr(ad, "proj") else float("nan")
            print(f"  step {step:>5}  loss {running/min(step,args.log_every):.4f}  "
                  f"n {n}  R {R}  α {an:+.3f}  proj.norm {pn:.2f}  "
                  f"nt {n_no_think/step:.2f}  ({time.time()-t0:.0f}s)",
                  flush=True)
            running = 0.0
        if args.eval_every and step % args.eval_every == 0:
            _run_heldout_eval(model, tok, args, thinking_id, eos_id, device,
                              band, tag=f" @step{step}")
            if args.autonomous_halt:
                _run_autonomous_eval(model, tok, args, thinking_id, eos_id, device,
                                     band, tag=f" @step{step}")
            if args.save:
                cfg["state_readonly_at_think"] = True
                cfg["use_latent_feedback_adapter"] = True
                torch.save({"state_dict": model.state_dict(), "config": cfg,
                            "step": step}, args.save)

    if args.save:
        cfg["state_readonly_at_think"] = True
        cfg["use_latent_feedback_adapter"] = True
        torch.save({"state_dict": model.state_dict(), "config": cfg,
                    "step": args.steps}, args.save)
        print(f"[saved] {args.save}", flush=True)

    # --- Final depth-matched eval: none vs R=depth on the DISJOINT heldout split.
    _run_heldout_eval(model, tok, args, thinking_id, eos_id, device, band,
                      tag=" FINAL")
    if args.autonomous_halt:
        _run_autonomous_eval(model, tok, args, thinking_id, eos_id, device, band,
                             tag=" FINAL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/pretrain_v9_step25432_tok5000134656.pt")
    ap.add_argument("--train_prefix", default="data/arith_pm_train")
    ap.add_argument("--heldout_prefix", default="data/arith_pm_heldout")
    ap.add_argument("--rungs", default="2,3,4,5,6,7,8")
    ap.add_argument("--eval_rungs", default="",
                    help="rungs to eval (default = train band)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--no_think_frac", type=float, default=0.25,
                    help="fraction of steps trained with R=0 (direct-solve) so "
                         "the no-think baseline is well-formed and fairly trained")
    ap.add_argument("--autonomous_halt", action="store_true",
                    help="train the gate to think the right number of steps then "
                         "HALT on its own (answer CE + gate halt-schedule BCE). "
                         "Use the m<=10 single-token corpus incl. an n=0 rung.")
    ap.add_argument("--gate_weight", type=float, default=1.0,
                    help="weight on the gate halt-schedule BCE (autonomous_halt)")
    ap.add_argument("--emit_threshold", type=float, default=0.5,
                    help="P(emit) threshold for autonomous decode eval")
    ap.add_argument("--per_hop", action="store_true",
                    help="per-hop supervision: force latent step j to emit f^j(s) "
                         "(needs an intermediates field + single-token values, "
                         "i.e. the m<=10 corpus). Makes the latent thread provably "
                         "load-bearing so depth doesn't degrade after ~2 hops")
    ap.add_argument("--eval_only", action="store_true",
                    help="skip training; just run the heldout evals on --base "
                         "(none arm = TRUE no-think via emit_threshold=0)")
    ap.add_argument("--no_ramp", action="store_true",
                    help="skip the depth ramp; sample the full band uniformly "
                         "from step 1 (use when continuing from an already-"
                         "curriculum-trained ckpt — pure consolidation)")
    ap.add_argument("--no_memory", action="store_true",
                    help="disable WorkingMemory so the latent feedback is the "
                         "clean out_norm(h) thread (matches the validated synthetic; "
                         "WM injection at think positions otherwise contaminates it)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--eval_max_gen", type=int, default=12)
    ap.add_argument("--eval_max_problems", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="run the heldout none-vs-R eval every N steps (0=off, "
                         "only at the end)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="checkpoints/latent_arith_real_v1.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.rungs, args.steps, args.accum, args.log_every = "2,3,4", 12, 2, 2
        args.eval_max_problems, args.save = 30, ""
    train(args)


if __name__ == "__main__":
    main()
