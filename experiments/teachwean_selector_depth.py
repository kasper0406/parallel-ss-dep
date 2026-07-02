"""
Teach-then-wean op-selector: is learnable random-access op-selection
SUPERVISION-FIXABLE?  (2026-06-25)

CONTEXT (decisive follow-up to experiments/op_selector_depth.py):
  On the heterogeneous multi-table chase `hetero_mt` (recall a DISTINCT random
  permutation per hop; program p_1..p_K given per example; answer
  = g_{p_K} o ... o g_{p_1}(s)), a shallow latent loop (L2 d128, R=K) hits
  K2=1.00 then collapses to chance by K6.  The wall is NOT compute/capacity:
    * an ORACLE that injects the correct op token each latent step SCALES to
      K6=1.00, and
    * a FIXED parameter-free counter-gather (gather program position r at latent
      step r) reaches the oracle ceiling.
  But EVERY *learned* op-selector collapses to chance under final-answer-only
  supervision (learned-absolute-position keys, identity-init adapter,
  RoPE+/-freeze+/-sharp): the attention sharpens onto the WRONG program
  positions (sel_acc ~ chance).  Diagnosed: the selection objective is
  NON-IDENTIFIABLE from final-answer-only supervision.

THE OPEN QUESTION (this file):
  Does PROCESS-SUPERVISION fix it?  At latent step r the correct selector target
  is just "attend to program position r" (one-hot at r) -- teach the attention to
  BE the counter, then REMOVE the teacher and see whether the learned Q/K HOLD
  that alignment or DRIFT back to chance.  This decides whether learnable
  random-access selection is supervision-fixable (-> process labels / RL are the
  lever for the real model) or fundamentally must be a fixed counter / permanent
  teacher.

MECHANISM:
  Reuse op_selector_depth.OpSelectorAttn (COLD init -- NO warm-start; the only
  thing allowed to create alignment is the aux supervision -- the clean test)
  and op_selector_depth.think_forward_opsel (the op-selector latent loop).  Add
  an AUXILIARY op-supervision loss: at each latent step r take the selector's
  attention distribution attn (B,K) and add NLL against the one-hot target
  position r (clamped to K-1).  Mean over steps and batch.
    total loss = answer_CE + w(t) * aux_op_NLL
  where w(t) follows a per-arm curriculum.

ARMS (--variant), L2 d128, all else identical to op_selector_depth:
  baseline               -- w=0 throughout (reproduce the ~0/N final-answer-only
                            floor; control).
  teach_persist          -- w=const 0.5 (soft permanent counter; sanity that the
                            aux target IS the right one -> ceiling).
  teach_then_wean        -- w linearly 1.0 -> 0.0 over first 60% of steps, then 0
                            (THE decisive arm: does the alignment SURVIVE removing
                            the teacher?).
  teach_then_wean_freeze -- same wean schedule, AND after the teach phase ends
                            FREEZE the selection params (k_proj / pos_key_emb /
                            step_q_emb that determine attn; keep out_proj/alpha/
                            v_proj trainable so the injection can still adapt) so
                            the taught alignment is held nearly fixed (attn can
                            still drift slightly via the trainable trunk
                            embedding feeding k_proj — the DECISIVE drift test is
                            the non-freeze teach_then_wean arm).

KEY DIAGNOSTIC: sel_acc = fraction of latent steps where argmax(attn)==r, on a
held-out eval batch at K=6, measured at TWO checkpoints: (a) end of teach phase
(~step 3000), (b) end of training (step 5000).  Drift lives here: high sel_acc at
3000 collapsing by 5000 -> the learned selector is not a stable attractor.

This file does NOT modify the validated primitives.  It imports the task
(make_multitable_chase_batch), the model builder (build), task_meta from
depth_via_iteration, and the OpSelectorAttn + think_forward_opsel op-selector
latent loop from op_selector_depth.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
      experiments/teachwean_selector_depth.py --variant teach_then_wean \
      --n_layers 2 --d_model 128 --steps 5000 \
      --out /tmp/teachwean_cells/teach_then_wean_s0.jsonl --seed 0
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.depth_via_iteration import (
    build, make_multitable_chase_batch, task_meta,
)
from experiments.op_selector_depth import OpSelectorAttn, think_forward_opsel


# ----------------------------------------------------------------------------
# FIXED-VALUE op-selector latent loop (isolate SELECTION from value-delivery)
# ----------------------------------------------------------------------------
# The imported think_forward_opsel injects out_proj(v_proj(attn @ prog_embeds))*a
# — a LEARNABLE value transform on top of the selection.  That re-introduces the
# documented "any learnable transform on the gathered value re-breaks to chance"
# failure (sel_counter_depth.py: counter_gather_adapt), so even PERFECT taught
# selection (sel_acc=1.0) leaves the answer at chance — the value path, not the
# selection, is the failure.  To cleanly test the OPEN question ("does teaching
# selection make the SELECTION learnable+holdable"), this loop injects the RAW
# attention-weighted op embedding (attn @ prog_embeds) — NO v_proj/out_proj/alpha
# — so a one-hot selection at step r injects EXACTLY embed(op-token at program
# position r) = the oracle's injection.  The only learnable thing left in the
# selector is the attention (q/k/pos_key/step_q), which is what we teach-then-wean.
def think_forward_opsel_fixedvalue(model, base_ids, R, thinking_id, opsel,
                                   prog_start, prog_len, return_attn=False):
    B, Lb = base_ids.shape
    if R == 0:
        logits, _h = model(base_ids, return_hidden=True)
        out = logits[:, -1, :]
        return (out, None) if return_attn else out

    base_emb = model.embed(base_ids)                                # (B, Lb, d)
    prog_embeds = base_emb[:, prog_start:prog_start + prog_len, :]  # (B, K, d)
    think_col = torch.full((B, 1), thinking_id, dtype=torch.long,
                           device=base_ids.device)
    ids = torch.cat([base_ids, think_col], dim=1)                  # (B, Lb+1)

    _logits0, h0 = model(base_ids, return_hidden=True)
    z = h0[:, -1:, :]                                               # (B, 1, d)

    attns = []
    logits = None
    for r in range(R):
        _op_learned, attn = opsel(prog_embeds, r)                  # use attn ONLY
        # raw attention-weighted op embedding — fixed value path (== oracle when
        # attn is one-hot at r).  Gradient flows to q/k via attn.
        op_fixed = torch.bmm(attn.unsqueeze(1), prog_embeds).squeeze(1)  # (B,d)
        slot_emb = z + op_fixed.unsqueeze(1)                        # (B,1,d)
        ie = torch.cat([base_emb, slot_emb], dim=1)               # (B, Lb+1, d)
        logits, h = model(ids, inputs_embeds=ie, return_hidden=True)
        z = h[:, -1:, :]
        if return_attn:
            attns.append(attn)
    out = logits[:, -1, :]
    if return_attn:
        return out, torch.stack(attns, dim=1)                      # (B,R,K)
    return out


# Module-level dispatch: train_cell sets this from args.fixed_value so the eval /
# measure / train call sites all route to the same loop.  False => the imported
# (learnable value-path) loop; True => the raw fixed-value loop above.
_FIXED_VALUE = False


def _opsel_fwd(*a, **k):
    fn = think_forward_opsel_fixedvalue if _FIXED_VALUE else think_forward_opsel
    return fn(*a, **k)


# ----------------------------------------------------------------------------
# Aux op-supervision loss + curriculum
# ----------------------------------------------------------------------------
def aux_op_loss(attn_stack: torch.Tensor, eps: float = 1e-9,
                label_noise: float = 0.0, coverage: float = 1.0) -> torch.Tensor:
    """Process-supervision NLL teaching the selector to be the counter.

    attn_stack: (B, R, K) softmax attention distributions over the K program
    positions, one per latent step r=0..R-1.  The correct target at step r is the
    one-hot at program position r, clamped to K-1 (for R>K eval; in training R==K
    so the clamp is inert).  Loss = mean over (batch, steps) of
    -log attn[b, r, target_r].  Minimised (->0) exactly when attn=onehot(r).

    REALISTIC-TRACE knobs (default off => byte-identical to clean counter):
      label_noise p: with per-(b,r) prob p, the target is a random WRONG program
        position (models a teacher CoT that mis-steps / our extraction misaligns).
      coverage c<1: only a per-(b,r) fraction c of steps carry an aux label (the
        rest contribute no gradient — models a partial / shorter teacher trace).
    Uses the global (seeded) RNG for reproducibility given torch.manual_seed.
    """
    B, R, K = attn_stack.shape
    device = attn_stack.device
    targets = torch.arange(R, device=device).clamp(max=K - 1)              # (R,)
    tgt = targets.view(1, R).expand(B, R).clone()                          # (B,R)
    if label_noise > 0.0:
        flip = torch.rand(B, R, device=device) < label_noise              # (B,R)
        rand_pos = torch.randint(0, K, (B, R), device=device)             # (B,R)
        # force WRONG: if the random draw landed on the true target, shift by 1.
        rand_pos = torch.where(rand_pos == tgt, (rand_pos + 1) % K, rand_pos)
        tgt = torch.where(flip, rand_pos, tgt)
    logp = torch.log(attn_stack.clamp_min(eps))                           # (B,R,K)
    sel = logp.gather(2, tgt.unsqueeze(2)).squeeze(2)                     # (B,R)
    nll = -sel                                                            # (B,R)
    if coverage < 1.0:
        mask = (torch.rand(B, R, device=device) < coverage).float()      # (B,R)
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)
    return nll.mean()


def aux_weight(variant: str, step: int, steps: int, teach_frac: float = 0.6,
               persist_w: float = 0.5, wean_w0: float = 1.0) -> float:
    """Per-arm aux-loss weight curriculum w(t).

    baseline:               0 throughout.
    teach_persist:          const persist_w (default 0.5).
    teach_then_wean(_freeze): linear wean_w0 -> 0 over [0, teach_frac*steps),
                              then 0.  Hits wean_w0 at t=0 and 0 at/after the
                              teach phase.
    """
    teach_end = teach_frac * steps
    if variant == "baseline":
        return 0.0
    if variant == "teach_persist":
        return persist_w
    if variant in ("teach_then_wean", "teach_then_wean_freeze"):
        if step >= teach_end:
            return 0.0
        return wean_w0 * (1.0 - step / teach_end)
    raise ValueError(variant)


def selection_params(opsel: OpSelectorAttn):
    """The params that determine the ATTENTION DISTRIBUTION (selection): the
    program-content key proj, the per-position key table, and the per-step query
    table.  v_proj/out_proj/alpha (injection CONTENT + magnitude) are excluded so
    the freeze arm can still adapt what it does with the selected op."""
    return [opsel.k_proj.weight, opsel.pos_key_emb.weight, opsel.step_q_emb.weight]


def freeze_selection_params(opsel: OpSelectorAttn):
    """Stop gradient on the selection params (k_proj/pos_key_emb/step_q_emb).
    With set_to_none zero_grad these get grad=None each step and AdamW skips them;
    out_proj/alpha/v_proj keep updating.  NOTE: this freezes the selector's OWN
    selection params, but attn can still drift slightly via the trunk embedding
    that feeds k_proj (model.embed stays trainable) — so a held sel_acc in the
    freeze arm is evidence of stability, not a tautology.  The DECISIVE drift
    test is the non-freeze teach_then_wean arm."""
    for p in selection_params(opsel):
        p.requires_grad_(False)


# ----------------------------------------------------------------------------
# sel_acc + answer-acc at a checkpoint (held-out batch at a given K, R=K)
# ----------------------------------------------------------------------------
@torch.no_grad()
def measure_sel_and_answer(model, opsel, N, L_ops, thinking_id, K, device,
                           n=512, seed=999):
    """Returns (sel_acc, answer_acc) on a fresh held-out batch at chain length K,
    latent R=K.  sel_acc = fraction of latent steps where argmax(attn)==target_r
    (target_r = min(r, K-1) = r since R==K)."""
    was_training = model.training
    model.eval()
    gg = torch.Generator().manual_seed(seed)
    ids, ans, _chain, _prog, _vocab = make_multitable_chase_batch(
        n, N, K, L_ops, device, gg, homogeneous=False)
    prog_start = L_ops * (2 * N + 1) + 1
    logits, attn_stack = _opsel_fwd(
        model, ids, K, thinking_id, opsel, prog_start, K, return_attn=True)
    R = attn_stack.shape[1]
    targets = torch.arange(R, device=device).clamp(max=K - 1)             # (R,)
    pred_pos = attn_stack.argmax(dim=-1)                                  # (n, R)
    sel_acc = (pred_pos == targets.view(1, R)).float().mean().item()
    answer_acc = (logits[:, :N].argmax(-1) == ans).float().mean().item()
    if was_training:
        model.train()
    return sel_acc, answer_acc


# ----------------------------------------------------------------------------
# Eval: accuracy vs K at R=K (matched-depth diagonal) and at fixed R
# (always via the op-selector latent loop, since every arm carries an opsel)
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_acc_vs_K(model, opsel, N, L_ops, thinking_id, R_or_diag, K_list, device,
                  n_eval=1024, batch=512, seed=12345):
    """R_or_diag: an int R (fixed latent steps) or the string 'diag' for R=K."""
    was_training = model.training
    model.eval()
    prog_start = L_ops * (2 * N + 1) + 1
    out = {}
    for K in K_list:
        gg = torch.Generator().manual_seed(seed + K)
        correct = 0
        done = 0
        R = K if R_or_diag == "diag" else int(R_or_diag)
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans, _chain, _prog, _vocab = make_multitable_chase_batch(
                b, N, K, L_ops, device, gg, homogeneous=False)
            logits = _opsel_fwd(
                model, ids, R, thinking_id, opsel, prog_start, K)
            pred = logits[:, :N].argmax(dim=-1)
            correct += (pred == ans).sum().item()
            done += b
        out[K] = correct / done
    if was_training:
        model.train()
    return out


# ----------------------------------------------------------------------------
# Train: same recipe as op_selector_depth (K-curriculum + uniform-K
# consolidation, final-answer-only) PLUS the aux op-supervision curriculum.
# ----------------------------------------------------------------------------
def train_cell(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Route every op-selector forward (train/eval/measure) through the fixed-value
    # raw-injection loop when requested (isolates SELECTION from value-delivery).
    global _FIXED_VALUE
    _FIXED_VALUE = bool(getattr(args, "fixed_value", False))

    thinking_id, vocab, max_T = task_meta(
        "hetero_mt", args.N, max(args.K, args.eval_K_max), args.L_ops)
    model = build(vocab, thinking_id, args.d_model, args.n_layers,
                  args.n_heads, args.d_head, max_T, device=device)
    prog_start = args.L_ops * (2 * args.N + 1) + 1

    # COLD-init op-selector (NO warm-start; aux supervision is the only thing
    # allowed to create alignment).
    opsel = OpSelectorAttn(
        args.d_model, max_steps=max(args.eval_K_max, args.K) + 2).to(device)
    params = list(model.parameters()) + list(opsel.parameters())
    nparams = sum(p.numel() for p in params)

    tag = (f"hetero_mt/{args.variant}/L{args.n_layers}/d{args.d_model}/"
           f"s{args.seed}")
    teach_end_int = int(args.teach_frac * args.steps)
    print(f"[train] {tag}  N={args.N} K={args.K} L_ops={args.L_ops}  "
          f"params={nparams:,}  thinking_id={thinking_id} max_T={max_T} "
          f"prog_start={prog_start}  teach_end={teach_end_int}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)
    g = torch.Generator().manual_seed(args.seed)
    ramp_steps = 0.6 * args.steps

    sel_acc_teachend = answer_acc_teachend = None
    frozen = False
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Freeze the selection params once the teach phase has ended (freeze arm).
        if (args.variant == "teach_then_wean_freeze" and not frozen
                and step >= teach_end_int):
            freeze_selection_params(opsel)
            frozen = True

        if step <= ramp_steps:
            frac = step / ramp_steps
            K_cur = max(1, min(args.K, int(round(1 + frac * (args.K - 1)))))
        else:
            K_cur = int(torch.randint(1, args.K + 1, (1,), generator=g).item())
        ids, ans, _chain, _prog, _vocab = make_multitable_chase_batch(
            args.batch, args.N, K_cur, args.L_ops, device, g, homogeneous=False)

        final_logits, attn_stack = _opsel_fwd(
            model, ids, K_cur, thinking_id, opsel, prog_start, K_cur,
            return_attn=True)
        loss_ans = F.cross_entropy(final_logits, ans)
        w = aux_weight(args.variant, step, args.steps, teach_frac=args.teach_frac,
                       persist_w=args.teach_weight, wean_w0=args.wean_start_weight)
        loss_aux = aux_op_loss(attn_stack, label_noise=args.label_noise,
                               coverage=args.label_coverage)
        loss = loss_ans + (w * loss_aux if w > 0.0 else 0.0 * loss_aux)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        # Checkpoint A: end of teach phase.  Measure sel_acc + answer acc at K6.
        if step == teach_end_int:
            sel_acc_teachend, answer_acc_teachend = measure_sel_and_answer(
                model, opsel, args.N, args.L_ops, thinking_id, 6, device,
                n=args.sel_n)
            print(f"  [teach-end @ {step}] {tag}  sel_acc(K6)="
                  f"{sel_acc_teachend:.3f}  K6_acc={answer_acc_teachend:.3f}",
                  flush=True)

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = (final_logits[:, :args.N].argmax(-1) == ans
                       ).float().mean().item()
            print(f"  {tag}  step {step:>5}  loss {loss_ans.item():.4f}  "
                  f"aux {loss_aux.item():.4f}  w {w:.3f}  acc {acc:.3f}  "
                  f"K_cur {K_cur}  a={opsel.alpha.item():.2f} "
                  f"o={opsel.out_proj.weight.norm().item():.2f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # In case teach_end coincided exactly with no logging or was 0 (smoke edge).
    if sel_acc_teachend is None:
        sel_acc_teachend, answer_acc_teachend = measure_sel_and_answer(
            model, opsel, args.N, args.L_ops, thinking_id, 6, device, n=args.sel_n)

    # Checkpoint B: final.
    sel_acc_final, answer_acc_final = measure_sel_and_answer(
        model, opsel, args.N, args.L_ops, thinking_id, 6, device, n=args.sel_n)

    # ----- evaluation -----
    K_list = list(range(1, args.eval_K_max + 1))
    res = {"task": "hetero_mt", "variant": args.variant,
           "n_layers": args.n_layers, "d_model": args.d_model,
           "n_heads": args.n_heads, "d_head": args.d_head, "N": args.N,
           "K_train": args.K, "L_ops": args.L_ops, "params": nparams,
           "steps": args.steps, "eval_K_max": args.eval_K_max,
           "seed": args.seed, "K_list": K_list,
           "fixed_value": bool(getattr(args, "fixed_value", False)),
           "label_noise": args.label_noise, "label_coverage": args.label_coverage,
           "teach_frac": args.teach_frac, "teach_end_step": teach_end_int,
           "sel_acc_teachend": sel_acc_teachend,
           "sel_acc_final": sel_acc_final,
           "K6_acc_teachend": answer_acc_teachend,
           "K6_acc_final": answer_acc_final}
    res["acc_ReqK"] = eval_acc_vs_K(model, opsel, args.N, args.L_ops, thinking_id,
                                    "diag", K_list, device, n_eval=args.n_eval)
    for R in [2, 4, 8]:
        res[f"acc_R{R}"] = eval_acc_vs_K(model, opsel, args.N, args.L_ops,
                                         thinking_id, R, K_list, device,
                                         n_eval=args.n_eval)

    print(f"[eval] {tag}  sel_acc teachend={sel_acc_teachend:.3f} "
          f"final={sel_acc_final:.3f}  K6 teachend={answer_acc_teachend:.3f} "
          f"final={answer_acc_final:.3f}")
    for k, v in res.items():
        if k.startswith("acc"):
            print("   " + k + ": " + " ".join(f"K{kk}={vv:.2f}"
                                               for kk, vv in v.items()))

    if args.save:
        sd = {"model": model.state_dict(), "opsel": opsel.state_dict()}
        torch.save({"state_dict": sd, "config": res}, args.save)
        print(f"[saved] {args.save}")
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[appended] {args.out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["baseline", "teach_persist", "teach_then_wean",
                             "teach_then_wean_freeze"],
                    required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--eval_K_max", type=int, default=8)
    ap.add_argument("--L_ops", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=0, help="0 = d_model//d_head")
    ap.add_argument("--d_head", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--sel_n", type=int, default=512,
                    help="held-out batch size for the sel_acc/K6 checkpoints")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teach_frac", type=float, default=0.6,
                    help="fraction of steps over which the teacher is active "
                         "(wean horizon / freeze trigger)")
    ap.add_argument("--teach_weight", type=float, default=0.5,
                    help="const aux weight for teach_persist")
    ap.add_argument("--wean_start_weight", type=float, default=1.0,
                    help="aux weight at t=0 for the wean arms")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--label_noise", type=float, default=0.0,
                    help="per-(b,r) prob the aux op-target is a random WRONG "
                         "program position (models a noisy teacher trace)")
    ap.add_argument("--label_coverage", type=float, default=1.0,
                    help="per-(b,r) fraction of steps that carry an aux label "
                         "(<1 models a partial / shorter teacher trace)")
    ap.add_argument("--fixed_value", action="store_true",
                    help="inject the RAW attention-weighted op embedding "
                         "(no v_proj/out_proj/alpha) so perfect selection == "
                         "oracle; isolates the SELECTION question from the "
                         "known-broken learnable value-delivery path")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.n_heads == 0:
        args.n_heads = max(1, args.d_model // args.d_head)
    if args.smoke:
        args.steps, args.batch, args.log_every = 200, 128, 50
        args.n_eval, args.sel_n = 512, 256
        args.K, args.eval_K_max = 4, 5
    train_cell(args)


if __name__ == "__main__":
    main()
