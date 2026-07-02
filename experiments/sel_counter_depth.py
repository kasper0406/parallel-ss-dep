"""
Counter-gather depth: does an EXPLICIT ITERATION COUNTER make per-step op
SELECTION learnable in a shallow latent loop?  (2026-06-25)

CONTEXT (the one open bottleneck from depth_via_iteration + op_selector_depth):
  A shallow latent loop (L2, R=K) CAN EXECUTE a chain of K distinct ops if it is
  TOLD which op each step (the ORACLE injects embed(OP_BASE+prog[:,r]) -> K6=1.00
  on hetero_mt).  But it CANNOT LEARN the per-step op-SELECTION from
  final-answer-only supervision: a learned attention op-selector failed (0/8
  solve K6).  The prior study's verdict: "the wall is LEARNING the per-step
  op-SELECTION (the step->program-position alignment), NOT the compute."

KEY HYPOTHESIS:
  The attention-selector failed because it had to LEARN the step->position
  alignment.  If we instead give the loop an EXPLICIT ITERATION COUNTER r and
  DETERMINISTICALLY gather the program op-token at program-position r FROM THE
  INPUT (positions prog_start..prog_start+K in base_ids), selection becomes a
  FIXED gather with NO alignment to learn.  This is NOT a label leak: the program
  op-tokens are already part of the model's input sequence, and only the op
  IDENTITY (a token in [OP_BASE, OP_BASE+L_ops)) is injected -- never the answer
  (a node id < N).  Same fairness bar as the oracle; the difference is the SOURCE
  (input gather via the counter) vs a ground-truth `prog` side-channel.

  ==> counter_gather is functionally identical to the oracle (it injects the same
  op-identity embedding) but realized as a DEPLOYABLE, end-to-end-trainable
  mechanism that needs no `prog` array.  A zero-init learnable adapter on the
  gathered op embedding keeps a cold-start no-op (byte-identical to baseline
  latent at init) while letting the model learn how to use the gathered op.

This file does NOT modify the validated primitives.  It IMPORTS the task
(make_multitable_chase_batch / make_hetero_chase_batch), the model builder
(build), task_meta from depth_via_iteration, the baseline latent loop
(think_forward) from latent_think, and the oracle loop (think_forward_oracle)
from op_selector_depth.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
      experiments/sel_counter_depth.py --task hetero_mt --variant counter_gather \
      --n_layers 2 --d_model 128 --steps 5000 --seed 0 \
      --out /tmp/selctr_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.depth_via_iteration import (
    build, make_multitable_chase_batch, make_hetero_chase_batch, task_meta,
)
from experiments.latent_think import think_forward
from experiments.op_selector_depth import think_forward_oracle


# ----------------------------------------------------------------------------
# COUNTER-GATHER latent loop (the main bet)
# ----------------------------------------------------------------------------
def think_forward_counter_gather(model, base_ids, R, thinking_id, prog_start,
                                 prog_len, adapter=None, return_steps=False):
    """Explicit-iteration-counter op-gather latent loop.

    At latent step r, an EXPLICIT counter deterministically gathers the program
    op-token at program-position r FROM THE INPUT (base_ids[:, prog_start + r]),
    embeds it, optionally transforms it through `adapter`, and ADDS it to the
    carried latent z before the trunk forward.  Selection is a FIXED gather (no
    alignment to learn), unlike the learned attention op-selector.

    NOT a label leak: base_ids[:, prog_start + r] == OP_BASE + prog[:, r] is the
    program op-token already present in the model's input -- the op IDENTITY (in
    [OP_BASE, OP_BASE+L_ops)), never the answer.  The injected embedding equals
    oracle's embed(OP_BASE+prog[:,r]); the only difference is the SOURCE.

    adapter (optional nn.Linear d->d): zero-init => cold-start byte-identical to
    baseline latent (think_forward mode='latent'); the model opts in via grad.
    None => raw additive == the oracle injection by construction.
    """
    B, Lb = base_ids.shape
    if R == 0:
        logits, _h = model(base_ids, return_hidden=True)
        out = logits[:, -1, :]
        return (out.unsqueeze(1) if return_steps else out)
    base_emb = model.embed(base_ids)                                # (B, Lb, d)
    think_col = torch.full((B, 1), thinking_id, dtype=torch.long,
                           device=base_ids.device)
    ids = torch.cat([base_ids, think_col], dim=1)                   # (B, Lb+1)
    _logits0, h0 = model(base_ids, return_hidden=True)
    z = h0[:, -1:, :]                                               # (B, 1, d)
    step_logits = []
    logits = None
    for r in range(R):
        rr = min(r, prog_len - 1)
        op_token = base_ids[:, prog_start + rr]                     # (B,) op id
        op_emb = model.embed(op_token)                             # (B, d)
        if adapter is not None:
            op_emb = adapter(op_emb)
        slot_emb = z + op_emb.unsqueeze(1)                        # (B, 1, d)
        ie = torch.cat([base_emb, slot_emb], dim=1)
        logits, h = model(ids, inputs_embeds=ie, return_hidden=True)
        z = h[:, -1:, :]
        if return_steps:
            step_logits.append(logits[:, -1, :])
    if return_steps:
        return torch.stack(step_logits, dim=1)                     # (B, R, vocab)
    return logits[:, -1, :]


# ----------------------------------------------------------------------------
# Batch dispatch
# ----------------------------------------------------------------------------
def _make_batch(task, B, N, K, L_ops, device, g):
    """Returns (ids, ans, chain, prog).  prog/chain are None for inorder."""
    if task == "hetero_mt":
        ids, ans, chain, prog, _vocab = make_multitable_chase_batch(
            B, N, K, L_ops, device, g, homogeneous=False)
        return ids, ans, chain, prog
    elif task == "hetero_inorder":
        # Baked-op heterogeneous chain, applied LEFT-TO-RIGHT (s FIRST) => the
        # linear-RNN recurrence folds it in a single pass.  No table-recall, no
        # data-dependent random access.  The "foldable" heterogeneous case.
        ids, ans, chain, prog, _vocab = make_hetero_chase_batch(
            B, N, K, L_ops, device, g, fold=True)
        return ids, ans, chain, prog
    raise ValueError(task)


def _prog_start(task, N, L_ops):
    if task == "hetero_mt":
        return L_ops * (2 * N + 1) + 1     # [tables, QUERY, prog.., s]
    if task == "hetero_inorder":
        return 1                            # [s, prog.., QUERY] (fold=True)
    raise ValueError(task)


def _task_meta_name(task):
    return "hetero_mt" if task == "hetero_mt" else "hetero_fold"


# ----------------------------------------------------------------------------
# Forward dispatch (one place so train + eval agree)
# ----------------------------------------------------------------------------
def _forward(variant, model, ids, R, thinking_id, task, N, L_ops, K,
             opt_modules, return_steps=False):
    """opt_modules: dict possibly holding 'adapter'."""
    if variant == "latent":
        mode = "none" if R == 0 else "latent"
        return think_forward(model, ids, R, thinking_id, mode=mode,
                             return_steps=return_steps)
    if variant == "oracle":
        # oracle only defined for hetero_mt (needs prog); reuse the validated fn.
        # Re-derive prog from input op tokens (identical values) to keep API.
        prog_start = _prog_start(task, N, L_ops)
        prog = ids[:, prog_start:prog_start + K] - N
        return think_forward_oracle(model, ids, R, thinking_id, prog, N,
                                    adapter=opt_modules.get("adapter"))
    if variant in ("counter_gather", "counter_gather_adapt"):
        prog_start = _prog_start(task, N, L_ops)
        return think_forward_counter_gather(
            model, ids, R, thinking_id, prog_start, K,
            adapter=opt_modules.get("adapter"), return_steps=return_steps)
    raise ValueError(variant)


# ----------------------------------------------------------------------------
# Eval: accuracy vs K at R=K (matched-depth diagonal) and at fixed R
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_acc_vs_K(model, variant, task, N, L_ops, thinking_id, R_or_diag,
                  K_list, device, opt_modules, n_eval=1024, batch=512,
                  seed=12345):
    model.eval()
    out = {}
    for K in K_list:
        gg = torch.Generator().manual_seed(seed + K)
        correct = 0
        total = 0
        done = 0
        R = K if R_or_diag == "diag" else int(R_or_diag)
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans, _chain, _prog = _make_batch(task, b, N, K, L_ops, device, gg)
            logits = _forward(variant, model, ids, R, thinking_id, task, N,
                              L_ops, K, opt_modules)
            pred = logits[:, :N].argmax(dim=-1)
            correct += (pred == ans).sum().item()
            total += b
            done += b
        out[K] = correct / total
    model.train()
    return out


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train_cell(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    thinking_id, vocab, max_T = task_meta(
        _task_meta_name(args.task), args.N, max(args.K, args.eval_K_max),
        args.L_ops)
    model = build(vocab, thinking_id, args.d_model, args.n_layers,
                  args.n_heads, args.d_head, max_T, device=device)
    prog_start = _prog_start(args.task, args.N, args.L_ops)

    opt_modules = {}
    params = list(model.parameters())
    use_adapter = (args.variant == "counter_gather_adapt") or \
                  (args.variant == "oracle" and args.oracle_adapter)
    if use_adapter:
        adapter = nn.Linear(args.d_model, args.d_model, bias=False).to(device)
        if args.adapter_init == "zero":
            nn.init.zeros_(adapter.weight)             # cold-start no-op
        elif args.adapter_init == "identity":
            nn.init.eye_(adapter.weight)               # cold-start == raw gather
        else:
            raise ValueError(args.adapter_init)
        opt_modules["adapter"] = adapter
        params = params + list(adapter.parameters())

    nparams = sum(p.numel() for p in params)
    tag = f"{args.task}/{args.variant}/L{args.n_layers}/d{args.d_model}/s{args.seed}"
    print(f"[train] {tag}  N={args.N} K={args.K} L_ops={args.L_ops}  "
          f"params={nparams:,}  thinking_id={thinking_id} max_T={max_T} "
          f"prog_start={prog_start} teach_wean={args.teach_wean}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)
    g = torch.Generator().manual_seed(args.seed)
    ramp_steps = 0.6 * args.steps
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if step <= ramp_steps:
            frac = step / ramp_steps
            K_cur = max(1, min(args.K, int(round(1 + frac * (args.K - 1)))))
        else:
            K_cur = int(torch.randint(1, args.K + 1, (1,), generator=g).item())
        ids, ans, chain, _prog = _make_batch(args.task, args.batch, args.N,
                                             K_cur, args.L_ops, device, g)
        # teach-then-wean: supervise the intermediate latent at step r to decode
        # the true partial result f_{p_1..p_r}(s) = chain[:, r]; anneal w 1->0.
        if args.teach_wean and chain is not None:
            step_logits = _forward(args.variant, model, ids, K_cur, thinking_id,
                                   args.task, args.N, args.L_ops, K_cur,
                                   opt_modules, return_steps=True)  # (B,R,V)
            R = step_logits.shape[1]
            final_logits = step_logits[:, -1, :]
            loss_final = F.cross_entropy(final_logits, ans)
            tgt = chain[:, :R]                                       # (B,R)
            loss_aux = F.cross_entropy(
                step_logits.reshape(-1, step_logits.shape[-1]),
                tgt.reshape(-1))
            w = max(0.0, 1.0 - step / (args.wean_frac * args.steps))
            loss = loss_final + w * loss_aux
        else:
            final_logits = _forward(args.variant, model, ids, K_cur, thinking_id,
                                    args.task, args.N, args.L_ops, K_cur,
                                    opt_modules)
            loss = F.cross_entropy(final_logits, ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = (final_logits[:, :args.N].argmax(-1) == ans).float().mean().item()
            extra = ""
            if "adapter" in opt_modules:
                extra = f"  adapt={opt_modules['adapter'].weight.norm().item():.2f}"
            if args.teach_wean:
                extra += f"  w={max(0.0,1.0-step/(args.wean_frac*args.steps)):.2f}"
            print(f"  {tag}  step {step:>5}  loss {loss.item():.4f}  "
                  f"acc {acc:.3f}  K_cur {K_cur}{extra}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ----- evaluation -----
    K_list = list(range(1, args.eval_K_max + 1))
    res = {"task": args.task, "variant": args.variant,
           "n_layers": args.n_layers, "d_model": args.d_model,
           "n_heads": args.n_heads, "d_head": args.d_head, "N": args.N,
           "K_train": args.K, "L_ops": args.L_ops, "params": nparams,
           "steps": args.steps, "eval_K_max": args.eval_K_max,
           "teach_wean": args.teach_wean, "adapter_init": args.adapter_init,
           "use_adapter": use_adapter, "seed": args.seed, "K_list": K_list}
    res["acc_ReqK"] = eval_acc_vs_K(model, args.variant, args.task, args.N,
                                    args.L_ops, thinking_id, "diag", K_list,
                                    device, opt_modules, n_eval=args.n_eval)
    for R in [2, 4, 8]:
        res[f"acc_R{R}"] = eval_acc_vs_K(model, args.variant, args.task, args.N,
                                         args.L_ops, thinking_id, R, K_list,
                                         device, opt_modules, n_eval=args.n_eval)

    print(f"[eval] {tag}")
    for k, v in res.items():
        if k.startswith("acc"):
            print("   " + k + ": " + " ".join(f"K{kk}={vv:.2f}"
                                              for kk, vv in v.items()),
                  flush=True)

    if args.save:
        sd = {"model": model.state_dict()}
        if "adapter" in opt_modules:
            sd["adapter"] = opt_modules["adapter"].state_dict()
        torch.save({"state_dict": sd, "config": res}, args.save)
        print(f"[saved] {args.save}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[appended] {args.out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["hetero_mt", "hetero_inorder"],
                    default="hetero_mt")
    ap.add_argument("--variant",
                    choices=["latent", "counter_gather", "counter_gather_adapt",
                             "oracle"],
                    required=True)
    ap.add_argument("--oracle_adapter", action="store_true")
    ap.add_argument("--teach_wean", action="store_true",
                    help="aux: supervise intermediate latent to chain[:,r], "
                         "anneal weight 1->0 over --wean_frac of steps")
    ap.add_argument("--wean_frac", type=float, default=0.6)
    ap.add_argument("--adapter_init", choices=["zero", "identity"],
                    default="zero")
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
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.n_heads == 0:
        args.n_heads = max(1, args.d_model // args.d_head)
    if args.smoke:
        args.steps, args.batch, args.log_every, args.n_eval = 150, 128, 50, 512
        args.K, args.eval_K_max = 4, 5
    train_cell(args)


if __name__ == "__main__":
    main()
