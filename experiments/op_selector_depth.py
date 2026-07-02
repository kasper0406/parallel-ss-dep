"""
Op-selector depth: does PER-STEP op-selection let a SHALLOW latent loop scale
HETEROGENEOUS depth?  (2026-06-25)

CONTEXT (decisive follow-up to experiments/depth_via_iteration.py):
  Latent thinking = f^R: it iterates ONE learned map.  That SOLVES homogeneous
  depth (same op x K) but COLLAPSES on the `hetero_mt` task (recall a DISTINCT
  permutation per hop from a per-example table, program p_1..p_K given per
  example, answer = g_{p_K} o ... o g_{p_1}(s)).  Measured baseline (L2 d128,
  latent R=K): K2=1.00, K3=0.55, K4=0.32, K5+=chance.  Depth/width don't help.

HYPOTHESIS:
  The collapse is because the latent loop has NO per-step op-selection — it
  re-applies one map.  If each latent iteration r can apply a DIFFERENT op by
  attending to the program token p_r (an explicit OP-SELECTOR indexed by the
  iteration step), a SHALLOW model should scale heterogeneous depth — moving the
  collapse knee from K2 out toward K6+.

MECHANISM (op-selector, adapted from model.LineSelectorAttn, 2026-06-03):
  At latent iteration r, a learned per-step query attends (softmax) over the
  PROGRAM token positions p_1..p_K — keyed by per-program-position index
  embeddings + projected verbatim op-token embeddings — and returns the selected
  op's verbatim INPUT embedding, ADDED to the carried latent z before the trunk
  forward.  So R iterations realize a SEQUENCE of distinct ops, not one repeated.
  Zero-init out_proj + alpha=1 => cold start is an EXACT no-op: the op-selector
  latent loop is byte-identical to the validated baseline latent loop at init,
  and the model opts in only via gradient (FiLM-alpha / LineSelectorAttn trick).

This file does NOT modify the validated primitives.  It imports the task
(make_multitable_chase_batch), the model builder (build), task_meta from
depth_via_iteration, and the baseline latent loop (think_forward) from
latent_think.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
      experiments/op_selector_depth.py --variant opsel \
      --n_layers 2 --d_model 128 --steps 5000 --out /tmp/opsel_results.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.depth_via_iteration import (
    build, make_multitable_chase_batch, task_meta,
)
from experiments.latent_think import think_forward


# ----------------------------------------------------------------------------
# OP-SELECTOR (adapted from model.LineSelectorAttn)
# ----------------------------------------------------------------------------
class OpSelectorAttn(nn.Module):
    """Per-latent-step op-selector for the heterogeneous chase.

    At latent iteration `r`, softly select program position `r` from the op-token
    sequence p_1..p_K (a learned per-step query over per-position index keys +
    projected op-token content) and return its verbatim op-token input embedding,
    to be ADDED to the carried latent z.  This gives the latent loop a DIFFERENT
    op identity each step instead of re-applying one map.

    Cold start: `out_proj.weight` is zero-init and `alpha`=1, so the returned
    additive term is EXACTLY zero at init — the op-selector latent loop is then
    byte-identical to the baseline latent loop, and the model opts in only via
    gradient (the LineSelectorAttn / FiLM-alpha zero-init-residual pattern).
    """

    def __init__(self, d_model: int, max_steps: int = 16):
        super().__init__()
        self.d_model = int(d_model)
        self.max_steps = int(max_steps)
        # Per-program-position index key (addressable by absolute step) and the
        # per-latent-step query table.
        self.pos_key_emb = nn.Embedding(self.max_steps, d_model)
        self.step_q_emb = nn.Embedding(self.max_steps, d_model)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)        # cold-start no-op
        self.alpha = nn.Parameter(torch.ones(1))

    def warm_start_alignment(self, scale: float = 4.0):
        """Initialize the per-step query table to the per-position key table and
        boost the position-key norm so q_r·key_p peaks at p==r from step 0 (the
        selector starts by correctly selecting program position r; out_proj still
        zero so it remains a cold-start no-op until gradient grows it).  Gives the
        learned selector the best chance — removes the random-mixture-noise window
        while the model is bootstrapping the chain."""
        with torch.no_grad():
            self.k_proj.weight.zero_()                # ignore content -> keys = pos
            nn.init.orthogonal_(self.pos_key_emb.weight)
            self.pos_key_emb.weight.mul_(scale)
            self.step_q_emb.weight.copy_(self.pos_key_emb.weight)

    def forward(self, prog_embeds: torch.Tensor, step_r: int):
        """prog_embeds: (B, K, d) verbatim op-token input embeddings, in program
        order.  step_r: int latent iteration (0-indexed).  Returns
        (out (B, d), attn (B, K))."""
        B, K, d = prog_embeds.shape
        device = prog_embeds.device
        pos = torch.arange(K, device=device).clamp(max=self.max_steps - 1)
        keys = self.k_proj(prog_embeds) + self.pos_key_emb(pos).unsqueeze(0)  # (B,K,d)
        vals = self.v_proj(prog_embeds)                                       # (B,K,d)
        r = min(int(step_r), self.max_steps - 1)
        q = self.step_q_emb(torch.tensor(r, device=device))                  # (d,)
        scores = torch.matmul(keys, q) / math.sqrt(d)                        # (B,K)
        attn = torch.softmax(scores, dim=-1)                                 # (B,K)
        sel = torch.matmul(attn.unsqueeze(1), vals).squeeze(1)              # (B,d)
        out = self.out_proj(sel) * self.alpha                               # zero at init
        return out, attn


# ----------------------------------------------------------------------------
# Latent loop WITH op-selector
# ----------------------------------------------------------------------------
def think_forward_oracle(model, base_ids, R, thinking_id, prog, N,
                         adapter=None):
    """ORACLE per-step op-injection: at latent step r, ADD the GROUND-TRUTH op
    token embedding embed(OP_BASE + prog[:, r]) to the carried latent z — no
    selection to learn.  This isolates "does per-step op IDENTITY help the
    shallow latent loop scale heterogeneous depth" from "can the model LEARN to
    select the op."  If even the oracle does NOT move the collapse knee, the wall
    is the per-step heterogeneous COMPUTE (the data-dependent 2-key table lookup
    at one position), not op-selection.

    `adapter` (optional nn.Linear d->d): a learned transform of the op embedding
    before injection.  None => raw additive (no params, maximal simplicity)."""
    B, Lb = base_ids.shape
    OP_BASE = N
    if R == 0:
        logits, _h = model(base_ids, return_hidden=True)
        return logits[:, -1, :]
    base_emb = model.embed(base_ids)
    think_col = torch.full((B, 1), thinking_id, dtype=torch.long,
                           device=base_ids.device)
    ids = torch.cat([base_ids, think_col], dim=1)
    _logits0, h0 = model(base_ids, return_hidden=True)
    z = h0[:, -1:, :]
    logits = None
    Kp = prog.shape[1]
    for r in range(R):
        rr = min(r, Kp - 1)
        op_emb = model.embed(OP_BASE + prog[:, rr])          # (B, d)
        if adapter is not None:
            op_emb = adapter(op_emb)
        slot_emb = z + op_emb.unsqueeze(1)
        ie = torch.cat([base_emb, slot_emb], dim=1)
        logits, h = model(ids, inputs_embeds=ie, return_hidden=True)
        z = h[:, -1:, :]
    return logits[:, -1, :]


def think_forward_opsel(model, base_ids, R, thinking_id, opsel,
                        prog_start, prog_len, return_attn=False):
    """Latent ponder for R steps; at step r the op-selector adds the selected
    program op embedding to the carried latent before the trunk forward.

    At cold start (zero out_proj) this is IDENTICAL to latent_think.think_forward
    with mode='latent'.

    prog_start / prog_len: absolute column range of the program tokens in
    base_ids (so the selector reads the verbatim op-token embeddings)."""
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
        op, attn = opsel(prog_embeds, r)                           # (B,d), (B,K)
        slot_emb = z + op.unsqueeze(1)                            # (B,1,d)
        ie = torch.cat([base_emb, slot_emb], dim=1)              # (B, Lb+1, d)
        logits, h = model(ids, inputs_embeds=ie, return_hidden=True)
        z = h[:, -1:, :]
        if return_attn:
            attns.append(attn)
    out = logits[:, -1, :]
    if return_attn:
        return out, torch.stack(attns, dim=1)                     # (B,R,K)
    return out


# ----------------------------------------------------------------------------
# Eval: accuracy vs K at R=K (the matched-depth diagonal) and at fixed R
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_acc_vs_K(model, N, L_ops, thinking_id, R_or_diag, K_list, device,
                  variant, opsel=None, adapter=None, n_eval=1024, batch=512,
                  seed=12345):
    """R_or_diag: an int R (fixed latent steps), or the string 'diag' for R=K."""
    model.eval()
    prog_start = L_ops * (2 * N + 1) + 1
    out = {}
    for K in K_list:
        gg = torch.Generator().manual_seed(seed + K)
        correct = 0
        total = 0
        done = 0
        R = K if R_or_diag == "diag" else int(R_or_diag)
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans, _chain, prog, _vocab = make_multitable_chase_batch(
                b, N, K, L_ops, device, gg, homogeneous=False)
            if variant in ("opsel", "opsel_warm"):
                logits = think_forward_opsel(
                    model, ids, R, thinking_id, opsel, prog_start, K)
            elif variant == "oracle":
                logits = think_forward_oracle(
                    model, ids, R, thinking_id, prog, N, adapter=adapter)
            else:
                mode = "none" if R == 0 else "latent"
                logits = think_forward(model, ids, R, thinking_id, mode=mode)
            pred = logits[:, :N].argmax(dim=-1)
            correct += (pred == ans).sum().item()
            total += b
            done += b
        out[K] = correct / total
    model.train()
    return out


# ----------------------------------------------------------------------------
# Train: same recipe as depth_via_iteration (K-curriculum + uniform-K
# consolidation, final-answer-only), on hetero_mt only.
# ----------------------------------------------------------------------------
def train_cell(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    thinking_id, vocab, max_T = task_meta(
        "hetero_mt", args.N, max(args.K, args.eval_K_max), args.L_ops)
    model = build(vocab, thinking_id, args.d_model, args.n_layers,
                  args.n_heads, args.d_head, max_T, device=device)
    prog_start = args.L_ops * (2 * args.N + 1) + 1

    opsel = None
    adapter = None
    params = list(model.parameters())
    if args.variant in ("opsel", "opsel_warm"):
        opsel = OpSelectorAttn(args.d_model,
                               max_steps=max(args.eval_K_max, args.K) + 2).to(device)
        if args.variant == "opsel_warm":
            opsel.warm_start_alignment()
        params = params + list(opsel.parameters())
    elif args.variant == "oracle" and args.oracle_adapter:
        adapter = nn.Linear(args.d_model, args.d_model, bias=False).to(device)
        nn.init.zeros_(adapter.weight)               # cold-start no-op
        params = params + list(adapter.parameters())

    nparams = sum(p.numel() for p in params)
    tag = f"hetero_mt/{args.variant}/L{args.n_layers}/d{args.d_model}/s{args.seed}"
    print(f"[train] {tag}  N={args.N} K={args.K} L_ops={args.L_ops}  "
          f"params={nparams:,}  thinking_id={thinking_id} max_T={max_T} "
          f"prog_start={prog_start}", flush=True)

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
        ids, ans, _chain, prog, _vocab = make_multitable_chase_batch(
            args.batch, args.N, K_cur, args.L_ops, device, g, homogeneous=False)
        if args.variant in ("opsel", "opsel_warm"):
            final_logits = think_forward_opsel(
                model, ids, K_cur, thinking_id, opsel, prog_start, K_cur)
        elif args.variant == "oracle":
            final_logits = think_forward_oracle(
                model, ids, K_cur, thinking_id, prog, args.N, adapter=adapter)
        else:
            final_logits = think_forward(model, ids, K_cur, thinking_id,
                                         mode="latent")
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
            if opsel is not None:
                extra = (f"  a={opsel.alpha.item():.2f} "
                         f"o={opsel.out_proj.weight.norm().item():.2f}")
            elif adapter is not None:
                extra = f"  adapt={adapter.weight.norm().item():.2f}"
            print(f"  {tag}  step {step:>5}  loss {loss.item():.4f}  "
                  f"acc {acc:.3f}  K_cur {K_cur}{extra}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ----- evaluation -----
    K_list = list(range(1, args.eval_K_max + 1))
    res = {"task": "hetero_mt", "variant": args.variant,
           "n_layers": args.n_layers, "d_model": args.d_model,
           "n_heads": args.n_heads, "d_head": args.d_head, "N": args.N,
           "K_train": args.K, "L_ops": args.L_ops, "params": nparams,
           "steps": args.steps, "eval_K_max": args.eval_K_max,
           "seed": args.seed, "K_list": K_list}
    res["acc_ReqK"] = eval_acc_vs_K(model, args.N, args.L_ops, thinking_id,
                                    "diag", K_list, device, args.variant,
                                    opsel=opsel, adapter=adapter,
                                    n_eval=args.n_eval)
    for R in [2, 4, 8]:
        res[f"acc_R{R}"] = eval_acc_vs_K(model, args.N, args.L_ops, thinking_id,
                                         R, K_list, device, args.variant,
                                         opsel=opsel, adapter=adapter,
                                         n_eval=args.n_eval)

    print(f"[eval] {tag}")
    for k, v in res.items():
        if k.startswith("acc"):
            print("   " + k + ": " + " ".join(f"K{kk}={vv:.2f}"
                                               for kk, vv in v.items()))

    if args.save:
        sd = {"model": model.state_dict()}
        if opsel is not None:
            sd["opsel"] = opsel.state_dict()
        if adapter is not None:
            sd["adapter"] = adapter.state_dict()
        torch.save({"state_dict": sd, "config": res}, args.save)
        print(f"[saved] {args.save}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[appended] {args.out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",
                    choices=["latent", "opsel", "opsel_warm", "oracle"],
                    required=True)
    ap.add_argument("--oracle_adapter", action="store_true",
                    help="oracle variant: learn a Linear transform of the op "
                         "embedding before injection (default raw additive)")
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
