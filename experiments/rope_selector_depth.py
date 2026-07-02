"""
RoPE op-selector depth: does FIXED relative-position geometry make a LEARNED
per-step op-selector a STABLE attractor (match counter_gather/oracle ~0.79),
where learned-ABSOLUTE-position keys + identity-init adapters DRIFTED to chance
(0.18)?  (2026-06-25)

CONTEXT (established, do not re-litigate; see project_depth_via_iteration.md):
  hetero_mt = recall a DISTINCT random permutation per hop by a data-dependent
  index; non-foldable, so the K hops must execute sequentially at the query
  position (depth).  On a shallow L2 d128 latent loop at R=K:
    - baseline `latent`:               K6 ~ 0.18, 0/4 seeds solve  (no selection)
    - ORACLE (inject true op id /step):K6 ~ 0.79, 3/4 seeds        (ceiling)
    - counter_gather (FIXED, param-free gather of program op at input pos
      prog_start+r):                   K6 ~ 0.79, 3/4 seeds = oracle EXACTLY
  CRITICAL prior finding: ANY *learnable* transform in the SELECTION path —
  learned attention with learned ABSOLUTE-position keys (op_selector_depth.py),
  OR even an IDENTITY-INIT adapter on the fixed gather — DRIFTS OFF the working
  fixed point and collapses to chance (0/4).  ⟹ fixed selection works; learned
  selection drifts.

THE QUESTION (user's hypothesis):
  The failed selector used LEARNED ABSOLUTE-position keys.  RoPE is different —
  its position geometry is FIXED (a rotation by position), only the Q/K
  projections are learned.  Does RoPE's fixed RELATIVE-position geometry make
  the learned selector a STABLE attractor that locks onto the step->program
  alignment (select program position r at latent step r = relative offset 0),
  escaping the drift?

MECHANISM (rope_selector):
  At latent step r, ONE learned base query vector q_base is rotated by position
  r; the program op-token embeddings are projected (learned k_proj) to keys and
  each rotated by its program position p in 0..K-1.  scores = rope(q,r)·rope(k,p)
  depend on (r-p) (RoPE relative-position property) -> selection is
  relative-position retrieval that peaks at p==r once content aligns.  The
  selected op embedding is the attention-weighted sum of the RAW program op-token
  embeddings (values = raw prog_embeds, NO learnable value/out transform), ADDED
  to the carried latent z before the trunk forward -- IDENTICAL injection form to
  oracle/counter_gather (so when attention is one-hot at p==r the injection ==
  oracle's embed(OP_BASE+prog[:,r])).  The ONLY learnable selection-path params
  are q_base and k_proj.  RoPE needs no per-step / per-position embedding table,
  so positions extend naturally past the trained K (length-gen for free).

  NOT a label leak: the selector only reads program op-token positions and
  injects op IDENTITIES (tokens in [OP_BASE, OP_BASE+L_ops)); the answer is a
  node id < N and is never injected.  Same fairness bar as oracle/counter_gather.

ARMS (run here; control/ceiling reuse the sel_counter_depth.py cells):
  - T1 rope_selector BARE:        does fixed-geometry + learned Q/K learn it?
  - T2 rope_selector + ANTI-DRIFT: freeze the selector (q_base, k_proj) after a
    short warmup so a once-aligned selector cannot drift off the fixed point
    (--variant rope_freeze --freeze_after N).  Cheapest of the two suggested
    anti-drifts; the prior drift was of the SELECTION, which freezing directly
    pins (vs an alpha-floor, which would only bound injection magnitude).

Imports the validated/established primitives (does NOT modify them):
  depth_via_iteration.build / make_multitable_chase_batch / task_meta,
  latent_think.think_forward, sel_counter_depth.{think_forward_counter_gather,
  _make_batch, _prog_start, _task_meta_name}.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
      experiments/rope_selector_depth.py --variant rope --n_layers 2 \
      --d_model 128 --steps 5000 --seed 0 --out /tmp/rope_results.jsonl
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

from experiments.depth_via_iteration import build, task_meta
from experiments.latent_think import think_forward
from experiments.sel_counter_depth import (
    think_forward_counter_gather, _make_batch, _prog_start, _task_meta_name,
)


# ----------------------------------------------------------------------------
# RoPE op-selector
# ----------------------------------------------------------------------------
class RopeOpSelector(nn.Module):
    """Per-latent-step op-selector with FIXED relative-position (RoPE) geometry.

    At latent step r: rotate ONE learned base query by position r; project the
    program op-token embeddings to keys (learned k_proj) and rotate each by its
    program position p; softmax(rope(q,r)·rope(k,p)) over p; return the
    attention-weighted sum of the RAW program op-token embeddings (values).

    The ONLY learnable selection params are `q_base` (d_head) and `k_proj`
    (d_model->d_head).  Values are the raw input embeddings (no learnable value /
    out transform) so a one-hot selection at p==r injects EXACTLY oracle's
    embed(OP_BASE+prog[:,r]).
    """

    def __init__(self, d_model: int, d_head: int | None = None,
                 base: float = 10000.0):
        super().__init__()
        self.d_model = int(d_model)
        self.d_head = int(d_head or d_model)
        assert self.d_head % 2 == 0, "RoPE needs an even head dim"
        self.base = float(base)
        self.k_proj = nn.Linear(self.d_model, self.d_head, bias=False)
        self.q_base = nn.Parameter(
            torch.randn(self.d_head) * (self.d_head ** -0.5))
        half = self.d_head // 2
        inv_freq = self.base ** (-torch.arange(0, half).float() / half)  # (half,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _angles(self, pos: torch.Tensor):
        """pos: (P,) -> (cos, sin) each (P, d_head), HF rotate_half convention."""
        pos = pos.float()
        freqs = torch.outer(pos, self.inv_freq)                 # (P, half)
        emb = torch.cat([freqs, freqs], dim=-1)                 # (P, d_head)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return x * cos + self._rotate_half(x) * sin

    def forward(self, prog_embeds: torch.Tensor, step_r: int):
        """prog_embeds: (B, K, d_model) verbatim op-token input embeddings, in
        program order.  step_r: int latent iteration (0-indexed).
        Returns (sel (B, d_model), attn (B, K))."""
        B, K, d = prog_embeds.shape
        device = prog_embeds.device
        keys = self.k_proj(prog_embeds)                          # (B, K, dh)
        cos_k, sin_k = self._angles(torch.arange(K, device=device))   # (K, dh)
        keys = self._rope(keys, cos_k.unsqueeze(0), sin_k.unsqueeze(0))
        cos_q, sin_q = self._angles(
            torch.tensor([step_r], device=device, dtype=torch.float32))  # (1,dh)
        q = self._rope(self.q_base.unsqueeze(0), cos_q, sin_q)   # (1, dh)
        scores = (keys * q.unsqueeze(1)).sum(-1) / math.sqrt(self.d_head)  # (B,K)
        attn = torch.softmax(scores, dim=-1)                    # (B, K)
        sel = torch.bmm(attn.unsqueeze(1), prog_embeds).squeeze(1)   # (B, d_model)
        return sel, attn


# ----------------------------------------------------------------------------
# Latent loop WITH the RoPE op-selector
# ----------------------------------------------------------------------------
def think_forward_rope(model, base_ids, R, thinking_id, selector, prog_start,
                       prog_len, return_steps=False, return_attn=False):
    """Latent ponder for R steps; at step r the RoPE selector adds the selected
    program op embedding to the carried latent z before the trunk forward.

    For R > prog_len the query step is clamped to prog_len-1 (re-apply the last
    program op) for parity with oracle / counter_gather's rr=min(r, prog_len-1);
    raw additive injection (== counter_gather / oracle form, no learned value
    transform)."""
    B, Lb = base_ids.shape
    if R == 0:
        logits, _h = model(base_ids, return_hidden=True)
        out = logits[:, -1, :]
        if return_attn:
            return out, None
        return out.unsqueeze(1) if return_steps else out

    base_emb = model.embed(base_ids)                            # (B, Lb, d)
    prog_embeds = base_emb[:, prog_start:prog_start + prog_len, :]   # (B, K, d)
    think_col = torch.full((B, 1), thinking_id, dtype=torch.long,
                           device=base_ids.device)
    ids = torch.cat([base_ids, think_col], dim=1)              # (B, Lb+1)

    _logits0, h0 = model(base_ids, return_hidden=True)
    z = h0[:, -1:, :]                                           # (B, 1, d)

    step_logits, attns = [], []
    logits = None
    for r in range(R):
        op, attn = selector(prog_embeds, min(r, prog_len - 1))  # (B,d), (B,K)
        slot_emb = z + op.unsqueeze(1)                         # (B,1,d)
        ie = torch.cat([base_emb, slot_emb], dim=1)           # (B, Lb+1, d)
        logits, h = model(ids, inputs_embeds=ie, return_hidden=True)
        z = h[:, -1:, :]
        if return_steps:
            step_logits.append(logits[:, -1, :])
        if return_attn:
            attns.append(attn)
    out = (torch.stack(step_logits, dim=1) if return_steps
           else logits[:, -1, :])
    if return_attn:
        return out, (torch.stack(attns, dim=1) if attns else None)
    return out


# ----------------------------------------------------------------------------
# Forward dispatch (one place so train + eval agree)
# ----------------------------------------------------------------------------
def _forward(variant, model, ids, R, thinking_id, task, N, L_ops, K,
             selector=None, return_steps=False):
    if variant == "latent":
        mode = "none" if R == 0 else "latent"
        return think_forward(model, ids, R, thinking_id, mode=mode,
                             return_steps=return_steps)
    if variant == "counter_gather":
        prog_start = _prog_start(task, N, L_ops)
        return think_forward_counter_gather(
            model, ids, R, thinking_id, prog_start, K, adapter=None,
            return_steps=return_steps)
    if variant in ("rope", "rope_freeze"):
        prog_start = _prog_start(task, N, L_ops)
        return think_forward_rope(model, ids, R, thinking_id, selector,
                                  prog_start, K, return_steps=return_steps)
    raise ValueError(variant)


# ----------------------------------------------------------------------------
# Eval: accuracy vs K at R=K (matched-depth diagonal) and at fixed R
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_acc_vs_K(model, variant, task, N, L_ops, thinking_id, R_or_diag,
                  K_list, device, selector=None, n_eval=1024, batch=512,
                  seed=12345):
    model.eval()
    out = {}
    for K in K_list:
        gg = torch.Generator().manual_seed(seed + K)
        correct = total = done = 0
        R = K if R_or_diag == "diag" else int(R_or_diag)
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans, _chain, _prog = _make_batch(task, b, N, K, L_ops, device, gg)
            logits = _forward(variant, model, ids, R, thinking_id, task, N,
                              L_ops, K, selector=selector)
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

    selector = None
    params = list(model.parameters())
    if args.variant in ("rope", "rope_freeze"):
        selector = RopeOpSelector(args.d_model, d_head=args.sel_d_head,
                                  base=args.rope_base).to(device)
        params = params + list(selector.parameters())

    nparams = sum(p.numel() for p in params)
    tag = f"{args.task}/{args.variant}/L{args.n_layers}/d{args.d_model}/s{args.seed}"
    print(f"[train] {tag}  N={args.N} K={args.K} L_ops={args.L_ops}  "
          f"params={nparams:,}  thinking_id={thinking_id} max_T={max_T} "
          f"prog_start={prog_start} freeze_after={args.freeze_after} "
          f"sel_d_head={args.sel_d_head} rope_base={args.rope_base}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)
    g = torch.Generator().manual_seed(args.seed)
    ramp_steps = 0.6 * args.steps
    frozen = False
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # anti-drift: freeze the selector after a short warmup so a once-aligned
        # selector cannot drift off the working fixed point.
        if (selector is not None and args.freeze_after > 0
                and not frozen and step > args.freeze_after):
            for p in selector.parameters():
                p.requires_grad_(False)
            frozen = True
            print(f"  {tag}  [froze selector at step {step}]", flush=True)

        if step <= ramp_steps:
            frac = step / ramp_steps
            K_cur = max(1, min(args.K, int(round(1 + frac * (args.K - 1)))))
        else:
            K_cur = int(torch.randint(1, args.K + 1, (1,), generator=g).item())
        ids, ans, _chain, _prog = _make_batch(args.task, args.batch, args.N,
                                              K_cur, args.L_ops, device, g)
        final_logits = _forward(args.variant, model, ids, K_cur, thinking_id,
                                args.task, args.N, args.L_ops, K_cur,
                                selector=selector)
        loss = F.cross_entropy(final_logits, ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            extra = ""
            with torch.no_grad():
                acc = (final_logits[:, :args.N].argmax(-1) == ans).float().mean().item()
                if selector is not None:
                    # selection-alignment probe at the current batch: argmax attn==r
                    base_emb = model.embed(ids)
                    pe = base_emb[:, prog_start:prog_start + K_cur, :]
                    hits = tot = 0
                    maxa = 0.0
                    for r in range(K_cur):
                        _sel, attn = selector(pe, r)
                        hits += (attn.argmax(-1) == r).sum().item()
                        tot += attn.shape[0]
                        maxa += attn.max(-1).values.mean().item()
                    sel_acc = hits / max(tot, 1)
                    maxa = maxa / max(K_cur, 1)
                    extra = (f"  sel_acc={sel_acc:.2f} maxattn={maxa:.2f}"
                             f" qn={selector.q_base.norm().item():.2f}"
                             f"{' FROZEN' if frozen else ''}")
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
           "freeze_after": args.freeze_after, "sel_d_head": args.sel_d_head,
           "rope_base": args.rope_base, "seed": args.seed, "K_list": K_list}
    res["acc_ReqK"] = eval_acc_vs_K(model, args.variant, args.task, args.N,
                                    args.L_ops, thinking_id, "diag", K_list,
                                    device, selector=selector, n_eval=args.n_eval)
    for R in [2, 4, 8]:
        res[f"acc_R{R}"] = eval_acc_vs_K(model, args.variant, args.task, args.N,
                                         args.L_ops, thinking_id, R, K_list,
                                         device, selector=selector,
                                         n_eval=args.n_eval)

    # final selection-alignment at K=eval_K_max (drift verdict)
    gg = torch.Generator().manual_seed(999)
    ids, _ans, _c, _p = _make_batch(args.task, 512, args.N, args.eval_K_max,
                                    args.L_ops, device, gg)
    if selector is not None:
        with torch.no_grad():
            base_emb = model.embed(ids)
            pe = base_emb[:, prog_start:prog_start + args.eval_K_max, :]
            hits = tot = 0
            for r in range(args.eval_K_max):
                _s, attn = selector(pe, r)
                hits += (attn.argmax(-1) == r).sum().item()
                tot += attn.shape[0]
            res["final_sel_acc"] = hits / max(tot, 1)

    print(f"[eval] {tag}")
    for k, v in res.items():
        if k.startswith("acc"):
            print("   " + k + ": " + " ".join(f"K{kk}={vv:.2f}"
                                              for kk, vv in v.items()),
                  flush=True)
    if "final_sel_acc" in res:
        print(f"   final_sel_acc(K{args.eval_K_max})={res['final_sel_acc']:.2f}",
              flush=True)

    if args.save:
        sd = {"model": model.state_dict()}
        if selector is not None:
            sd["selector"] = selector.state_dict()
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
                    choices=["rope", "rope_freeze", "latent", "counter_gather"],
                    required=True)
    ap.add_argument("--freeze_after", type=int, default=0,
                    help="freeze selector (q_base,k_proj) after this step (0=never)")
    ap.add_argument("--sel_d_head", type=int, default=0,
                    help="selector head dim (0 => d_model)")
    ap.add_argument("--rope_base", type=float, default=10000.0)
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
    if args.sel_d_head == 0:
        args.sel_d_head = args.d_model
    if args.variant == "rope_freeze" and args.freeze_after == 0:
        args.freeze_after = int(0.4 * args.steps)   # sensible default
    if args.smoke:
        args.steps, args.batch, args.log_every, args.n_eval = 150, 128, 50, 512
        args.K, args.eval_K_max = 4, 5
        if args.variant == "rope_freeze":
            args.freeze_after = 60
    train_cell(args)


if __name__ == "__main__":
    main()
