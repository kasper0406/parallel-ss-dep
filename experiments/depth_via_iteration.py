"""
Depth-via-iteration: can a SHALLOW trunk + latent thinking (R feedback steps)
simulate the DEPTH of a deeper trunk?  (2026-06-25)

HYPOTHESIS
  Latent thinking iterates ONE learned function (effectively f^R at the think
  slot).  Iterating a single function can replicate depth only for HOMOGENEOUS
  computation (the same operation applied K times).  HETEROGENEOUS computation
  — a *stack of distinct per-step operations* — cannot be produced by f^R; it
  needs genuine depth (distinct per-step transforms).  So:
    (a) HOMOGENEOUS: shallow + R=K  ~=  deep + R=0     (predicted: yes)
    (b) HETEROGENEOUS: shallow + R   <<  deep + R=0     (predicted: shallow fails)

TWO SYNTHETIC TASKS (both = K data-dependent sequential hops at ONE position,
the validated non-foldable pointer-chase structure; the ONLY difference is
homogeneous-vs-heterogeneous):

  HOMOGENEOUS  (= the validated pointer-chase, reused from latent_think):
    One random permutation f on N nodes, presented as shuffled (i, f(i)) pairs,
    then [QUERY, s].  answer = f^K(s).  Same op f at every hop.

  HETEROGENEOUS (op-program chase, new here):
    A BAKED library of L_ops fixed permutations g_0..g_{L_ops-1} on N nodes
    (the SAME across all examples — learned into weights, like a CPU's opcode
    set).  Each example draws a PROGRAM p_1..p_K of op-ids and a start s:
       sequence = [p_1, ..., p_K, QUERY, s]   (s LAST -> non-foldable)
       answer   = g_{p_K}( ... g_{p_2}( g_{p_1}(s) ) ).
    A DIFFERENT op each hop, specified per example.  CAVEAT (empirically
    measured, see report): because the K ops occupy K SEQUENCE POSITIONS and the
    op-composite is independent of s, the linear-RNN recurrence FOLDS the
    composite along the program positions BEFORE reading s — so this task is in
    fact POSITIONALLY FOLDABLE (a single forward, even L=1, solves it up to the
    trained K).  It therefore does NOT require depth-at-position and is only used
    here to demonstrate the folding regime, NOT as the heterogeneity test.
    'hetero_fold' (s FIRST) is the trivially-foldable positive control proving
    the ops + program-reading are learnable.

  The TRUE heterogeneity test is `hetero_mt` (multi-table RECALL program):
  random-per-example tables (non-memorizable) + data-dependent per-hop lookups
  + s LAST make it NON-foldable, so the K distinct ops must execute sequentially
  AT THE QUERY POSITION = depth.  `homo` (pointer-chase) is its homogeneous
  twin; `homo_mt` is the recall+context-matched homogeneous control.
  Final-answer-only supervision (no per-hop labels), depth curriculum +
  uniform-K consolidation, matching the validated latent recipe.

fp32, single GPU.  Imports the validated primitives from latent_think /
model.py; does NOT modify them.

Run ONE cell (model x task x variant) and append a JSON result:
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
    experiments/depth_via_iteration.py \
    --task hetero --variant latent --n_layers 2 --d_model 128 \
    --out /tmp/depthiter_results.jsonl
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

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM
from experiments.latent_think import make_pointer_chase_batch, think_forward


# ----------------------------------------------------------------------------
# HETEROGENEOUS task: multi-table pointer-chase with a per-example op program
# ----------------------------------------------------------------------------
def hetero_layout(N: int, L_ops: int):
    """Token id assignments for the heterogeneous task."""
    OP_BASE = N                       # op tokens N .. N+L_ops-1 (table tag + program ref)
    QUERY = N + L_ops
    THINK = N + L_ops + 1
    PAD = N + L_ops + 2
    vocab = N + L_ops + 3
    return OP_BASE, QUERY, THINK, PAD, vocab


def make_multitable_chase_batch(B: int, N: int, K: int, L_ops: int,
                                device="cuda", generator: torch.Generator | None = None,
                                homogeneous: bool = False):
    """TRUE heterogeneous depth-at-position task (non-foldable, non-memorizable).

    L_ops permutations f_0..f_{L_ops-1} RANDOM per example, each presented as a
    tagged table [OP_j, (i,f_j(i))*N].  Then [QUERY, p_1..p_K, s].
    answer = f_{p_K}( ... f_{p_1}(s) ).

    Unlike the baked-op program, the per-hop lookup is DATA-DEPENDENT (the row
    looked up in table p_{r+1} is f_{p_r}(...)(s), unknown until computed) AND
    the tables are recalled (random per example, not memorizable), so the K hops
    CANNOT be folded along the program positions — they must execute
    sequentially AT THE QUERY POSITION.  This is the heterogeneous analog of
    pointer-chase.
    """
    OP_BASE, QUERY, THINK, PAD, vocab = hetero_layout(N, L_ops)
    g = generator
    perm = torch.rand(B, L_ops, N, generator=g).argsort(dim=2)          # (B,L,N)
    order = torch.rand(B, L_ops, N, generator=g).argsort(dim=2)
    tgt = perm.gather(2, order)
    pairs = torch.stack([order, tgt], dim=3).reshape(B, L_ops, 2 * N)
    tags = (OP_BASE + torch.arange(L_ops)).view(1, L_ops, 1).expand(B, L_ops, 1)
    blocks = torch.cat([tags, pairs], dim=2)                            # (B,L,2N+1)
    tables = blocks.reshape(B, L_ops * (2 * N + 1))

    if homogeneous:
        # SAME op repeated K times (CONTROL: matches the multi-table recall +
        # context burden of hetero_mt but applies ONE op each hop -> isolates
        # heterogeneity from recall load).
        one = torch.randint(0, L_ops, (B, 1), generator=g)
        prog = one.expand(B, K).contiguous()
    else:
        prog = torch.randint(0, L_ops, (B, K), generator=g)             # (B,K)
    s = torch.randint(0, N, (B, 1), generator=g)
    x = s.clone()
    chain_cols = []
    bidx = torch.arange(B)
    for r in range(K):
        j = prog[:, r]
        x = perm[bidx, j, x[:, 0]].unsqueeze(1)
        chain_cols.append(x.clone())
    chain = torch.cat(chain_cols, dim=1) if chain_cols else s[:, :0]
    answers = chain[:, -1] if K > 0 else s.squeeze(1)
    query_col = torch.full((B, 1), QUERY, dtype=torch.long)
    prog_tok = OP_BASE + prog
    ids = torch.cat([tables, query_col, prog_tok, s], dim=1)
    return ids.to(device), answers.to(device), chain.to(device), prog.to(device), vocab


_BAKED_OPS: dict = {}


def get_baked_ops(N: int, L_ops: int, seed: int = 1234) -> torch.Tensor:
    """L_ops FIXED random permutations on N nodes, shared across ALL examples
    (the baked op-library). perms[j, i] = g_j(i).  Deterministic from seed so
    train and eval use the IDENTICAL library."""
    key = (N, L_ops, seed)
    if key not in _BAKED_OPS:
        g = torch.Generator().manual_seed(seed)
        _BAKED_OPS[key] = torch.rand(L_ops, N, generator=g).argsort(dim=1)
    return _BAKED_OPS[key]


def make_hetero_chase_batch(B: int, N: int, K: int, L_ops: int,
                            device="cuda", generator: torch.Generator | None = None,
                            fold: bool = False):
    """Returns (ids, answers (B,), chain (B,K), prog (B,K), vocab).

    Sequence (default, non-foldable): [p_1, ..., p_K, QUERY, s]
    Sequence (fold positive control): [s, p_1, ..., p_K, QUERY]
    answer = g_{p_K} o ... o g_{p_1} (s).  chain[:,r] = g_{p_r}o..og_{p_1}(s).
    Ops are BAKED (fixed library, no per-example tables).
    """
    OP_BASE, QUERY, THINK, PAD, vocab = hetero_layout(N, L_ops)
    g = generator
    perms = get_baked_ops(N, L_ops)                                    # (L_ops,N)
    prog = torch.randint(0, L_ops, (B, K), generator=g)               # (B,K) op ids
    s = torch.randint(0, N, (B, 1), generator=g)                      # start node

    # apply the program: x_{r} = g_{p_r}(x_{r-1})
    x = s.clone()                                                      # (B,1)
    chain_cols = []
    for r in range(K):
        j = prog[:, r]                                                 # (B,)
        x = perms[j, x[:, 0]].unsqueeze(1)                            # g_j(x)
        chain_cols.append(x.clone())
    chain = torch.cat(chain_cols, dim=1) if chain_cols else s[:, :0]
    answers = chain[:, -1] if K > 0 else s.squeeze(1)

    query_col = torch.full((B, 1), QUERY, dtype=torch.long)
    prog_tok = OP_BASE + prog                                          # op tokens
    if fold:
        ids = torch.cat([s, prog_tok, query_col], dim=1)             # s FIRST
    else:
        ids = torch.cat([prog_tok, query_col, s], dim=1)             # s LAST
    return ids.to(device), answers.to(device), chain.to(device), prog.to(device), vocab


# ----------------------------------------------------------------------------
# Model builder (explicit thinking_id / vocab so it works for both tasks)
# ----------------------------------------------------------------------------
def build(vocab, thinking_id, d_model, n_layers, n_heads, d_head, max_T,
          device="cuda"):
    model = TinyLM(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=DeltaNetAttention,
        max_T=max_T, feedback_mode="none", use_memory=False,
        thinking_token_id=thinking_id, state_readonly_at_think=True,
        output_gate=False, activation_checkpointing=False,
    ).to(device)
    return model


# ----------------------------------------------------------------------------
# Batch dispatch (task-agnostic interface): returns (ids, answers, vocab, thinking_id, max_T)
# ----------------------------------------------------------------------------
def task_meta(task, N, K_max, L_ops):
    """K_max must be the LARGEST K ever fed (train or eval) so the positional
    embedding table is big enough (the hetero program length scales with K)."""
    if task == "homo":
        thinking_id = N + 1
        vocab = N + 3
        max_T = 2 * N + 3                       # 2N pairs + QUERY + s + think slot
    elif task in ("hetero", "hetero_fold"):
        _, _, thinking_id, _, vocab = hetero_layout(N, L_ops)
        max_T = K_max + 2 + 1 + 2               # prog + QUERY + s + think (+margin)
    elif task in ("hetero_mt", "homo_mt"):
        _, _, thinking_id, _, vocab = hetero_layout(N, L_ops)
        max_T = L_ops * (2 * N + 1) + 1 + K_max + 1 + 2   # tables + QUERY + prog + s + think
    else:
        raise ValueError(task)
    return thinking_id, vocab, max_T


def make_batch(task, B, N, K, L_ops, device, g):
    if task == "homo":
        ids, ans, chain, vocab = make_pointer_chase_batch(B, N, K, device, g)
        return ids, ans, chain
    elif task in ("hetero_mt", "homo_mt"):
        ids, ans, chain, _prog, vocab = make_multitable_chase_batch(
            B, N, K, L_ops, device, g, homogeneous=(task == "homo_mt"))
        return ids, ans, chain
    else:
        ids, ans, chain, _prog, vocab = make_hetero_chase_batch(
            B, N, K, L_ops, device, g, fold=(task == "hetero_fold"))
        return ids, ans, chain


# ----------------------------------------------------------------------------
# Eval: accuracy vs K, at a given latent R (0 = no-think plain forward)
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_acc_vs_K(model, task, N, L_ops, thinking_id, R, K_list,
                  device, n_eval=1024, batch=512, seed=12345):
    model.eval()
    out = {}
    for K in K_list:
        gg = torch.Generator().manual_seed(seed + K)
        correct = 0
        total = 0
        done = 0
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans, _chain = make_batch(task, b, N, K, L_ops, device, gg)
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
# Train: depth curriculum (ramp K 1->K_max over 60%) + uniform-K consolidation
# ----------------------------------------------------------------------------
def train_cell(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    thinking_id, vocab, max_T = task_meta(
        args.task, args.N, max(args.K, args.eval_K_max), args.L_ops)
    model = build(vocab, thinking_id, args.d_model, args.n_layers,
                  args.n_heads, args.d_head, max_T, device=device)
    nparams = model.num_params()
    tag = f"{args.task}/{args.variant}/L{args.n_layers}/d{args.d_model}"
    print(f"[train] {tag}  N={args.N} K={args.K} L_ops={args.L_ops}  "
          f"params={nparams:,}  thinking_id={thinking_id} max_T={max_T}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
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
            # consolidation: uniform K in [1, K_max] to avoid forgetting shallow rungs
            K_cur = int(torch.randint(1, args.K + 1, (1,), generator=g).item())
        ids, ans, _chain = make_batch(args.task, args.batch, args.N, K_cur,
                                      args.L_ops, device, g)
        R_cur = 0 if args.variant == "nothink" else K_cur
        mode = "none" if args.variant == "nothink" else "latent"
        final_logits = think_forward(model, ids, R_cur, thinking_id, mode=mode)
        loss = F.cross_entropy(final_logits, ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = (final_logits[:, :args.N].argmax(-1) == ans).float().mean().item()
            print(f"  {tag}  step {step:>5}  loss {loss.item():.4f}  "
                  f"acc {acc:.3f}  K_cur {K_cur}  ({time.time()-t0:.0f}s)")

    # ----- evaluation -----
    K_list = list(range(1, args.eval_K_max + 1))
    res = {"task": args.task, "variant": args.variant, "n_layers": args.n_layers,
           "d_model": args.d_model, "n_heads": args.n_heads, "d_head": args.d_head,
           "N": args.N, "K_train": args.K, "L_ops": args.L_ops,
           "params": nparams, "steps": args.steps, "eval_K_max": args.eval_K_max,
           "K_list": K_list}
    if args.variant == "nothink":
        res["acc_R0"] = eval_acc_vs_K(model, args.task, args.N, args.L_ops,
                                      thinking_id, 0, K_list, device,
                                      n_eval=args.n_eval)
    else:
        for R in [0, 2, 4, 8]:
            res[f"acc_R{R}"] = eval_acc_vs_K(model, args.task, args.N, args.L_ops,
                                             thinking_id, R, K_list, device,
                                             n_eval=args.n_eval)
        # R=K diagonal (matched depth)
        diag = {}
        for K in K_list:
            d = eval_acc_vs_K(model, args.task, args.N, args.L_ops, thinking_id,
                              K, [K], device, n_eval=args.n_eval)
            diag[K] = d[K]
        res["acc_ReqK"] = diag

    print(f"[eval] {tag}")
    for k, v in res.items():
        if k.startswith("acc"):
            print(f"   {k}: " + " ".join(f"K{kk}={vv:.2f}" for kk, vv in v.items()))

    if args.save:
        torch.save({"state_dict": model.state_dict(), "config": res}, args.save)
        print(f"[saved] {args.save}")
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[appended] {args.out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",
                    choices=["homo", "hetero", "hetero_fold", "hetero_mt", "homo_mt"],
                    required=True)
    ap.add_argument("--variant", choices=["nothink", "latent"], required=True)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--K", type=int, default=6, help="max chain length trained")
    ap.add_argument("--eval_K_max", type=int, default=8)
    ap.add_argument("--L_ops", type=int, default=3, help="# ops (hetero only)")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=0, help="0 = d_model//d_head")
    ap.add_argument("--d_head", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2500)
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
        args.steps, args.batch, args.log_every, args.n_eval = 200, 128, 50, 512
        args.K, args.eval_K_max = 4, 5
    train_cell(args)


if __name__ == "__main__":
    main()
