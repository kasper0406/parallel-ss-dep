"""
Latent-space thinking — validation harness (2026-05-28).

GOAL: prove a HIGH-BANDWIDTH, NON-CoT thinking primitive is load-bearing,
on a task that provably requires sequential depth, with a small DeltaNet
co-trained from scratch.

MECHANISM ("latent ponder"):
  At a "think" slot appended after the query, we feed the trunk's OWN
  continuous hidden state back as the next input embedding (Coconut-style),
  for R refinement iterations. The think slot runs in state-readonly mode
  (DeltaNet b_proj β=0, via TinyLM(state_readonly_at_think=True)): it READS
  the recurrent state (full context) but never WRITES to it, so the
  long-range bindings the linear-RNN state carries are preserved. Each step
  feeds a full d_model continuous vector — maximum thinking bandwidth, not a
  1-of-vocab discrete token.

TASK (pointer-chase, needs depth + recall):
  A random permutation f on N nodes is presented as shuffled (i, f(i)) pairs,
  then [QUERY, s]. The label is f^K(s) — K sequential hops. The function
  table must be held in the recurrent state (recall); computing f^K(s) at a
  single position needs K sequential steps (depth). A fixed-depth forward can
  do ~1 hop; latent ponder steps add the missing depth WITHOUT corrupting the
  table state.

SUCCESS CRITERION:
  with-latent-think accuracy >> no-think (R=0) accuracy at K > trunk depth,
  AND ablating the latent channel (token-mode / R=0) collapses to the floor,
  AND a state-WRITE think variant corrupts the table (recall drops).

Usage:
  PYTHONPATH=. .venv/bin/python experiments/latent_think.py --smoke
  PYTHONPATH=. .venv/bin/python experiments/latent_think.py \
      --N 12 --K 4 --R_train 5 --steps 4000 --batch 256
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


# ----------------------------------------------------------------------------
# Task: pointer-chase f^K(start)
# ----------------------------------------------------------------------------
def make_pointer_chase_batch(B: int, N: int, K: int,
                             device: torch.device | str = "cuda",
                             generator: torch.Generator | None = None):
    """Returns (input_ids (B, 2N+2), answers (B,), chain (B, K), vocab_size).

    Token layout: nodes 0..N-1, QUERY=N, THINK=N+1, PAD=N+2.
    Sequence: [i0,f(i0), i1,f(i1), ..., QUERY, s]; label = f^K(s).
    `chain[:, r]` = f^{r+1}(s) — the intermediate hop targets for deep
    (per-step) latent supervision.
    """
    QUERY = N
    vocab = N + 3                      # nodes + QUERY + THINK + PAD
    g = generator
    # f: random permutation per row. perm[b, i] = f(i).
    perm = torch.rand(B, N, generator=g).argsort(dim=1)        # (B, N)
    # present pairs in shuffled source order
    order = torch.rand(B, N, generator=g).argsort(dim=1)       # (B, N) sources
    tgt = perm.gather(1, order)                                # f(source)
    pairs = torch.stack([order, tgt], dim=2).reshape(B, 2 * N)  # interleaved
    s = torch.randint(0, N, (B, 1), generator=g)               # start node
    # chain = [f(s), f^2(s), ..., f^K(s)]
    x = s.clone()
    chain_cols = []
    for _ in range(K):
        x = perm.gather(1, x)
        chain_cols.append(x.clone())
    chain = torch.cat(chain_cols, dim=1) if chain_cols else s[:, :0]
    answers = chain[:, -1] if K > 0 else s.squeeze(1)
    query_col = torch.full((B, 1), QUERY, dtype=torch.long)
    ids = torch.cat([pairs, query_col, s], dim=1)              # (B, 2N+2)
    return ids.to(device), answers.to(device), chain.to(device), vocab


def make_fixedpoint_chase_batch(B: int, N: int, L_max: int,
                                device: torch.device | str = "cuda",
                                generator: torch.Generator | None = None):
    """Adaptive-halting task: follow f from s until an ABSORBING node a
    (f(a)=a); the answer is a, reached in a VARIABLE number of hops L.

    Construction per row: pick a simple path s=v0 -> v1 -> ... -> vL=a of
    random length L in [1, L_max]; set f(vL)=vL (absorbing); every off-path
    node maps to a (so a is the unique fixed point and reachable from all).
    Returns (ids (B, 2N+2), answer=a (B,), L (B,), vocab).
    """
    QUERY = N
    vocab = N + 3
    g = generator
    ids = torch.empty(B, 2 * N + 2, dtype=torch.long)
    answer = torch.empty(B, dtype=torch.long)
    Ls = torch.empty(B, dtype=torch.long)
    for b in range(B):
        L = int(torch.randint(1, L_max + 1, (1,), generator=g).item())
        nodes = torch.randperm(N, generator=g)
        path = nodes[:L + 1]                       # v0..vL  (L hops)
        a = int(path[-1].item())
        f = torch.full((N,), a, dtype=torch.long)  # off-path -> a
        for r in range(L):
            f[int(path[r].item())] = int(path[r + 1].item())
        f[a] = a                                   # absorbing
        order = torch.randperm(N, generator=g)
        pairs = torch.stack([order, f[order]], dim=1).reshape(-1)  # 2N
        s = int(path[0].item())
        ids[b] = torch.cat([pairs, torch.tensor([QUERY, s])])
        answer[b] = a
        Ls[b] = L
    return ids.to(device), answer.to(device), Ls.to(device), vocab


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(vocab, N, d_model, n_layers, n_heads, d_head, max_T,
                state_readonly=True, output_gate=False, device="cuda"):
    thinking_id = N + 1
    model = TinyLM(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=DeltaNetAttention,
        max_T=max_T,
        feedback_mode="none",
        use_memory=False,
        thinking_token_id=thinking_id,
        state_readonly_at_think=state_readonly,
        output_gate=output_gate,
        activation_checkpointing=False,
    ).to(device)
    return model, thinking_id


# ----------------------------------------------------------------------------
# Latent thinking forward
# ----------------------------------------------------------------------------
def think_forward(model, base_ids, R, thinking_id, mode="latent",
                  return_steps=False):
    """Compute answer logits at the final position after R latent steps.

    mode:
      'none'   — R is ignored; plain forward, predict from the last position.
      'latent' — feed the trunk's own hidden back as the think-slot input
                 embedding for R iterations (high-bandwidth latent ponder).
      'token'  — append a think slot but keep its input = embed(THINK) every
                 step (discrete homogeneous baseline; tests bandwidth).

    Returns answer_logits (B, vocab), or if return_steps: (B, R, vocab)
    stacked per-step logits at the think slot (for deep supervision).
    """
    B, Lb = base_ids.shape
    if mode == "none" or R == 0:
        logits, _h = model(base_ids, return_hidden=True)
        out = logits[:, -1, :]
        return out.unsqueeze(1) if return_steps else out

    base_emb = model.embed(base_ids)                       # (B, Lb, d)
    think_col = torch.full((B, 1), thinking_id, dtype=torch.long,
                           device=base_ids.device)
    ids = torch.cat([base_ids, think_col], dim=1)          # (B, Lb+1)
    think_emb = model.embed(think_col)                     # (B, 1, d)

    # init latent from the query position's hidden
    _logits0, h0 = model(base_ids, return_hidden=True)
    z = h0[:, -1:, :]                                       # (B, 1, d)

    step_logits = []
    for _ in range(R):
        if mode == "latent":
            slot_emb = z
        elif mode == "token":
            slot_emb = think_emb
        else:
            raise ValueError(mode)
        ie = torch.cat([base_emb, slot_emb], dim=1)        # (B, Lb+1, d)
        logits, h = model(ids, inputs_embeds=ie, return_hidden=True)
        z = h[:, -1:, :]
        step_logits.append(logits[:, -1, :])
    if return_steps:
        return torch.stack(step_logits, dim=1)             # (B, R, vocab)
    return step_logits[-1]


def think_forward_gated(model, base_ids, R, thinking_id):
    """Latent ponder for R steps, returning per-step answer logits AND the
    per-step halt-gate at the think slot.  Returns (step_logits (B,R,V),
    step_gate_logits (B,R)).  The gate is read from model._last_gate_logits
    at the think-slot position (requires output_gate=True)."""
    B, Lb = base_ids.shape
    base_emb = model.embed(base_ids)
    think_col = torch.full((B, 1), thinking_id, dtype=torch.long,
                           device=base_ids.device)
    ids = torch.cat([base_ids, think_col], dim=1)
    _logits0, h0 = model(base_ids, return_hidden=True)
    z = h0[:, -1:, :]
    step_logits, step_gates = [], []
    for _ in range(R):
        ie = torch.cat([base_emb, z], dim=1)
        logits, h = model(ids, inputs_embeds=ie, return_hidden=True)
        z = h[:, -1:, :]
        step_logits.append(logits[:, -1, :])
        step_gates.append(model._last_gate_logits[:, -1])     # (B,)
    return torch.stack(step_logits, dim=1), torch.stack(step_gates, dim=1)


@torch.no_grad()
def eval_halt(model, N, L_max, thinking_id, device, R_max, n_eval=2048,
              batch=512, seed=777):
    """Run the adaptive loop: step until gate(halt) fires (or R_max), then
    emit. Report answer accuracy and how well the halt step matches L."""
    model.eval()
    gg = torch.Generator().manual_seed(seed)
    ans_correct = 0
    halt_correct = 0          # halted at exactly L
    halt_mae = 0.0
    total = 0
    done = 0
    while done < n_eval:
        b = min(batch, n_eval - done)
        ids, ans, Ls, _ = make_fixedpoint_chase_batch(b, N, L_max, device, gg)
        step_logits, step_glogits = think_forward_gated(
            model, ids, R_max, thinking_id)
        halt = (torch.sigmoid(step_glogits) > 0.5)            # (b, R_max)
        # first step where halt fires; if never, use last step
        any_halt = halt.any(dim=1)
        first = torch.where(any_halt, halt.float().argmax(dim=1),
                            torch.full_like(Ls, R_max - 1))
        idx = first.clamp(max=R_max - 1)
        emitted = step_logits[torch.arange(b), idx][:, :N].argmax(dim=-1)
        ans_correct += (emitted == ans).sum().item()
        halt_step = (idx + 1)                                  # hops taken
        halt_correct += (halt_step == Ls).sum().item()
        halt_mae += (halt_step - Ls).abs().float().sum().item()
        total += b
        done += b
    model.train()
    return {"answer_acc": ans_correct / total,
            "halt_exact": halt_correct / total,
            "halt_mae": halt_mae / total}


def train_fixedpoint(args):
    """Adaptive-halting latent thinking: the gate learns WHEN to stop."""
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    max_T = 2 * args.N + 3
    vocab = args.N + 3
    R_max = args.K
    model, thinking_id = build_model(
        vocab, args.N, args.d_model, args.n_layers, args.n_heads,
        args.d_head, max_T, state_readonly=not args.state_write,
        output_gate=True, device=device)
    print(f"[latent-halt] N={args.N} L_max={args.K} L={args.n_layers} "
          f"d={args.d_model}  params={model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.k_curriculum:
            frac = min(1.0, step / (0.6 * args.steps))
            L_cur = max(1, min(args.K, int(round(1 + frac * (args.K - 1)))))
        else:
            L_cur = args.K
        ids, ans, Ls, _ = make_fixedpoint_chase_batch(
            args.batch, args.N, L_cur, device, g)
        R = L_cur
        step_logits, step_glogits = think_forward_gated(
            model, ids, R, thinking_id)
        B = ids.shape[0]
        # answer supervision at the true halt step (step L): decode -> a
        ans_idx = (Ls - 1).clamp(max=R - 1)
        ans_logits = step_logits[torch.arange(B), ans_idx]
        loss_ans = F.cross_entropy(ans_logits, ans)
        # gate halt supervision: fire (1) at steps r >= L, else 0
        rr = torch.arange(1, R + 1, device=device).unsqueeze(0)   # (1,R) hops
        gate_tgt = (rr >= Ls.unsqueeze(1)).float()                # (B,R)
        loss_gate = F.binary_cross_entropy_with_logits(step_glogits, gate_tgt)
        loss = loss_ans + args.gate_weight * loss_gate
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                aacc = (ans_logits[:, :args.N].argmax(-1) == ans).float().mean().item()
            print(f"  step {step:>5}  loss {loss.item():.4f} "
                  f"(ans {loss_ans.item():.3f} gate {loss_gate.item():.3f})  "
                  f"ans_acc {aacc:.3f}  L_cur {L_cur}  ({time.time()-t0:.0f}s)")
    res = eval_halt(model, args.N, args.K, thinking_id, device, R_max=args.K + 3)
    print("\n=== ADAPTIVE HALT EVAL (fixed-point chase, variable L) ===")
    for k, v in res.items():
        print(f"  {k:>12}: {v:.3f}")
    if args.save:
        torch.save({"state_dict": model.state_dict(), "config": {
            "N": args.N, "K": args.K, "task": "fixedpoint",
            "thinking_id": thinking_id}}, args.save)
        print(f"[saved] {args.save}")
    return model, res


# ----------------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, N, K, thinking_id, R_list, device, n_eval=2048,
             batch=512, seed=12345):
    model.eval()
    results = {}
    modes = [("none", 0)] + [("latent", R) for R in R_list] \
            + [("token", max(R_list))]
    for mode, R in modes:
        correct = 0
        total = 0
        gg = torch.Generator().manual_seed(seed)  # same eval set every mode
        done = 0
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans, _chain, _ = make_pointer_chase_batch(b, N, K, device, gg)
            logits = think_forward(model, ids, R, thinking_id, mode=mode)
            pred = logits[:, :N].argmax(dim=-1)   # answer is a node id
            correct += (pred == ans).sum().item()
            total += b
            done += b
        key = f"{mode}{'' if mode=='none' else f'_R{R}'}"
        results[key] = correct / total
    model.train()
    return results


@torch.no_grad()
def eval_per_hop(model, N, R_steps, thinking_id, device, n_eval=2048,
                 batch=512, seed=999):
    """Run R_steps latent steps; report accuracy of each step's decoded
    latent vs the true hop f^r(s). When R_steps > K_train this measures
    LENGTH GENERALIZATION (does the model keep applying f past trained depth?)."""
    model.eval()
    gg = torch.Generator().manual_seed(seed)
    correct = torch.zeros(R_steps, device=device)
    total = 0
    done = 0
    while done < n_eval:
        b = min(batch, n_eval - done)
        # chain of length R_steps so every step has a ground-truth hop
        ids, _ans, chain, _ = make_pointer_chase_batch(b, N, R_steps, device, gg)
        step_logits = think_forward(model, ids, R_steps, thinking_id,
                                    mode="latent", return_steps=True)
        pred = step_logits[:, :, :N].argmax(dim=-1)                    # (b,R)
        correct += (pred == chain).float().sum(dim=0)
        total += b
        done += b
    model.train()
    return (correct / total).tolist()


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    max_T = 2 * args.N + 3            # pairs + QUERY + s + think slot
    vocab = args.N + 3
    model, thinking_id = build_model(
        vocab, args.N, args.d_model, args.n_layers, args.n_heads,
        args.d_head, max_T, state_readonly=not args.state_write,
        device=device)
    print(f"[latent-think] N={args.N} K={args.K} R_train={args.R_train} "
          f"L={args.n_layers} d={args.d_model}  params={model.num_params():,}  "
          f"state_readonly={not args.state_write}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)

    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.k_curriculum:
            # Ramp chain length 1 -> K over the first 60% of steps, then hold.
            frac = min(1.0, step / (0.6 * args.steps))
            K_cur = max(1, min(args.K, int(round(1 + frac * (args.K - 1)))))
        else:
            K_cur = args.K
        R_cur = K_cur if args.deep_supervision or args.k_curriculum else args.R_train
        ids, ans, chain, _ = make_pointer_chase_batch(
            args.batch, args.N, K_cur, device, g)
        if args.deep_supervision and args.train_mode != "none":
            # Supervise each latent step r to decode f^r(s). R aligns with hop r.
            step_logits = think_forward(model, ids, R_cur, thinking_id,
                                        mode=args.train_mode, return_steps=True)
            R = step_logits.shape[1]
            tgt = chain[:, :R]                          # (B, R) hop targets
            loss = F.cross_entropy(
                step_logits.reshape(-1, step_logits.shape[-1]),
                tgt.reshape(-1))
            final_logits = step_logits[:, -1, :]
        else:
            # Final-answer-only supervision (realistic: no per-hop labels).
            final_logits = think_forward(model, ids, R_cur, thinking_id,
                                         mode=args.train_mode)
            loss = F.cross_entropy(final_logits, ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = (final_logits[:, :args.N].argmax(-1) == ans).float().mean().item()
            print(f"  step {step:>5}  loss {loss.item():.4f}  "
                  f"train_acc(final) {acc:.3f}  ({time.time()-t0:.0f}s)")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "step": args.steps,
                    "config": {"N": args.N, "K": args.K, "d_model": args.d_model,
                               "n_layers": args.n_layers, "n_heads": args.n_heads,
                               "d_head": args.d_head, "vocab": vocab,
                               "thinking_id": thinking_id}},
                   args.save)
        print(f"[saved] {args.save}")

    R_list = [int(x) for x in args.eval_R.split(",")]
    res = evaluate(model, args.N, args.K, thinking_id, R_list, device)
    print("\n=== EVAL (pointer-chase f^K(start), final-answer acc) ===")
    for k, v in res.items():
        print(f"  {k:>16}: {v:.3f}")
    if args.K > 0:
        R_probe = max(args.K, args.extrapolate_R)
        per_hop = eval_per_hop(model, args.N, R_probe, thinking_id, device)
        print(f"\n=== PER-HOP latent thread acc (trained K={args.K}; "
              f"step r decodes f^r(s); r>{args.K} = extrapolation) ===")
        print("  " + "  ".join(f"h{r+1}:{a:.2f}" for r, a in enumerate(per_hop)))
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--R_train", type=int, default=5)
    ap.add_argument("--eval_R", type=str, default="0,1,2,3,4,5,6,8")
    ap.add_argument("--extrapolate_R", type=int, default=0,
                    help="probe per-hop thread out to this many steps (>K)")
    ap.add_argument("--train_mode", type=str, default="latent",
                    choices=["latent", "token", "none"])
    ap.add_argument("--deep_supervision", action="store_true",
                    help="supervise each latent step r to decode f^r(s)")
    ap.add_argument("--k_curriculum", action="store_true",
                    help="ramp chain length 1->K over training (R tracks K)")
    ap.add_argument("--task", type=str, default="chase",
                    choices=["chase", "fixedpoint"],
                    help="chase=f^K(s); fixedpoint=adaptive-halt to absorbing a")
    ap.add_argument("--gate_weight", type=float, default=1.0)
    ap.add_argument("--state_write", action="store_true",
                    help="disable state-readonly (think writes to recurrence)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.N, args.K, args.R_train = 8, 3, 3
        args.steps, args.batch, args.log_every = 300, 128, 50
    if args.task == "fixedpoint":
        train_fixedpoint(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
