"""
Latent-thinking on real arithmetic-chain reasoning — transfer test (2026-05-28).

Ports the validated latent-ponder primitive (`latent_think.py`) onto the
arithmetic-chain reasoning task — the exact task class where discrete-token
thinking scored 0/80 on every rung (`THINKING_DEMONSTRATION_2026_05_28.md`).

Task: a chain  v0 = seed; v1 = v0 OP a1; ...; vn = v(n-1) OP an  (OP in +/-),
encoded as real arithmetic TEXT tokens. The model reads the chain and must
emit the final value vn. Depth = n_steps; one forward can do a few ops, deeper
chains need the latent ponder steps to supply the missing sequential compute.
Answer is bounded so it is a single token (clean accuracy, mirrors the
synthetic decode-from-latent), but the PROBLEM is a real multi-token sequence
and the model is the real DeltaNet (optionally + FiLM + gate).

Recipe (validated on the synthetic): latent ponder, state-readonly, depth
curriculum (ramp n 1->N), final-answer-only supervision. R latent steps = n.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/latent_arith.py --smoke
  PYTHONPATH=. .venv/bin/python experiments/latent_arith.py \
      --max_n 6 --k_curriculum --n_layers 4 --d_model 256 --steps 8000
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
from experiments.latent_think import think_forward

# token layout
PLUS, MINUS, AMARK, THINK, PAD = 10, 11, 12, 13, 14
ANS_BASE = 15
VRANGE = 64                              # answers in [-VRANGE, VRANGE]
VOCAB = ANS_BASE + 2 * VRANGE + 1        # = 144
THINKING_ID = THINK


def gen_arith_batch(B, n_steps, device, g):
    """Returns (ids (B, 1+2n+1), answer_token (B,)). Chain of +/- ops."""
    seed = torch.randint(0, 10, (B, 1), generator=g)
    val = seed.clone()
    toks = [seed]
    for _ in range(n_steps):
        op = torch.randint(0, 2, (B, 1), generator=g)         # 0=+,1=-
        operand = torch.randint(0, 10, (B, 1), generator=g)
        op_tok = torch.where(op == 0, torch.full_like(op, PLUS),
                             torch.full_like(op, MINUS))
        toks.append(op_tok)
        toks.append(operand)
        val = torch.where(op == 0, val + operand, val - operand)
    toks.append(torch.full((B, 1), AMARK))
    ids = torch.cat(toks, dim=1)
    val = val.clamp(-VRANGE, VRANGE)
    ans_tok = (ANS_BASE + val + VRANGE).squeeze(1)
    return ids.to(device), ans_tok.to(device)


def build_model(max_T, n_layers, d_model, n_heads, d_head, feedback,
                state_readonly=True, device="cuda"):
    kw = {}
    if feedback == "film":
        kw = dict(feedback_mode="film", feedback_pairs=((1, n_layers - 1),),
                  feedback_self_k=3)
    model = TinyLM(
        vocab_size=VOCAB, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_head=d_head, attention_cls=DeltaNetAttention, max_T=max_T,
        use_memory=False, thinking_token_id=THINKING_ID,
        state_readonly_at_think=state_readonly,
        activation_checkpointing=False, **kw,
    ).to(device)
    return model


@torch.no_grad()
def evaluate(model, n_steps, R_list, device, n_eval=2048, batch=512, seed=4321):
    model.eval()
    res = {}
    modes = [("none", 0)] + [("latent", R) for R in R_list] \
            + [("token", max(R_list))]
    for mode, R in modes:
        correct = 0
        total = 0
        gg = torch.Generator().manual_seed(seed)
        done = 0
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans = gen_arith_batch(b, n_steps, device, gg)
            logits = think_forward(model, ids, R, THINKING_ID, mode=mode)
            pred = logits[:, ANS_BASE:ANS_BASE + 2 * VRANGE + 1].argmax(-1) + ANS_BASE
            correct += (pred == ans).sum().item()
            total += b
            done += b
        res[f"{mode}{'' if mode=='none' else f'_R{R}'}"] = correct / total
    model.train()
    return res


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    max_T = 1 + 2 * args.max_n + 1 + 1        # seed + ops + AMARK + think slot
    model = build_model(max_T, args.n_layers, args.d_model, args.n_heads,
                        args.d_head, args.feedback,
                        state_readonly=not args.state_write, device=device)
    print(f"[latent-arith] max_n={args.max_n} L={args.n_layers} d={args.d_model} "
          f"fb={args.feedback}  params={model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.k_curriculum:
            ramp_end = 0.6 * args.steps
            if step < ramp_end:
                frac = step / ramp_end
                n_cur = max(1, min(args.max_n,
                                   int(round(1 + frac * (args.max_n - 1)))))
            else:
                # Consolidation: sample depth uniformly so the model retains
                # ALL depths (a strict ramp forgets shallow rungs).
                n_cur = int(torch.randint(1, args.max_n + 1, (1,),
                                          generator=g).item())
        else:
            n_cur = args.max_n
        R = n_cur
        ids, ans = gen_arith_batch(args.batch, n_cur, device, g)
        logits = think_forward(model, ids, R, THINKING_ID, mode=args.train_mode)
        loss = F.cross_entropy(logits, ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                sl = logits[:, ANS_BASE:ANS_BASE + 2 * VRANGE + 1]
                acc = ((sl.argmax(-1) + ANS_BASE) == ans).float().mean().item()
            print(f"  step {step:>5}  loss {loss.item():.4f}  acc {acc:.3f}  "
                  f"n_cur {n_cur}  ({time.time()-t0:.0f}s)")

    print("\n=== PER-RUNG EVAL (latent-think vs no-think) ===")
    print(f"{'n':>3} | {'none':>6} | {'latent_Rn':>9} | {'token_Rn':>8} | delta")
    for n in range(1, args.max_n + 1):
        res = evaluate(model, n, [n], device)
        d = res[f"latent_R{n}"] - res["none"]
        print(f"{n:>3} | {res['none']:>6.3f} | {res[f'latent_R{n}']:>9.3f} | "
              f"{res[f'token_R{n}']:>8.3f} | {d:+.3f}")
    if args.save:
        torch.save({"state_dict": model.state_dict(),
                    "config": {"max_n": args.max_n, "d_model": args.d_model,
                               "n_layers": args.n_layers, "feedback": args.feedback,
                               "vocab": VOCAB, "thinking_id": THINKING_ID}},
                   args.save)
        print(f"[saved] {args.save}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_n", type=int, default=6)
    ap.add_argument("--k_curriculum", action="store_true")
    ap.add_argument("--train_mode", type=str, default="latent",
                    choices=["latent", "token", "none"])
    ap.add_argument("--state_write", action="store_true")
    ap.add_argument("--feedback", type=str, default="none",
                    choices=["none", "film"])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_n, args.steps, args.batch = 3, 400, 128
        args.k_curriculum = True
    train(args)


if __name__ == "__main__":
    main()
