"""
Train + evaluate the latent-thinking DeltaNet on one of the two tasks.

Recipe (the validated one): a depth curriculum that ramps the chain length
1 -> n_max over the first 60% of training, with the number of latent think
steps R tracking the current depth, plus optional per-hop supervision. Eval
reports, per depth n, accuracy with NO thinking vs R=n latent thinking (the
"lift"), on a DISJOINT held-out set.

  PYTHONPATH not needed — self-contained.
  python train.py --task homogeneous   --steps 3000
  python train.py --task heterogeneous --steps 3000

Use CUDA_VISIBLE_DEVICES=0 to pin to a free GPU.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from model import DeltaNetLM, ModelConfig
from tasks import make_batch, max_seq_len, vocab_for  # noqa: F401


# Disjoint train/eval splits: distinct base seeds for the two stream generators.
TRAIN_SEED_BASE = 1000
EVAL_SEED_BASE = 9_000_000


def build(task: str, V: int, n_max: int, args, device):
    vocab = vocab_for(task, V)
    # +1 for the single appended think slot the latent loop adds.
    max_T = max_seq_len(task, V, n_max) + 1
    cfg = ModelConfig(
        vocab_size=vocab.size,
        max_T=max_T,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        thinking_id=vocab.THINK,
        state_readonly=not args.state_write,
        use_memory=not args.no_memory,
        mem_size=args.mem_size,
    )
    return DeltaNetLM(cfg).to(device), vocab


@torch.no_grad()
def evaluate(model, task, V, n_list, device, n_eval=2048, batch=512):
    """For each depth n: accuracy with mode 'none' (R=0) and 'latent' (R=n).
    Also reports the 'token' control (R=n_max constant-embedding feedback)."""
    model.eval()
    rows = {}
    for n in n_list:
        out = {}
        for mode, R in [("none", 0), ("latent", n), ("token", n)]:
            correct = total = done = 0
            g = torch.Generator().manual_seed(EVAL_SEED_BASE + n)  # fixed eval set
            while done < n_eval:
                b = min(batch, n_eval - done)
                ids, ans, _chain, vocab = make_batch(task, b, V, n, device, g)
                logits = model.think_forward(ids, R, mode=mode)
                pred = logits[:, :V].argmax(dim=-1)
                correct += (pred == ans).sum().item()
                total += b
                done += b
            out[mode] = correct / total
        rows[n] = out
    model.train()
    return rows


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    V, n_max = args.V, args.n_max
    model, vocab = build(args.task, V, n_max, args, device)
    print(f"[{args.task}] V={V} n_max={n_max} L={args.n_layers} d={args.d_model} "
          f"mem={'off' if args.no_memory else args.mem_size} "
          f"state_readonly={not args.state_write}  params={model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)

    g = torch.Generator().manual_seed(TRAIN_SEED_BASE + args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # depth curriculum: ramp 1 -> n_max over first 60%, then hold (with a
        # consolidation phase that samples uniformly to avoid forgetting shallow).
        frac = min(1.0, step / (0.6 * args.steps))
        n_top = max(1, min(n_max, int(round(1 + frac * (n_max - 1)))))
        if step > 0.6 * args.steps:
            # consolidation: sample depth uniformly in [1, n_max]
            n_cur = int(torch.randint(1, n_max + 1, (1,)).item())
        else:
            n_cur = n_top
        R = n_cur

        ids, ans, chain, _ = make_batch(args.task, args.batch, V, n_cur, device, g)
        if args.deep_supervision:
            step_logits = model.think_forward(ids, R, mode="latent",
                                              return_steps=True)   # (B, R, Vsz)
            Rs = step_logits.shape[1]
            tgt = chain[:, :Rs]
            loss = F.cross_entropy(step_logits.reshape(-1, step_logits.shape[-1]),
                                   tgt.reshape(-1))
            final_logits = step_logits[:, -1, :]
        else:
            final_logits = model.think_forward(ids, R, mode="latent")
            loss = F.cross_entropy(final_logits, ans)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = (final_logits[:, :V].argmax(-1) == ans).float().mean().item()
            print(f"  step {step:>5}  loss {loss.item():.4f}  "
                  f"train_acc {acc:.3f}  n_cur {n_cur}  ({time.time()-t0:.0f}s)")

    n_list = list(range(1, n_max + 1))
    rows = evaluate(model, args.task, V, n_list, device)
    print(f"\n=== EVAL: {args.task} (held-out) — accuracy by depth ===")
    print(f"{'depth':>6} | {'no-think':>9} | {'latent R=n':>11} | "
          f"{'lift':>6} | {'token R=n':>9}")
    print("-" * 56)
    for n in n_list:
        r = rows[n]
        lift = r["latent"] - r["none"]
        print(f"{n:>6} | {r['none']:>9.3f} | {r['latent']:>11.3f} | "
              f"{lift:>+6.3f} | {r['token']:>9.3f}")
    avg_lift = sum(rows[n]["latent"] - rows[n]["none"] for n in n_list) / len(n_list)
    deep = [n for n in n_list if n >= max(2, n_max // 2)]
    deep_lift = sum(rows[n]["latent"] - rows[n]["none"] for n in deep) / len(deep)
    print("-" * 56)
    print(f"  mean lift (all depths): {avg_lift:+.3f}   "
          f"mean lift (deep n>={deep[0]}): {deep_lift:+.3f}")

    if args.save:
        torch.save({"state_dict": model.state_dict(),
                    "cfg": model.cfg.__dict__,
                    "task": args.task, "V": V, "n_max": n_max,
                    "eval": rows}, args.save)
        print(f"[saved] {args.save}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["homogeneous", "heterogeneous"],
                    default="homogeneous")
    ap.add_argument("--V", type=int, default=10, help="number of value tokens (0..V-1)")
    ap.add_argument("--n_max", type=int, default=6, help="max depth / chain length")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=48)
    ap.add_argument("--mem_size", type=int, default=32)
    ap.add_argument("--no_memory", action="store_true")
    ap.add_argument("--state_write", action="store_true",
                    help="let think steps WRITE the recurrent state (default: readonly)")
    ap.add_argument("--deep_supervision", action="store_true",
                    help="supervise each latent step to decode the r-th intermediate")
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.V, args.n_max, args.steps, args.batch = 8, 3, 300, 128
        args.log_every = 50
    train(args)


if __name__ == "__main__":
    main()
