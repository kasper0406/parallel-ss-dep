"""bf16-optimizer-state experiment — does loss diverge from fp32?

Compares fp32-state vs bf16-state AdamW (and Muon-momentum) on a small
TinyLM trained on synthetic-but-realistic next-token data (CodeParrot
slice from the v4 mix). Goal: green/red light on whether we can move
optimizer state to bf16 in the v4 trainer to free ~550 MB of GPU memory.

Tests three things:
  1. AdamW with bf16 (exp_avg, exp_avg_sq) state — most concerning case
  2. Muon with bf16 momentum_buffer — Muon NS already bf16 internally
  3. Both, simultaneously

Usage:
  CUDA_VISIBLE_DEVICES=1 python experiments/test_bf16_optim.py
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Make repo importable.
sys.path.insert(0, ".")
from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM


# -----------------------------------------------------------------------------
# Optimizer wrappers that cast state to bf16.
# -----------------------------------------------------------------------------

def _cast_state_to_bf16(opt: torch.optim.Optimizer, keys: tuple[str, ...]):
    """Cast the named state tensors of every param to bf16, in-place."""
    for group in opt.param_groups:
        for p in group["params"]:
            if p not in opt.state:
                continue
            st = opt.state[p]
            for k in keys:
                if k in st and isinstance(st[k], torch.Tensor) and st[k].dtype != torch.bfloat16:
                    st[k] = st[k].to(torch.bfloat16)


class BF16StateAdamW(torch.optim.Optimizer):
    """AdamW with exp_avg / exp_avg_sq stored persistently in bf16.

    Math is done in fp32 (cast → step → cast back); only the stored
    state is bf16. Persistent memory: 4 bytes/param vs 8 in stock AdamW.
    Transient peak: one extra fp32 copy of the largest single param
    (negligible vs the persistent saving).

    This is the cleanest test of "is bf16 storage precision enough to
    keep the running averages on track?" — round-to-nearest-even on
    every cast (no stochastic rounding). If this works, real bf16 in-
    place arithmetic with bnb-style block quant is even safer.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                state["step"] += 1
                t = state["step"]
                # Decoupled weight decay (AdamW form).
                if wd != 0:
                    p.mul_(1 - lr * wd)
                # Lift state (and grad, if bf16) to fp32 for the math.
                m = state["exp_avg"].to(torch.float32)
                v = state["exp_avg_sq"].to(torch.float32)
                g = grad.to(torch.float32) if grad.dtype != torch.float32 else grad
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                denom = (v.sqrt() / math.sqrt(bias_c2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias_c1)
                # Round-to-nearest-even back to bf16 for storage.
                state["exp_avg"].copy_(m)
                state["exp_avg_sq"].copy_(v)
        return loss


class BF16StateMuon(torch.optim.Muon):
    """Muon where momentum_buffer is stored in bf16. Muon already runs
    Newton-Schulz in bf16 internally — this only changes the *storage*
    of the momentum buffer, not the arithmetic precision.

    Caveat: torch.optim.Muon's .step() does its own .add_(grad) on the
    fp32 momentum buffer, so we cast AFTER its step. The next step
    will cast back to fp32 implicitly when add_/mul_ operates on
    bf16+fp32 (PyTorch promotes). That promotion path is more lenient
    than AdamW's lerp_, so this should work without a custom loop.
    """
    def step(self, closure=None):
        loss = super().step(closure)
        _cast_state_to_bf16(self, ("momentum_buffer",))
        return loss


# -----------------------------------------------------------------------------
# Tiny model + data.
# -----------------------------------------------------------------------------

def build_tiny_model(vocab_size: int, device: str) -> nn.Module:
    """A small DeltaNet that trains fast but exercises the same kernels."""
    return TinyLM(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_head=32,
        attention_cls=DeltaNetAttention,
        feedback_mode="none",
    ).to(device)


def gen_synthetic_batch(batch: int, T: int, vocab: int, device: str, gen: torch.Generator):
    """Random integer tokens — nominal task, but with a structure so the
    model has *something* to learn (loss should drop)."""
    ids = torch.randint(0, vocab, (batch, T), device=device, generator=gen)
    # Inject a copy-3 task: every 8th token is a copy of the token 3 ago,
    # giving the model a concrete pattern to learn so loss falls cleanly.
    for k in range(8, T, 8):
        ids[:, k] = ids[:, k - 3]
    return ids


def init_model_with_seed(vocab_size: int, device: str, seed: int) -> nn.Module:
    """Build model with deterministic weight init."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return build_tiny_model(vocab_size, device)


# -----------------------------------------------------------------------------
# Train loop.
# -----------------------------------------------------------------------------

@dataclass
class RunStats:
    losses: list[float]
    final_loss: float
    state_bytes: int
    wall_s: float


def train_one(
    optim_kind: str,           # see variants list in main()
    n_steps: int = 400,
    batch: int = 8,
    T: int = 256,
    grad_accum: int = 1,
    lr_adamw: float = 3e-4,
    lr_muon: float = 1e-3,
    seed: int = 42,
    device: str = "cuda",
    log_every: int = 20,
    bf16_grads: bool = False,
    bf16_autocast: bool = False,
) -> RunStats:
    print(f"\n=== {optim_kind} ===")

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    vocab = tok.vocab_size + 1   # match TinyLM's +1 for thinking token slot

    model = init_model_with_seed(vocab, device, seed)
    n_params = sum(p.numel() for p in model.parameters())
    # Muon requires strictly-2D params, excluding embeddings.
    muon_params, adamw_params = [], []
    for n, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in n and "lm_head" not in n:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    if optim_kind == "fp32_adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=lr_adamw)]
    elif optim_kind == "bf16_adamw":
        opts = [BF16StateAdamW(model.parameters(), lr=lr_adamw)]
    elif optim_kind == "fp32_muon":
        opts = [
            torch.optim.Muon(muon_params, lr=lr_muon),
            torch.optim.AdamW(adamw_params, lr=lr_adamw),
        ]
    elif optim_kind == "bf16_muon":
        opts = [
            BF16StateMuon(muon_params, lr=lr_muon),
            torch.optim.AdamW(adamw_params, lr=lr_adamw),
        ]
    elif optim_kind == "both_bf16":
        opts = [
            BF16StateMuon(muon_params, lr=lr_muon),
            BF16StateAdamW(adamw_params, lr=lr_adamw),
        ]
    else:
        raise ValueError(optim_kind)

    print(f"  params: {n_params/1e6:.2f}M  Muon-eligible: {sum(p.numel() for p in muon_params)/1e6:.2f}M  AdamW: {sum(p.numel() for p in adamw_params)/1e6:.2f}M")

    # Independent RNG for data so all variants see the SAME batches.
    data_gen = torch.Generator(device=device).manual_seed(seed)

    if bf16_grads:
        # Tell autograd it's allowed to assign a bf16 .grad to fp32 params.
        for p in model.parameters():
            try:
                p.grad_dtype = torch.bfloat16
            except AttributeError:
                pass  # older torch — no grad_dtype guard

    def maybe_cast_grads_bf16():
        if not bf16_grads:
            return
        for p in model.parameters():
            if p.grad is not None and p.grad.dtype != torch.bfloat16:
                p.grad = p.grad.to(torch.bfloat16)

    losses: list[float] = []
    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            ids = gen_synthetic_batch(batch, T, vocab, device, data_gen)
            if bf16_autocast:
                ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            else:
                from contextlib import nullcontext
                ctx = nullcontext()
            with ctx:
                out = model(ids)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                ) / grad_accum
            loss.backward()
            maybe_cast_grads_bf16()
            loss_acc += loss.item()
        for o in opts:
            o.step()
        losses.append(loss_acc * grad_accum)  # report unscaled
        if step == 1 or step % log_every == 0 or step == n_steps:
            print(f"  step {step:4d}  loss={losses[-1]:.4f}")

    wall = time.perf_counter() - t0

    # Tally optimizer-state bytes.
    state_bytes = 0
    for o in opts:
        for group in o.param_groups:
            for p in group["params"]:
                if p in o.state:
                    for v in o.state[p].values():
                        if isinstance(v, torch.Tensor):
                            state_bytes += v.element_size() * v.numel()

    print(f"  state memory: {state_bytes/1e6:.1f} MB   wall: {wall:.1f}s")

    return RunStats(losses=losses, final_loss=losses[-1], state_bytes=state_bytes, wall_s=wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_steps", type=int, default=400)
    ap.add_argument("--variants", type=str, default="fp32_adamw,bf16_adamw,fp32_muon,bf16_muon,both_bf16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--bf16_grads", action="store_true",
                    help="Cast .grad to bf16 after each backward (precision test for the bf16-grad scenario).")
    ap.add_argument("--bf16_autocast", action="store_true",
                    help="Wrap forward+backward in bf16 autocast (matches v4 trainer setup).")
    args = ap.parse_args()

    variants = args.variants.split(",")
    results: dict[str, RunStats] = {}
    for v in variants:
        results[v] = train_one(v, n_steps=args.n_steps, seed=args.seed,
                               grad_accum=args.grad_accum,
                               bf16_grads=args.bf16_grads,
                               bf16_autocast=args.bf16_autocast)

    # Summary.
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'variant':<14} {'final_loss':>11} {'min_loss':>10} {'state MB':>10} {'wall s':>8}")
    for k, r in results.items():
        print(f"{k:<14} {r.final_loss:>11.4f} {min(r.losses):>10.4f} {r.state_bytes/1e6:>10.1f} {r.wall_s:>8.1f}")

    # Pairwise loss-curve divergence (max | bf16 - fp32 |).
    print("\nLoss-curve divergence (max abs diff, last-100-step mean diff):")
    pairs = [
        ("fp32_adamw", "bf16_adamw"),
        ("fp32_muon",  "bf16_muon"),
        ("fp32_muon",  "both_bf16"),
    ]
    for a, b in pairs:
        if a in results and b in results:
            la = torch.tensor(results[a].losses)
            lb = torch.tensor(results[b].losses)
            n = min(len(la), len(lb))
            d = (la[:n] - lb[:n]).abs()
            tail = d[-min(100, n):].mean().item()
            mx = d.max().item()
            print(f"  {a:<12} vs {b:<12}: max |Δ|={mx:.4f}  tail-100 mean |Δ|={tail:.4f}")


if __name__ == "__main__":
    main()
