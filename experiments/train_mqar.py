"""
MQAR training driver — tests recall ability across architectures.

The killer test for the `(R, c)` semidirect-product scan: can the
conjugated KV memory actually retrieve past tokens, the way DeltaNet/
linear-attention can but pure SO(n) and AUSSM cannot?

Usage:
    python experiments/train_mqar.py --arches linear,ortho,rotconj,deltanet \\
        --T 256 --steps 5000 --batch 256 --n_pairs 16 --vocab 64
"""
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import (
    LinearAttention, HeisenbergAttention, SoftmaxAttention,
    DeltaNetAttention, GatedDeltaNetAttention, Mamba2Attention,
    OrthogonalScanAttention, RotConjAttention,
)
from experiments.bf16_optim import BF16StateAdamW
from experiments.model import TinyLM
from experiments.speed_knobs import apply_speed_knobs
from experiments.tasks.mqar import make_batch as mqar_batch


ARCHES = {
    "linear":     LinearAttention,
    "heisenberg": HeisenbergAttention,
    "softmax":    SoftmaxAttention,
    "deltanet":   DeltaNetAttention,
    "gateddelta": GatedDeltaNetAttention,
    "mamba2":     Mamba2Attention,
    "ortho":      OrthogonalScanAttention,
    "rotconj":    RotConjAttention,
}


@dataclasses.dataclass
class RunResult:
    arch: str
    T: int
    n_pairs: int
    vocab: int
    steps: int
    final_train_loss: float
    final_val_loss: float
    recall_acc: float
    secs: float
    params: int
    label: str = ""


def _val(model, T, n_pairs, vocab, batch_size, device, use_memory=False):
    model.eval()
    with torch.no_grad():
        x, y, mask = mqar_batch(batch_size, T, vocab_size=vocab,
                                n_pairs=n_pairs, device=device)
        # Memory reads happen at query positions (where the mask is 1).
        read_mask = mask.bool() if use_memory else None
        logits = model(x, mem_read_mask=read_mask)
        # Slice to MQAR-vocab columns so the recall@1 / CE comparison is
        # fair across mem-on (vocab+1) and mem-off (vocab) builds.
        logits = logits[..., :vocab]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction="none",
        ).reshape(batch_size, T)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        preds = logits.argmax(dim=-1)
        correct = ((preds == y) & mask).float().sum()
        total = mask.sum().clamp_min(1)
        recall = (correct / total).item()
    model.train()
    return loss.item(), recall


def _parse_xattn_spec(spec: str) -> tuple:
    """Parse '2:8,9,10,11;4:8,9,10,11' → ((2,(8,9,10,11)),(4,(8,9,10,11)))."""
    if not spec:
        return ()
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        tgt_s, src_s = chunk.split(":")
        srcs = tuple(int(x) for x in src_s.split(",") if x.strip())
        out.append((int(tgt_s), srcs))
    return tuple(out)


def _parse_pairs_spec(spec: str) -> tuple:
    """Parse '2,28;3,27' → ((2,28),(3,27)).  Empty → ()."""
    if not spec:
        return ()
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        t_s, s_s = chunk.split(",")
        out.append((int(t_s), int(s_s)))
    return tuple(out)


def train_one(arch, T, n_pairs, vocab, steps, batch_size,
              d_model, n_layers, n_heads, d_head, lr, log_every,
              device="cuda", seed=0, feedback="none",
              use_memory=False, mem_size=1024, mem_dim=0,
              mem_read_alpha_init=1.0,
              feedback_pairs="", feedback_self_k=0,
              feedback_xattn="", feedback_xattn_form="attn",
              feedback_xattn_heads=4, label=None):
    torch.manual_seed(seed)
    cls = ARCHES[arch]
    # When use_memory=True we need a thinking_token_id slot; reserve the
    # last id of the vocab (MQAR's actual tokens are in [0, vocab-1] but the
    # generator only uses values < n_pairs*2 + queries, so the top slot is
    # always free). The token doesn't actually appear in MQAR inputs — we
    # drive reads via mem_read_mask instead — but the model needs a valid id.
    vocab_eff = vocab + (1 if use_memory else 0)
    thinking_id = vocab_eff - 1 if use_memory else None
    fb_pairs = _parse_pairs_spec(feedback_pairs)
    fb_xattn = _parse_xattn_spec(feedback_xattn)
    # Param-count check before allocating
    model = TinyLM(
        vocab_size=vocab_eff, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=cls,
        max_T=T,                              # learnable abs pos embed
        feedback_mode=feedback,
        feedback_pairs=fb_pairs,
        feedback_self_k=int(feedback_self_k),
        feedback_xattn_pairs=fb_xattn,
        feedback_xattn_form=feedback_xattn_form,
        feedback_xattn_heads=int(feedback_xattn_heads),
        use_memory=use_memory,
        mem_size=mem_size,
        mem_dim=mem_dim if mem_dim > 0 else d_model,
        mem_read_alpha_init=mem_read_alpha_init,
        thinking_token_id=thinking_id,
        activation_checkpointing=False,
    ).to(device)
    # NOTE: MQAR runs in fp32 (no bf16 autocast). MQAR's masked loss only
    # carries gradient at ~25% of positions, so the per-token gradient is
    # ~4× smaller than production pretrain (where every token contributes
    # via packed-sequence CE). At that signal level bf16's 7-bit mantissa
    # rounds the per-token update to noise and the model COLLAPSES to
    # uniform logits — verified by direct probe (Phase 1, 2026-05-16): a
    # 30L×576 + FiLM K=3 run at lr=1e-3 with bf16+ckpt drove logit_std from
    # 0.58 → 0.34 over 20 steps while loss stayed flat at 6.40. fp32
    # forward keeps things stable. apply_speed_knobs still gives us TF32 on
    # the residual fp32 matmuls (~free).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Plain fp32 AdamW — keep MQAR pipeline simple and noise-free.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=lr * 0.1,
    )

    tag = label if label else arch
    suffix = []
    if use_memory:
        suffix.append("+mem")
    if fb_pairs:
        suffix.append(f"+FiLM{fb_pairs}K={feedback_self_k}")
    if fb_xattn:
        suffix.append(f"+xattn[{feedback_xattn_form}]")
    print(f"\n[{tag}{''.join(suffix)}]  "
          f"L={n_layers} d={d_model}  T={T}  n_pairs={n_pairs}  vocab={vocab}  "
          f"params={model.num_params():,}")
    print(f"{'step':>6}  {'train_loss':>11}  {'val_loss':>9}  {'recall':>8}")

    t0 = time.perf_counter()
    last_train_loss = float("nan")
    for step in range(1, steps + 1):
        x, y, mask = mqar_batch(batch_size, T, vocab_size=vocab,
                                n_pairs=n_pairs, device=device)
        read_mask = mask.bool() if use_memory else None
        logits = model(x, mem_read_mask=read_mask)
        # Logits cover vocab_eff slots; CE targets MQAR tokens in [0, vocab).
        # Take only the MQAR-vocab columns for CE (the extra slot for the
        # thinking-token id never appears in targets).
        loss_logits = logits[..., :vocab] if use_memory else logits
        loss = F.cross_entropy(
            loss_logits.reshape(-1, vocab), y.reshape(-1), reduction="none",
        ).reshape(batch_size, T)
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        last_train_loss = loss.item()

        if step % log_every == 0 or step == steps:
            # Use the training batch size for periodic val so we don't OOM
            # at wider-d configs (the original 512/1024 eval batches were
            # tuned for the 128d MQAR experiments).
            v_loss, recall = _val(model, T, n_pairs, vocab, batch_size,
                                   device, use_memory=use_memory)
            extra = ""
            if use_memory:
                extra = f"  memα={model.memory.read_alpha.item():+.4f}"
            print(f"{step:>6d}  {last_train_loss:>11.4f}  "
                  f"{v_loss:>9.4f}  {recall:>8.3f}{extra}")

    v_loss, recall = _val(model, T, n_pairs, vocab,
                           min(2 * batch_size, 256), device,
                           use_memory=use_memory)
    secs = time.perf_counter() - t0
    return RunResult(
        arch=arch, T=T, n_pairs=n_pairs, vocab=vocab, steps=steps,
        final_train_loss=last_train_loss, final_val_loss=v_loss,
        recall_acc=recall, secs=secs, params=model.num_params(),
        label=tag,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arches", type=str, default="linear,ortho,rotconj,deltanet")
    p.add_argument("--T", type=int, nargs="+", default=[256])
    p.add_argument("--n_pairs", type=int, default=16)
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feedback", type=str, default="none",
                   choices=["none", "additive", "film", "predictive"])
    p.add_argument("--use_memory", action="store_true",
                   help="Add the bounded working-memory module; reads at "
                        "query positions.")
    p.add_argument("--mem_size", type=int, default=256,
                   help="Max entries in the write-gated memory buffer.")
    p.add_argument("--mem_dim", type=int, default=0,
                   help="0 = use d_model.")
    p.add_argument("--mem_read_alpha_init", type=float, default=1.0,
                   help="Init for the WM read-injection α gate. 0.0 = "
                        "zero-init-residual bootstrap (FiLM-α pattern); "
                        "1.0 = legacy un-gated injection.")
    args = p.parse_args()

    arches = args.arches.split(",")
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"capability={torch.cuda.get_device_capability(0)}")

    results = []
    for T in args.T:
        for a in arches:
            r = train_one(
                arch=a, T=T, n_pairs=args.n_pairs, vocab=args.vocab,
                steps=args.steps, batch_size=args.batch,
                d_model=args.d_model, n_layers=args.n_layers,
                n_heads=args.n_heads, d_head=args.d_head,
                lr=args.lr, log_every=args.log_every, seed=args.seed,
                feedback=args.feedback,
                use_memory=args.use_memory,
                mem_size=args.mem_size,
                mem_dim=args.mem_dim,
                mem_read_alpha_init=args.mem_read_alpha_init,
            )
            results.append(r)

    print("\n" + "=" * 80)
    print(f"{'arch':<12} {'T':>5} {'pairs':>5} {'recall':>8} "
          f"{'val_loss':>9} {'params':>10} {'secs':>7}")
    print("-" * 80)
    for r in results:
        print(f"{r.arch:<12} {r.T:>5} {r.n_pairs:>5} {r.recall_acc:>8.3f} "
              f"{r.final_val_loss:>9.4f} {r.params:>10,} {r.secs:>7.1f}")

    print("\nMQAR ranking by recall:")
    by_T = {}
    for r in results:
        by_T.setdefault(r.T, []).append(r)
    for T, rs in sorted(by_T.items()):
        ordered = sorted(rs, key=lambda r: r.recall_acc, reverse=True)
        print(f"  T={T}:")
        for r in ordered:
            print(f"    {r.arch:<12}  recall={r.recall_acc:.3f}  "
                  f"loss={r.final_val_loss:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
