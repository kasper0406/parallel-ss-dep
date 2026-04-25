"""
Minimal language modelling training driver — practical demonstration.

The point: show that the hybrid scaffold trains stably on real text at
~135M-param scale and reaches competitive perplexity vs pure DeltaNet.

Uses TinyStories (HuggingFaceH4/TinyStories, ~500MB of children's
stories) tokenised with SmolLM2's tokeniser (49152 vocab). This is the
smallest realistic LM benchmark; if hybrid can't match DeltaNet here,
distillation is futile.

Usage:
    python experiments/train_lm.py --arch deltanet --steps 5000
    python experiments/train_lm.py --arch hybrid   --steps 5000
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from experiments.layers import (
    DeltaNetAttention, DeltaNetNegEigAttention,
    OrthogonalScanAttention,
)
from experiments.model import TinyLM


_NAME_TO_CLS = {
    "deltanet":   DeltaNetAttention,
    "deltanet_negeig": DeltaNetNegEigAttention,
    "ortho":      OrthogonalScanAttention,
}


def build_arch(name: str, n_layers: int):
    """Map an arch name to either a single attention class or a per-layer list.

    For shorthand patterns:
      hybrid        — alternating ortho/deltanet (50/50, ortho first).
      hybrid_25_75  — 1 ortho + 3 deltanet, repeating.
      hybrid_75_25  — 3 ortho + 1 deltanet, repeating.
    Or use --layers for an explicit comma-separated list.
    """
    if name in _NAME_TO_CLS:
        return dict(attention_cls=_NAME_TO_CLS[name])
    if name == "hybrid":
        cls = [OrthogonalScanAttention if i % 2 == 0 else DeltaNetAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_25_75":
        # 1 ortho + 3 deltanet pattern, ortho at every 4th position.
        cls = [OrthogonalScanAttention if i % 4 == 0 else DeltaNetAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_75_25":
        # 3 ortho + 1 deltanet pattern.
        cls = [DeltaNetAttention if i % 4 == 0 else OrthogonalScanAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_negeig":
        cls = [OrthogonalScanAttention if i % 2 == 0 else DeltaNetNegEigAttention
               for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    raise ValueError(f"unknown arch: {name}")


def parse_layers_arg(spec: str) -> list:
    """Parse comma-separated --layers spec into a class list."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [_NAME_TO_CLS[p] for p in parts]


class TokenisedStream(IterableDataset):
    """Streaming IterableDataset of fixed-length tokenised chunks."""

    def __init__(self, dataset, tokenizer, block_size, text_field="text",
                 shuffle_buffer=1024):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        buf: list[int] = []
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.bos_token_id
        if eos is None:
            eos = 0
        for example in self.dataset:
            text = example[self.text_field]
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= self.block_size + 1:
                chunk = buf[: self.block_size + 1]
                buf = buf[self.block_size :]
                inputs = torch.tensor(chunk[:-1], dtype=torch.long)
                targets = torch.tensor(chunk[1:], dtype=torch.long)
                yield inputs, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default=None,
                   choices=["deltanet", "deltanet_negeig", "ortho",
                            "hybrid", "hybrid_25_75", "hybrid_75_25",
                            "hybrid_negeig"])
    p.add_argument("--layers", type=str, default=None,
                   help="explicit comma-separated layer arch list, "
                        "e.g. 'ortho,deltanet,deltanet,deltanet,ortho,...'. "
                        "Overrides --arch.")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--tokenizer", type=str,
                   default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                   help="HF dataset id; supports also 'codeparrot/codeparrot-clean' "
                        "and bigcode/the-stack-smol/data/python")
    p.add_argument("--dataset_config", type=str, default=None,
                   help="Optional config name (e.g. 'python' for the-stack-smol)")
    p.add_argument("--text_field", type=str, default="text",
                   help="Field in the dataset that contains the text "
                        "('text' for TinyStories, 'content' for codeparrot/the-stack)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    arch_label = args.arch if args.arch else f"layers={args.layers}"
    print(f"GPU: {torch.cuda.get_device_name(0)}  arch={arch_label}")

    # 1. Tokeniser + dataset.
    from transformers import AutoTokenizer
    from datasets import load_dataset
    print(f"Loading tokeniser {args.tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  vocab size: {tok.vocab_size}")

    print(f"Loading dataset {args.dataset} (streaming) ...")
    ds_kwargs = dict(streaming=True)
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    try:
        train_stream = load_dataset(args.dataset, split="train", **ds_kwargs)
    except ValueError:
        # Some datasets only have "train".
        train_stream = load_dataset(args.dataset, **ds_kwargs)["train"]
    try:
        val_stream = load_dataset(args.dataset, split="validation", **ds_kwargs)
    except (ValueError, KeyError):
        # No validation split — split off a slice of train as held-out.
        # We just take a separate streaming pass with a different seed-shuffle.
        try:
            val_stream = load_dataset(args.dataset, split="test", **ds_kwargs)
        except (ValueError, KeyError):
            print("  no val/test split — using shuffled train tail as validation")
            val_stream = load_dataset(args.dataset, split="train",
                                      **ds_kwargs).shuffle(seed=42).skip(10_000)

    train_ds = TokenisedStream(train_stream, tok, args.T,
                               text_field=args.text_field)
    val_ds = TokenisedStream(val_stream, tok, args.T,
                             text_field=args.text_field)
    train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=1)

    # 2. Model.
    if args.layers:
        cls_list = parse_layers_arg(args.layers)
        n_layers_actual = len(cls_list)
        attn_kw = dict(attention_cls_per_layer=cls_list)
    else:
        if args.arch is None:
            raise SystemExit("specify --arch or --layers")
        attn_kw = build_arch(args.arch, args.n_layers)
        n_layers_actual = args.n_layers
    model = TinyLM(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_layers=n_layers_actual,
        n_heads=args.n_heads, d_head=args.d_head, **attn_kw,
    ).to("cuda")
    print(f"  params: {model.num_params() / 1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1,
    )

    # 3. Train loop.
    print(f"\n{'step':>6}  {'tok/s':>8}  {'tloss':>8}  {'lr':>9}")
    t0 = time.perf_counter()
    train_iter = iter(train_loader)
    last_log = t0
    last_log_step = 0
    losses = []
    for step in range(1, args.steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to("cuda"), y.to("cuda")
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, tok.vocab_size), y.reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        losses.append(loss.item())

        if step % args.log_every == 0 or step == args.steps:
            now = time.perf_counter()
            tok_per_sec = (step - last_log_step) * args.batch * args.T / (now - last_log)
            print(f"{step:>6d}  {tok_per_sec:>8.0f}  "
                  f"{sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:])):>8.4f}  "
                  f"{scheduler.get_last_lr()[0]:>9.2e}")
            last_log = now
            last_log_step = step

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to("cuda"), vy.to("cuda")
                    vlogits = model(vx)
                    vloss = F.cross_entropy(
                        vlogits.reshape(-1, tok.vocab_size), vy.reshape(-1),
                    )
                    val_loss += vloss.item() * vx.numel()
                    n_val += vx.numel()
                    if n_val >= 64 * args.T:                # cap val
                        break
            val_loss /= n_val
            ppl = float(torch.tensor(val_loss).exp())
            print(f"        VAL  loss={val_loss:.4f}  ppl={ppl:.2f}")
            model.train()

    secs = time.perf_counter() - t0
    print(f"\nDone in {secs:.0f}s ({secs/args.steps*1000:.0f} ms/step avg).")


if __name__ == "__main__":
    sys.exit(main())
