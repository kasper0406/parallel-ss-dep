"""
Lagged-cached vs 2-pass val PPL parity check (Phase 20.5 follow-up).

Settles whether the cheap lagged-cached decode protocol used in the
708 M sparse-(2, 34) FiLM checkpoint produces the same val PPL as the
training-faithful 2-pass forward. If yes, the ≤1 % decode-tax claim in
LATENCY_REPORT.md is fully defensible. If not, we accept the 2-pass
+99 % tax.

Two protocols are evaluated on the *same* deterministic 10 K-token
slice of the codeparrot tail (matching the training-time val tail —
`shuffle(seed=42).skip(10_000)`).

  1. **2-pass** (gold reference).  Calls model(x) which goes through
     the `feedback_pairs` 2-pass path used at training time. Should
     reproduce ~34.26 PPL within sampling noise.

  2. **Lagged-cached** (deployment proxy). For each T-length val
     chunk, runs a token-by-token streaming forward where at every
     step the FiLM at layer 2 reads the *previous step's pass-2
     layer-34 output* as a proxy for training's pass-1 layer-34
     output. At step 0 the lagged input is zeros (matching the
     `_shift_right_by_1` convention that training also uses).

Outputs CE / PPL under each plus the gap. Designed to be self-
contained and reproducible — does not modify any other module.
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
import torch.nn.functional as F
from fla.models.utils import Cache as FLACache

from experiments.decode_bench import (
    _block_with_cache, load_dn_or_film,
)


CKPT_FILM = "checkpoints/sparse_2_34_708M_muon.pt"


# ---------------------------------------------------------------------------
# Val tail — matches training-time val. codeparrot has no validation
# split, so train_lm.py uses `train.shuffle(seed=42).skip(10_000)`.
# ---------------------------------------------------------------------------


def make_val_chunks(tokenizer, T: int, n_tokens: int,
                     dataset: str = "codeparrot/codeparrot-clean",
                     text_field: str = "content",
                     skip: int = 10_000) -> torch.Tensor:
    """Return (N, T+1) int64 tensor of contiguous tokenised chunks,
    drawn from the codeparrot val tail. Total covered tokens = N*T ≥ n_tokens.

    Uses the EXACT same `TokenisedStream` machinery + `DataLoader` that
    `train_lm.py` uses, so the val tokens we score are bit-identical to
    those scored at training time. This reproduces the reported 34.26 PPL.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from experiments.train_lm import TokenisedStream
    val_stream = (
        load_dataset(dataset, split="train", streaming=True)
        .shuffle(seed=42)
        .skip(skip)
    )
    val_ds = TokenisedStream(val_stream, tokenizer, T, text_field=text_field)
    # Note: TokenisedStream yields (input, target) pairs of length T each;
    # we recombine them into (T+1)-length chunks for use by both the 2-pass
    # and lagged-cached eval paths (input = chunk[:-1], target = chunk[1:]).
    loader = DataLoader(val_ds, batch_size=1, num_workers=1)
    target = max(1, math.ceil(n_tokens / T))
    chunks: list[torch.Tensor] = []
    for x, y in loader:
        # x: (1, T), y: (1, T). y[t] should equal training chunk[t+1].
        # Concatenate x[0] with last token of y[0] to get the (T+1)-length
        # chunk that produced this (x, y).
        chunk = torch.cat([x[0], y[0, -1:]], dim=0)
        chunks.append(chunk)
        if len(chunks) >= target:
            break
    out = torch.stack(chunks, dim=0).to(torch.long)         # (N, T+1)
    return out


# ---------------------------------------------------------------------------
# 2-pass eval — the gold reference. Use the model's training-faithful
# forward. With `feedback_pairs=((2,34),)` the forward routes through
# the sparse-pair 2-pass path in TinyLM.forward.
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_2pass_ppl(model, chunks: torch.Tensor, batch: int = 4) -> tuple:
    """chunks: (N, T+1) int64 on CPU. Returns (mean_ce, n_tokens, ppl)."""
    device = next(model.parameters()).device
    losses_sum = 0.0
    n_tokens = 0
    N = chunks.shape[0]
    for i in range(0, N, batch):
        batch_chunks = chunks[i : i + batch].to(device)
        x = batch_chunks[:, :-1]
        y = batch_chunks[:, 1:]
        logits = model(x)
        # Per-token CE summed (so we can weight properly across batches).
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            reduction="sum",
        )
        losses_sum += float(loss.item())
        n_tokens += y.numel()
    mean_ce = losses_sum / n_tokens
    ppl = math.exp(mean_ce)
    return mean_ce, n_tokens, ppl


# ---------------------------------------------------------------------------
# Lagged-cached eval — token-by-token streaming, mirroring the
# decode_bench.py lagged-cached protocol exactly. Init lagged=zeros,
# starting cache empty.
# ---------------------------------------------------------------------------


@torch.no_grad()
def lagged_cached_logits_one_seq(model, tokens: torch.Tensor,
                                  target: int = 2, source: int = 34
                                  ) -> torch.Tensor:
    """tokens: (1, T) int64 on device. Returns logits (1, T, V).

    Streams the sequence one token at a time using the lagged-cached
    decode protocol. At step 0 the lagged source is zeros, matching
    the `_shift_right_by_1` convention of training. At step t>0 the
    FiLM input is the previous step's pass-2 layer-34 output.
    """
    device = tokens.device
    B, T = tokens.shape
    assert B == 1, "streaming impl is single-sequence"
    cache = FLACache(seen_tokens=0)
    d_model = model.embed.embedding_dim
    lagged = torch.zeros(B, 1, d_model, device=device)
    out_logits = torch.empty(B, T, model.lm_head.out_features,
                              device=device, dtype=model.lm_head.weight.dtype)
    for t in range(T):
        next_token = tokens[:, t : t + 1]
        x = model.embed(next_token)
        new_lagged_source_out = None
        for L, blk in enumerate(model.blocks):
            if L == target:
                x = model.sparse_feedback[str(target)](x, lagged)
            x = _block_with_cache(blk, x, past=cache, use_cache=True)
            if L == source:
                new_lagged_source_out = x
        h = model.out_norm(x)
        out_logits[:, t : t + 1] = model.lm_head(h)
        lagged = new_lagged_source_out
    return out_logits


@torch.no_grad()
def eval_lagged_cached_ppl(model, chunks: torch.Tensor) -> tuple:
    """Streaming PPL eval. chunks: (N, T+1) int64 on CPU."""
    device = next(model.parameters()).device
    target, source = model.feedback_pairs[0]
    losses_sum = 0.0
    n_tokens = 0
    N = chunks.shape[0]
    t_start = time.perf_counter()
    last_print = t_start
    for i in range(N):
        chunk = chunks[i : i + 1].to(device)
        x = chunk[:, :-1]
        y = chunk[:, 1:]
        logits = lagged_cached_logits_one_seq(model, x,
                                                target=target, source=source)
        # CE per chunk.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            y.reshape(-1),
            reduction="sum",
        )
        losses_sum += float(loss.item())
        n_tokens += y.numel()
        now = time.perf_counter()
        if now - last_print > 30 or i == N - 1:
            elapsed = now - t_start
            rate = (i + 1) * x.shape[1] / elapsed
            eta = (N - 1 - i) * x.shape[1] / max(rate, 1e-6)
            partial_ce = losses_sum / max(1, n_tokens)
            print(f"  [{i+1:3d}/{N}]  partial CE={partial_ce:.4f}  "
                  f"PPL={math.exp(partial_ce):.2f}  "
                  f"({rate:.0f} tok/s, ETA {eta:.0f}s)")
            last_print = now
    mean_ce = losses_sum / n_tokens
    ppl = math.exp(mean_ce)
    return mean_ce, n_tokens, ppl


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=CKPT_FILM)
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n_tokens", type=int, default=10_000,
                   help="Total target val tokens (T-aligned chunks rounded up).")
    p.add_argument("--batch_2pass", type=int, default=4,
                   help="Batch size for the 2-pass eval (parallel along N).")
    p.add_argument("--out", type=str, default="bench_film_ppl.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", type=str, default="both",
                   choices=["both", "2pass", "lagged"])
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading checkpoint: {args.ckpt}")
    model = load_dn_or_film(args.ckpt, device=args.device)
    cfg_d = {
        "d_model": model.embed.embedding_dim,
        "n_layers": len(model.blocks),
        "feedback_pairs": list(model.feedback_pairs),
        "feedback_mode": model.feedback_mode,
    }
    print(f"  config: {cfg_d}")

    print(f"Loading codeparrot val tail (skip=10_000) at T={args.T} ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    chunks = make_val_chunks(tok, T=args.T, n_tokens=args.n_tokens,
                              skip=10_000)
    n_chunks, _ = chunks.shape
    target_tokens = n_chunks * args.T
    print(f"  {n_chunks} chunks × T={args.T} = {target_tokens} eval tokens")

    results = {
        "ckpt": args.ckpt,
        "T": args.T,
        "n_chunks": n_chunks,
        "n_eval_tokens": target_tokens,
        "config": cfg_d,
    }

    if args.mode in ("both", "2pass"):
        print("\n--- 2-pass val PPL (training-faithful, gold reference) ---")
        t0 = time.perf_counter()
        mean_ce, n, ppl = eval_2pass_ppl(model, chunks, batch=args.batch_2pass)
        elapsed = time.perf_counter() - t0
        print(f"  CE  = {mean_ce:.4f}")
        print(f"  PPL = {ppl:.4f}")
        print(f"  ({n} tokens evaluated in {elapsed:.1f}s)")
        results["2pass"] = {"ce": mean_ce, "ppl": ppl, "n_tokens": n,
                              "elapsed_s": elapsed}

    if args.mode in ("both", "lagged"):
        print("\n--- Lagged-cached val PPL (deployment protocol) ---")
        t0 = time.perf_counter()
        mean_ce_l, n_l, ppl_l = eval_lagged_cached_ppl(model, chunks)
        elapsed_l = time.perf_counter() - t0
        print(f"  CE  = {mean_ce_l:.4f}")
        print(f"  PPL = {ppl_l:.4f}")
        print(f"  ({n_l} tokens evaluated in {elapsed_l:.1f}s)")
        results["lagged_cached"] = {"ce": mean_ce_l, "ppl": ppl_l,
                                     "n_tokens": n_l, "elapsed_s": elapsed_l}

    if args.mode == "both":
        gap_ce = results["lagged_cached"]["ce"] - results["2pass"]["ce"]
        gap_ppl_abs = results["lagged_cached"]["ppl"] - results["2pass"]["ppl"]
        gap_ppl_rel = gap_ppl_abs / results["2pass"]["ppl"]
        print("\n--- Comparison ---")
        print(f"  ΔCE          = {gap_ce:+.4f} nats/token")
        print(f"  ΔPPL (abs)   = {gap_ppl_abs:+.4f}")
        print(f"  ΔPPL (rel)   = {gap_ppl_rel*100:+.3f} %")
        results["gap"] = {
            "delta_ce_nats_per_tok": gap_ce,
            "delta_ppl_abs": gap_ppl_abs,
            "delta_ppl_rel": gap_ppl_rel,
        }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
