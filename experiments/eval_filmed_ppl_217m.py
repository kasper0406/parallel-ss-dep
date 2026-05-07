"""
Phase 21c — eval the 217 M self-feeding-trained sparse-(2, 28) FiLM model.

Three quantities are reported on a deterministic 1 K-token codeparrot val
slice (matching training's val tail: `train.shuffle(seed=42).skip(10_000)`,
T=512, batch 1):

  1. **Training-protocol PPL** — runs `model(x)` which executes the K-iter
     self-feeding forward (the model produces its own FiLM input from the
     previous iter's source-layer output). This is the PPL the training
     loss curves are measuring.

  2. **2-pass PPL** (gold reference) — disables self-feeding for the
     forward and runs the standard 2-pass FiLM (pass-1 vanilla, pass-2
     with lag-1 of pass-1 source). This matches the *unmodified* sparse-
     FiLM training protocol from Phase 17 / 20. Provides a sanity-check
     reference: a self-feeding-trained model ought to produce roughly
     the same PPL whether you run it through K-iter or standard 2-pass.

  3. **Lagged-cached deployment PPL** — token-by-token streaming forward
     where, at each step, the FiLM at the target layer reads the previous
     step's pass-2 source-layer output as a lag-1 proxy. This is the
     deployment cost-1× protocol. The whole point of self-feeding is to
     close the gap between this number and the training-protocol number.

Self-consistency check at end of training: ‖prev_iter.source - final.source‖
averaged over a val batch, normalised by ‖final.source‖. A small relative
norm (< ~5%) indicates the iteration has converged and lagged-cached
deployment will track the training distribution.

Outputs JSON with all three PPLs, the self-consistency norm, and gaps.

Usage:
    CUDA_VISIBLE_DEVICES=1 ./.venv/bin/python experiments/eval_filmed_ppl_217m.py \
        --ckpt checkpoints/film_self_k2_2_28_30L_217M.pt \
        --n_tokens 1024 \
        --out bench_film_self_k2_ppl.json
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

from experiments.layers import DeltaNetAttention
from experiments.model import (
    TinyLM, _shift_right_by_k, _shift_right_by_1,
)
from experiments.decode_bench import _block_with_cache


# ---------------------------------------------------------------------------
# Loader for 217M sparse-FiLM checkpoint, including possible self-feeding
# config field.
# ---------------------------------------------------------------------------


def load_film_217m(path: str, device: str = "cuda",
                    override_self_k: int | None = None) -> TinyLM:
    """Load a TinyLM 217M sparse-FiLM checkpoint, optionally overriding the
    self-feeding K (e.g. set to 0 for plain 2-pass eval).

    For surprise_modulated checkpoints, override_self_k is *ignored* — we
    always use the trained K because the surprise signal depends on having
    multiple no-grad iterations to compute the inter-iter delta. The
    fallback case (when no surprise tensor is provided, e.g. lagged-cached
    deploy) uses α₀·σ(bias) inside the FeedbackProjection forward.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    alpha_mode = cfg.get("feedback_alpha_mode", "scalar")
    self_k_ckpt = cfg.get("feedback_self_k", 0)
    if override_self_k is not None and alpha_mode == "surprise_modulated":
        print(f"  [info] surprise_modulated checkpoint: ignoring "
              f"override_self_k={override_self_k}, using trained K={self_k_ckpt}.")
        self_k = self_k_ckpt
    else:
        self_k = (override_self_k if override_self_k is not None
                  else self_k_ckpt)
    model = TinyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_head=cfg["d_head"],
        attention_cls=DeltaNetAttention,
        max_T=0,                     # DN has no positional embedding
        feedback_mode=cfg.get("feedback_mode", "film"),
        feedback_pairs=cfg.get("feedback_pairs", ()),
        feedback_self_k=self_k,
        feedback_alpha_mode=alpha_mode,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # Set layer_idx on every fla DeltaNet layer for cache plumbing.
    for L, blk in enumerate(model.blocks):
        blk.attn.layer.layer_idx = L
    return model


# ---------------------------------------------------------------------------
# Val tail — same as 708M eval. Reuses TokenisedStream from train_lm.py
# so the val tokens are bit-identical to those scored at training time.
# ---------------------------------------------------------------------------


def make_val_chunks(tokenizer, T: int, n_tokens: int,
                     dataset: str = "codeparrot/codeparrot-clean",
                     text_field: str = "content",
                     skip: int = 10_000) -> torch.Tensor:
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from experiments.train_lm import TokenisedStream
    val_stream = (
        load_dataset(dataset, split="train", streaming=True)
        .shuffle(seed=42)
        .skip(skip)
    )
    val_ds = TokenisedStream(val_stream, tokenizer, T, text_field=text_field)
    loader = DataLoader(val_ds, batch_size=1, num_workers=1)
    target = max(1, math.ceil(n_tokens / T))
    chunks: list[torch.Tensor] = []
    for x, y in loader:
        chunk = torch.cat([x[0], y[0, -1:]], dim=0)
        chunks.append(chunk)
        if len(chunks) >= target:
            break
    out = torch.stack(chunks, dim=0).to(torch.long)         # (N, T+1)
    return out


# ---------------------------------------------------------------------------
# Eval routines.
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_model_forward_ppl(model, chunks: torch.Tensor, batch: int = 4) -> tuple:
    """PPL using the model's currently-configured forward (training-protocol).

    For a self-feeding-trained model with feedback_self_k=K, this runs K
    iterations and computes loss on the last iter's logits. For a standard
    2-pass model, this runs the standard 2-pass forward.
    """
    device = next(model.parameters()).device
    losses_sum = 0.0
    n_tokens = 0
    N = chunks.shape[0]
    for i in range(0, N, batch):
        batch_chunks = chunks[i : i + batch].to(device)
        x = batch_chunks[:, :-1]
        y = batch_chunks[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
            reduction="sum",
        )
        losses_sum += float(loss.item())
        n_tokens += y.numel()
    mean_ce = losses_sum / n_tokens
    ppl = math.exp(mean_ce)
    return mean_ce, n_tokens, ppl


@torch.no_grad()
def lagged_cached_logits_one_seq(model, tokens: torch.Tensor,
                                  target: int, source: int) -> torch.Tensor:
    """Stream a (1, T) sequence one token at a time using the deployment
    lagged-cached protocol. Identical to the 708M eval's path but reused
    here. Returns logits (1, T, V).

    At step 0 the lagged FiLM input is zeros (matches `_shift_right_by_1`
    behavior at t=0 of training). At step t>0 the FiLM input at the target
    layer is the previous step's pass-2 source-layer output.
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


@torch.no_grad()
def eval_self_consistency(model, chunks: torch.Tensor,
                            batch: int = 4) -> dict:
    """Run the model's K-iter self-feeding forward and return self-consistency
    norms: ‖prev_iter_src - final_iter_src‖ averaged over a batch, both
    absolute and relative to ‖final_iter_src‖.

    Only meaningful when model.feedback_self_k > 0. For K=0 (standard 2-pass),
    returns empty dict.
    """
    if model.feedback_self_k == 0:
        return {"feedback_self_k": 0, "note": "no self-feeding; skipped"}
    device = next(model.parameters()).device
    abs_norms: list[float] = []
    rel_norms: list[float] = []
    final_norms: list[float] = []
    N = chunks.shape[0]
    for i in range(0, N, batch):
        batch_chunks = chunks[i : i + batch].to(device)
        x = batch_chunks[:, :-1]
        _ = model(x)
        prev = model._last_self_feed_prev_src    # iter K-1 (no_grad'd)
        final = model._last_self_feed_final_src  # iter K-1 (the final pass)
        for s in prev.keys():
            diff = (final[s] - prev[s]).reshape(-1).float()
            tgt = final[s].reshape(-1).float()
            abs_n = diff.norm().item()
            tgt_n = tgt.norm().item()
            abs_norms.append(abs_n)
            final_norms.append(tgt_n)
            rel_norms.append(abs_n / max(tgt_n, 1e-12))
    return {
        "feedback_self_k": model.feedback_self_k,
        "n_batches": len(abs_norms),
        "abs_norm_mean": float(sum(abs_norms) / len(abs_norms)),
        "final_norm_mean": float(sum(final_norms) / len(final_norms)),
        "rel_norm_mean": float(sum(rel_norms) / len(rel_norms)),
    }


@torch.no_grad()
def collect_surprise_alpha_distribution(model, chunks: torch.Tensor,
                                          batch: int = 4) -> dict:
    """For surprise_modulated checkpoints, collect surprise(t) and α(t)
    distributions over the val slice.

    Returns a dict with per-target histograms (numpy lists), summary
    stats (mean/std/min/max/p5/p50/p95), and ASCII plots for quick eyeballing.
    Only meaningful when model.feedback_alpha_mode == 'surprise_modulated'.
    Returns an empty dict for scalar α models.
    """
    if getattr(model, "feedback_alpha_mode", "scalar") != "surprise_modulated":
        return {"feedback_alpha_mode": "scalar",
                "note": "scalar α; no per-token distribution to collect"}
    if model.feedback_self_k == 0:
        return {"feedback_self_k": 0,
                "note": "no self-feeding; cannot compute surprise"}
    import numpy as np
    device = next(model.parameters()).device
    surprise_acc: dict = {}
    alpha_t_acc: dict = {}
    N = chunks.shape[0]
    for i in range(0, N, batch):
        batch_chunks = chunks[i : i + batch].to(device)
        x = batch_chunks[:, :-1]
        _ = model(x)
        for t, sup in model._last_surprise_per_target.items():
            surprise_acc.setdefault(t, []).append(sup.float().reshape(-1).cpu().numpy())
        for t, at in model._last_alpha_t_per_target.items():
            alpha_t_acc.setdefault(t, []).append(at.float().reshape(-1).cpu().numpy())
    out: dict = {"feedback_alpha_mode": "surprise_modulated", "per_target": {}}
    for t in surprise_acc:
        sup_all = np.concatenate(surprise_acc[t])
        at_all = np.concatenate(alpha_t_acc[t])

        def _summary(a):
            qs = np.quantile(a.astype(np.float64),
                              [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
            return {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "min": float(a.min()),
                "max": float(a.max()),
                "p5": qs[0], "p25": qs[1], "p50": qs[2],
                "p75": qs[3], "p95": qs[4],
                "n": int(a.size),
            }

        def _ascii_hist(a, n_bins=20, width=40, label=""):
            counts, edges = np.histogram(a.astype(np.float64), bins=n_bins)
            mx = max(counts.max(), 1)
            lines = [f"  {label} histogram (n={a.size}, "
                     f"min={a.min():.4f}, max={a.max():.4f}):"]
            for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                bar = "█" * int(width * c / mx) if mx > 0 else ""
                lines.append(f"    {lo:8.4f} – {hi:8.4f} | {c:6d} {bar}")
            return "\n".join(lines)

        out["per_target"][int(t)] = {
            "surprise": _summary(sup_all),
            "alpha_t": _summary(at_all),
            "surprise_hist_ascii": _ascii_hist(sup_all, label="surprise(t)"),
            "alpha_t_hist_ascii": _ascii_hist(at_all, label="α(t)"),
            "surprise_hist": np.histogram(
                sup_all.astype(np.float64), bins=40)[0].tolist(),
            "surprise_hist_edges": np.histogram(
                sup_all.astype(np.float64), bins=40)[1].tolist(),
            "alpha_t_hist": np.histogram(
                at_all.astype(np.float64), bins=40)[0].tolist(),
            "alpha_t_hist_edges": np.histogram(
                at_all.astype(np.float64), bins=40)[1].tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to the 217M sparse-FiLM checkpoint.")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--n_tokens", type=int, default=1024,
                   help="Total target val tokens (T-aligned chunks rounded up).")
    p.add_argument("--batch", type=int, default=2,
                   help="Batch size for non-streaming evals.")
    p.add_argument("--out", type=str, default="bench_film_self_ppl.json")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading checkpoint: {args.ckpt}")

    # First load with the checkpoint's own self-K config (training-protocol
    # PPL & self-consistency).
    model_self = load_film_217m(args.ckpt, device=args.device)
    cfg_d = {
        "d_model": model_self.embed.embedding_dim,
        "n_layers": len(model_self.blocks),
        "feedback_pairs": list(model_self.feedback_pairs),
        "feedback_mode": model_self.feedback_mode,
        "feedback_self_k": model_self.feedback_self_k,
        "feedback_alpha_mode": getattr(model_self, "feedback_alpha_mode", "scalar"),
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

    # 1. Training-protocol PPL — uses the model's K-iter self-feeding forward.
    print("\n--- Training-protocol PPL (model's K-iter self-feeding forward) ---")
    t0 = time.perf_counter()
    ce, n, ppl = eval_model_forward_ppl(model_self, chunks, batch=args.batch)
    elapsed = time.perf_counter() - t0
    print(f"  CE  = {ce:.4f}")
    print(f"  PPL = {ppl:.4f}")
    print(f"  ({n} tokens, {elapsed:.1f}s)")
    results["training_protocol"] = {
        "ce": ce, "ppl": ppl, "n_tokens": n, "elapsed_s": elapsed,
    }

    # 2. Self-consistency norms.
    print("\n--- Self-consistency norms (prev_iter vs final_iter source state) ---")
    t0 = time.perf_counter()
    sc = eval_self_consistency(model_self, chunks, batch=args.batch)
    elapsed_sc = time.perf_counter() - t0
    print(f"  {sc}")
    print(f"  ({elapsed_sc:.1f}s)")
    results["self_consistency"] = sc

    # 2b. surprise(t) and α(t) distributions for surprise_modulated
    # checkpoints. Empty dict for scalar-α checkpoints.
    if getattr(model_self, "feedback_alpha_mode", "scalar") == "surprise_modulated":
        print("\n--- surprise(t) and α(t) distributions ---")
        t0 = time.perf_counter()
        dist = collect_surprise_alpha_distribution(model_self, chunks,
                                                     batch=args.batch)
        elapsed_d = time.perf_counter() - t0
        # Print ASCII histograms for quick eyeballing.
        for t, target_d in dist.get("per_target", {}).items():
            print(f"  Target layer {t}:")
            print(f"    surprise(t): "
                  f"mean={target_d['surprise']['mean']:.4f}  "
                  f"std={target_d['surprise']['std']:.4f}  "
                  f"min={target_d['surprise']['min']:.4f}  "
                  f"max={target_d['surprise']['max']:.4f}  "
                  f"p50={target_d['surprise']['p50']:.4f}  "
                  f"p95={target_d['surprise']['p95']:.4f}")
            print(f"    α(t):        "
                  f"mean={target_d['alpha_t']['mean']:.4f}  "
                  f"std={target_d['alpha_t']['std']:.4f}  "
                  f"min={target_d['alpha_t']['min']:.4f}  "
                  f"max={target_d['alpha_t']['max']:.4f}  "
                  f"p50={target_d['alpha_t']['p50']:.4f}  "
                  f"p95={target_d['alpha_t']['p95']:.4f}")
            print(target_d["surprise_hist_ascii"])
            print(target_d["alpha_t_hist_ascii"])
        print(f"  ({elapsed_d:.1f}s)")
        results["surprise_alpha_distribution"] = dist

    # 3. 2-pass PPL — same model weights, but with feedback_self_k=0 (standard
    # 2-pass forward). Sanity-checks that self-feeding-trained weights still
    # produce a coherent PPL when run in 2-pass mode.
    print("\n--- 2-pass PPL (gold reference, training-faithful 2-pass) ---")
    del model_self
    torch.cuda.empty_cache()
    model_2pass = load_film_217m(args.ckpt, device=args.device, override_self_k=0)
    t0 = time.perf_counter()
    ce_2p, n_2p, ppl_2p = eval_model_forward_ppl(model_2pass, chunks, batch=args.batch)
    elapsed_2p = time.perf_counter() - t0
    print(f"  CE  = {ce_2p:.4f}")
    print(f"  PPL = {ppl_2p:.4f}")
    print(f"  ({n_2p} tokens, {elapsed_2p:.1f}s)")
    results["2pass"] = {
        "ce": ce_2p, "ppl": ppl_2p, "n_tokens": n_2p, "elapsed_s": elapsed_2p,
    }

    # 4. Lagged-cached PPL — token-by-token streaming with previous-step
    # pass-2 source as the lagged FiLM input. Reuses the same model_2pass
    # (the cell forward is identical; the FiLM module is the same params).
    print("\n--- Lagged-cached deployment PPL (1× decode cost protocol) ---")
    t0 = time.perf_counter()
    ce_lc, n_lc, ppl_lc = eval_lagged_cached_ppl(model_2pass, chunks)
    elapsed_lc = time.perf_counter() - t0
    print(f"  CE  = {ce_lc:.4f}")
    print(f"  PPL = {ppl_lc:.4f}")
    print(f"  ({n_lc} tokens, {elapsed_lc:.1f}s)")
    results["lagged_cached"] = {
        "ce": ce_lc, "ppl": ppl_lc, "n_tokens": n_lc, "elapsed_s": elapsed_lc,
    }

    # 5. Comparisons.
    gap_lc_vs_train_abs = ppl_lc - ppl
    gap_lc_vs_train_rel = gap_lc_vs_train_abs / ppl
    gap_lc_vs_2p_abs = ppl_lc - ppl_2p
    gap_lc_vs_2p_rel = gap_lc_vs_2p_abs / ppl_2p
    print("\n--- Comparison ---")
    print(f"  PPL (training-protocol K={cfg_d['feedback_self_k']})  = {ppl:.4f}")
    print(f"  PPL (2-pass gold)               = {ppl_2p:.4f}")
    print(f"  PPL (lagged-cached deployment)  = {ppl_lc:.4f}")
    print(f"  ΔPPL lagged vs training-protocol: {gap_lc_vs_train_abs:+.4f} "
          f"({gap_lc_vs_train_rel*100:+.2f} %)")
    print(f"  ΔPPL lagged vs 2-pass:            {gap_lc_vs_2p_abs:+.4f} "
          f"({gap_lc_vs_2p_rel*100:+.2f} %)")
    results["gap"] = {
        "lc_vs_training_abs": gap_lc_vs_train_abs,
        "lc_vs_training_rel": gap_lc_vs_train_rel,
        "lc_vs_2pass_abs": gap_lc_vs_2p_abs,
        "lc_vs_2pass_rel": gap_lc_vs_2p_rel,
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
