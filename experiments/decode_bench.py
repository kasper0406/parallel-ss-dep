"""
Decode-latency, prefill-latency, and inference-state memory benchmark for
708M DN baseline vs 708M sparse-(2,34) FiLM DN, with a 360M Transformer
reference for cross-architecture deployment-memory comparison.

Implements a custom decode loop:

  For DN-baseline:
    - Prefill: walk every TinyLM block once, threading an fla `Cache`
      through each `DeltaNet` layer. After prefill, the cache holds the
      recurrent + conv state at every layer.
    - Decode step: feed the current single token through the same per-block
      walk, propagating the cache.

  For sparse-(2,34) FiLM:
    - Prefill: do 2-pass forward as in training. Pass-1 (vanilla) collects
      layer-34 output. Pass-2 (with FiLM at layer 2 input) is the "real"
      forward; capture the cache + last position layer-34 output. Note
      pass-1 only needs layers 0..34, so we run a partial pass-1.
    - Decode step: at the *current* decode step we apply the FiLM to layer
      2's input using the LAGGED pass-2 layer-34 output cached from the
      previous decode step (an approximation — at decode time we no longer
      have a separate pass-1, so we use pass-2's lagged output as a proxy.
      Cost: ~1× model forward per decode step, NOT 2×).

  For Transformer:
    - Standard KV-cache. Each `SoftmaxAttention` is run incrementally with
      its own KV state stored externally.

Outputs:
    - decode_results.json with median + p95 ms/token at T = 512, 2048, 4096, 8192
    - prefill_results.json with tokens/s
    - state_memory.json with end-of-context state size in MB

Usage:
    python experiments/decode_bench.py --bench decode --T 4096
    python experiments/decode_bench.py --bench prefill --T 8192
    python experiments/decode_bench.py --bench memory  # analytical
    python experiments/decode_bench.py --bench all     # everything
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.models.utils import Cache as FLACache

from experiments.layers import DeltaNetAttention, SoftmaxAttention
from experiments.model import (
    Block, FeedbackProjection, GLU, RMSNorm, TinyLM, _shift_right_by_1,
)


CKPT_DN = "checkpoints/dn_36L_708M_muon.pt"
CKPT_FILM = "checkpoints/sparse_2_34_708M_muon.pt"
CKPT_TX = "checkpoints/transformer_30L_360M_muon.pt"

CONTEXT_LENGTHS = (512, 2048, 4096, 8192)


# ---------------------------------------------------------------------------
# Model loading.
# ---------------------------------------------------------------------------


def load_dn_or_film(path: str, device: str = "cuda") -> TinyLM:
    """Load a TinyLM checkpoint (DN baseline or sparse-FiLM DN)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TinyLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_head=cfg["d_head"],
        attention_cls=DeltaNetAttention,
        max_T=0,                     # DN has no positional embedding
        feedback_mode=cfg["feedback_mode"],
        feedback_pairs=cfg["feedback_pairs"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.thinking_token_id = cfg.get("thinking_token_id")
    model.eval()
    # Set layer_idx on every fla DeltaNet layer for cache plumbing.
    for L, blk in enumerate(model.blocks):
        # blk.attn is a `_FlaWrapper`; the actual fla layer is blk.attn.layer.
        blk.attn.layer.layer_idx = L
    return model


def load_or_build_transformer(device: str = "cuda",
                               max_T: int = 8192,
                               from_ckpt: bool = False) -> TinyLM:
    """Build a Transformer reference at 360M shape.

    If from_ckpt=True, attempt to load the 360M Muon Transformer checkpoint
    and tile/extrapolate the position embedding to max_T (random for
    positions beyond 512). The checkpoint's quality is not retained at
    long T, but for *latency reference* this is fine and noted in the
    report.
    """
    if from_ckpt and os.path.exists(CKPT_TX):
        ckpt = torch.load(CKPT_TX, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        d_model, n_layers, n_heads, d_head = (
            cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_head"]
        )
        cfg_max_T = max(max_T, cfg["max_T"])
        model = TinyLM(
            vocab_size=cfg["vocab_size"],
            d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, d_head=d_head,
            attention_cls=SoftmaxAttention,
            max_T=cfg_max_T,
        ).to(device)
        # Load matching params; pad pos_embed to cfg_max_T.
        sd = ckpt["state_dict"]
        # Extend pos_embed
        old_pos = sd["pos_embed.weight"]
        if old_pos.shape[0] < cfg_max_T:
            new_pos = torch.empty(cfg_max_T, d_model, device=old_pos.device,
                                   dtype=old_pos.dtype)
            new_pos[:old_pos.shape[0]] = old_pos
            # Random extrapolation for positions beyond the trained range.
            torch.nn.init.normal_(new_pos[old_pos.shape[0]:],
                                   std=old_pos.std().item())
            sd["pos_embed.weight"] = new_pos
        model.load_state_dict(sd)
        model.eval()
        return model
    # Random init at 360M shape, max_T sized for the longest context.
    d_model, n_layers, n_heads, d_head = 768, 30, 12, 64
    model = TinyLM(
        vocab_size=49152,
        d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head,
        attention_cls=SoftmaxAttention,
        max_T=max_T,
    ).to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Decode / prefill loops for the DN+FiLM TinyLM family.
# ---------------------------------------------------------------------------


def _block_with_cache(blk: Block, x: torch.Tensor,
                       past: FLACache | None,
                       use_cache: bool) -> torch.Tensor:
    """Run one TinyLM block, threading past_key_values through the fla cell."""
    # blk.attn is a `_FlaWrapper`; blk.attn.layer is the fla DeltaNet.
    attn_in = blk.attn_norm(x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = blk.attn.layer(
            hidden_states=attn_in,
            past_key_values=past,
            use_cache=use_cache,
        )
    if isinstance(out, tuple):
        attn_out = out[0]
    else:
        attn_out = out
    attn_out = attn_out.to(x.dtype)
    x = x + attn_out
    x = x + blk.mlp(blk.mlp_norm(x))
    return x


@torch.inference_mode()
def stateful_prefill_dn(model: TinyLM,
                         input_ids: torch.Tensor) -> tuple:
    """Run prefill on the DN baseline, return (last_logits, cache).

    The full forward is one pass since there is no FiLM.
    """
    assert model.feedback_mode == "none"
    cache = FLACache(seen_tokens=0)
    x = model.embed(input_ids)
    for L, blk in enumerate(model.blocks):
        x = _block_with_cache(blk, x, past=cache, use_cache=True)
    h = model.out_norm(x)
    logits = model.lm_head(h[:, -1:])
    return logits, cache


@torch.inference_mode()
def stateful_prefill_film(model: TinyLM,
                           input_ids: torch.Tensor,
                           target: int = 2,
                           source: int = 34) -> tuple:
    """Run prefill on the sparse-FiLM DN, return (last_logits, cache,
    last_layer_source_pass2).

    Frees pass-1 intermediates before returning so the steady-state
    inference memory holds only the per-layer cache + the (1, 1, D)
    lagged-source tensor.
    """
    assert model.feedback_pairs == ((target, source),)
    # Pass 1: vanilla forward, collect output at the source layer.
    # NOTE: pass-1 does NOT need to populate the cache (we discard it).
    # But pass-1 also only needs layers 0..source (and we run through
    # `source` to get its output). Saves running layers source+1..N-1 in pass-1.
    x_p1 = model.embed(input_ids)
    pass1_dummy_cache = FLACache(seen_tokens=0)
    pass1_source_out = None
    for L, blk in enumerate(model.blocks):
        x_p1 = _block_with_cache(blk, x_p1, past=pass1_dummy_cache, use_cache=False)
        if L == source:
            pass1_source_out = x_p1
            break        # pass-1 only needed up to the source layer
    assert pass1_source_out is not None
    # Lag-1 source state for pass-2 FiLM at layer `target`.
    src_state_lagged = _shift_right_by_1(pass1_source_out)
    # Free pass-1 intermediates now that we've snapshotted the lagged state.
    del x_p1, pass1_source_out, pass1_dummy_cache

    # Pass 2: real forward with FiLM at layer `target` input. Cache the
    # recurrent state at every layer.
    cache = FLACache(seen_tokens=0)
    x = model.embed(input_ids)
    last_pass2_source_out = None
    for L, blk in enumerate(model.blocks):
        if L == target:
            x = model.sparse_feedback[str(target)](x, src_state_lagged)
        x = _block_with_cache(blk, x, past=cache, use_cache=True)
        if L == source:
            # Take only the last position; clone so we don't hold the full
            # (1, T, D) tensor alive through the next block's residual chain.
            last_pass2_source_out = x[:, -1:].clone()
    assert last_pass2_source_out is not None
    # Free src_state_lagged once we've consumed it.
    del src_state_lagged
    h = model.out_norm(x)
    logits = model.lm_head(h[:, -1:])
    del x, h
    return logits, cache, last_pass2_source_out


@torch.inference_mode()
def decode_step_dn(model: TinyLM,
                    next_token: torch.Tensor,
                    cache: FLACache) -> torch.Tensor:
    """One decode step on the DN baseline. Returns logits (B, 1, V)."""
    x = model.embed(next_token)
    for L, blk in enumerate(model.blocks):
        x = _block_with_cache(blk, x, past=cache, use_cache=True)
    h = model.out_norm(x)
    logits = model.lm_head(h)
    return logits


@torch.inference_mode()
def decode_step_film(model: TinyLM,
                      next_token: torch.Tensor,
                      cache: FLACache,
                      lagged_source_out: torch.Tensor,
                      target: int = 2,
                      source: int = 34) -> tuple:
    """One decode step on the sparse-FiLM DN with the lagged-cached source.

    Returns (logits, new_lagged_source_out) where new_lagged_source_out
    is *this* step's pass-2 output at the source layer (will be the lag
    input for the next step).
    """
    x = model.embed(next_token)
    new_lagged_source_out = None
    for L, blk in enumerate(model.blocks):
        if L == target:
            # Apply FiLM with the lagged source state.
            x = model.sparse_feedback[str(target)](x, lagged_source_out)
        x = _block_with_cache(blk, x, past=cache, use_cache=True)
        if L == source:
            new_lagged_source_out = x      # (B, 1, D)
    assert new_lagged_source_out is not None
    h = model.out_norm(x)
    logits = model.lm_head(h)
    return logits, new_lagged_source_out


@torch.inference_mode()
def stateful_prefill_film_2pass(model: TinyLM,
                                 input_ids: torch.Tensor,
                                 target: int = 2,
                                 source: int = 34) -> tuple:
    """Prefill that returns *two* caches (pass-1 cache + pass-2 cache),
    plus the lag-1 source state. Used by the 2-pass decode protocol.
    Pass-1 cache only covers layers 0..source.

    Returns: (last_logits, cache_p1, cache_p2, last_pass1_source_out)
    """
    assert model.feedback_pairs == ((target, source),)
    cache_p1 = FLACache(seen_tokens=0)
    x_p1 = model.embed(input_ids)
    pass1_source_out = None
    for L, blk in enumerate(model.blocks):
        x_p1 = _block_with_cache(blk, x_p1, past=cache_p1, use_cache=True)
        if L == source:
            pass1_source_out = x_p1
            break
    assert pass1_source_out is not None
    src_state_lagged = _shift_right_by_1(pass1_source_out)
    last_pass1_source_out = pass1_source_out[:, -1:].clone()
    del x_p1, pass1_source_out

    cache_p2 = FLACache(seen_tokens=0)
    x = model.embed(input_ids)
    for L, blk in enumerate(model.blocks):
        if L == target:
            x = model.sparse_feedback[str(target)](x, src_state_lagged)
        x = _block_with_cache(blk, x, past=cache_p2, use_cache=True)
    del src_state_lagged
    h = model.out_norm(x)
    logits = model.lm_head(h[:, -1:])
    del x, h
    return logits, cache_p1, cache_p2, last_pass1_source_out


@torch.inference_mode()
def decode_step_film_2pass(model: TinyLM,
                            next_token: torch.Tensor,
                            cache_p1: FLACache,
                            cache_p2: FLACache,
                            lagged_pass1_source: torch.Tensor,
                            target: int = 2,
                            source: int = 34) -> tuple:
    """Pessimistic 2-pass decode: at each step, run pass-1 over the new
    token through layers 0..source (with cache_p1) to get *this* step's
    pass-1 source output, then run pass-2 through all layers (with cache_p2)
    using the PREVIOUS step's pass-1 source as the lag-1 FiLM input.

    Mirrors training exactly — the lag-1 input is true pass-1 (not the
    pass-2 proxy used in `decode_step_film`).

    Returns: (logits, new_lagged_pass1_source)
    """
    # Pass 1: walk layers 0..source on the new token.
    x_p1 = model.embed(next_token)
    new_pass1_source_out = None
    for L, blk in enumerate(model.blocks):
        x_p1 = _block_with_cache(blk, x_p1, past=cache_p1, use_cache=True)
        if L == source:
            new_pass1_source_out = x_p1.clone()    # (B, 1, D), pass-1 at this step
            break

    # Pass 2: walk all layers with FiLM at `target` reading the PREVIOUS
    # step's pass-1 source (the true lag-1 input).
    x = model.embed(next_token)
    for L, blk in enumerate(model.blocks):
        if L == target:
            x = model.sparse_feedback[str(target)](x, lagged_pass1_source)
        x = _block_with_cache(blk, x, past=cache_p2, use_cache=True)
    h = model.out_norm(x)
    logits = model.lm_head(h)
    return logits, new_pass1_source_out


@torch.inference_mode()
def _decode_pass1_step(model: TinyLM,
                        next_token: torch.Tensor,
                        cache_p1: FLACache,
                        source: int = 34) -> torch.Tensor:
    """Run only pass-1 (layers 0..source) on a single new token. Returns
    (B, 1, D) — pass-1 source-layer output at this step (i.e. the lag-1
    FiLM input for *next* step's pass-2).
    """
    x_p1 = model.embed(next_token)
    new_pass1_source_out = None
    for L, blk in enumerate(model.blocks):
        x_p1 = _block_with_cache(blk, x_p1, past=cache_p1, use_cache=True)
        if L == source:
            new_pass1_source_out = x_p1.clone()
            break
    assert new_pass1_source_out is not None
    return new_pass1_source_out


@torch.inference_mode()
def _decode_pass2_step(model: TinyLM,
                        next_token: torch.Tensor,
                        cache_p2: FLACache,
                        lagged_pass1_source: torch.Tensor,
                        target: int = 2) -> torch.Tensor:
    """Run only pass-2 (all layers, with FiLM at `target`) on a single
    new token. Returns logits (B, 1, V).
    """
    x = model.embed(next_token)
    for L, blk in enumerate(model.blocks):
        if L == target:
            x = model.sparse_feedback[str(target)](x, lagged_pass1_source)
        x = _block_with_cache(blk, x, past=cache_p2, use_cache=True)
    h = model.out_norm(x)
    logits = model.lm_head(h)
    return logits


@torch.inference_mode()
def decode_step_film_2pass_overlap(model: TinyLM,
                                    next_token: torch.Tensor,
                                    cache_p1: FLACache,
                                    cache_p2: FLACache,
                                    lagged_pass1_source: torch.Tensor,
                                    stream_pass1: torch.cuda.Stream,
                                    stream_pass2: torch.cuda.Stream,
                                    target: int = 2,
                                    source: int = 34) -> tuple:
    """Async-overlapped 2-pass decode step.

    Pass-1 and pass-2 at the same decode step are mutually independent
    (pass-2 reads the *previous* step's pass-1 output via
    `lagged_pass1_source`). We dispatch them on two CUDA streams so they
    can co-execute on the SMs.

    Logical equivalence to `decode_step_film_2pass`:
      - Same input cache state, same `lagged_pass1_source`,
        same `next_token` ⇒ produces same `(logits, new_pass1_source)`.

    Returns: (logits, new_lagged_pass1_source)
    """
    current_stream = torch.cuda.current_stream()

    # Both new streams need to wait for any in-flight work on the current
    # stream that produced our inputs (cache state, lagged_pass1_source,
    # next_token).
    stream_pass1.wait_stream(current_stream)
    stream_pass2.wait_stream(current_stream)

    # Pass 2 (the real model output) on stream_pass2.
    with torch.cuda.stream(stream_pass2):
        logits = _decode_pass2_step(
            model, next_token, cache_p2, lagged_pass1_source, target=target,
        )

    # Pass 1 (produces lag-1 FiLM input for *next* step) on stream_pass1.
    with torch.cuda.stream(stream_pass1):
        new_pass1_source_out = _decode_pass1_step(
            model, next_token, cache_p1, source=source,
        )

    # Make the current stream wait for both branches to finish before the
    # caller consumes their outputs (logits, new_pass1_source_out, and the
    # mutated caches).
    current_stream.wait_stream(stream_pass2)
    current_stream.wait_stream(stream_pass1)
    return logits, new_pass1_source_out


# ---------------------------------------------------------------------------
# Decode / prefill loops for the Transformer reference.
#
# We don't have plumbing in `SoftmaxAttention` for KV cache reuse, so we use
# a simple manual KV cache: maintain (K, V) tensors of shape (B, H, T, D)
# externally, append per step, and run the attention with `is_causal=False`
# but with the right key/value tensors.
# ---------------------------------------------------------------------------


_TX_KV_DTYPE = torch.bfloat16   # bf16 KV cache, matching production deployments


@torch.inference_mode()
def stateful_prefill_transformer(model: TinyLM,
                                  input_ids: torch.Tensor) -> tuple:
    """Run a Transformer prefill, return (last_logits, kv_cache_list).

    kv_cache_list[L] = (K, V) of shape (B, H, T, D_head). Stored in bf16.
    """
    B, T = input_ids.shape
    pos = torch.arange(T, device=input_ids.device)
    x = model.embed(input_ids) + model.pos_embed(pos)
    kv_cache_list = []
    for blk in model.blocks:
        attn_in = blk.attn_norm(x)
        attn = blk.attn   # SoftmaxAttention
        H, Dh = attn.n_heads, attn.d_head
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = attn.W_q(attn_in).view(B, T, H, Dh).transpose(1, 2)
            k = attn.W_k(attn_in).view(B, T, H, Dh).transpose(1, 2)
            v = attn.W_v(attn_in).view(B, T, H, Dh).transpose(1, 2)
            # Causal SDPA across the whole prefill.
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).contiguous().view(B, T, H * Dh).to(x.dtype)
        x = x + attn.W_o(o)
        x = x + blk.mlp(blk.mlp_norm(x))
        # Persist KV cache in bf16 (matches typical production deployment).
        kv_cache_list.append((k.to(_TX_KV_DTYPE), v.to(_TX_KV_DTYPE)))
    h = model.out_norm(x)
    logits = model.lm_head(h[:, -1:])
    return logits, kv_cache_list


@torch.inference_mode()
def decode_step_transformer(model: TinyLM,
                             next_token: torch.Tensor,
                             kv_cache_list: list,
                             pos_idx: int) -> tuple:
    """One decode step on the Transformer. pos_idx = absolute position.
    Appends to kv_cache_list in-place. Returns logits (B, 1, V).
    """
    B = next_token.shape[0]
    pos = torch.tensor([pos_idx], device=next_token.device)
    x = model.embed(next_token) + model.pos_embed(pos).unsqueeze(0)
    for L, blk in enumerate(model.blocks):
        attn_in = blk.attn_norm(x)
        attn = blk.attn
        H, Dh = attn.n_heads, attn.d_head
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q = attn.W_q(attn_in).view(B, 1, H, Dh).transpose(1, 2)
            k_new = attn.W_k(attn_in).view(B, 1, H, Dh).transpose(1, 2)
            v_new = attn.W_v(attn_in).view(B, 1, H, Dh).transpose(1, 2)
        K_old, V_old = kv_cache_list[L]
        K = torch.cat([K_old, k_new.to(_TX_KV_DTYPE)], dim=2)
        V = torch.cat([V_old, v_new.to(_TX_KV_DTYPE)], dim=2)
        kv_cache_list[L] = (K, V)
        # Single-query attention over the full sequence (bf16).
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = F.scaled_dot_product_attention(q, K, V, is_causal=False)
        o = o.transpose(1, 2).contiguous().view(B, 1, H * Dh).to(x.dtype)
        x = x + attn.W_o(o)
        x = x + blk.mlp(blk.mlp_norm(x))
    h = model.out_norm(x)
    logits = model.lm_head(h)
    return logits


# ---------------------------------------------------------------------------
# Timing utilities.
# ---------------------------------------------------------------------------


def percentile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)


def median(xs: list[float]) -> float:
    return percentile(xs, 0.5)


# ---------------------------------------------------------------------------
# Benchmark harness.
# ---------------------------------------------------------------------------


@dataclass
class TimingResult:
    name: str
    T: int
    # Decode: list of per-step ms (one entry per measured step).
    decode_ms: list[float]
    # Prefill: list of full-prefill seconds (one entry per measured run).
    prefill_secs: list[float]


def bench_dn(model: TinyLM, T: int, n_warmup: int = 3, n_meas: int = 10,
              n_decode: int = 32) -> TimingResult:
    """Bench DN-baseline: prefill of length T, then `n_decode` decode steps."""
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    decode_ms: list[float] = []
    prefill_secs: list[float] = []

    for it in range(n_warmup + n_meas):
        torch.manual_seed(0xABCDEF + it)
        ids = torch.randint(0, vocab, (1, T), device=device)
        torch.cuda.synchronize()

        # Prefill.
        t0 = time.perf_counter()
        logits, cache = stateful_prefill_dn(model, ids)
        torch.cuda.synchronize()
        prefill_secs.append(time.perf_counter() - t0)

        # Pick next token (greedy is fine — we only need *something*).
        next_tok = logits.argmax(dim=-1)         # (1, 1)

        # Decode steps.
        per_step_ms: list[float] = []
        for s in range(n_decode):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = decode_step_dn(model, next_tok, cache)
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1000)
            next_tok = logits.argmax(dim=-1)
        if it >= n_warmup:
            decode_ms.extend(per_step_ms)
        del cache, logits

    return TimingResult(
        name="DN-baseline",
        T=T,
        decode_ms=decode_ms,
        prefill_secs=prefill_secs[n_warmup:],
    )


def bench_film(model: TinyLM, T: int, n_warmup: int = 3, n_meas: int = 10,
                n_decode: int = 32) -> TimingResult:
    """Bench sparse-(2,34) FiLM DN."""
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    decode_ms: list[float] = []
    prefill_secs: list[float] = []
    target, source = model.feedback_pairs[0]

    for it in range(n_warmup + n_meas):
        torch.manual_seed(0xABCDEF + it)
        ids = torch.randint(0, vocab, (1, T), device=device)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        logits, cache, lagged_src = stateful_prefill_film(
            model, ids, target=target, source=source,
        )
        torch.cuda.synchronize()
        prefill_secs.append(time.perf_counter() - t0)

        next_tok = logits.argmax(dim=-1)

        per_step_ms: list[float] = []
        for s in range(n_decode):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, lagged_src = decode_step_film(
                model, next_tok, cache, lagged_src,
                target=target, source=source,
            )
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1000)
            next_tok = logits.argmax(dim=-1)
        if it >= n_warmup:
            decode_ms.extend(per_step_ms)
        del cache, logits

    return TimingResult(
        name="Sparse-(2,34)-FiLM DN",
        T=T,
        decode_ms=decode_ms,
        prefill_secs=prefill_secs[n_warmup:],
    )


def bench_film_2pass(model: TinyLM, T: int, n_warmup: int = 3, n_meas: int = 10,
                      n_decode: int = 32) -> TimingResult:
    """Bench sparse-(2,34) FiLM DN with the *exact* 2-pass-per-decode-step
    protocol (gold-matching, but ~2× the cost of lagged-cached).
    """
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    decode_ms: list[float] = []
    prefill_secs: list[float] = []
    target, source = model.feedback_pairs[0]

    for it in range(n_warmup + n_meas):
        torch.manual_seed(0xABCDEF + it)
        ids = torch.randint(0, vocab, (1, T), device=device)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        logits, c1, c2, lagged_p1 = stateful_prefill_film_2pass(
            model, ids, target=target, source=source,
        )
        torch.cuda.synchronize()
        prefill_secs.append(time.perf_counter() - t0)

        next_tok = logits.argmax(dim=-1)

        per_step_ms: list[float] = []
        for s in range(n_decode):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, lagged_p1 = decode_step_film_2pass(
                model, next_tok, c1, c2, lagged_p1,
                target=target, source=source,
            )
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1000)
            next_tok = logits.argmax(dim=-1)
        if it >= n_warmup:
            decode_ms.extend(per_step_ms)
        del c1, c2, logits

    return TimingResult(
        name="Sparse-(2,34)-FiLM DN [2-pass]",
        T=T,
        decode_ms=decode_ms,
        prefill_secs=prefill_secs[n_warmup:],
    )


def bench_film_pass1_only(model: TinyLM, T: int, n_warmup: int = 3,
                            n_meas: int = 10, n_decode: int = 32) -> TimingResult:
    """Bench pass-1 alone (layers 0..source on a single new token) — the
    cheap branch of the 2-pass. Used to confirm pass-1 is much cheaper
    than pass-2, which is required for the overlap protocol to be useful.
    """
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    decode_ms: list[float] = []
    target, source = model.feedback_pairs[0]

    for it in range(n_warmup + n_meas):
        torch.manual_seed(0xABCDEF + it)
        ids = torch.randint(0, vocab, (1, T), device=device)
        torch.cuda.synchronize()

        _, c1, _, _ = stateful_prefill_film_2pass(
            model, ids, target=target, source=source,
        )
        next_tok = torch.randint(0, vocab, (1, 1), device=device)
        torch.cuda.synchronize()

        per_step_ms: list[float] = []
        for s in range(n_decode):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _decode_pass1_step(model, next_tok, c1, source=source)
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1000)
        if it >= n_warmup:
            decode_ms.extend(per_step_ms)
        del c1

    return TimingResult(
        name="Sparse-(2,34)-FiLM DN [pass-1 only]",
        T=T, decode_ms=decode_ms, prefill_secs=[],
    )


def bench_film_pass2_only(model: TinyLM, T: int, n_warmup: int = 3,
                            n_meas: int = 10, n_decode: int = 32) -> TimingResult:
    """Bench pass-2 alone (full forward through all layers, FiLM at
    `target`) on a single new token, with a stale `lagged_pass1_source`
    that we don't update — only the pass-2 work is measured.
    """
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    decode_ms: list[float] = []
    target, source = model.feedback_pairs[0]

    for it in range(n_warmup + n_meas):
        torch.manual_seed(0xABCDEF + it)
        ids = torch.randint(0, vocab, (1, T), device=device)
        torch.cuda.synchronize()

        _, _, c2, lagged_p1 = stateful_prefill_film_2pass(
            model, ids, target=target, source=source,
        )
        next_tok = torch.randint(0, vocab, (1, 1), device=device)
        torch.cuda.synchronize()

        per_step_ms: list[float] = []
        for s in range(n_decode):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _decode_pass2_step(model, next_tok, c2, lagged_p1, target=target)
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1000)
        if it >= n_warmup:
            decode_ms.extend(per_step_ms)
        del c2

    return TimingResult(
        name="Sparse-(2,34)-FiLM DN [pass-2 only]",
        T=T, decode_ms=decode_ms, prefill_secs=[],
    )


@torch.inference_mode()
def validate_overlap_equivalence(model: TinyLM, T: int = 512,
                                  n_decode: int = 64,
                                  seed: int = 0xABCDEF) -> dict:
    """Verify the async-overlap 2-pass decode produces *identical* outputs
    to the sequential 2-pass decode at every step. Same RNG seed, greedy
    sampling, T-token prefill, n_decode steps.

    Returns a dict of diagnostics (max abs logit diff per step, token
    match count, etc.).
    """
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    target, source = model.feedback_pairs[0]

    torch.manual_seed(seed)
    ids = torch.randint(0, vocab, (1, T), device=device)

    # Sequential 2-pass reference.
    logits_seq, c1_seq, c2_seq, lagged_seq = stateful_prefill_film_2pass(
        model, ids, target=target, source=source,
    )
    next_tok_seq = logits_seq.argmax(dim=-1)
    seq_tokens: list[int] = [int(next_tok_seq.item())]
    seq_logits_per_step: list[torch.Tensor] = []
    for _ in range(n_decode):
        new_logits, lagged_seq = decode_step_film_2pass(
            model, next_tok_seq, c1_seq, c2_seq, lagged_seq,
            target=target, source=source,
        )
        seq_logits_per_step.append(new_logits.detach().clone())
        next_tok_seq = new_logits.argmax(dim=-1)
        seq_tokens.append(int(next_tok_seq.item()))

    # Async-overlap 2-pass under test.
    logits_ov, c1_ov, c2_ov, lagged_ov = stateful_prefill_film_2pass(
        model, ids.clone(), target=target, source=source,
    )
    next_tok_ov = logits_ov.argmax(dim=-1)
    ov_tokens: list[int] = [int(next_tok_ov.item())]
    ov_logits_per_step: list[torch.Tensor] = []
    stream_pass1 = torch.cuda.Stream(device=device)
    stream_pass2 = torch.cuda.Stream(device=device)
    for _ in range(n_decode):
        new_logits, lagged_ov = decode_step_film_2pass_overlap(
            model, next_tok_ov, c1_ov, c2_ov, lagged_ov,
            stream_pass1=stream_pass1, stream_pass2=stream_pass2,
            target=target, source=source,
        )
        ov_logits_per_step.append(new_logits.detach().clone())
        next_tok_ov = new_logits.argmax(dim=-1)
        ov_tokens.append(int(next_tok_ov.item()))

    # Diagnostics.
    n_match = sum(int(a == b) for a, b in zip(seq_tokens, ov_tokens))
    max_diffs = [
        float((a.float() - b.float()).abs().max().item())
        for a, b in zip(seq_logits_per_step, ov_logits_per_step)
    ]
    return {
        "T_prefill": T, "n_decode": n_decode,
        "tokens_seq": seq_tokens,
        "tokens_overlap": ov_tokens,
        "tokens_match": n_match,
        "tokens_total": len(seq_tokens),
        "max_logit_diff_per_step": max_diffs,
        "max_logit_diff_overall": max(max_diffs) if max_diffs else 0.0,
    }


def bench_film_2pass_overlap(model: TinyLM, T: int, n_warmup: int = 3,
                               n_meas: int = 10, n_decode: int = 32) -> TimingResult:
    """Bench sparse-(2,34) FiLM DN with the *async-overlap* 2-pass decode
    protocol. Pass 1 and pass 2 run on separate CUDA streams; they're
    independent at any decode step (pass 2 reads pass 1's output from the
    PREVIOUS step). End-of-step sync re-establishes the token dependency.
    """
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    decode_ms: list[float] = []
    prefill_secs: list[float] = []
    target, source = model.feedback_pairs[0]

    stream_pass1 = torch.cuda.Stream(device=device)
    stream_pass2 = torch.cuda.Stream(device=device)

    for it in range(n_warmup + n_meas):
        torch.manual_seed(0xABCDEF + it)
        ids = torch.randint(0, vocab, (1, T), device=device)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        logits, c1, c2, lagged_p1 = stateful_prefill_film_2pass(
            model, ids, target=target, source=source,
        )
        torch.cuda.synchronize()
        prefill_secs.append(time.perf_counter() - t0)

        next_tok = logits.argmax(dim=-1)

        per_step_ms: list[float] = []
        for s in range(n_decode):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits, lagged_p1 = decode_step_film_2pass_overlap(
                model, next_tok, c1, c2, lagged_p1,
                stream_pass1=stream_pass1, stream_pass2=stream_pass2,
                target=target, source=source,
            )
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1000)
            next_tok = logits.argmax(dim=-1)
        if it >= n_warmup:
            decode_ms.extend(per_step_ms)
        del c1, c2, logits

    return TimingResult(
        name="Sparse-(2,34)-FiLM DN [2-pass overlap]",
        T=T,
        decode_ms=decode_ms,
        prefill_secs=prefill_secs[n_warmup:],
    )


def bench_transformer(model: TinyLM, T: int, n_warmup: int = 3, n_meas: int = 10,
                       n_decode: int = 32) -> TimingResult:
    """Bench Transformer reference."""
    device = next(model.parameters()).device
    vocab = model.embed.num_embeddings
    decode_ms: list[float] = []
    prefill_secs: list[float] = []

    for it in range(n_warmup + n_meas):
        torch.manual_seed(0xABCDEF + it)
        ids = torch.randint(0, vocab, (1, T), device=device)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        logits, kv_cache = stateful_prefill_transformer(model, ids)
        torch.cuda.synchronize()
        prefill_secs.append(time.perf_counter() - t0)

        next_tok = logits.argmax(dim=-1)
        pos_idx = T  # next position is T

        per_step_ms: list[float] = []
        for s in range(n_decode):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = decode_step_transformer(model, next_tok, kv_cache, pos_idx)
            torch.cuda.synchronize()
            per_step_ms.append((time.perf_counter() - t0) * 1000)
            next_tok = logits.argmax(dim=-1)
            pos_idx += 1
        if it >= n_warmup:
            decode_ms.extend(per_step_ms)
        del kv_cache, logits

    return TimingResult(
        name="Transformer-360M",
        T=T,
        decode_ms=decode_ms,
        prefill_secs=prefill_secs[n_warmup:],
    )


# ---------------------------------------------------------------------------
# Memory analysis (analytical).
# ---------------------------------------------------------------------------


def state_memory_dn(n_layers: int, n_heads: int, d_head_k: int, d_head_v: int,
                     conv_size: int, d_model: int) -> dict:
    """Analytical inference-time state per batch element for DN.

    Per layer:
      - recurrent_state: (n_heads, d_head_k, d_head_v) fp32
      - conv_state for q, k, v: 3 × (d_model, conv_size) bf16
        (the q_conv1d, k_conv1d, v_conv1d caches in fla DeltaNet)
    """
    rec = n_heads * d_head_k * d_head_v * 4         # fp32
    conv = 3 * d_model * conv_size * 2              # bf16
    per_layer = rec + conv
    total = n_layers * per_layer
    return {
        "per_layer_recurrent_bytes": rec,
        "per_layer_conv_bytes": conv,
        "per_layer_total_bytes": per_layer,
        "n_layers": n_layers,
        "total_bytes": total,
        "total_MB": total / (1024 * 1024),
    }


def state_memory_film(n_layers: int, n_heads: int, d_head_k: int, d_head_v: int,
                       conv_size: int, d_model: int) -> dict:
    """DN state + 1 cached layer-source output (B, 1, d_model) bf16."""
    base = state_memory_dn(n_layers, n_heads, d_head_k, d_head_v, conv_size,
                            d_model)
    fb_bytes = d_model * 2                              # bf16
    base["fb_lagged_source_bytes"] = fb_bytes
    base["total_bytes"] = base["total_bytes"] + fb_bytes
    base["total_MB"] = base["total_bytes"] / (1024 * 1024)
    return base


def state_memory_transformer(n_layers: int, n_heads: int, d_head: int,
                              T: int) -> dict:
    """Transformer KV cache per batch element.

    2 (K + V) × n_layers × n_heads × T × d_head, in bf16 (2 bytes).
    """
    bytes_total = 2 * n_layers * n_heads * T * d_head * 2
    return {
        "n_layers": n_layers, "n_heads": n_heads, "d_head": d_head, "T": T,
        "total_bytes": bytes_total,
        "total_MB": bytes_total / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# CLI + report writer.
# ---------------------------------------------------------------------------


def fmt_ms(median_ms: float, p95_ms: float) -> str:
    return f"{median_ms:.2f} (p95 {p95_ms:.2f})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["decode", "prefill", "memory", "all"],
                   default="all")
    p.add_argument("--T", type=int, nargs="*", default=None,
                   help="Override context lengths.")
    p.add_argument("--n_warmup", type=int, default=2)
    p.add_argument("--n_meas", type=int, default=5,
                   help="Outer iterations for prefill timing.")
    p.add_argument("--n_decode", type=int, default=24,
                   help="Decode steps per outer iteration; total measured steps "
                        "is n_meas × n_decode.")
    p.add_argument("--out", type=str, default="bench_decode.json")
    p.add_argument("--include_transformer", action="store_true", default=True)
    p.add_argument("--no_transformer", dest="include_transformer",
                   action="store_false")
    p.add_argument("--transformer_from_ckpt", action="store_true",
                   help="Load real 360M Transformer checkpoint (extends "
                        "pos-embed to required max_T with random init).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--include_dn", action="store_true", default=True)
    p.add_argument("--no_dn", dest="include_dn", action="store_false")
    p.add_argument("--include_film", action="store_true", default=True)
    p.add_argument("--no_film", dest="include_film", action="store_false")
    p.add_argument("--overlap_only", action="store_true",
                   help="Run only the async-overlap 2-pass FiLM bench (and "
                        "validation), then exit. Skips DN, lagged FiLM, and "
                        "Transformer benches.")
    p.add_argument("--validate_overlap", action="store_true",
                   help="Run the 2-pass-vs-overlap equivalence check on the "
                        "FiLM model (T=512, 64 decode steps).")
    p.add_argument("--include_overlap", action="store_true", default=True,
                   help="Include the 2-pass async-overlap bench (default on).")
    p.add_argument("--no_overlap", dest="include_overlap", action="store_false")
    p.add_argument("--include_pass_isolation", action="store_true",
                   help="Bench pass-1-only and pass-2-only standalone "
                        "(diagnostic — confirms pass-2 dominates).")
    args = p.parse_args()

    Ts = args.T or list(CONTEXT_LENGTHS)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Context lengths: {Ts}")

    results: dict = {"decode": {}, "prefill": {}, "memory": {}}

    # Fast-path: overlap-only bench. Loads FiLM ckpt, runs validation
    # (T=512, 64 steps) and then the async-overlap decode bench at every
    # requested context length. Optionally also pass-1/pass-2 standalone.
    if args.overlap_only:
        print("\n--- Sparse-(2,34) FiLM DN [2-pass async-overlap] (overlap_only) ---")
        film = load_dn_or_film(CKPT_FILM, device=args.device)
        print(f"  params: {film.num_params() / 1e6:.1f} M")

        print("\n  validating overlap == sequential 2-pass (T=512, 64 steps)...")
        v = validate_overlap_equivalence(film, T=512, n_decode=64)
        print(f"    tokens match: {v['tokens_match']}/{v['tokens_total']}  "
              f"max |Δlogit| over 64 steps = {v['max_logit_diff_overall']:.2e}")
        results.setdefault("validation", {})["overlap_vs_sequential"] = {
            "T_prefill": v["T_prefill"], "n_decode": v["n_decode"],
            "tokens_match": v["tokens_match"],
            "tokens_total": v["tokens_total"],
            "max_logit_diff_overall": v["max_logit_diff_overall"],
            "tokens_seq": v["tokens_seq"],
            "tokens_overlap": v["tokens_overlap"],
        }

        if args.include_pass_isolation:
            print("\n  pass-1-only / pass-2-only diagnostic bench:")
            for T in Ts:
                r1 = bench_film_pass1_only(film, T, n_warmup=args.n_warmup,
                                            n_meas=args.n_meas, n_decode=args.n_decode)
                r2 = bench_film_pass2_only(film, T, n_warmup=args.n_warmup,
                                            n_meas=args.n_meas, n_decode=args.n_decode)
                results["decode"][f"FILMP1_T={T}"] = {
                    "median_ms": median(r1.decode_ms),
                    "p95_ms": percentile(r1.decode_ms, 0.95),
                    "n_steps": len(r1.decode_ms),
                }
                results["decode"][f"FILMP2_T={T}"] = {
                    "median_ms": median(r2.decode_ms),
                    "p95_ms": percentile(r2.decode_ms, 0.95),
                    "n_steps": len(r2.decode_ms),
                }
                print(f"    T={T:5d}: pass-1 {median(r1.decode_ms):.2f} ms/tok  "
                      f"pass-2 {median(r2.decode_ms):.2f} ms/tok  "
                      f"max(p1,p2) {max(median(r1.decode_ms), median(r2.decode_ms)):.2f} ms")

        for T in Ts:
            print(f"  overlap T={T} ...")
            r = bench_film_2pass_overlap(film, T, n_warmup=args.n_warmup,
                                          n_meas=args.n_meas, n_decode=args.n_decode)
            results["decode"][f"FILM2POV_T={T}"] = {
                "median_ms": median(r.decode_ms),
                "p95_ms": percentile(r.decode_ms, 0.95),
                "n_steps": len(r.decode_ms),
            }
            results["prefill"][f"FILM2POV_T={T}"] = {
                "median_secs": median(r.prefill_secs),
                "p95_secs": percentile(r.prefill_secs, 0.95),
                "median_tok_per_s": T / median(r.prefill_secs),
            }
            print(f"    decode {fmt_ms(median(r.decode_ms), percentile(r.decode_ms, 0.95))} ms/tok")
        del film
        torch.cuda.empty_cache()

        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {out}")
        return 0

    # Models — load lazily to avoid holding everything at once.
    if args.bench in ("decode", "prefill", "all"):
        if args.include_dn:
            print("\n--- DN baseline (708M) ---")
            dn = load_dn_or_film(CKPT_DN, device=args.device)
            print(f"  params: {dn.num_params() / 1e6:.1f} M")
            for T in Ts:
                print(f"  T={T} ...")
                r = bench_dn(dn, T, n_warmup=args.n_warmup,
                              n_meas=args.n_meas, n_decode=args.n_decode)
                results["decode"][f"DN_T={T}"] = {
                    "median_ms": median(r.decode_ms),
                    "p95_ms": percentile(r.decode_ms, 0.95),
                    "n_steps": len(r.decode_ms),
                }
                results["prefill"][f"DN_T={T}"] = {
                    "median_secs": median(r.prefill_secs),
                    "p95_secs": percentile(r.prefill_secs, 0.95),
                    "median_tok_per_s": T / median(r.prefill_secs),
                }
                print(f"    decode {fmt_ms(median(r.decode_ms), percentile(r.decode_ms, 0.95))} ms/tok  "
                      f"prefill {median(r.prefill_secs)*1000:.1f} ms ({T/median(r.prefill_secs):.0f} tok/s)")
            del dn
            torch.cuda.empty_cache()

        if args.include_film:
            print("\n--- Sparse-(2,34) FiLM DN (708M) [lagged-cached, 1×] ---")
            film = load_dn_or_film(CKPT_FILM, device=args.device)
            print(f"  params: {film.num_params() / 1e6:.1f} M")
            for T in Ts:
                print(f"  T={T} ...")
                r = bench_film(film, T, n_warmup=args.n_warmup,
                                n_meas=args.n_meas, n_decode=args.n_decode)
                results["decode"][f"FILM_T={T}"] = {
                    "median_ms": median(r.decode_ms),
                    "p95_ms": percentile(r.decode_ms, 0.95),
                    "n_steps": len(r.decode_ms),
                }
                results["prefill"][f"FILM_T={T}"] = {
                    "median_secs": median(r.prefill_secs),
                    "p95_secs": percentile(r.prefill_secs, 0.95),
                    "median_tok_per_s": T / median(r.prefill_secs),
                }
                print(f"    decode {fmt_ms(median(r.decode_ms), percentile(r.decode_ms, 0.95))} ms/tok  "
                      f"prefill {median(r.prefill_secs)*1000:.1f} ms ({T/median(r.prefill_secs):.0f} tok/s)")
            print(f"\n--- Sparse-(2,34) FiLM DN [2-pass-per-decode, gold-matching, 2×] ---")
            for T in Ts:
                print(f"  T={T} ...")
                r = bench_film_2pass(film, T, n_warmup=args.n_warmup,
                                       n_meas=args.n_meas, n_decode=args.n_decode)
                results["decode"][f"FILM2P_T={T}"] = {
                    "median_ms": median(r.decode_ms),
                    "p95_ms": percentile(r.decode_ms, 0.95),
                    "n_steps": len(r.decode_ms),
                }
                results["prefill"][f"FILM2P_T={T}"] = {
                    "median_secs": median(r.prefill_secs),
                    "p95_secs": percentile(r.prefill_secs, 0.95),
                    "median_tok_per_s": T / median(r.prefill_secs),
                }
                print(f"    decode {fmt_ms(median(r.decode_ms), percentile(r.decode_ms, 0.95))} ms/tok  "
                      f"prefill {median(r.prefill_secs)*1000:.1f} ms ({T/median(r.prefill_secs):.0f} tok/s)")
            if args.include_overlap:
                print(f"\n--- Sparse-(2,34) FiLM DN [2-pass async-overlap] ---")
                v = validate_overlap_equivalence(film, T=512, n_decode=64)
                print(f"  validation tokens match: {v['tokens_match']}/{v['tokens_total']}  "
                      f"max |Δlogit| = {v['max_logit_diff_overall']:.2e}")
                results.setdefault("validation", {})["overlap_vs_sequential"] = {
                    "T_prefill": v["T_prefill"], "n_decode": v["n_decode"],
                    "tokens_match": v["tokens_match"],
                    "tokens_total": v["tokens_total"],
                    "max_logit_diff_overall": v["max_logit_diff_overall"],
                    "tokens_seq": v["tokens_seq"],
                    "tokens_overlap": v["tokens_overlap"],
                }
                for T in Ts:
                    print(f"  T={T} ...")
                    r = bench_film_2pass_overlap(film, T, n_warmup=args.n_warmup,
                                                   n_meas=args.n_meas, n_decode=args.n_decode)
                    results["decode"][f"FILM2POV_T={T}"] = {
                        "median_ms": median(r.decode_ms),
                        "p95_ms": percentile(r.decode_ms, 0.95),
                        "n_steps": len(r.decode_ms),
                    }
                    results["prefill"][f"FILM2POV_T={T}"] = {
                        "median_secs": median(r.prefill_secs),
                        "p95_secs": percentile(r.prefill_secs, 0.95),
                        "median_tok_per_s": T / median(r.prefill_secs),
                    }
                    print(f"    decode {fmt_ms(median(r.decode_ms), percentile(r.decode_ms, 0.95))} ms/tok  "
                          f"prefill {median(r.prefill_secs)*1000:.1f} ms ({T/median(r.prefill_secs):.0f} tok/s)")
            del film
            torch.cuda.empty_cache()

        if args.include_transformer:
            print("\n--- Transformer reference (360M) ---")
            max_T = max(Ts) + args.n_decode + 4
            tx = load_or_build_transformer(
                device=args.device, max_T=max_T,
                from_ckpt=args.transformer_from_ckpt,
            )
            print(f"  params: {tx.num_params() / 1e6:.1f} M")
            for T in Ts:
                print(f"  T={T} ...")
                r = bench_transformer(tx, T, n_warmup=args.n_warmup,
                                       n_meas=args.n_meas, n_decode=args.n_decode)
                results["decode"][f"TX_T={T}"] = {
                    "median_ms": median(r.decode_ms),
                    "p95_ms": percentile(r.decode_ms, 0.95),
                    "n_steps": len(r.decode_ms),
                }
                results["prefill"][f"TX_T={T}"] = {
                    "median_secs": median(r.prefill_secs),
                    "p95_secs": percentile(r.prefill_secs, 0.95),
                    "median_tok_per_s": T / median(r.prefill_secs),
                }
                print(f"    decode {fmt_ms(median(r.decode_ms), percentile(r.decode_ms, 0.95))} ms/tok  "
                      f"prefill {median(r.prefill_secs)*1000:.1f} ms ({T/median(r.prefill_secs):.0f} tok/s)")
            del tx
            torch.cuda.empty_cache()

    if args.bench in ("memory", "all"):
        print("\n--- State memory (analytical) ---")
        # 708M DN config:
        dn_mem = state_memory_dn(
            n_layers=36, n_heads=16, d_head_k=64, d_head_v=64,
            conv_size=4, d_model=1024,
        )
        film_mem = state_memory_film(
            n_layers=36, n_heads=16, d_head_k=64, d_head_v=64,
            conv_size=4, d_model=1024,
        )
        results["memory"]["DN_708M"] = dn_mem
        results["memory"]["FILM_708M"] = film_mem
        # 360M Transformer at each context length:
        for T in Ts:
            tx_mem = state_memory_transformer(
                n_layers=30, n_heads=12, d_head=64, T=T,
            )
            results["memory"][f"TX_360M_T={T}"] = tx_mem
            print(f"  TX 360M @ T={T}: {tx_mem['total_MB']:.1f} MB KV cache")
        print(f"  DN  708M state: {dn_mem['total_MB']:.2f} MB (constant in T)")
        print(f"  FiLM 708M state: {film_mem['total_MB']:.2f} MB (constant in T; "
              f"+{film_mem['fb_lagged_source_bytes']} B for lagged FB)")

        # Empirical state-only memory delta. Allocate the model with no
        # prefill, then prefill at T=8192, and measure the cuda allocator
        # delta. This includes activation overhead in the prefill itself —
        # we want only the *post-prefill* persistent state, so we measure
        # at end of prefill *after* freeing logits/x intermediate tensors.
        if args.include_dn or args.include_film:
            print("\n--- State memory (empirical Δ at T=8192) ---")
            T_emp = max(Ts)

            def _measure(load_fn, prefill_fn, name: str):
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                m = load_fn()
                m.eval()
                # Warmup: do a one-shot prefill then drop, to settle the
                # autocast bf16-weight cache + triton kernel allocator state.
                # Otherwise the *first* prefill in the program incurs a
                # one-time ~32 MB bookkeeping bump that gets attributed to
                # the cache, but subsequent prefills don't.
                ids_warm = torch.randint(0, m.embed.num_embeddings,
                                          (1, 1024), device=args.device)
                ret_warm = prefill_fn(m, ids_warm)
                del ret_warm, ids_warm
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                ids = torch.randint(0, m.embed.num_embeddings,
                                     (1, T_emp), device=args.device)
                ret = prefill_fn(m, ids)
                # ret = (logits, cache, ...) — keep only the cache (and any
                # FiLM lagged-source tensor). Logits aren't part of the
                # persistent inference state.
                cache = ret[1]
                if len(ret) > 2:
                    fb = ret[2]
                else:
                    fb = None
                del ret
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated()
                delta = after - base
                peak = torch.cuda.max_memory_allocated() - base
                results["memory"][name + "_empirical"] = {
                    "post_prefill_delta_bytes": delta,
                    "post_prefill_delta_MB": delta / (1024 * 1024),
                    "peak_during_prefill_delta_bytes": peak,
                    "peak_during_prefill_delta_MB": peak / (1024 * 1024),
                }
                print(f"  {name} post-prefill state Δ: {delta/(1024*1024):.2f} MB; "
                      f"peak Δ during prefill: {peak/(1024*1024):.1f} MB")
                del m, cache, fb, ids
                gc.collect()
                torch.cuda.empty_cache()

            if args.include_dn:
                _measure(
                    lambda: load_dn_or_film(CKPT_DN, device=args.device),
                    stateful_prefill_dn, "DN_708M",
                )
            if args.include_film:
                _measure(
                    lambda: load_dn_or_film(CKPT_FILM, device=args.device),
                    stateful_prefill_film, "FILM_708M",
                )
            if args.include_transformer:
                # Transformer KV cache is tensor-based, easy to measure too.
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                tx_max_T = T_emp + 4
                m = load_or_build_transformer(
                    device=args.device, max_T=tx_max_T,
                    from_ckpt=args.transformer_from_ckpt,
                )
                m.eval()
                ids_warm = torch.randint(0, m.embed.num_embeddings,
                                          (1, 1024), device=args.device)
                ret_warm = stateful_prefill_transformer(m, ids_warm)
                del ret_warm, ids_warm
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                ids = torch.randint(0, m.embed.num_embeddings,
                                     (1, T_emp), device=args.device)
                ret_tx = stateful_prefill_transformer(m, ids)
                kv = ret_tx[1]
                del ret_tx
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated()
                delta = after - base
                peak = torch.cuda.max_memory_allocated() - base
                results["memory"]["TX_360M_T=" + str(T_emp) + "_empirical"] = {
                    "post_prefill_delta_bytes": delta,
                    "post_prefill_delta_MB": delta / (1024 * 1024),
                    "peak_during_prefill_delta_bytes": peak,
                    "peak_during_prefill_delta_MB": peak / (1024 * 1024),
                }
                print(f"  TX 360M @ T={T_emp} post-prefill state Δ: "
                      f"{delta/(1024*1024):.2f} MB; peak Δ: {peak/(1024*1024):.1f} MB")
                del m, kv, ids
                gc.collect()
                torch.cuda.empty_cache()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    sys.exit(main())
