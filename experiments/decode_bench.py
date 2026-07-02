"""DECODE-COST benchmark — honestly test the repo's "O(1)/constant-memory
decode" moat claim.

Compares single-stream (batch=1) autoregressive decode of:
  OURS:     lean linearized DeltaNet (bounded recurrent state, state-passing
            incremental decode via TinyLM.prefill + TinyLM.forward_step).
  BASELINE: HuggingFaceTB/SmolLM2-360M (HF transformer, growing KV cache).

Protocol per context length L:
  - prefill L tokens (excluded from per-token timing),
  - generate G tokens autoregressively (state-passing for ours, KV-cache for
    the transformer),
  - measure STEADY-STATE decode latency (median ms / generated token, after a
    warmup) and PEAK decode-phase GPU memory.

The decode-phase peak is measured by resetting the CUDA peak-memory tracker
*after* prefill (so the prefill O(L) transient — which is not the moat claim —
does not mask the decode footprint). We also record the cache size (allocated
memory growth from prefill) which is the cleanest single number for "does the
per-token state grow with context".

bf16 for both (deployment dtype). Eager for both (no torch.compile) — fair,
and the deployed forward_step path is eager. Single GPU.

Run:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. HF_HUB_OFFLINE=1 \
    .venv/bin/python experiments/decode_bench.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time

import torch

MIB = 2 ** 20


def _sync():
    torch.cuda.synchronize()


def _oom(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _fla_state_bytes(fla_cache) -> float:
    """Sum the bytes of the recurrent + conv state held in an FLA Cache.

    This is the TRUE bounded decode state for our model. We measure it
    directly (rather than via allocated-memory deltas) because prefill
    also returns a transient (1, L, vocab) logits tensor whose size grows
    with L — that transient is NOT part of the decode footprint (it is
    freed on the first decode step) and would otherwise contaminate a
    naive allocated-delta read. FLA's Cache is indexable per layer; each
    entry is a dict of state tensors."""
    seen, tot = set(), 0
    try:
        n = len(fla_cache)
    except TypeError:
        return 0.0
    for i in range(n):
        entry = fla_cache[i]
        vals = entry.values() if isinstance(entry, dict) else \
            (entry if isinstance(entry, (list, tuple)) else [entry])
        for v in vals:
            items = v if isinstance(v, (list, tuple)) else [v]
            for t in items:
                if torch.is_tensor(t) and id(t) not in seen:
                    seen.add(id(t))
                    tot += t.numel() * t.element_size()
    return tot / MIB


# --------------------------------------------------------------------------- #
# OURS — lean linearized DeltaNet, state-passing incremental decode.
# --------------------------------------------------------------------------- #
def load_ours(ckpt_path: str):
    from experiments.eval_bracket_structure import build_model_from_ckpt
    model, cfg = build_model_from_ckpt(ckpt_path)
    model.eval()
    model._film_bypass = True          # deploy convention (K=1 at decode)
    model.to(torch.bfloat16)           # deployment dtype
    return model, cfg


def prefill_state_only(model, input_ids, chunk: int = 0):
    """Build the SAME bounded FLA recurrent state as `TinyLM.prefill` but
    skip the all-position `lm_head` transient (a real server only needs the
    last-position logits — exactly what `logits_to_keep=1` gives the
    transformer baseline, so this keeps the comparison fair). Only valid for
    the lean config (no FiLM / WM / PKM / gate / think-index). Validated to
    produce decode argmax IDENTICAL to the real `prefill` (B>1, single-shot
    AND chunked). `chunk>0` processes the prompt in segments, continuing the
    recurrent state across them (bounds prefill activation memory)."""
    from fla.models.utils import Cache as FLACache
    B, T = input_ids.shape
    x = model.embed(input_ids)
    if model.max_T > 0:
        pos = torch.arange(T, device=input_ids.device)
        x = x + model.pos_embed(pos)
    fla = FLACache(seen_tokens=0)
    segs = range(0, T, chunk) if (chunk and T > chunk) else [0]
    seg_len = chunk if (chunk and T > chunk) else T
    for s in segs:
        h = x[:, s:s + seg_len]
        for L, blk in enumerate(model.blocks):
            h = model._step_block(blk, h, past=fla, layer_idx=L, think_mask=None)
            h = model._maybe_pkm(h, L)
    return {"fla_cache": fla, "seen": int(T),
            "lagged_sources": None, "think_run_len": None}


@torch.no_grad()
def bench_ours_batched(model, vocab, L, B, G, warmup, chunk, device="cuda"):
    """Batched decode: B sequences in parallel, prefill L then generate G.
    Returns per-seq latency, aggregate throughput, peak decode memory."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weight_mib = torch.cuda.memory_allocated() / MIB
        prompt = torch.randint(0, vocab, (B, L), device=device, dtype=torch.long)
        # No outer autocast: `model` is already cast to bf16 (load_ours), so
        # every op here runs bf16-native — matches the transformer arm below,
        # which is also timed with no autocast. `_FlaWrapper.forward_step`
        # (layers.py) still wraps its own inner FLA-kernel call in a narrow
        # autocast (safety net for non-bf16 callers); that is untouched and
        # unaffected by removing this outer one. See DECODE_COST_BENCH.md
        # 2026-07-01 autocast-asymmetry correction note for the equivalence
        # check that justified this removal (argmax-sequence-identical over
        # a 16-step decode probe; raw logits differ by <0.44 due to
        # RMSNorm's reduction running fp32-promoted under autocast vs
        # native bf16 without it — immaterial in-distribution).
        cache = prefill_state_only(model, prompt, chunk=chunk)
        _sync()
        cache_mib = _fla_state_bytes(cache["fla_cache"])
        next_tok = torch.zeros((B, 1), dtype=torch.long, device=device)
        for _ in range(warmup):
            _, cache = model.forward_step(next_tok, cache)
        _sync()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(G):
            _sync(); t0 = time.perf_counter()
            _, cache = model.forward_step(next_tok, cache)
            _sync(); times.append((time.perf_counter() - t0) * 1e3)
        peak = torch.cuda.max_memory_allocated() / MIB
        total_s = sum(times) / 1e3
        return dict(per_seq_ms=statistics.median(times),
                    throughput_tok_s=(B * G) / total_s,
                    peak_decode_mib=peak, cache_mib=cache_mib, weight_mib=weight_mib)
    except Exception as e:  # noqa: BLE001
        if _oom(e):
            torch.cuda.empty_cache(); return {"oom": True}
        raise


@torch.no_grad()
def bench_transformer_batched(model, vocab, L, B, G, warmup, device="cuda",
                              chunk=0):
    """chunk>0 prefills the prompt in segments carried through
    past_key_values — the same accommodation bench_ours_batched gets via
    prefill_state_only(chunk=...). Without it, a one-shot (B, L) prefill's
    activation transient can OOM at cells whose KV cache would fit at
    decode, mislabelling a prefill-transient OOM as a decode-state OOM."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weight_mib = torch.cuda.memory_allocated() / MIB
        prompt = torch.randint(0, vocab, (B, L), device=device, dtype=torch.long)
        ltk = _supports_ltk(model)
        kw = dict(use_cache=True)
        if ltk:
            kw["logits_to_keep"] = 1
        if chunk and chunk > 0:
            past = None
            for i in range(0, L, chunk):
                seg = prompt[:, i:i + chunk]
                out = (model(seg, past_key_values=past, **kw)
                       if past is not None else model(seg, **kw))
                past = out.past_key_values
                del out
        else:
            out = model(prompt, **kw)
            past = out.past_key_values
        _sync()
        cache_mib = torch.cuda.memory_allocated() / MIB - weight_mib
        next_tok = torch.zeros((B, 1), dtype=torch.long, device=device)
        for _ in range(warmup):
            out = model(next_tok, past_key_values=past, **kw)
            past = out.past_key_values
        _sync()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(G):
            _sync(); t0 = time.perf_counter()
            out = model(next_tok, past_key_values=past, **kw)
            past = out.past_key_values
            _sync(); times.append((time.perf_counter() - t0) * 1e3)
        peak = torch.cuda.max_memory_allocated() / MIB
        total_s = sum(times) / 1e3
        return dict(per_seq_ms=statistics.median(times),
                    throughput_tok_s=(B * G) / total_s,
                    peak_decode_mib=peak, cache_mib=cache_mib, weight_mib=weight_mib)
    except Exception as e:  # noqa: BLE001
        if _oom(e):
            torch.cuda.empty_cache(); return {"oom": True}
        raise


@torch.no_grad()
def bench_ours(model, vocab, L, G, warmup, device="cuda"):
    """Returns dict(ms_per_tok, peak_decode_mib, cache_mib, weight_mib) or
    {'oom': True}."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weight_mib = torch.cuda.memory_allocated() / MIB

        prompt = torch.randint(0, vocab, (1, L), device=device, dtype=torch.long)
        # No outer autocast — see the note in bench_ours_batched above and
        # the 2026-07-01 correction in DECODE_COST_BENCH.md.
        prefill_out = model.prefill(prompt)
        cache = prefill_out[0]
        # Drop the transient (1, L, vocab) prefill-logits tensor so it does
        # not contaminate the decode-phase peak below; measure the true
        # bounded recurrent state directly from the FLA cache.
        del prefill_out
        _sync()
        cache_mib = _fla_state_bytes(cache["fla_cache"])

        next_tok = torch.zeros((1, 1), dtype=torch.long, device=device)

        # Warmup (kernel autotune / allocator settle) — excluded from timing.
        for _ in range(warmup):
            _, cache = model.forward_step(next_tok, cache)
        _sync()

        # Reset peak AFTER prefill so we measure the decode-phase footprint
        # (weights + recurrent state + per-step activation), not the O(L)
        # prefill transient.
        torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(G):
            _sync()
            t0 = time.perf_counter()
            _, cache = model.forward_step(next_tok, cache)
            _sync()
            times.append((time.perf_counter() - t0) * 1e3)
        peak_decode_mib = torch.cuda.max_memory_allocated() / MIB
        return dict(
            ms_per_tok=statistics.median(times),
            ms_p10=sorted(times)[len(times) // 10],
            peak_decode_mib=peak_decode_mib,
            cache_mib=cache_mib,
            weight_mib=weight_mib,
        )
    except Exception as e:  # noqa: BLE001
        if _oom(e):
            torch.cuda.empty_cache()
            return {"oom": True}
        raise


# --------------------------------------------------------------------------- #
# BASELINE — SmolLM2-360M HF transformer, growing KV cache.
# --------------------------------------------------------------------------- #
def load_transformer(name="HuggingFaceTB/SmolLM2-360M"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16).cuda().eval()
    return model


@torch.no_grad()
def bench_transformer(model, vocab, L, G, warmup, device="cuda"):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        weight_mib = torch.cuda.memory_allocated() / MIB

        prompt = torch.randint(0, vocab, (1, L), device=device, dtype=torch.long)
        # logits_to_keep=1 → only last-position logits at prefill (realistic
        # serving; avoids an O(L*vocab) prefill-logits transient that would
        # otherwise dwarf the KV cache and confound the memory read).
        try:
            out = model(prompt, use_cache=True, logits_to_keep=1)
        except TypeError:
            out = model(prompt, use_cache=True)
        past = out.past_key_values
        _sync()
        cache_mib = torch.cuda.memory_allocated() / MIB - weight_mib

        next_tok = torch.zeros((1, 1), dtype=torch.long, device=device)

        for _ in range(warmup):
            out = model(next_tok, past_key_values=past, use_cache=True,
                        logits_to_keep=1) if _supports_ltk(model) else \
                  model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
        _sync()
        torch.cuda.reset_peak_memory_stats()

        ltk = _supports_ltk(model)
        times = []
        for _ in range(G):
            _sync()
            t0 = time.perf_counter()
            if ltk:
                out = model(next_tok, past_key_values=past, use_cache=True,
                            logits_to_keep=1)
            else:
                out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            _sync()
            times.append((time.perf_counter() - t0) * 1e3)
        peak_decode_mib = torch.cuda.max_memory_allocated() / MIB
        return dict(
            ms_per_tok=statistics.median(times),
            ms_p10=sorted(times)[len(times) // 10],
            peak_decode_mib=peak_decode_mib,
            cache_mib=cache_mib,
            weight_mib=weight_mib,
        )
    except Exception as e:  # noqa: BLE001
        if _oom(e):
            torch.cuda.empty_cache()
            return {"oom": True}
        raise


_LTK_CACHE = {}


def _supports_ltk(model) -> bool:
    key = id(model)
    if key not in _LTK_CACHE:
        import inspect
        try:
            sig = inspect.signature(model.forward)
            _LTK_CACHE[key] = "logits_to_keep" in sig.parameters
        except (TypeError, ValueError):
            _LTK_CACHE[key] = False
    return _LTK_CACHE[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/linearize/linearized_stage3.pt")
    ap.add_argument("--transformer", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--lengths", default="256,512,1024,2048,4096,8192,16384,32768,65536,131072")
    ap.add_argument("--batches", default="1,8,32,64,128,256")
    ap.add_argument("--batch_lengths", default="2048,8192,32768")
    ap.add_argument("--prefill_chunk", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default="checkpoints/decode_bench/results.json")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    batch_lengths = [int(x) for x in args.batch_lengths.split(",") if x.strip()]
    device = "cuda"
    assert torch.cuda.is_available(), "needs CUDA"
    gpu = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {gpu}  free={free/MIB:.0f} MiB / {total/MIB:.0f} MiB")

    # ---- OURS ----
    print("\n[loading OURS]", args.ckpt)
    ours, cfg = load_ours(args.ckpt)
    vocab = int(cfg["vocab_size"])
    ours_params = sum(p.numel() for p in ours.parameters()) / 1e6
    print(f"  OURS: {cfg.get('n_layers')}L x {cfg.get('d_model')}d, "
          f"{ours_params:.1f}M params, feedback={cfg.get('feedback')}, "
          f"use_memory={getattr(ours, 'use_memory', False)}")

    ours_rows = {}
    for L in lengths:
        r = bench_ours(ours, vocab, L, args.gen, args.warmup, device)
        ours_rows[L] = r
        if r.get("oom"):
            print(f"  L={L:>7}: OOM")
        else:
            print(f"  L={L:>7}: {r['ms_per_tok']:7.3f} ms/tok  "
                  f"peak={r['peak_decode_mib']:8.1f} MiB  "
                  f"cache={r['cache_mib']:7.2f} MiB")

    print("\n  [OURS batched B x L]")
    ours_batched = {}
    for L in batch_lengths:
        for B in batches:
            r = bench_ours_batched(ours, vocab, L, B, args.gen, args.warmup,
                                   args.prefill_chunk, device)
            ours_batched[(B, L)] = r
            if r.get("oom"):
                print(f"  B={B:>4} L={L:>6}: OOM")
            else:
                print(f"  B={B:>4} L={L:>6}: {r['throughput_tok_s']:9.1f} tok/s  "
                      f"{r['per_seq_ms']:7.2f} ms/tok/seq  "
                      f"peak={r['peak_decode_mib']:8.1f} MiB  state={r['cache_mib']:7.2f} MiB")

    del ours
    gc.collect()
    torch.cuda.empty_cache()

    # ---- TRANSFORMER ----
    print("\n[loading TRANSFORMER]", args.transformer)
    xf = load_transformer(args.transformer)
    xf_params = sum(p.numel() for p in xf.parameters()) / 1e6
    print(f"  {args.transformer}: {xf_params:.1f}M params")

    xf_rows = {}
    for L in lengths:
        r = bench_transformer(xf, vocab, L, args.gen, args.warmup, device)
        xf_rows[L] = r
        if r.get("oom"):
            print(f"  L={L:>7}: OOM")
        else:
            print(f"  L={L:>7}: {r['ms_per_tok']:7.3f} ms/tok  "
                  f"peak={r['peak_decode_mib']:8.1f} MiB  "
                  f"cache={r['cache_mib']:7.2f} MiB")

    print("\n  [TRANSFORMER batched B x L]")
    xf_batched = {}
    for L in batch_lengths:
        for B in batches:
            r = bench_transformer_batched(xf, vocab, L, B, args.gen, args.warmup,
                                          device, chunk=args.prefill_chunk)
            xf_batched[(B, L)] = r
            if r.get("oom"):
                print(f"  B={B:>4} L={L:>6}: OOM")
            else:
                print(f"  B={B:>4} L={L:>6}: {r['throughput_tok_s']:9.1f} tok/s  "
                      f"{r['per_seq_ms']:7.2f} ms/tok/seq  "
                      f"peak={r['peak_decode_mib']:8.1f} MiB  KV={r['cache_mib']:8.1f} MiB")

    # ---- TABLE ----
    print("\n" + "=" * 100)
    hdr = (f"{'L':>7} | {'ours ms/tok':>11} | {'ours peak MiB':>13} | "
           f"{'xf ms/tok':>10} | {'xf peak MiB':>11} | {'speedup':>8} | {'mem ratio':>9}")
    print(hdr)
    print("-" * len(hdr))
    table = []
    for L in lengths:
        o, x = ours_rows[L], xf_rows[L]
        o_ms = "OOM" if o.get("oom") else f"{o['ms_per_tok']:.3f}"
        o_pk = "OOM" if o.get("oom") else f"{o['peak_decode_mib']:.1f}"
        x_ms = "OOM" if x.get("oom") else f"{x['ms_per_tok']:.3f}"
        x_pk = "OOM" if x.get("oom") else f"{x['peak_decode_mib']:.1f}"
        if not o.get("oom") and not x.get("oom"):
            spd = f"{x['ms_per_tok'] / o['ms_per_tok']:.2f}x"
            mr = f"{x['peak_decode_mib'] / o['peak_decode_mib']:.2f}x"
        elif x.get("oom") and not o.get("oom"):
            spd, mr = "xf-OOM", "xf-OOM"
        else:
            spd, mr = "-", "-"
        print(f"{L:>7} | {o_ms:>11} | {o_pk:>13} | {x_ms:>10} | {x_pk:>11} | "
              f"{spd:>8} | {mr:>9}")
        table.append(dict(L=L, ours=o, transformer=x, speedup=spd, mem_ratio=mr))

    # ---- BATCHED TABLE ----
    print("\n" + "=" * 118)
    print("BATCHED (throughput = total tok/s across B parallel seqs; "
          "per-seq ms/tok; peak MiB)")
    bhdr = (f"{'B':>4} {'L':>6} | {'ours tok/s':>11} {'ours ms/tok':>11} "
            f"{'ours MiB':>9} | {'xf tok/s':>11} {'xf ms/tok':>11} {'xf MiB':>9} "
            f"| {'tput x':>7} {'mem x':>7}")
    print(bhdr)
    print("-" * len(bhdr))
    btable = []
    for L in batch_lengths:
        for B in batches:
            o, x = ours_batched[(B, L)], xf_batched[(B, L)]
            o_oom, x_oom = o.get("oom"), x.get("oom")
            o_tp = "OOM" if o_oom else f"{o['throughput_tok_s']:.1f}"
            o_ms = "OOM" if o_oom else f"{o['per_seq_ms']:.2f}"
            o_pk = "OOM" if o_oom else f"{o['peak_decode_mib']:.0f}"
            x_tp = "OOM" if x_oom else f"{x['throughput_tok_s']:.1f}"
            x_ms = "OOM" if x_oom else f"{x['per_seq_ms']:.2f}"
            x_pk = "OOM" if x_oom else f"{x['peak_decode_mib']:.0f}"
            if not o_oom and not x_oom:
                tputx = f"{o['throughput_tok_s'] / x['throughput_tok_s']:.2f}x"
                memx = f"{x['peak_decode_mib'] / o['peak_decode_mib']:.2f}x"
            elif x_oom and not o_oom:
                tputx, memx = "xf-OOM", "xf-OOM"
            else:
                tputx, memx = "-", "-"
            print(f"{B:>4} {L:>6} | {o_tp:>11} {o_ms:>11} {o_pk:>9} | "
                  f"{x_tp:>11} {x_ms:>11} {x_pk:>9} | {tputx:>7} {memx:>7}")
            btable.append(dict(B=B, L=L, ours=o, transformer=x,
                               tput_ratio=tputx, mem_ratio=memx))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(
            gpu=gpu, ours_ckpt=args.ckpt, transformer=args.transformer,
            ours_params_m=ours_params, transformer_params_m=xf_params,
            gen=args.gen, warmup=args.warmup, vocab=vocab,
            ours_config=cfg, lengths=lengths, table=table,
            batches=batches, batch_lengths=batch_lengths,
            prefill_chunk=args.prefill_chunk, batched_table=btable,
        ), f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
