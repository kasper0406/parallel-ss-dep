"""torch.profiler harness for the pretrain hot path.

Runs N training steps of the real model config under torch.profiler and
prints the top CUDA ops by self-time, plus a forward/backward/optimizer
breakdown. Use it to quantify where the ~400 ms/step goes and to A/B
test speed knobs (--compile, --feedback_self_k_warmup_steps).

Needs a free GPU — do NOT run co-resident with a real training job, the
numbers will be polluted by contention.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python \\
        experiments/profile_train.py \\
        --data_mix configs/pretrain_mix_v2_with_cve.yaml \\
        --steps 30 --warmup 8 --compile

The model/data flags mirror train_lm.py's launcher so the profile
reflects the actual run. Only the knobs under test need to be passed;
the rest default to the v3 launcher values.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile


def _build(args):
    from transformers import AutoTokenizer
    from experiments.data_mix import load_sources_from_yaml, MixedSourceStream
    from experiments.model_builder import build_model_from_args
    from experiments.optim_utils import build_optimizer
    from experiments.speed_knobs import apply_speed_knobs

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    base_vocab = tok.vocab_size
    vocab_size = ((base_vocab + 1 + 63) // 64) * 64
    thinking_token_id = base_vocab

    # argparse namespace the builders expect — mirror the v3 launcher.
    ns = argparse.Namespace(
        arch="deltanet", layers=None, d_model=576, n_layers=30, d_head=64,
        n_heads=9, max_T=args.T, feedback="film", feedback_pairs="2,28",
        feedback_xattn="", feedback_xattn_heads=1, feedback_xattn_form="film",
        feedback_lag=1, feedback_position="pre", feedback_per_channel_alpha=False,
        feedback_self_k=3, feedback_alpha_mode="scalar", aux_brackets=False,
        aux_max_depth=0, output_gate=True, enable_thinking_token=False,
        think_decision="gate", use_memory=True, mem_size=1024, mem_dim=0,
        activation_checkpointing=args.activation_checkpointing,
        layer_drop_max=0.0, load_ckpt=None,
    )
    model, _ = build_model_from_args(
        ns, vocab_size=vocab_size, thinking_token_id=thinking_token_id)
    apply_speed_knobs(model, bf16=True, tf32=True, compile_model=args.compile)
    opts, _ = build_optimizer(
        model, optimizer="muon", lr=3e-4, lr_muon=1e-3, alpha_wd=0.0,
        steps=args.steps, wd=0.01, verbose=False)

    sources = load_sources_from_yaml(args.data_mix)
    stream = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=args.T,
        thinking_token_id=thinking_token_id, think_burst_prob=0.5,
        think_max_bursts=2, think_max_burst_depth=6, base_seed=0)
    return model, opts, iter(stream), thinking_token_id


def _step(model, opts, batch, film_bypass):
    import torch.nn.functional as F
    model._film_bypass = film_bypass
    x, y = batch
    x, y = x.cuda(), y.cuda()
    for o in opts:
        o.zero_grad(set_to_none=True)
    logits = model(x)
    V = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                           ignore_index=-100)
    loss.backward()
    for o in opts:
        o.step()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mix", required=True)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=7)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8,
                    help="Steps to run before profiling (kernel autotune "
                         "+ compile warmup land here).")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--film_bypass", action="store_true",
                    help="Profile the K-self-feed-bypassed (1-pass) path.")
    ap.add_argument("--activation_checkpointing", action="store_true")
    args = ap.parse_args()

    print(f"[profile] building model (compile={args.compile} "
          f"film_bypass={args.film_bypass} T={args.T} batch={args.batch})")
    model, opts, data_iter, _ = _build(args)
    model.train()

    def next_batch():
        xs, ys = [], []
        for _ in range(args.batch):
            xx, yy = next(data_iter)
            xs.append(xx); ys.append(yy)
        return torch.stack(xs), torch.stack(ys)

    print(f"[profile] warmup {args.warmup} steps ...")
    for _ in range(args.warmup):
        _step(model, opts, next_batch(), args.film_bypass)
    torch.cuda.synchronize()

    # Wall-clock timing (clean, no profiler overhead).
    t0 = time.perf_counter()
    for _ in range(args.steps):
        _step(model, opts, next_batch(), args.film_bypass)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    tok_per_step = args.batch * args.T
    print(f"[profile] wall: {wall/args.steps*1000:.1f} ms/step  "
          f"{tok_per_step*args.steps/wall:,.0f} tok/s")

    # Profiled run for the op breakdown.
    print(f"[profile] profiling {args.steps} steps ...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(args.steps):
            _step(model, opts, next_batch(), args.film_bypass)
        torch.cuda.synchronize()

    print("\n=== top 20 CUDA ops by self-time ===")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
