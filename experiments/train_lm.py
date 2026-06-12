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
import contextlib
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from experiments.layers import (
    DeltaNetAttention,
    SoftmaxAttention, Mamba2Attention,
)
from experiments.model import TinyLM
from experiments.aux_brackets import compute_bracket_deltas, bracket_depth
from experiments.gist_loss import build_gist_heads, parse_horizons
from experiments.thinking import (
    ThinkContinuation,
    ThinkContinuationQueue,
    ThinkReplay,
    ThinkReplayQueue,
    build_continuation_batch,
    build_replay_batch,
    choose_explore_actions,
    choose_think_actions,
    cross_entropy_masking_token,
    latent_cotrain_loss,
    mask_token_logit,
)


from experiments.gate_calibration import compute_gate_calibration_loss
# LatentReasoningCotrain is imported lazily inside main() — its dependency chain
# (latent_sft → eval_bracket_structure → train_lm) would otherwise be circular.
from experiments.build_arch import build_arch, parse_layers_arg, _NAME_TO_CLS  # noqa: F401


# ---------------------------------------------------------------------------
# Per-layer learning diagnostics. Logged every --log_every steps to answer a
# specific question: are early layers gradient-starved (the vanishing-gradient
# signature) or is the gradient healthy and "slow early layers" just a misread
# of the logit-lens? Cheap — grad norms are free post-backward; the
# update-ratio clones one matrix per block on log steps only.

def _block_repr_weight(blk):
    """First 2D weight in a block — the representative matrix tracked for the
    update-to-weight ratio."""
    return next((p for p in blk.parameters() if p.ndim == 2), None)


def _block_grad_norms(model) -> list[float]:
    """Per-block total gradient L2 norm. Call after backward(), before clip."""
    out = []
    for blk in model.blocks:
        sq = 0.0
        for p in blk.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
        out.append(sq ** 0.5)
    return out


def _block_weight_snapshot(model) -> list:
    """Clone each block's representative weight (call pre-step)."""
    snap = []
    for blk in model.blocks:
        w = _block_repr_weight(blk)
        snap.append(None if w is None else w.detach().clone())
    return snap


def _block_update_ratios(model, snapshot) -> list[float]:
    """‖ΔW‖/‖W‖ for each block's representative weight (call post-step)."""
    out = []
    for blk, w_before in zip(model.blocks, snapshot):
        w_after = _block_repr_weight(blk)
        if w_before is None or w_after is None:
            out.append(float("nan"))
            continue
        delta = float((w_after.detach() - w_before).float().norm())
        denom = max(float(w_before.float().norm()), 1e-9)
        out.append(delta / denom)
    return out


def _nonthink_forward_loss(model, x, y, args, step, bracket_deltas,
                           doc_ids=None, gist_horizons=None, fwd_model=None):
    """Forward + LM loss for the non-thinking-token (pretrain) path.

    Returns (logits, ce_per_token, lm_loss, aux_loss, gate_aux_loss,
    gist_loss). Factored out of the step loop so gradient accumulation
    can run it once per microbatch. Mirrors the inline forward +
    gate/plain-loss branches exactly.

    When `gist_horizons` is set and --gist_loss_weight > 0, the model
    computes the multi-horizon trunk gist loss INSIDE its forward and
    returns it as a scalar (model._gist_loss_enabled gates it). Not
    supported together with --aux_brackets.

    When --gate_entropy_aux_weight > 0, the gate logit gets an auxiliary
    BCE target derived from the SAME forward's per-position next-token
    entropy (detached): target_t = exp(-H_t/T). High entropy ⇒ low target
    ⇒ gate trained to close (think). Costs nothing extra — no second
    forward, just turns the gate into a free predictive-uncertainty head.
    """
    # `fwd_model` is the (possibly DDP-wrapped) callable used for the
    # loss-bearing forward so DDP's grad-sync hooks fire; `model` stays the raw
    # module for attribute reads (_last_gate, pkm_layer, …) which DDP doesn't
    # proxy. Default fwd_model=model preserves the single-GPU path exactly.
    fwd_model = fwd_model if fwd_model is not None else model
    want_gist = (gist_horizons is not None
                 and getattr(args, "gist_loss_weight", 0.0) > 0.0)
    if args.aux_brackets:
        if want_gist:
            raise SystemExit("--gist_loss_weight is not supported "
                             "together with --aux_brackets.")
        logits, aux_logits = fwd_model(x, return_aux=True, doc_ids=doc_ids)
        depth = bracket_depth(x, bracket_deltas).clamp(0, args.aux_max_depth)
        aux_loss = F.cross_entropy(
            aux_logits.reshape(-1, args.aux_max_depth + 1),
            depth.reshape(-1),
        )
        gist_loss = logits.new_zeros(())
    elif want_gist:
        # The model computes the gist loss inside the (compiled) forward
        # and returns it as a scalar — see TinyLM._finalize. This keeps
        # the hidden state from ever crossing the compile boundary.
        logits, gist_loss = fwd_model(x, doc_ids=doc_ids)
        aux_loss = logits.new_zeros(())
    else:
        logits = fwd_model(x, doc_ids=doc_ids)
        aux_loss = logits.new_zeros(())
        gist_loss = logits.new_zeros(())
    V = logits.shape[-1]
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, V), y.reshape(-1), reduction="none",
    ).reshape(y.shape)                                                   # (B, T)
    if args.output_gate:
        g = model._last_gate                                             # (B, T)
        if args.gate_warmup_steps > 0:
            progress = min(1.0, step / args.gate_warmup_steps)
            gate_floor = (1.0 - progress) * 1.0 + progress * args.gate_floor_min
        else:
            gate_floor = args.gate_floor_min
        g_eff = g.clamp(min=gate_floor) if gate_floor > 0.0 else g
        # Ponder cost. The CE term uses the floor-clamped g_eff (so real-token
        # loss keeps weight >= floor — the gate_floor_min anti-collapse fix).
        # The THINK-cost term, however, must penalise the RAW gate g, not g_eff:
        # clamp() zeros the gradient to g whenever g < floor, so with the cost
        # on g_eff the *raw* gate (the one generation uses, with no clamp) is
        # never penalised for thinking -> it over-thinks at deploy (think_frac
        # ~0.6 observed). Costing the raw g restores that gradient so the gate
        # learns to emit unless CE > gate_lambda. Default off (backwards-compat).
        ponder_gate = g if getattr(args, "gate_ponder_raw", False) else g_eff
        gate_terms = g_eff * ce_per_token + (1.0 - ponder_gate) * args.gate_lambda
        valid = (y != -100).float()
        denom = valid.sum().clamp(min=1.0)
        lm_loss = (gate_terms * valid).sum() / denom
    else:
        valid = (y != -100).float()
        denom = valid.sum().clamp(min=1.0)
        lm_loss = (ce_per_token * valid).sum() / denom
    # Entropy-grounded gate target (CE-reduction self-reward, cheap form).
    gate_aux_loss = torch.zeros((), device=logits.device)
    if args.output_gate and args.gate_entropy_aux_weight > 0.0:
        gate_logits = model._last_gate_logits                            # (B, T)
        # logsumexp + p·logp = stable entropy. We detach the source logits
        # because the target is a SELF-supervised signal — gradient must
        # flow into the gate head, not into the LM head.
        lse = torch.logsumexp(logits.detach(), dim=-1)                   # (B, T)
        # H_t = lse - sum(p * raw_logit) = lse - mean over support
        # using p = softmax(logits): H = lse - sum(p*logits)
        p = (logits.detach() - lse.unsqueeze(-1)).exp()                  # (B, T, V)
        H = lse - (p * logits.detach()).sum(dim=-1)                      # (B, T) ≥ 0
        T = max(args.gate_entropy_aux_temperature, 1e-6)
        target = torch.exp(-H / T).clamp(0.0, 1.0)                       # (B, T)
        c = args.gate_entropy_aux_target_clamp
        if c > 0.0:
            target = target.clamp(c, 1.0 - c)
        # BCE-with-logits over valid (non-ignored) positions only.
        valid = (y != -100).float()
        bce = F.binary_cross_entropy_with_logits(
            gate_logits, target, reduction="none",
        )
        denom = valid.sum().clamp(min=1.0)
        gate_aux_loss = (bce * valid).sum() / denom
    return logits, ce_per_token, lm_loss, aux_loss, gate_aux_loss, gist_loss


def _z_loss_term(logits, weight):
    """z-loss regulariser: weight * mean(logsumexp(logits)^2)."""
    if weight <= 0.0:
        return logits.new_zeros(())
    return weight * (torch.logsumexp(logits, dim=-1) ** 2).mean()


def _pkm_diversity_loss(pkm) -> torch.Tensor:
    """Negative entropy of the per-head slot-selection distribution.

    Aggregates the per-head top-k retrievals into a (n_heads, n_slots)
    histogram and returns the mean of (-H_per_head). Minimising this loss
    increases entropy — encouraging the model to spread retrievals across
    the full table rather than concentrate on a handful of hot slots.

    Operates on the STASHED detached indices/weights from the last forward,
    so this aux loss does NOT backprop through the router. It only affects
    the value-table grads (which see a slightly more diverse retrieval
    pattern across a training run).
    """
    if pkm._last_slot_idx is None:
        # Forward hasn't been called yet (curriculum order during start);
        # return a no-op zero.
        device = next(pkm.parameters()).device
        return torch.zeros((), device=device)
    slot_idx = pkm._last_slot_idx       # (B, T, H, top_k), int64
    weights = pkm._last_weights         # (B, T, H, top_k), float
    H = pkm.n_heads
    n_slots = pkm.n_keys * pkm.n_keys
    B, T, _, tk = slot_idx.shape
    # Build a per-head slot mass: (H, n_slots).
    # Flatten (B,T,top_k) → mass scatter-adds into slot bins.
    device = slot_idx.device
    mass = torch.zeros(H, n_slots, device=device, dtype=torch.float32)
    for h in range(H):
        idx_h = slot_idx[:, :, h, :].reshape(-1)
        w_h = weights[:, :, h, :].reshape(-1).float()
        mass[h].scatter_add_(0, idx_h, w_h)
    # Normalise per head so each row sums to 1.
    mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    # H(p) = -sum p log p ; we MAXIMISE entropy → minimise neg_entropy.
    neg_entropy = (mass * (mass.clamp_min(1e-12).log())).sum(dim=-1)  # (H,)
    return neg_entropy.mean()


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
    from experiments.train_lm_args import build_parser
    p = build_parser()
    args = p.parse_args()

    # --- DDP (batch-parallel, grads-only all-reduce; no NVLink needed) -------
    # Activated when launched under torchrun (WORLD_SIZE>1). Each rank holds a
    # full model copy on its own GPU and processes a disjoint data shard; only
    # gradients cross the PCIe bus (NCCL all-reduce). world_size==1 (plain
    # `python ...`) is byte-identical to the legacy single-GPU path.
    import os as _os
    ddp_world_size = int(_os.environ.get("WORLD_SIZE", "1"))
    ddp_rank = int(_os.environ.get("RANK", "0"))
    ddp_local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
    is_ddp = ddp_world_size > 1
    is_main = ddp_rank == 0
    if is_ddp:
        if args.enable_thinking_token:
            raise SystemExit(
                "DDP (WORLD_SIZE>1) is wired for the non-thinking-token "
                "(pretrain) path only; the thinking-token queue path is not "
                "DDP-safe (per-rank dynamic queues desync). Run it single-GPU."
            )
        torch.cuda.set_device(ddp_local_rank)
        torch.distributed.init_process_group(backend="nccl")
        print(f"[ddp] rank {ddp_rank}/{ddp_world_size} "
              f"local_rank={ddp_local_rank} device=cuda:{ddp_local_rank}",
              flush=True)

    def _mainprint(*a, **k):
        if is_main:
            print(*a, **k)

    if args.enable_thinking_token and args.output_gate:
        raise SystemExit(
            "--enable_thinking_token and --output_gate are mutually exclusive. "
            "Use --think_decision gate to train the discrete THINKING path "
            "with a binary gate head."
        )
    if args.enable_thinking_token and args.think_lambda < 0:
        raise SystemExit("--think_lambda must be non-negative.")
    if (args.enable_thinking_token and args.think_lambda_start is not None
            and args.think_lambda_start < 0):
        raise SystemExit("--think_lambda_start must be non-negative.")
    if args.enable_thinking_token and args.think_curriculum_steps < 0:
        raise SystemExit("--think_curriculum_steps must be non-negative.")
    if args.enable_thinking_token and args.think_queue_max <= 0:
        raise SystemExit("--think_queue_max must be positive.")
    if args.enable_thinking_token and not (0.0 <= args.think_explore_prob <= 1.0):
        raise SystemExit("--think_explore_prob must be in [0, 1].")
    if args.enable_thinking_token and not (0.0 <= args.think_explore_start_prob <= 1.0):
        raise SystemExit("--think_explore_start_prob must be in [0, 1].")
    if args.enable_thinking_token and args.think_min_fresh_rows < 0:
        raise SystemExit("--think_min_fresh_rows must be non-negative.")
    if args.enable_thinking_token and not (0.0 < args.think_gate_threshold < 1.0):
        raise SystemExit("--think_gate_threshold must be in (0, 1).")
    if (args.enable_thinking_token and args.think_gate_threshold_start is not None
            and not (0.0 < args.think_gate_threshold_start < 1.0)):
        raise SystemExit("--think_gate_threshold_start must be in (0, 1).")
    if args.enable_thinking_token and args.think_max_new_per_step < 0:
        raise SystemExit("--think_max_new_per_step must be non-negative.")
    if args.enable_thinking_token and args.think_safety_max_depth < 0:
        raise SystemExit("--think_safety_max_depth must be non-negative.")
    if (args.enable_thinking_token and args.think_safety_max_depth_start is not None
            and args.think_safety_max_depth_start < 0):
        raise SystemExit("--think_safety_max_depth_start must be non-negative.")
    if args.enable_thinking_token and args.think_replay_weight < 0:
        raise SystemExit("--think_replay_weight must be non-negative.")
    if args.enable_thinking_token and args.think_aux_loss_scale < 0:
        raise SystemExit("--think_aux_loss_scale must be non-negative.")
    if args.grad_accum < 1:
        raise SystemExit("--grad_accum must be >= 1.")
    if args.grad_accum > 1 and args.enable_thinking_token:
        raise SystemExit(
            "--grad_accum > 1 is only supported on the non-thinking-token "
            "(pretrain) path; the thinking-token path has its own "
            "--think_queue_accum_steps."
        )
    if args.enable_thinking_token and args.think_queue_accum_steps < 0:
        raise SystemExit("--think_queue_accum_steps must be non-negative.")
    if args.enable_thinking_token and args.think_queue_accum_max_steps < 0:
        raise SystemExit("--think_queue_accum_max_steps must be non-negative.")
    if (args.enable_thinking_token and args.think_queue_accum_max_steps > 0
            and args.think_queue_accum_max_steps < args.think_queue_accum_steps):
        raise SystemExit(
            "--think_queue_accum_max_steps must be >= --think_queue_accum_steps."
        )
    if args.enable_thinking_token and args.think_queue_drain_target < -1:
        raise SystemExit("--think_queue_drain_target must be >= -1.")
    if args.enable_thinking_token and args.think_backpressure_target < -1:
        raise SystemExit("--think_backpressure_target must be >= -1.")
    if args.enable_thinking_token and args.think_backpressure_max < 0:
        raise SystemExit("--think_backpressure_max must be non-negative.")
    if args.enable_thinking_token and args.think_backpressure_lambda < 0:
        raise SystemExit("--think_backpressure_lambda must be non-negative.")
    if args.enable_thinking_token and args.think_backpressure_threshold < 0:
        raise SystemExit("--think_backpressure_threshold must be non-negative.")
    if args.enable_thinking_token and args.think_backpressure_explore < 0:
        raise SystemExit("--think_backpressure_explore must be non-negative.")
    if args.enable_thinking_token and args.think_gate_emit_weight < 0:
        raise SystemExit("--think_gate_emit_weight must be non-negative.")
    if args.enable_thinking_token and args.aux_brackets:
        raise SystemExit(
            "--enable_thinking_token currently supports the standard LM loss "
            "path only; disable aux losses for thinking experiments."
        )

    torch.manual_seed(args.seed)
    arch_label = args.arch if args.arch else f"layers={args.layers}"
    print(f"GPU: {torch.cuda.get_device_name(0)}  arch={arch_label}")

    # 1. Tokeniser + dataset.
    from transformers import AutoTokenizer
    from datasets import load_dataset
    print(f"Loading tokeniser {args.tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    thinking_token_id = None
    added_thinking_tokens = 0
    if args.enable_thinking_token:
        added_thinking_tokens = tok.add_special_tokens(
            {"additional_special_tokens": [args.thinking_token]}
        )
        thinking_token_id = tok.convert_tokens_to_ids(args.thinking_token)
        if thinking_token_id is None or thinking_token_id < 0:
            raise SystemExit(
                f"failed to add/resolve thinking token {args.thinking_token!r}"
            )
    if args.data_mix:
        # Mixed-corpus pretrain. Reserve one slot above the base tokenizer
        # vocab for the think token; round model vocab up to multiple-of-64
        # so embedding / lm_head dims are GPU-friendly.
        thinking_token_id = int(tok.vocab_size)
        model_vocab_size = ((int(tok.vocab_size) + 1 + 63) // 64) * 64
    else:
        model_vocab_size = len(tok)
    print(f"  vocab size: base={tok.vocab_size}, model={model_vocab_size}"
          f"{f', thinking_id={thinking_token_id}' if thinking_token_id is not None else ''}")

    if args.data_mix:
        print(f"Loading data mix from {args.data_mix} (streaming) ...")
        from experiments.data_mix import (
            MixedSourceStream, load_sources_from_yaml,
        )
        sources = load_sources_from_yaml(args.data_mix)
        print(f"  {len(sources)} sources:")
        for s in sources:
            print(f"    - {s.name:30s} weight={s.weight:.3f}  id={s.dataset_id}")
        train_ds = MixedSourceStream(
            sources=sources, tokenizer=tok, block_size=args.T,
            thinking_token_id=thinking_token_id,
            think_burst_prob=args.think_burst_prob,
            think_max_bursts=args.think_max_bursts,
            think_max_burst_depth=args.think_max_burst_depth,
            # Per-rank seed offset → each DDP rank streams a disjoint shard.
            base_seed=args.seed + ddp_rank * 100_003,
            mask_eos_in_targets=bool(args.mask_eos_in_targets),
            emit_doc_ids=True,
        )
        # Val: same sources, different seed, burst injection off so val PPL
        # reflects the clean data distribution.
        val_ds = MixedSourceStream(
            sources=sources, tokenizer=tok, block_size=args.T,
            thinking_token_id=thinking_token_id,
            think_burst_prob=0.0,
            base_seed=args.seed + 999_983,
            mask_eos_in_targets=bool(args.mask_eos_in_targets),
            emit_doc_ids=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=1)
    else:
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
            val_stream = load_dataset(args.dataset, split="validation",
                                       **ds_kwargs)
        except (ValueError, KeyError):
            # No validation split — split off a slice of train as held-out.
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
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                   num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=1)

    # 2. Model — see experiments/model_builder.py.
    from experiments.model_builder import build_model_from_args
    model, _build_info = build_model_from_args(
        args, vocab_size=model_vocab_size,
        thinking_token_id=thinking_token_id,
    )
    fb_pairs = _build_info.fb_pairs
    fb_xattn_pairs = _build_info.fb_xattn_pairs
    n_layers_actual = _build_info.n_layers
    aux_dim = _build_info.aux_dim
    # ---- Speed knobs (must run AFTER model is built but BEFORE the train
    # loop touches it). See experiments/speed_knobs.py.
    # The latent co-train aux losses run extra eager forwards at short/odd
    # shapes; under torch.compile (strict mode, no silent fallback) that
    # reproduces the documented Inductor symbolic-shape assertion (2026-05-27
    # gate-calibration smoke, bug #1) as a hard crash mid-run. Auto-disable
    # compile rather than relying on every launcher remembering --no-compile.
    if bool(args.compile) and (getattr(args, "latent_cotrain_weight", 0.0) > 0.0
                               or getattr(args, "latent_reasoning_weight", 0.0) > 0.0):
        print("[compile] AUTO-DISABLED: --latent_cotrain_weight/"
              "--latent_reasoning_weight run variable-shape extra forwards "
              "that crash Inductor under strict compile. Pass --no-compile "
              "to silence this message.")
        args.compile = False
    from experiments.speed_knobs import apply_speed_knobs
    apply_speed_knobs(model, bf16=bool(args.bf16), tf32=bool(args.tf32),
                      compile_model=bool(args.compile),
                      compile_mode=args.compile_mode)
    if fb_xattn_pairs:
        n_total_pairs = sum(len(srcs) for _, srcs in fb_xattn_pairs)
        feedback_desc = (f"xattn[{args.feedback_xattn_form}]"
                         f"(targets={len(fb_xattn_pairs)},"
                         f"src-edges={n_total_pairs},"
                         f"heads={args.feedback_xattn_heads})")
    else:
        feedback_desc = f"{args.feedback}"
    print(f"  params: {model.num_params() / 1e6:.1f}M  aux_dim={aux_dim}  "
          f"feedback={feedback_desc}")

    if args.aux_brackets:
        print("Computing bracket-deltas table for tokenizer ...")
        bracket_deltas = compute_bracket_deltas(tok)
        print(f"  table shape: {bracket_deltas.shape}, "
              f"non-zero count: {(bracket_deltas != 0).sum().item()}")
    else:
        bracket_deltas = None

    # Trunk multi-horizon gist heads (v7, see experiments/gist_loss.py).
    # Attached as a model submodule so they ride the existing optimizer,
    # state_dict and resume paths with no special handling. Must be
    # attached BEFORE build_optimizer so the optimizer picks them up.
    gist_horizons = None
    if args.gist_loss_weight > 0.0:
        gist_horizons = parse_horizons(args.gist_horizons)
        model.gist_heads = build_gist_heads(args.d_model,
                                            gist_horizons).cuda()
        # The model computes the gist loss inside its own forward (see
        # TinyLM._finalize) — it reads these two attributes.
        model._gist_horizons = gist_horizons
        model._gist_loss_enabled = True
        # build_model_from_args loaded --load_ckpt BEFORE gist_heads
        # existed, so its gist_heads.* keys (if any) were dropped as
        # "unexpected". Re-load just those now. The v7.1 pretrain ckpt
        # has none → fresh heads, which is correct for Phase C.
        if args.load_ckpt is not None:
            _ck = torch.load(args.load_ckpt, map_location="cuda",
                             weights_only=False)
            _sd = (_ck["state_dict"] if isinstance(_ck, dict)
                   and "state_dict" in _ck else _ck)
            _gh = {k[len("gist_heads."):]: v for k, v in _sd.items()
                   if k.startswith("gist_heads.")}
            if _gh:
                model.gist_heads.load_state_dict(_gh)
                print(f"  gist_heads: restored from {args.load_ckpt!r}")
            else:
                print("  gist_heads: fresh (absent from loaded ckpt)")
        print(f"  trunk gist loss ON: horizons={gist_horizons} "
              f"weight={args.gist_loss_weight}")

    # DDP wrap: AFTER compile (apply_speed_knobs replaced model.forward) and
    # gist-head attach, BEFORE the optimizer (which reads model.parameters() —
    # DDP shares the same parameter tensors, so the optimizer is built on the
    # raw model and `ddp_model` is used ONLY for the loss-bearing forward).
    #   * find_unused_parameters=True: the FiLM K-warmup bypass and the gate /
    #     PKM / gist heads make the set of grad-receiving params vary per step;
    #     DDP errors without this.
    #   * broadcast_buffers=False: no BN running stats to sync (PKM uses LN).
    ddp_model = model
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(model, device_ids=[ddp_local_rank],
                        output_device=ddp_local_rank,
                        find_unused_parameters=False,
                        broadcast_buffers=False,
                        gradient_as_bucket_view=True)
        # static_graph: REQUIRED with --activation_checkpointing. Reentrant
        # checkpoint fires each param's grad hook twice per backward; DDP's
        # default reducer rejects the second mark ("marked ready twice",
        # e.g. gate_head.bias). static_graph tells DDP the param-usage set is
        # fixed across iterations (true at K=3 steady state), so it tolerates
        # the double-fire and also subsumes find_unused_parameters.
        ddp_model._set_static_graph()
        # bf16 gradient compression: this rig has NO P2P (GPU0<->GPU1 is PHB /
        # chipset-not-supported), so all-reduce is host-staged at ~4 GB/s.
        # Compressing grads fp32->bf16 halves the per-step all-reduce bytes
        # (~570ms -> ~300ms for the 600M-param grad), recovering most of the
        # 2x. Lossless enough for grad sync (bf16 has fp32 exponent range).
        if not args.ddp_no_bf16_compress:
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks \
                import bf16_compress_hook
            ddp_model.register_comm_hook(state=None, hook=bf16_compress_hook)
            print("[ddp] bf16_compress_hook registered (PCIe-bound link)",
                  flush=True)
        print(f"[ddp] wrapped model on cuda:{ddp_local_rank}", flush=True)

    # Optimizer construction — see experiments/optim_utils.py.
    from experiments.optim_utils import build_optimizer
    opts, scheds = build_optimizer(
        model, optimizer=args.optimizer, lr=args.lr, lr_muon=args.lr_muon,
        alpha_wd=args.alpha_wd, steps=args.steps, wd=args.wd,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup_steps,
        decay_frac=args.lr_decay_frac,
        bf16_optim_state=args.bf16_optim_state,
        pkm_value_lr_mult=float(getattr(args, "pkm_value_lr_mult", 1.0)),
    )
    # Backwards-compat aliases used elsewhere in the loop.
    opt = opts[0]
    scheduler = scheds[0]

    # Resume support: fast-forward LR scheduler to match --start_step so the
    # cosine schedule continues from where the ckpt left off. Optimizer
    # momenta are still fresh (we don't save them in mid-eval ckpts) — a
    # brief loss-spike transient is expected on resume.
    if args.start_step > 0:
        for _ in range(args.start_step):
            for s in scheds:
                s.step()
        print(f"Fast-forwarded LR scheduler by {args.start_step} steps; "
              f"resumed lr={scheduler.get_last_lr()[0]:.2e}")

    # 3. Train loop.
    print(f"\n{'step':>6}  {'tok/s':>8}  {'tloss':>8}  {'lr':>9}")
    t0 = time.perf_counter()
    train_iter = iter(train_loader)
    last_log = t0
    last_log_step = 0
    losses = []
    # Mid-training eval state.
    mid_eval_controller = None
    tokens_seen = 0
    next_eval_at = 0
    tokens_at_last_probe = 0
    next_feature_probe_at = (int(args.feature_probe_every_tokens)
                             if getattr(args, "feature_probe_every_tokens", 0) > 0
                             else 0)
    if next_feature_probe_at:
        print(f"\nPer-feature usefulness probe enabled every "
              f"{args.feature_probe_every_tokens:,} tokens "
              f"(ablation-delta CE for WM/PKM, FiLM α, gate fire-rate).")
    if args.probe_humaneval_every_tokens > 0:
        if not pathlib.Path(args.probe_humaneval_path).exists():
            print(f"  [probe] {args.probe_humaneval_path} not found; "
                  f"run experiments/build_probe_dataset.py first. "
                  f"Disabling probe.")
            args.probe_humaneval_every_tokens = 0
    if args.mid_eval_every_tokens > 0:
        from experiments.eval_callback import EvalStopController, run_eval
        mid_eval_controller = EvalStopController(
            stop_threshold=args.auto_stop_threshold,
            k_consecutive_flat=args.auto_stop_k,
        )
        next_eval_at = int(args.mid_eval_every_tokens)
        print(f"\nMid-training eval enabled: HumanEval @ "
              f"{args.mid_eval_n_problems} problems every "
              f"{args.mid_eval_every_tokens:,} tokens. "
              f"auto_stop={args.auto_stop} (Δ<{args.auto_stop_threshold:.3f} "
              f"for {args.auto_stop_k} consecutive intervals).")
    losses_gate_window: list[tuple] = []  # (mean_g, emit_frac, raw_ce) per step
    think_queue = (ThinkContinuationQueue(args.think_queue_max)
                   if args.enable_thinking_token else None)
    think_replay_queue = (ThinkReplayQueue(args.think_queue_max)
                          if args.enable_thinking_token else None)
    think_stats_window: list[dict[str, float]] = []
    think_closed_traj_window: list[float] = []
    think_queue_batch = args.think_queue_batch or 1
    think_replay_batch = args.think_replay_batch or think_queue_batch
    if args.enable_thinking_token:
        print(f"  THINKING queue: max={args.think_queue_max:,} CPU records, "
              f"packed_cont={think_queue_batch}, packed_replay={think_replay_batch}, "
              f"accum_steps={args.think_queue_accum_steps}, "
              f"accum_max={args.think_queue_accum_max_steps or args.think_queue_accum_steps}, "
              f"drain_target={args.think_queue_drain_target}, "
              f"decision={args.think_decision}, "
              f"priority={'on' if args.think_prioritize_queue else 'off'}")
    pad_token_id = tok.eos_token_id
    if pad_token_id is None:
        pad_token_id = tok.bos_token_id if tok.bos_token_id is not None else 0

    # Depth-matched latent-reasoning co-train (2026-06-05 fix). Loads the
    # depth-bound pointer-chase corpus once; emits one answer-span latent loss
    # per optimizer step (last microbatch) at R=depth with a curriculum.
    _latent_reasoner = None
    if getattr(args, "latent_reasoning_weight", 0.0) > 0.0:
        from experiments.latent_reasoning_cotrain import LatentReasoningCotrain
        if thinking_token_id is None:
            raise SystemExit("--latent_reasoning_weight needs a thinking token "
                             "(set --thinking_token or use --data_mix).")
        _rr_rungs = [int(x) for x in args.latent_reasoning_rungs.split(",")
                     if x.strip()]
        _latent_reasoner = LatentReasoningCotrain(
            train_prefix=args.latent_reasoning_train_prefix,
            rungs=_rr_rungs, tok=tok, thinking_id=int(thinking_token_id),
            eos_id=int(tok.eos_token_id if tok.eos_token_id is not None
                       else pad_token_id),
            device="cuda", max_len=int(args.latent_reasoning_max_len),
            no_ramp=bool(args.latent_reasoning_no_ramp),
            gate_weight=float(getattr(args, "latent_reasoning_gate_weight", 0.0)),
            seed=int(args.seed))
        print(f"Latent-reasoning co-train ON: weight={args.latent_reasoning_weight} "
              f"rungs={_latent_reasoner.rungs} "
              f"n/step={args.latent_reasoning_n} "
              f"(examples/rung: "
              f"{ {n: len(_latent_reasoner.data[n]) for n in _latent_reasoner.rungs} })")

    # TensorBoard writer — no-op context when --tb_dir is not set. Under DDP
    # only rank 0 writes (multiple ranks → the same event file would corrupt).
    if args.tb_dir and is_main:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=args.tb_dir)
        print(f"TensorBoard logging → {args.tb_dir}")
    else:
        tb = None

    def thinking_schedule(step: int) -> dict[str, float]:
        if step <= args.think_warmup_steps:
            curriculum = 0.0
        elif args.think_curriculum_steps > 0:
            curriculum = min(
                1.0,
                (step - args.think_warmup_steps)
                / float(args.think_curriculum_steps),
            )
        else:
            curriculum = 1.0
        explore_prob = (
            args.think_explore_start_prob
            + curriculum
            * (args.think_explore_prob - args.think_explore_start_prob)
        )
        lambda_start = (
            args.think_lambda if args.think_lambda_start is None
            else args.think_lambda_start
        )
        lambda_eff = lambda_start + curriculum * (args.think_lambda - lambda_start)
        gate_threshold_start = (
            args.think_gate_threshold
            if args.think_gate_threshold_start is None
            else args.think_gate_threshold_start
        )
        gate_threshold = (
            gate_threshold_start
            + curriculum * (args.think_gate_threshold - gate_threshold_start)
        )
        depth_start = (
            args.think_safety_max_depth
            if args.think_safety_max_depth_start is None
            else args.think_safety_max_depth_start
        )
        if args.think_safety_max_depth <= 0 and depth_start <= 0:
            safety_max_depth = 0
        else:
            depth_eff_float = (
                depth_start
                + curriculum * (args.think_safety_max_depth - depth_start)
            )
            safety_max_depth = max(1, int(round(depth_eff_float)))
        return {
            "curriculum": float(curriculum),
            "explore_prob": float(explore_prob),
            "lambda": float(lambda_eff),
            "gate_threshold": float(gate_threshold),
            "safety_max_depth": float(safety_max_depth),
            "queue_pressure": 0.0,
            "backpressure_target": 0.0,
        }

    def apply_queue_backpressure(schedule: dict[str, float]) -> dict[str, float]:
        if think_queue is None or think_replay_queue is None:
            return schedule
        target = args.think_backpressure_target
        if target < 0:
            target = args.think_queue_drain_target
        uses_backpressure = (
            target >= 0
            and (
                args.think_backpressure_lambda > 0.0
                or args.think_backpressure_threshold > 0.0
                or args.think_backpressure_explore > 0.0
            )
        )
        if not uses_backpressure:
            schedule["backpressure_target"] = float(max(0, target))
            return schedule
        target = max(1, target)
        backlog = max(len(think_queue), len(think_replay_queue))
        pressure = max(0.0, (backlog - target) / float(target))
        if args.think_backpressure_max > 0.0:
            pressure = min(pressure, args.think_backpressure_max)
        schedule = dict(schedule)
        schedule["queue_pressure"] = float(pressure)
        schedule["backpressure_target"] = float(target)
        if pressure <= 0.0:
            return schedule
        schedule["lambda"] = float(
            schedule["lambda"] + args.think_backpressure_lambda * pressure
        )
        if args.think_backpressure_threshold > 0.0:
            divisor = 1.0 + args.think_backpressure_threshold * pressure
            schedule["gate_threshold"] = float(
                max(1e-4, min(0.9999, schedule["gate_threshold"] / divisor))
            )
        if args.think_backpressure_explore > 0.0:
            divisor = 1.0 + args.think_backpressure_explore * pressure
            schedule["explore_prob"] = float(schedule["explore_prob"] / divisor)
        return schedule

    def new_thinking_stats(schedule: dict[str, float]) -> dict[str, float]:
        return {
            "cont_items": 0.0,
            "cont_think": 0.0,
            "cont_explore": 0.0,
            "forced_emit": 0.0,
            "closed": 0.0,
            "replay_items": 0.0,
            "replay_think": 0.0,
            **schedule,
        }

    def merge_thinking_stats(dst: dict[str, float], src: dict[str, float]) -> None:
        summed = {
            "cont_items", "cont_think", "cont_explore", "forced_emit",
            "closed", "replay_items", "replay_think",
        }
        for key, value in src.items():
            if key in summed:
                dst[key] = dst.get(key, 0.0) + float(value)
            else:
                dst[key] = float(value)

    def process_thinking_aux_batch(
        step: int,
        cont_items: list[ThinkContinuation],
        replay_items: list[ThinkReplay],
        schedule: dict[str, float],
    ) -> tuple[torch.Tensor, float, dict[str, float]]:
        if not cont_items and not replay_items:
            return torch.zeros((), device="cuda"), 0.0, new_thinking_stats(schedule)
        assert think_queue is not None and think_replay_queue is not None
        rows = []
        cont_targets = cont_last = None
        replay_targets = replay_last = replay_is_think = None
        if cont_items:
            cont_x, cont_targets, cont_last = build_continuation_batch(
                cont_items, block_size=args.T, pad_token_id=pad_token_id,
                device="cuda",
            )
            rows.append(cont_x)
        if replay_items:
            replay_x, replay_targets, replay_last, replay_is_think = build_replay_batch(
                replay_items, block_size=args.T, pad_token_id=pad_token_id,
                thinking_token_id=int(thinking_token_id), device="cuda",
            )
            rows.append(replay_x)
        aux_x = torch.cat(rows, dim=0)
        if args.think_checkpointing:
            from torch.utils.checkpoint import checkpoint
            # model is a nn.Module, which checkpoint can wrap.
            # We use use_reentrant=False for modern compatibility.
            aux_logits = checkpoint(model, aux_x, use_reentrant=False)
        else:
            aux_logits = model(aux_x)
        loss_terms: list[torch.Tensor] = []
        stats = new_thinking_stats(schedule)
        lambda_eff = float(schedule["lambda"])
        gate_threshold = float(schedule["gate_threshold"])
        safety_max_depth = int(schedule["safety_max_depth"])
        explore_prob = float(schedule["explore_prob"])

        if cont_items:
            assert cont_targets is not None and cont_last is not None
            row = torch.arange(len(cont_items), device=aux_logits.device)
            cont_logits = aux_logits[row, cont_last]
            forced = torch.zeros(len(cont_items), dtype=torch.bool,
                                 device=aux_logits.device)
            if safety_max_depth > 0:
                forced |= torch.tensor(
                    [item.depth >= safety_max_depth for item in cont_items],
                    dtype=torch.bool, device=aux_logits.device,
                )
            if args.think_queue_ttl > 0:
                forced |= torch.tensor(
                    [step - item.origin_step >= args.think_queue_ttl
                     for item in cont_items],
                    dtype=torch.bool, device=aux_logits.device,
                )
            if args.think_decision == "gate":
                cont_gate = model._last_gate[row, cont_last].detach()
                cont_think = (cont_gate < gate_threshold) & ~forced
                cont_gate_logits = model._last_gate_logits[row, cont_last]
                cont_think_nll = F.binary_cross_entropy_with_logits(
                    cont_gate_logits,
                    torch.zeros_like(cont_gate_logits),
                    reduction="none",
                )
            else:
                cont_think = choose_think_actions(
                    cont_logits.detach(), int(thinking_token_id),
                    args.think_policy, args.think_threshold,
                    args.think_temperature, allow_think=~forced,
                )
                think_targets = torch.full_like(cont_targets, int(thinking_token_id))
                cont_think_nll = F.cross_entropy(
                    cont_logits, think_targets, reduction="none",
                )
            cont_explore = torch.zeros_like(cont_think)
            if explore_prob > 0.0:
                cont_explore = (
                    torch.rand_like(cont_think.float()) < explore_prob
                ) & ~forced
                cont_think = cont_think | cont_explore
            cont_answer_nll = cross_entropy_masking_token(
                cont_logits, cont_targets, int(thinking_token_id),
                reduction="none",
            )
            closed_traj: list[float] = []
            for i, item in enumerate(cont_items):
                if bool(cont_think[i].item()):
                    next_ctx = (item.context_ids + [int(thinking_token_id)])[-args.T:]
                    think_queue.enqueue(ThinkContinuation(
                        context_ids=next_ctx,
                        target_id=item.target_id,
                        depth=item.depth + 1,
                        accum_nll=(
                            item.accum_nll
                            + float(cont_think_nll[i].detach().item())
                        ),
                        accum_cost=item.accum_cost + lambda_eff,
                        origin_step=item.origin_step,
                        decision_context_ids=(
                            item.decision_context_ids or item.context_ids
                        ),
                        immediate_nll=item.immediate_nll,
                    ))
                else:
                    answer_after_think = float(cont_answer_nll[i].detach().item())
                    traj = item.accum_nll + item.accum_cost + answer_after_think
                    closed_traj.append(traj)
                    comparable_traj = item.accum_cost + answer_after_think
                    beneficial = (
                        comparable_traj + args.think_advantage_margin
                        < float(item.immediate_nll)
                    )
                    think_replay_queue.enqueue(ThinkReplay(
                        context_ids=item.decision_context_ids or item.context_ids,
                        target_id=item.target_id,
                        target_is_thinking=beneficial,
                    ))
                    loss_terms.append(cont_answer_nll[i])
            if closed_traj:
                think_closed_traj_window.extend(closed_traj)
            stats.update({
                "cont_items": float(len(cont_items)),
                "cont_think": float(cont_think.float().sum().item()),
                "cont_explore": float(cont_explore.float().sum().item()),
                "forced_emit": float(forced.float().sum().item()),
                "closed": float(len(closed_traj)),
            })

        if replay_items:
            assert replay_targets is not None
            assert replay_last is not None and replay_is_think is not None
            start = len(cont_items)
            row = torch.arange(start, start + len(replay_items),
                               device=aux_logits.device)
            replay_logits = aux_logits[row, replay_last]
            if args.think_decision == "gate":
                replay_gate_logits = model._last_gate_logits[row, replay_last]
                emit_targets = (~replay_is_think).float()
                gate_ce = F.binary_cross_entropy_with_logits(
                    replay_gate_logits, emit_targets, reduction="none",
                )
                replay_answer_ce = cross_entropy_masking_token(
                    replay_logits, replay_targets, int(thinking_token_id),
                    reduction="none",
                )
                replay_ce = torch.where(
                    replay_is_think,
                    gate_ce + lambda_eff,
                    gate_ce + replay_answer_ce,
                )
            else:
                replay_ce = F.cross_entropy(
                    replay_logits, replay_targets, reduction="none",
                )
                replay_ce = replay_ce + lambda_eff * replay_is_think.float()
            loss_terms.append(args.think_replay_weight * replay_ce)
            stats["replay_items"] = float(len(replay_items))
            stats["replay_think"] = float(replay_is_think.float().sum().item())

        if not loss_terms:
            return torch.zeros((), device=aux_logits.device), 0.0, stats
        loss_sum = torch.stack([term.sum() for term in loss_terms]).sum()
        count = float(sum(term.numel() for term in loss_terms))
        return loss_sum, count, stats

    for step in range(args.start_step + 1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        # Streams may yield (x, y) or (x, y, doc_ids); doc_ids drives
        # cross-document state isolation in the model (None = one document
        # per row, the leak-free default for the non-data_mix path).
        x, y, *_rest = batch
        x, y = x.to("cuda"), y.to("cuda")
        doc_ids = _rest[0].to("cuda") if _rest else None
        # K-self-feed curriculum: bypass FiLM (1-pass forward) until the
        # warmup boundary, then run the configured --feedback_self_k.
        if args.feedback_self_k_warmup_steps > 0:
            bypass = step <= args.feedback_self_k_warmup_steps
            if bypass != model._film_bypass:
                model._film_bypass = bypass
                if not bypass:
                    print(f"[step {step}] FiLM K-self-feed curriculum: "
                          f"warmup over, enabling feedback_self_k="
                          f"{args.feedback_self_k}")
        # PKM ε-greedy curriculum: linear anneal from --pkm_epsilon_start
        # to 0 over --pkm_epsilon_warmup_steps. Forces every slot to get
        # gradient early; the learned router takes over after warmup.
        if (getattr(args, "use_pkm", False)
                and getattr(args, "pkm_epsilon_start", 0.0) > 0.0):
            warm = max(1, int(getattr(args, "pkm_epsilon_warmup_steps", 0)))
            progress = min(1.0, step / warm) if warm > 0 else 1.0
            eps = float(args.pkm_epsilon_start) * (1.0 - progress)
            model.pkm_layer.random_slot_epsilon = eps
        # PKM α-floor curriculum: linear anneal of the additive
        # sign-preserving floor on the output gate. Forces a minimum PKM
        # contribution during the value-table-bootstrap window so values
        # get meaningful gradient before α can shrink. Synced with ε.
        if (getattr(args, "use_pkm", False)
                and getattr(args, "pkm_alpha_floor_start", 0.0) > 0.0):
            warm = max(1, int(getattr(args, "pkm_alpha_floor_warmup_steps", 0)))
            progress = min(1.0, step / warm) if warm > 0 else 1.0
            floor = float(args.pkm_alpha_floor_start) * (1.0 - progress)
            model.pkm_layer.alpha_floor = floor
        for o in opts:
            o.zero_grad(set_to_none=True)
        pre_think_stats: dict[str, float] | None = None
        if args.enable_thinking_token:
            base_schedule = thinking_schedule(step)
            schedule = apply_queue_backpressure(base_schedule)
            pre_think_stats = new_thinking_stats(schedule)
            if args.think_queue_accum_steps > 0:
                assert think_queue is not None and think_replay_queue is not None
                fresh_token_budget = max(1, x.numel())
                accum_max_steps = (
                    args.think_queue_accum_max_steps
                    or args.think_queue_accum_steps
                )
                accum_step = 0
                while accum_step < accum_max_steps:
                    must_do_minimum = accum_step < args.think_queue_accum_steps
                    should_drain = (
                        args.think_queue_drain_target >= 0
                        and (
                            len(think_queue) > args.think_queue_drain_target
                            or len(think_replay_queue) > args.think_queue_drain_target
                        )
                    )
                    if not must_do_minimum and not should_drain:
                        break
                    cont_n = min(think_queue_batch, len(think_queue))
                    replay_n = min(think_replay_batch, len(think_replay_queue))
                    if cont_n == 0 and replay_n == 0:
                        break
                    cont_items = think_queue.pop_batch(cont_n)
                    replay_items = think_replay_queue.pop_batch(replay_n)
                    aux_schedule = apply_queue_backpressure(base_schedule)
                    aux_loss_sum, aux_count, aux_stats = process_thinking_aux_batch(
                        step, cont_items, replay_items, aux_schedule,
                    )
                    merge_thinking_stats(pre_think_stats, aux_stats)
                    if aux_count > 0:
                        aux_denom = (
                            aux_count
                            if args.think_aux_normalize == "aux_items"
                            else fresh_token_budget
                        )
                        (args.think_aux_loss_scale
                         * aux_loss_sum / max(1.0, aux_denom)).backward()
                    accum_step += 1
        packed_cont_items: list[ThinkContinuation] = []
        packed_replay_items: list[ThinkReplay] = []
        packed_cont_last = packed_cont_targets = None
        packed_replay_last = packed_replay_targets = packed_replay_is_think = None
        fresh_offset = 0
        fresh_n = x.shape[0]
        if args.enable_thinking_token:
            assert think_queue is not None and think_replay_queue is not None
            if args.think_queue_accum_steps > 0:
                n_cont = 0
                n_replay = 0
            elif args.think_prioritize_queue:
                queue_capacity = max(0, x.shape[0] - args.think_min_fresh_rows)
                n_cont = min(len(think_queue), queue_capacity)
                n_replay = min(len(think_replay_queue), queue_capacity - n_cont)
            else:
                max_aux_rows = max(0, x.shape[0] - args.think_min_fresh_rows)
                n_cont = min(think_queue_batch, len(think_queue), max_aux_rows)
                n_replay = min(think_replay_batch, len(think_replay_queue),
                               max_aux_rows - n_cont)
            packed_cont_items = think_queue.pop_batch(n_cont)
            packed_replay_items = think_replay_queue.pop_batch(n_replay)
            fresh_n = x.shape[0] - n_cont - n_replay
            packed_rows = []
            packed_targets = []
            if packed_cont_items:
                cont_x, packed_cont_targets, packed_cont_last = build_continuation_batch(
                    packed_cont_items, block_size=args.T,
                    pad_token_id=pad_token_id, device=x.device,
                )
                packed_rows.append(cont_x)
                packed_targets.append(torch.zeros_like(cont_x))
            if packed_replay_items:
                replay_x, packed_replay_targets, packed_replay_last, packed_replay_is_think = build_replay_batch(
                    packed_replay_items, block_size=args.T,
                    pad_token_id=pad_token_id,
                    thinking_token_id=int(thinking_token_id),
                    device=x.device,
                )
                packed_rows.append(replay_x)
                packed_targets.append(torch.zeros_like(replay_x))
            if fresh_n > 0:
                packed_rows.append(x[:fresh_n])
                packed_targets.append(y[:fresh_n])
            if packed_rows:
                x = torch.cat(packed_rows, dim=0)
                y = torch.cat(packed_targets, dim=0)
            fresh_offset = n_cont + n_replay
        # Gated loss (Phase 23): L = mean(g_t * CE_t + (1-g_t) * λ).
        # g_t is stored in model._last_gate by the forward pass (side effect).
        # When output_gate is off, fall back to standard mean CE.
        if args.enable_thinking_token:
            if args.aux_brackets:
                logits, aux_logits = model(x, return_aux=True)
            else:
                logits = model(x)
                aux_logits = None
            if args.aux_brackets:
                depth = bracket_depth(x, bracket_deltas).clamp(0, args.aux_max_depth)
                aux_loss = F.cross_entropy(
                    aux_logits.reshape(-1, args.aux_max_depth + 1),
                    depth.reshape(-1),
                )
            else:
                aux_loss = torch.zeros((), device="cuda")
            V = logits.shape[-1]
            ce_per_token = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1),
                reduction="none",
            ).reshape(y.shape)                                               # (B, T)
            schedule = apply_queue_backpressure(thinking_schedule(step))
            think_curriculum = schedule["curriculum"]
            think_explore_prob_eff = schedule["explore_prob"]
            think_lambda_eff = schedule["lambda"]
            think_gate_threshold_eff = schedule["gate_threshold"]
            think_safety_max_depth_eff = int(schedule["safety_max_depth"])
            cont_loss_terms: list[torch.Tensor] = []
            replay_loss_terms: list[torch.Tensor] = []
            cont_stats = new_thinking_stats(schedule)
            cont_stats["cont_items"] = float(len(packed_cont_items))
            cont_stats["replay_items"] = float(len(packed_replay_items))
            if packed_cont_items:
                row = torch.arange(len(packed_cont_items), device=logits.device)
                cont_logits = logits[row, packed_cont_last]
                forced = torch.zeros(len(packed_cont_items), dtype=torch.bool,
                                     device=logits.device)
                if think_safety_max_depth_eff > 0:
                    forced |= torch.tensor(
                        [item.depth >= think_safety_max_depth_eff
                         for item in packed_cont_items],
                        dtype=torch.bool, device=logits.device,
                    )
                if args.think_queue_ttl > 0:
                    forced |= torch.tensor(
                        [step - item.origin_step >= args.think_queue_ttl
                         for item in packed_cont_items],
                        dtype=torch.bool, device=logits.device,
                )
                if args.think_decision == "gate":
                    cont_gate = model._last_gate[row, packed_cont_last].detach()
                    cont_think = (cont_gate < think_gate_threshold_eff) & ~forced
                else:
                    cont_think = choose_think_actions(
                        cont_logits.detach(), int(thinking_token_id),
                        args.think_policy, args.think_threshold,
                        args.think_temperature, allow_think=~forced,
                    )
                cont_explore = torch.zeros_like(cont_think)
                if think_explore_prob_eff > 0.0:
                    cont_explore = (
                        torch.rand_like(cont_think.float()) < think_explore_prob_eff
                    ) & ~forced
                    cont_think = cont_think | cont_explore
                think_targets = torch.full_like(packed_cont_targets,
                                                int(thinking_token_id))
                if args.think_decision == "gate":
                    cont_gate_logits = model._last_gate_logits[row, packed_cont_last]
                    cont_think_nll = F.binary_cross_entropy_with_logits(
                        cont_gate_logits,
                        torch.zeros_like(cont_gate_logits),
                        reduction="none",
                    )
                else:
                    cont_think_nll = F.cross_entropy(
                        cont_logits, think_targets, reduction="none"
                    )
                cont_answer_nll = cross_entropy_masking_token(
                    cont_logits, packed_cont_targets, int(thinking_token_id),
                    reduction="none",
                )
                closed_traj: list[float] = []
                for i, item in enumerate(packed_cont_items):
                    if bool(cont_think[i].item()):
                        next_ctx = (item.context_ids + [int(thinking_token_id)])[-args.T:]
                        think_queue.enqueue(ThinkContinuation(
                            context_ids=next_ctx,
                            target_id=item.target_id,
                            depth=item.depth + 1,
                            accum_nll=(item.accum_nll
                                       + float(cont_think_nll[i].detach().item())),
                            accum_cost=item.accum_cost + float(think_lambda_eff),
                            origin_step=item.origin_step,
                            decision_context_ids=(item.decision_context_ids
                                                  or item.context_ids),
                            immediate_nll=item.immediate_nll,
                        ))
                    else:
                        answer_after_think = float(cont_answer_nll[i].detach().item())
                        traj = item.accum_nll + item.accum_cost + answer_after_think
                        closed_traj.append(traj)
                        comparable_traj = item.accum_cost + answer_after_think
                        beneficial = (
                            comparable_traj + args.think_advantage_margin
                            < float(item.immediate_nll)
                        )
                        think_replay_queue.enqueue(ThinkReplay(
                            context_ids=(item.decision_context_ids
                                         or item.context_ids),
                            target_id=item.target_id,
                            target_is_thinking=beneficial,
                        ))
                        cont_loss_terms.append(cont_answer_nll[i])
                if closed_traj:
                    think_closed_traj_window.extend(closed_traj)
                cont_stats.update({
                    "cont_think": float(cont_think.float().sum().item()),
                    "cont_explore": float(cont_explore.float().sum().item()),
                    "forced_emit": float(forced.float().sum().item()),
                    "closed": float(len(closed_traj)),
                })
            if packed_replay_items:
                start = len(packed_cont_items)
                row = torch.arange(start, start + len(packed_replay_items),
                                   device=logits.device)
                replay_logits = logits[row, packed_replay_last]
                if args.think_decision == "gate":
                    replay_gate_logits = model._last_gate_logits[row, packed_replay_last]
                    emit_targets = (~packed_replay_is_think).float()
                    gate_ce = F.binary_cross_entropy_with_logits(
                        replay_gate_logits, emit_targets, reduction="none"
                    )
                    replay_answer_ce = cross_entropy_masking_token(
                        replay_logits, packed_replay_targets,
                        int(thinking_token_id), reduction="none",
                    )
                    replay_ce = torch.where(
                        packed_replay_is_think,
                        gate_ce + think_lambda_eff,
                        gate_ce + replay_answer_ce,
                    )
                else:
                    replay_ce = F.cross_entropy(
                        replay_logits, packed_replay_targets, reduction="none"
                    )
                    replay_ce = (
                        replay_ce
                        + think_lambda_eff * packed_replay_is_think.float()
                    )
                replay_loss_terms.append(args.think_replay_weight * replay_ce)
                cont_stats["replay_think"] = float(
                    packed_replay_is_think.float().sum().item()
                )
            allow_think = step > args.think_warmup_steps
            fresh_logits = logits[fresh_offset:]
            fresh_y = y[fresh_offset:]
            if fresh_n > 0:
                if args.think_decision == "gate":
                    fresh_gate = model._last_gate[fresh_offset:].detach()
                    think_mask = fresh_gate < think_gate_threshold_eff
                    if not allow_think:
                        think_mask = torch.zeros_like(think_mask)
                else:
                    think_mask = choose_think_actions(
                        fresh_logits.detach(), int(thinking_token_id),
                        args.think_policy, args.think_threshold,
                        args.think_temperature, allow_think=allow_think,
                    )
            else:
                think_mask = torch.zeros((0, args.T), dtype=torch.bool,
                                         device=logits.device)
            explore_mask = torch.zeros_like(think_mask)
            free_slots = think_queue.max_len - len(think_queue)
            if fresh_n > 0:
                answer_ce_all = cross_entropy_masking_token(
                    fresh_logits.reshape(-1, V), fresh_y.reshape(-1),
                    int(thinking_token_id), reduction="none",
                ).reshape(fresh_y.shape)
                if allow_think and think_explore_prob_eff > 0.0:
                    explore_mask = choose_explore_actions(
                        answer_ce_all.detach(),
                        probability=think_explore_prob_eff,
                        mode=args.think_explore_mode,
                        top_frac=args.think_explore_top_frac,
                        min_score=args.think_explore_min_ce,
                    )
                    think_mask = think_mask | explore_mask
                fresh_loss_sum = answer_ce_all.sum()
            else:
                answer_ce_all = torch.zeros((0, args.T), device=logits.device)
                fresh_loss_sum = torch.zeros((), device=logits.device)
            if free_slots <= 0:
                think_mask = torch.zeros_like(think_mask)
                explore_mask = torch.zeros_like(explore_mask)
            else:
                think_flat = think_mask.reshape(-1)
                max_new = free_slots
                if args.think_max_new_per_step > 0:
                    max_new = min(max_new, int(args.think_max_new_per_step))
                n_think = int(think_flat.sum().item())
                if n_think > max_new:
                    idx = think_flat.nonzero(as_tuple=False).flatten()
                    if answer_ce_all.numel():
                        scores = answer_ce_all.detach().reshape(-1)[idx]
                        keep_idx = idx[torch.topk(scores, k=max_new).indices]
                    else:
                        keep_idx = idx[:max_new]
                    limited = torch.zeros_like(think_flat)
                    limited[keep_idx] = True
                    think_mask = limited.reshape_as(think_mask)
                    explore_mask = explore_mask & think_mask
            extra_loss_sum = torch.zeros((), device=logits.device)
            extra_count = 0
            if cont_loss_terms:
                extra_loss_sum = extra_loss_sum + torch.stack(cont_loss_terms).sum()
                extra_count += len(cont_loss_terms)
            if replay_loss_terms:
                extra_loss_sum = extra_loss_sum + torch.cat(replay_loss_terms).sum()
                extra_count += int(sum(t.numel() for t in replay_loss_terms))
            gate_emit_items = 0.0
            if (args.think_decision == "gate" and args.think_gate_emit_weight > 0
                    and fresh_n > 0 and think_mask.numel() > 0):
                emit_mask = ~think_mask.detach()
                gate_logits_fresh = model._last_gate_logits[fresh_offset:]
                if emit_mask.any():
                    emit_ce = F.binary_cross_entropy_with_logits(
                        gate_logits_fresh[emit_mask],
                        torch.ones_like(gate_logits_fresh[emit_mask]),
                        reduction="sum",
                    )
                    extra_loss_sum = (
                        extra_loss_sum
                        + args.think_gate_emit_weight * emit_ce
                    )
                    weighted_emit_count = (
                        args.think_gate_emit_weight
                        * float(emit_mask.float().sum().item())
                    )
                    extra_count += weighted_emit_count
                    gate_emit_items = float(emit_mask.float().sum().item())
            denom = fresh_y.numel() + extra_count
            lm_loss = (fresh_loss_sum + extra_loss_sum) / max(1, denom)
            think_coords = think_mask.detach().nonzero(as_tuple=False).cpu().tolist()
            x_cpu = x[fresh_offset:].detach().cpu()
            y_cpu = fresh_y.detach().cpu()
            answer_ce_cpu = answer_ce_all.detach().cpu()
            fresh_logits_cpu = fresh_logits.detach().cpu()
            fresh_gate_logits_cpu = (
                model._last_gate_logits[fresh_offset:].detach().cpu()
                if args.think_decision == "gate" else None
            )
            for b, t_idx in think_coords:
                ctx_ids = x_cpu[b, :t_idx + 1].tolist() + [int(thinking_token_id)]
                decision_ctx = x_cpu[b, :t_idx + 1].tolist()
                if args.think_decision == "gate":
                    assert fresh_gate_logits_cpu is not None
                    think_nll = F.binary_cross_entropy_with_logits(
                        fresh_gate_logits_cpu[b, t_idx].unsqueeze(0),
                        torch.zeros(1),
                        reduction="none",
                    )[0].item()
                else:
                    think_nll = F.cross_entropy(
                        fresh_logits_cpu[b, t_idx].unsqueeze(0),
                        torch.tensor([int(thinking_token_id)]),
                        reduction="none",
                    )[0].item()
                think_queue.enqueue(ThinkContinuation(
                    context_ids=ctx_ids[-args.T:],
                    target_id=int(y_cpu[b, t_idx].item()),
                    depth=1,
                    accum_nll=float(think_nll),
                    accum_cost=float(think_lambda_eff),
                    origin_step=step,
                    decision_context_ids=decision_ctx[-args.T:],
                    immediate_nll=float(answer_ce_cpu[b, t_idx].item()),
                ))
            think_stats = {
                "normal_items": float(fresh_y.numel()),
                "normal_think": float(think_mask.float().sum().item()),
                "normal_explore": float(explore_mask.float().sum().item()),
                "answer_ce": (float(answer_ce_all.mean().detach().item())
                              if answer_ce_all.numel() else 0.0),
                "queue_len": float(len(think_queue)),
                "queue_mean_depth": float(think_queue.mean_depth()),
                "queue_max_depth": float(think_queue.max_depth()),
                "queue_dropped": float(think_queue.dropped),
                "replay_queue_len": float(len(think_replay_queue)),
                "gate_emit_items": gate_emit_items,
                **cont_stats,
            }
            if pre_think_stats is not None:
                merge_thinking_stats(pre_think_stats, think_stats)
                think_stats = pre_think_stats
            think_stats_window.append(think_stats)
            loss = lm_loss + args.aux_weight * aux_loss
            loss = loss + _z_loss_term(logits, args.z_loss)
            loss.backward()
        else:
            # Non-thinking (pretrain) path: gradient accumulation over
            # --grad_accum microbatches, one optimizer step per `step`.
            # The gate/plain-loss branches live in _nonthink_forward_loss.
            n_micro = max(1, args.grad_accum)
            _latent_cotrain_diag = None
            _latent_reasoning_diag = None
            _gate_calib_diag = None
            for micro in range(n_micro):
                if micro > 0:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)
                    x, y, *_rest = batch
                    x, y = x.to("cuda"), y.to("cuda")
                    doc_ids = _rest[0].to("cuda") if _rest else None
                # DDP: only all-reduce grads on the LAST microbatch; no_sync()
                # suppresses the reduce on the intermediates (correctness +
                # avoids n_micro× redundant comms). nullcontext when single-GPU.
                _is_last_micro = (micro == n_micro - 1)
                _sync_ctx = (ddp_model.no_sync() if (is_ddp and not _is_last_micro)
                             else contextlib.nullcontext())
                with _sync_ctx:
                    logits, ce_per_token, lm_loss, aux_loss, gate_aux_loss, \
                        gist_loss = _nonthink_forward_loss(
                            model, x, y, args, step, bracket_deltas,
                            doc_ids=doc_ids, gist_horizons=gist_horizons,
                            fwd_model=ddp_model)
                    # Snapshot the MAIN-forward gate NOW, before any aux loss
                    # (latent_cotrain / gate_calibration) runs an extra forward
                    # that clobbers model._last_gate(_logits) — the documented
                    # 2026-05-27 footgun. _last_gate is detached for diag;
                    # _last_gate_logits is the GRAD-CARRYING tensor the
                    # gate-calibration BCE flows into, so keep it un-detached.
                    main_gate = getattr(model, "_last_gate", None)
                    main_gate_logits = getattr(model, "_last_gate_logits", None)
                    # Snapshot the WM injection from the MAIN forward too: the
                    # aux extra-forwards (latent_cotrain / gate_calibration) call
                    # model() and clobber memory._last_injection with their own
                    # (N, Lmax, d) shape, which then mismatches x at the
                    # wm(inj=) diagnostic below (IndexError).
                    main_wm_inj = getattr(getattr(model, "memory", None),
                                          "_last_injection", None)
                    loss = lm_loss + args.aux_weight * aux_loss
                    loss = loss + _z_loss_term(logits, args.z_loss)
                    if args.output_gate and args.gate_entropy_aux_weight > 0.0:
                        loss = loss + args.gate_entropy_aux_weight * gate_aux_loss
                    if args.gist_loss_weight > 0.0:
                        loss = loss + args.gist_loss_weight * gist_loss
                    # v9: latent-thinking co-training — grad CE on the
                    # post-R-latent-think prediction so the trunk learns to do
                    # useful computation during thinking. Logs mean Δlogp (the
                    # "is thinking becoming useful" signal: climbs from ≈-7
                    # toward 0/positive). Requires --no-compile (extra forwards).
                    # Fire ONCE per optimizer step (last microbatch only): the
                    # latent loss runs ~5 extra eager forwards, so doing it on
                    # every microbatch was the main throughput killer. A single
                    # 24-position sample per step is plenty of signal. Scale by
                    # n_micro so the per-step gradient still matches
                    # --latent_cotrain_weight after the (loss / n_micro).backward().
                    if (getattr(args, "latent_cotrain_weight", 0.0) > 0.0
                            and _is_last_micro
                            and thinking_token_id is not None
                            and int(thinking_token_id) != int(pad_token_id)):
                        _lc = latent_cotrain_loss(
                            model, x, y, R=args.latent_cotrain_R,
                            thinking_token_id=int(thinking_token_id),
                            sample_frac=args.latent_cotrain_sample_frac,
                            max_positions=args.latent_cotrain_max_positions,
                            max_prefix_len=128,
                            selective=bool(getattr(
                                args, "latent_cotrain_selective", False)),
                            pad_id=int(pad_token_id))
                        if _lc is not None:
                            _lc_loss, _lc_delta, _lc_n = _lc
                            loss = loss + (n_micro * args.latent_cotrain_weight) \
                                * _lc_loss
                            _latent_cotrain_diag = (_lc_delta, _lc_n)
                    # Depth-matched latent-REASONING co-train (the 2026-06-05 fix):
                    # answer-span CE on pointer-chase at R=depth, clean latent
                    # thread (WM off + film bypass inside the helper). Fires once
                    # per optimizer step; scaled by n_micro so the per-step
                    # gradient matches --latent_reasoning_weight after /n_micro.
                    if (_latent_reasoner is not None
                            and _is_last_micro):
                        _lr_loss, _lr_rung = _latent_reasoner.step(
                            model, step, args.steps,
                            int(args.latent_reasoning_n))
                        loss = loss + (n_micro * args.latent_reasoning_weight) \
                            * _lr_loss
                        _latent_reasoning_diag = (float(_lr_loss.detach()),
                                                  int(_lr_rung))
                    # Gate-calibration: train the OUTPUT GATE (not the trunk) to
                    # fire think exactly where a latent think raises
                    # logp(true_next). The BCE flows ONLY into the grad-carrying
                    # gate-logit snapshot taken BEFORE the latent extra forward
                    # clobbered model._last_gate_logits. Fire once/step (last
                    # microbatch) and scale by n_micro to keep the per-step
                    # gradient equal to --gate_calibration_weight after the
                    # (loss / n_micro).backward(). Default weight 0 = OFF.
                    if (getattr(args, "gate_calibration_weight", 0.0) > 0.0
                            and args.output_gate
                            and _is_last_micro
                            and main_gate_logits is not None
                            and thinking_token_id is not None
                            and int(thinking_token_id) != int(pad_token_id)):
                        _gc = compute_gate_calibration_loss(
                            model, x, y, main_gate_logits,
                            thinking_token_id=int(thinking_token_id),
                            latent_R=int(args.gate_calibration_R),
                            sample_frac=float(args.gate_calibration_sample_frac),
                            max_positions=int(args.gate_calibration_max_positions),
                            sigma_low=float(args.gate_calibration_sigma_low),
                            sigma_high=float(args.gate_calibration_sigma_high),
                            # EOS targets are already -100 under
                            # --mask_eos_in_targets and filtered by the
                            # helper's targets!=-100 check, so eos_id=None.
                            eos_id=None)
                        if _gc is not None:
                            loss = loss + (n_micro * args.gate_calibration_weight) \
                                * _gc.loss
                            _gate_calib_diag = (_gc.target_frac_pos,
                                                _gc.mean_sigma, _gc.mean_delta,
                                                _gc.n_positions)
                    # PKM diversity-bonus: -H(slot-selection distribution) per
                    # head, averaged across batch and heads. We MAXIMISE entropy
                    # so the auxiliary loss is NEGATIVE entropy. This is the
                    # direct fix for v5-pkm's "4 % of slots cover 95 % of mass"
                    # failure mode. The slot indices are detached upstream so the
                    # router itself isn't trained to produce high entropy — we
                    # only nudge the *distribution* (via the value-table grad
                    # this implies for diverse retrievals).
                    if (getattr(args, "use_pkm", False)
                            and getattr(args, "pkm_diversity_weight", 0.0) > 0.0):
                        div_loss = _pkm_diversity_loss(model.pkm_layer)
                        loss = loss + args.pkm_diversity_weight * div_loss
                    (loss / n_micro).backward()
        _log_this_step = (step % args.log_every == 0 or step == args.steps)
        # Per-layer diagnostics: grad norms must be read before clip; the
        # weight snapshot must be taken before opt.step().
        _blk_gnorms = _block_grad_norms(model) if _log_this_step else None
        _blk_wsnap = _block_weight_snapshot(model) if _log_this_step else None
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        _blk_uratios = (_block_update_ratios(model, _blk_wsnap)
                        if _log_this_step else None)
        losses.append(lm_loss.item())  # track LM loss alone for comparison
        if args.output_gate:
            g_detached = (main_gate.detach() if main_gate is not None
                          else model._last_gate.detach())
            ce_detached = ce_per_token.detach()
            emit_mask = g_detached > 0.5
            # CE averaged only over emitted positions — the fair comparison metric.
            emit_ce = (ce_detached[emit_mask].mean().item()
                       if emit_mask.any() else float("nan"))
            losses_gate_window.append((
                float(g_detached.mean().item()),
                float(emit_mask.float().mean().item()),
                float(ce_detached.mean().item()),
                emit_ce,
            ))

        if step % args.log_every == 0 or step == args.steps:
            now = time.perf_counter()
            # ×ddp_world_size so this is GLOBAL throughput (all ranks), directly
            # comparable to the single-GPU baseline. Per-rank does batch*T*ga.
            tok_per_sec = ((step - last_log_step) * args.batch * args.T
                           * args.grad_accum * ddp_world_size / (now - last_log))
            tloss_avg = sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:]))
            line = (f"{step:>6d}  {tok_per_sec:>8.0f}  "
                    f"{tloss_avg:>8.4f}  "
                    f"{scheduler.get_last_lr()[0]:>9.2e}")
            if args.gist_loss_weight > 0.0:
                line += f"  gist={gist_loss.item():.4f}"
            if getattr(args, "latent_cotrain_weight", 0.0) > 0.0 and \
                    _latent_cotrain_diag is not None:
                _d, _n = _latent_cotrain_diag
                line += f"  latent(Δlogp={_d:+.3f},n={_n})"
            if _latent_reasoner is not None and \
                    _latent_reasoning_diag is not None:
                _rl, _rr = _latent_reasoning_diag
                line += f"  reason(loss={_rl:.3f},R={_rr})"
            if getattr(args, "gate_calibration_weight", 0.0) > 0.0 and \
                    _gate_calib_diag is not None:
                _t1, _sg, _gd, _gn = _gate_calib_diag
                # tgt1 = fraction where latent think helped (BCE target=1);
                # σ = mean gate sigmoid at scored positions; Δlogp = mean
                # latent-think benefit. tgt1>σ ⇒ gate UNDER-fires (miscal).
                line += (f"  gc(tgt1={_t1:.2f},σ={_sg:.2f},"
                         f"Δlogp={_gd:+.2f},n={_gn})")
            if args.feedback != "none" or fb_xattn_pairs:
                alphas = model.feedback_alphas()
                if not alphas:
                    pass
                elif isinstance(alphas[0], tuple):
                    # Sparse-FiLM:    (target, source_int, alpha)
                    # Cross-attn:     (target, sources_tuple, alpha)
                    if fb_xattn_pairs:
                        # Show first 4 (target<-K:α) entries to keep line short.
                        head_n = 4
                        items = [f"{t}<-{len(srcs)}:{a:+.3f}"
                                 for t, srcs, a in alphas[:head_n]]
                        suffix = f",… ({len(alphas) - head_n} more)" if len(alphas) > head_n else ""
                        line += "  α=[" + ",".join(items) + suffix + "]"
                    else:
                        line += "  α=[" + ",".join(
                            f"{t}<-{s}:{a:+.3f}" for t, s, a in alphas
                        ) + "]"
                elif isinstance(alphas[0], list):
                    # Dense multi-scale: max |α| per layer.
                    summary = [max(abs(a) for a in row) for row in alphas]
                    line += f"  max|α|=[{','.join(f'{a:.3f}' for a in summary)}]"
                else:
                    line += f"  α=[{','.join(f'{a:+.3f}' for a in alphas)}]"
            if args.output_gate and losses_gate_window:
                recent = losses_gate_window[-args.log_every:]
                mean_g = sum(r[0] for r in recent) / len(recent)
                emit_frac = sum(r[1] for r in recent) / len(recent)
                raw_ce = sum(r[2] for r in recent) / len(recent)
                valid_emit = [r[3] for r in recent if r[3] == r[3]]  # drop NaN
                emit_ce = sum(valid_emit) / len(valid_emit) if valid_emit else float("nan")
                if args.gate_warmup_steps > 0:
                    progress = min(1.0, step / args.gate_warmup_steps)
                    floor_now = (1.0 - progress) * 1.0 + progress * args.gate_floor_min
                    gf = f",floor={floor_now:.2f}"
                else:
                    gf = (f",floor={args.gate_floor_min:.2f}"
                          if args.gate_floor_min > 0 else "")
                ece_str = f"{emit_ce:.4f}" if emit_ce == emit_ce else "nan"
                line += (f"  gate(g={mean_g:.3f},emit={emit_frac:.2f},"
                         f"emit_ce={ece_str},ce={raw_ce:.4f}{gf})")
            if args.enable_thinking_token and think_stats_window:
                recent = think_stats_window[-args.log_every:]
                normal_items = sum(r.get("normal_items", 0.0) for r in recent)
                normal_think = sum(r.get("normal_think", 0.0) for r in recent)
                normal_explore = sum(r.get("normal_explore", 0.0) for r in recent)
                cont_items = sum(r.get("cont_items", 0.0) for r in recent)
                cont_think = sum(r.get("cont_think", 0.0) for r in recent)
                cont_explore = sum(r.get("cont_explore", 0.0) for r in recent)
                replay_items = sum(r.get("replay_items", 0.0) for r in recent)
                replay_think = sum(r.get("replay_think", 0.0) for r in recent)
                gate_emit_items = sum(r.get("gate_emit_items", 0.0) for r in recent)
                forced_emit = sum(r.get("forced_emit", 0.0) for r in recent)
                answer_ce = sum(r.get("answer_ce", 0.0) for r in recent) / len(recent)
                think_rate = ((normal_think + cont_think)
                              / max(1.0, normal_items + cont_items))
                explore_rate = ((normal_explore + cont_explore)
                                / max(1.0, normal_items + cont_items))
                forced_rate = forced_emit / max(1.0, cont_items)
                queue_len = recent[-1].get("queue_len", 0.0)
                q_mean_depth = recent[-1].get("queue_mean_depth", 0.0)
                q_max_depth = recent[-1].get("queue_max_depth", 0.0)
                q_dropped = recent[-1].get("queue_dropped", 0.0)
                replay_rate = replay_think / max(1.0, replay_items)
                curr = recent[-1].get("curriculum", 1.0)
                explore_prob = recent[-1].get("explore_prob", args.think_explore_prob)
                lambda_eff = recent[-1].get("lambda", args.think_lambda)
                gate_thr = recent[-1].get("gate_threshold", args.think_gate_threshold)
                q_pressure = recent[-1].get("queue_pressure", 0.0)
                depth_cap = recent[-1].get("safety_max_depth", args.think_safety_max_depth)
                traj_recent = think_closed_traj_window[-args.log_every:]
                traj_nll = (sum(traj_recent) / len(traj_recent)
                            if traj_recent else float("nan"))
                traj_str = f"{traj_nll:.4f}" if traj_nll == traj_nll else "nan"
                line += (f"  think(rate={think_rate:.3f},ans_ce={answer_ce:.4f},"
                          f"traj={traj_str},q={queue_len:.0f},"
                          f"depth={q_mean_depth:.2f}/{q_max_depth:.0f},"
                          f"work={cont_items:.0f}/{replay_items:.0f},"
                          f"forced={forced_rate:.3f},explore={explore_rate:.3f},"
                          f"replay_think={replay_rate:.3f},"
                          f"gate_emit={gate_emit_items:.0f},"
                          f"curr={curr:.2f},eps={explore_prob:.3f},"
                          f"lam={lambda_eff:.3f},thr={gate_thr:.3f},"
                          f"bp={q_pressure:.2f},"
                          f"cap={depth_cap:.0f},"
                          f"drop={q_dropped:.0f})")
            if _blk_gnorms is not None:
                gn, ur = _blk_gnorms, _blk_uratios
                n = len(gn)
                mid = n // 2
                line += (f"  gnorm(L0={gn[0]:.2e},L{mid}={gn[mid]:.2e},"
                         f"L{n-1}={gn[n-1]:.2e},last/first="
                         f"{gn[-1] / max(gn[0], 1e-12):.1f})")
                line += (f"  uratio(L0={ur[0]:.1e},L{mid}={ur[mid]:.1e},"
                         f"L{n-1}={ur[n-1]:.1e})")
            # PKM live diagnostics (v7.1): is the table actually waking up?
            #   αL      = learned scalar gate (init 0)
            #   αeff    = α + sign(α)·alpha_floor (the magnitude that
            #             actually scales PKM output in the forward)
            #   row     = mean row-norm of value table / expected init norm.
            #             >1 means rows have GROWN from init; <1 means they
            #             shrunk. =1 exactly means the table is frozen.
            #             (Replaces the misleading `v_std` diagnostic which
            #             is invariant under updates that preserve overall
            #             Gaussian distribution — frozen and learning-but-
            #             centred values both gave std≈1.)
            #   slots/H = unique slots hit this microbatch (out of n_keys²)
            #   top     = mass on the single hottest slot (lower=more diverse)
            #   ε       = current ε-greedy exploration rate
            #   φ       = current α-floor (decaying from start to 0)
            if (getattr(args, "use_pkm", False)
                    and hasattr(model, "pkm_layer")):
                pkm = model.pkm_layer
                with torch.no_grad():
                    aL = float(pkm.out_alpha.detach()) if pkm.use_output_gate else float("nan")
                    floor = float(getattr(pkm, "alpha_floor", 0.0))
                    sign = 1.0 if aL >= 0.0 or abs(aL) < 1e-3 else -1.0
                    aEff = aL + sign * floor if pkm.use_output_gate else float("nan")
                    # Row-norm drift: mean over rows of ||v_row|| / expected_init.
                    rn_mean = float(torch.stack([
                        emb.weight.float().norm(dim=-1).mean()
                        for emb in pkm.values
                    ]).mean())
                    init_norm = float(pkm._expected_init_row_norm)
                    rn_ratio = rn_mean / max(init_norm, 1e-9)
                    eps = float(getattr(pkm, "random_slot_epsilon", 0.0))
                    n_slots = pkm.n_keys * pkm.n_keys
                    if pkm._last_slot_idx is not None:
                        idx = pkm._last_slot_idx       # (B, T, H, top_k)
                        w = pkm._last_weights          # (B, T, H, top_k)
                        H_ = pkm.n_heads
                        slot_mass = torch.zeros(H_, n_slots,
                                                 device=idx.device, dtype=torch.float32)
                        for h_ in range(H_):
                            slot_mass[h_].scatter_add_(
                                0, idx[:, :, h_, :].reshape(-1),
                                w[:, :, h_, :].reshape(-1).float())
                        slot_mass = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                        unique_hit = int((slot_mass > 0).sum(dim=-1).float().mean())
                        top_share = float(slot_mass.max(dim=-1).values.mean())
                    else:
                        unique_hit, top_share = 0, float("nan")
                line += (f"  pkm(αL={aL:+.3f},αeff={aEff:+.3f},"
                         f"row={rn_ratio:.3f},"
                         f"slots/H={unique_hit}/{n_slots},top={top_share:.3f},"
                         f"ε={eps:.2f},φ={floor:.2f})")
            # WorkingMemory (dynamic RAG) liveness. WM is read/injected ONLY at
            # think positions; the supervised gradient comes from the LAST think
            # in each burst predicting the next REAL token (that target is NOT
            # masked). inj = mean ‖injection‖ at think positions (0 ⇒ WM inert /
            # no think tokens in batch), think% = fraction of think positions in
            # the batch, Wproj = ‖W_proj.weight‖ (drifts from init as WM learns).
            if (getattr(args, "use_memory", False)
                    and hasattr(model, "memory")
                    and getattr(model.memory, "_last_injection", None) is not None):
                with torch.no_grad():
                    inj = main_wm_inj                                    # (B,T,d) main fwd
                    tmask = (x == int(thinking_token_id))                # (B,T)
                    n_think = int(tmask.sum())
                    if (inj is not None and inj.shape[:2] == tmask.shape
                            and n_think > 0):
                        inj_norm = float(
                            inj[tmask].float().norm(dim=-1).mean())
                    else:
                        inj_norm = 0.0
                    think_frac = float(tmask.float().mean())
                    wproj = float(model.memory.W_proj.weight.float().norm())
                line += (f"  wm(inj={inj_norm:.3f},think%={think_frac*100:.1f},"
                         f"Wproj={wproj:.2f})")
            if is_main:
                print(line)
            if tb is not None:
                tb.add_scalar("train/loss", tloss_avg, step)
                tb.add_scalar("train/ppl", float(torch.tensor(tloss_avg).exp()), step)
                tb.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                tb.add_scalar("train/tok_per_sec", tok_per_sec, step)
                if _blk_gnorms is not None:
                    for L, (g, u) in enumerate(zip(_blk_gnorms, _blk_uratios)):
                        tb.add_scalar(f"layer_grad_norm/L{L:02d}", g, step)
                        tb.add_scalar(f"layer_update_ratio/L{L:02d}", u, step)
                    tb.add_scalar("layer_grad_norm/last_over_first",
                                  _blk_gnorms[-1] / max(_blk_gnorms[0], 1e-12),
                                  step)
                if (getattr(args, "use_pkm", False)
                        and hasattr(model, "pkm_layer")):
                    pkm = model.pkm_layer
                    with torch.no_grad():
                        if pkm.use_output_gate:
                            tb.add_scalar("pkm/alpha_learned",
                                          float(pkm.out_alpha.detach()), step)
                        floor = float(getattr(pkm, "alpha_floor", 0.0))
                        tb.add_scalar("pkm/alpha_floor", floor, step)
                        rn_mean = float(torch.stack([
                            emb.weight.float().norm(dim=-1).mean()
                            for emb in pkm.values
                        ]).mean())
                        init_norm = float(pkm._expected_init_row_norm)
                        tb.add_scalar("pkm/row_norm_mean", rn_mean, step)
                        tb.add_scalar("pkm/row_norm_ratio_vs_init",
                                      rn_mean / max(init_norm, 1e-9), step)
                        tb.add_scalar("pkm/epsilon",
                                      float(getattr(pkm, "random_slot_epsilon", 0.0)),
                                      step)
                        if pkm._last_slot_idx is not None:
                            n_slots = pkm.n_keys * pkm.n_keys
                            idx = pkm._last_slot_idx
                            w_ = pkm._last_weights
                            H_ = pkm.n_heads
                            slot_mass = torch.zeros(H_, n_slots,
                                                     device=idx.device,
                                                     dtype=torch.float32)
                            for h_ in range(H_):
                                slot_mass[h_].scatter_add_(
                                    0, idx[:, :, h_, :].reshape(-1),
                                    w_[:, :, h_, :].reshape(-1).float())
                            slot_mass = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                            uniq = (slot_mass > 0).sum(dim=-1).float().mean()
                            top = slot_mass.max(dim=-1).values.mean()
                            ent = -(slot_mass * slot_mass.clamp_min(1e-12).log()).sum(dim=-1).mean()
                            tb.add_scalar("pkm/unique_slots_per_head", float(uniq), step)
                            tb.add_scalar("pkm/top_slot_share", float(top), step)
                            tb.add_scalar("pkm/slot_entropy", float(ent), step)
                if args.output_gate and losses_gate_window:
                    tb.add_scalar("gate/think_frac", 1.0 - emit_frac, step)
                    tb.add_scalar("gate/mean_gate", mean_g, step)
                    tb.add_scalar("gate/raw_ce", raw_ce, step)
                    if emit_ce == emit_ce:  # not NaN
                        tb.add_scalar("gate/emit_ce", emit_ce, step)
                    if args.gate_warmup_steps > 0 or args.gate_floor_min > 0:
                        if args.gate_warmup_steps > 0:
                            progress = min(1.0, step / args.gate_warmup_steps)
                            gate_floor = (1.0 - progress) + progress * args.gate_floor_min
                        else:
                            gate_floor = args.gate_floor_min
                        tb.add_scalar("gate/floor", gate_floor, step)
                if (getattr(args, "gate_calibration_weight", 0.0) > 0.0
                        and _gate_calib_diag is not None):
                    _t1, _sg, _gd, _gn = _gate_calib_diag
                    tb.add_scalar("gate_calib/target_frac_pos", _t1, step)
                    tb.add_scalar("gate_calib/mean_sigma", _sg, step)
                    tb.add_scalar("gate_calib/mean_delta_logp", _gd, step)
                if (getattr(args, "latent_cotrain_weight", 0.0) > 0.0
                        and _latent_cotrain_diag is not None):
                    _d, _n = _latent_cotrain_diag
                    tb.add_scalar("latent_cotrain/delta_logp", _d, step)
                if (_latent_reasoner is not None
                        and _latent_reasoning_diag is not None):
                    _rl, _rr = _latent_reasoning_diag
                    tb.add_scalar("latent_reasoning/loss", _rl, step)
                    tb.add_scalar("latent_reasoning/rung", _rr, step)
                if args.enable_thinking_token and think_stats_window:
                    tb.add_scalar("think/rate", think_rate, step)
                    tb.add_scalar("think/explore_rate", explore_rate, step)
                    tb.add_scalar("think/explore_prob", explore_prob, step)
                    tb.add_scalar("think/curriculum", curr, step)
                    tb.add_scalar("think/lambda", lambda_eff, step)
                    tb.add_scalar("think/gate_threshold", gate_thr, step)
                    tb.add_scalar("think/queue_pressure", q_pressure, step)
                    tb.add_scalar("think/safety_max_depth", depth_cap, step)
                    tb.add_scalar("think/replay_think_rate", replay_rate, step)
                    tb.add_scalar("think/answer_ce", answer_ce, step)
                    if traj_nll == traj_nll:
                        tb.add_scalar("think/trajectory_nll", traj_nll, step)
                    tb.add_scalar("think/queue_len", queue_len, step)
                    tb.add_scalar("think/queue_mean_depth", q_mean_depth, step)
                    tb.add_scalar("think/queue_max_depth", q_max_depth, step)
                    tb.add_scalar("think/forced_emit_rate", forced_rate, step)
                    tb.add_scalar("think/queue_dropped", q_dropped, step)
                    tb.add_scalar("think/cont_items", cont_items, step)
                    tb.add_scalar("think/replay_items", replay_items, step)
                # FiLM α values.
                if args.feedback != "none" or fb_xattn_pairs:
                    for entry in model.feedback_alphas():
                        if isinstance(entry, tuple) and len(entry) == 3:
                            t_idx, s_idx, alpha = entry
                            label = (f"alpha/t{t_idx}"
                                     if isinstance(s_idx, tuple)
                                     else f"alpha/t{t_idx}_s{s_idx}")
                            tb.add_scalar(label, alpha, step)
            last_log = now
            last_log_step = step

        # DDP: rank 0 owns validation (collective-free; other ranks just wait
        # at the next backward all-reduce until rank 0 rejoins).
        if (step % args.val_every == 0 or step == args.steps) and is_main:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vx, vy, *_vrest = vbatch
                    vx, vy = vx.to("cuda"), vy.to("cuda")
                    vdoc_ids = _vrest[0].to("cuda") if _vrest else None
                    vlogits = model(vx, doc_ids=vdoc_ids)
                    if args.enable_thinking_token:
                        vloss = cross_entropy_masking_token(
                            vlogits.reshape(-1, vlogits.shape[-1]), vy.reshape(-1),
                            int(thinking_token_id),
                            reduction="mean",
                        )
                    else:
                        vloss = F.cross_entropy(
                            vlogits.reshape(-1, vlogits.shape[-1]), vy.reshape(-1),
                        )
                    val_loss += vloss.item() * vx.numel()
                    n_val += vx.numel()
                    if n_val >= 64 * args.T:                # cap val
                        break
            val_loss /= n_val
            ppl = float(torch.tensor(val_loss).exp())
            print(f"        VAL  loss={val_loss:.4f}  ppl={ppl:.2f}")
            if tb is not None:
                tb.add_scalar("val/loss", val_loss, step)
                tb.add_scalar("val/ppl", ppl, step)
            model.train()
            torch.cuda.empty_cache()

        # Per-feature usefulness probe: ablate each mechanism on a held-out
        # batch and log the CE rise (load-bearing iff Δce > 0). Runs no_grad
        # and restores every poked attribute, so it can't perturb training.
        if (next_feature_probe_at and is_main
                and tokens_seen >= next_feature_probe_at):
            from experiments.feature_probe import (
                run_feature_probe, format_feature_probe)
            try:
                _fp_batch = next(iter(val_loader))
                _fpx, _fpy, *_fprest = _fp_batch
                _fpx, _fpy = _fpx.to("cuda"), _fpy.to("cuda")
                _fpdoc = _fprest[0].to("cuda") if _fprest else None
                _fp_metrics = run_feature_probe(
                    model, _fpx, _fpy, doc_ids=_fpdoc,
                    thinking_token_id=thinking_token_id)
                print("        " + format_feature_probe(_fp_metrics))
                if tb is not None:
                    for _k, _v in _fp_metrics.items():
                        tb.add_scalar(f"probe/{_k}", float(_v), step)
            except Exception as _e:  # never let the probe kill a 20h run
                print(f"        [feature-probe] skipped ({_e})")

            # WM load-bearing signal — the natural-text probe batch above has
            # zero think tokens, so WM's ablation-delta there is ≈0 BY DESIGN
            # (WM reads only at think positions). The real WM signal comes from
            # held-out long-context recall WITH think tokens: recall_full vs
            # recall with the WM read mean-ablated. delta > 0 ⇒ WM load-bearing.
            _wm_recall_path = getattr(
                args, "feature_probe_wm_recall_path", "") or ""
            if (_wm_recall_path and getattr(model, "memory", None) is not None
                    and thinking_token_id is not None):
                try:
                    from experiments.eval_longctx_recall import (
                        eval_longctx_recall)
                    _wm_n = int(getattr(args, "feature_probe_wm_recall_n", 64))
                    _full = eval_longctx_recall(
                        model, tok, _wm_recall_path, n=_wm_n,
                        wm_ablate="none")
                    _abl = eval_longctx_recall(
                        model, tok, _wm_recall_path, n=_wm_n,
                        wm_ablate="mean")
                    _wm_recall = _full["recall"]
                    _wm_recall_delta = _wm_recall - _abl["recall"]
                    print(f"        [wm-recall] recall={_wm_recall:.3f} "
                          f"Δ(full-ablated)={_wm_recall_delta:+.3f} "
                          f"think_frac={_full['think_frac']:.3f} "
                          f"(n={int(_full['n_total'])})")
                    if tb is not None:
                        tb.add_scalar("probe/wm_recall", _wm_recall, step)
                        tb.add_scalar("probe/wm_recall_delta",
                                      _wm_recall_delta, step)
                        tb.add_scalar("probe/wm_recall_think_frac",
                                      _full["think_frac"], step)
                except Exception as _e:  # never let the probe kill a 20h run
                    print(f"        [wm-recall] skipped ({_e})")
                model.train()
            torch.cuda.empty_cache()
            while next_feature_probe_at <= tokens_seen:
                next_feature_probe_at += int(args.feature_probe_every_tokens)

        if (args.probe_humaneval_every_tokens > 0 and is_main
                and tokens_seen - tokens_at_last_probe
                    >= args.probe_humaneval_every_tokens):
            from experiments.probe_humaneval import run_humaneval_probe
            n_probe = (args.probe_humaneval_n_problems
                       if args.probe_humaneval_n_problems > 0 else None)
            try:
                res = run_humaneval_probe(
                    model, tok,
                    probe_path=args.probe_humaneval_path,
                    max_gen=args.probe_humaneval_max_gen,
                    n_problems=n_probe,
                    use_thinking=bool(args.output_gate
                                       and thinking_token_id is not None),
                    thinking_token_id=thinking_token_id,
                    gate_floor=float(args.gate_floor_min),
                    min_emit_before_eos=30,
                )
                print(f"        PROBE  pass@1={res['pass_rate']*100:.1f}% "
                      f"({res['n_passed']}/{res['n_total']})  "
                      f"emit={res['mean_emit_tokens']:.0f}tok  "
                      f"t={res['elapsed_s']:.1f}s  "
                      f"@tok={tokens_seen/1e6:.1f}M")
                if tb is not None:
                    tb.add_scalar("probe/pass_rate", res["pass_rate"], step)
                    tb.add_scalar("probe/n_passed", res["n_passed"], step)
            except Exception as e:
                print(f"        PROBE  ERROR: {e}")
            tokens_at_last_probe = tokens_seen
            torch.cuda.empty_cache()

        # ---- Mid-training HumanEval hook (auto_stop). ----
        # tokens_seen is the rough count of tokens consumed by the train
        # loop so far. At every `mid_eval_every_tokens` boundary, save a
        # ckpt, shell out to eval_humaneval.py, log pass-rate to TB,
        # append to controller, optionally stop.
        # Under DDP every rank consumes batch*T*grad_accum tokens per step, so
        # the global token count scales with world size.
        tokens_seen = step * args.batch * args.T * args.grad_accum * ddp_world_size
        if (mid_eval_controller is not None and is_main
                and tokens_seen >= next_eval_at):
            # Snapshot to a numbered ckpt — kept for the curve plot.
            stem = pathlib.Path(args.save_ckpt or "checkpoints/pretrain.pt").stem
            mid_path = pathlib.Path("checkpoints") / (
                f"{stem}_step{step}_tok{tokens_seen}.pt")
            mid_path.parent.mkdir(parents=True, exist_ok=True)
            _save_cfg = dict(
                vocab_size=model_vocab_size,
                tokenizer_base_vocab_size=tok.vocab_size,
                d_model=args.d_model, n_heads=args.n_heads,
                d_head=args.d_head, n_layers=n_layers_actual,
                max_T=args.max_T, feedback_mode=args.feedback,
                feedback_pairs=fb_pairs,
                feedback_self_k=args.feedback_self_k,
                feedback_alpha_mode=args.feedback_alpha_mode,
                arch=args.arch, layers_spec=args.layers,
                tokenizer=args.tokenizer,
                thinking_token_id=thinking_token_id,
                use_memory=bool(args.use_memory),
                mem_size=int(args.mem_size) if args.use_memory else 0,
                mem_dim=(int(args.mem_dim) if args.mem_dim > 0
                          else int(args.d_model)) if args.use_memory else 0,
                use_pkm=bool(getattr(args, "use_pkm", False)),
                pkm_after_layer=int(getattr(args, "pkm_after_layer", 14)),
                pkm_n_keys=int(getattr(args, "pkm_n_keys", 256)),
                pkm_n_heads=int(getattr(args, "pkm_n_heads", 4)),
                pkm_k_dim=int(getattr(args, "pkm_k_dim", 128)),
                pkm_top_k=int(getattr(args, "pkm_top_k", 32)),
                pkm_value_bf16=bool(getattr(args, "pkm_value_bf16", True)),
                # v7 PKM-bootstrap-fix package.
                pkm_score_norm=str(getattr(args, "pkm_score_norm", "layer")),
                pkm_value_init_std=float(getattr(args, "pkm_value_init_std", 1.0)),
                pkm_use_output_gate=bool(getattr(args, "pkm_use_output_gate", True)),
                output_gate=bool(args.output_gate
                                  or (args.enable_thinking_token
                                      and args.think_decision == "gate")),
                state_readonly_at_think=bool(
                    getattr(args, "state_readonly_at_think", False)),
                use_latent_feedback_adapter=bool(
                    getattr(args, "use_latent_feedback_adapter", False)),
                retrieval_input_additive=False,
            )
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "config": _save_cfg}, str(mid_path))
            print(f"\n[mid-eval] saved ckpt at step={step} tokens={tokens_seen:,}"
                  f" → {mid_path}")

            # Decide whether to run the HumanEval subprocess. Two skip paths,
            # both leave the ckpt on disk and advance the counter (resume
            # artifact > HumanEval signal during pretrain):
            #   1. --mid_eval_save_only: explicit user opt-out.
            #   2. Auto-skip: trainer is using nearly all of GPU memory; the
            #      eval subprocess would OOM trying to load its own model copy
            #      on the same device (observed in v4 at step 1526).
            skip_eval = False
            skip_reason = ""
            if args.mid_eval_save_only:
                skip_eval = True
                skip_reason = "--mid_eval_save_only"
            elif args.mid_eval_min_free_gib > 0:
                free_b, _ = torch.cuda.mem_get_info()
                free_gib = free_b / (1024 ** 3)
                if free_gib < args.mid_eval_min_free_gib:
                    skip_eval = True
                    skip_reason = (
                        f"free GPU memory {free_gib:.2f} GiB < "
                        f"{args.mid_eval_min_free_gib:.2f} GiB — eval "
                        f"subprocess would OOM")

            if skip_eval:
                print(f"[mid-eval] SKIPPED HumanEval ({skip_reason}). "
                      f"Ckpt is on disk; advancing counter.")
                from experiments.eval_callback import EvalResult
                res = EvalResult(
                    humaneval_pass_rate=float("nan"),
                    mbpp_pass_rate=None,
                    tokens_seen=tokens_seen, step=step,
                    ckpt_path=str(mid_path),
                    raw_log_tail=f"<skipped: {skip_reason}>",
                )
            else:
                print(f"[mid-eval] running HumanEval (max_problems="
                      f"{args.mid_eval_n_problems}) ...")
                model.eval()
                res = run_eval(
                    str(mid_path), tokens_seen=tokens_seen, step=step,
                    n_problems=args.mid_eval_n_problems,
                    max_gen=args.mid_eval_max_gen,
                    use_thinking=bool(args.use_memory),
                    emit_threshold=0.5,
                    min_emit_before_eos=int(args.mid_eval_min_emit_before_eos),
                    gate_floor=float(args.gate_floor_min),
                )
                model.train()
            mid_eval_controller.append(res)
            print(f"[mid-eval] {mid_eval_controller.summary_line()}")
            if not skip_eval and res.humaneval_pass_rate != res.humaneval_pass_rate:  # NaN
                print("[mid-eval] WARNING: humaneval=NaN — eval subprocess "
                      "did not emit a parseable `pass@k =` line. Last 2 kB "
                      "of its stdout/stderr:")
                print(res.raw_log_tail)
            if tb is not None:
                tb.add_scalar("eval/humaneval", res.humaneval_pass_rate, step)
                tb.add_scalar("eval/humaneval_vs_tokens",
                              res.humaneval_pass_rate, tokens_seen // 1_000_000)
            # Advance the next-eval threshold to the next interval; if we
            # blew past several intervals (e.g. small batch * long step),
            # snap forward by multiples.
            while next_eval_at <= tokens_seen:
                next_eval_at += int(args.mid_eval_every_tokens)
            if args.auto_stop and mid_eval_controller.should_stop():
                print(f"[mid-eval] AUTO STOP — pass-rate plateaued. "
                      f"Last {args.auto_stop_k} intervals each gained "
                      f"<{args.auto_stop_threshold:.3f}. Stopping at "
                      f"step={step} tokens={tokens_seen:,}.")
                break

    secs = time.perf_counter() - t0
    print(f"\nDone in {secs:.0f}s ({secs/args.steps*1000:.0f} ms/step avg).")
    if tb is not None:
        tb.close()

    if args.save_ckpt and is_main:
        ckpt = {
            "state_dict": model.state_dict(),
            "step": locals().get("step", 0),
            "config": {
                "vocab_size": model_vocab_size,
                "tokenizer_base_vocab_size": tok.vocab_size,
                "use_memory": bool(args.use_memory),
                "mem_size": (int(args.mem_size) if args.use_memory else 0),
                "mem_dim": ((int(args.mem_dim) if args.mem_dim > 0
                             else int(args.d_model)) if args.use_memory else 0),
                "use_pkm": bool(getattr(args, "use_pkm", False)),
                "pkm_after_layer": int(getattr(args, "pkm_after_layer", 14)),
                "pkm_n_keys": int(getattr(args, "pkm_n_keys", 256)),
                "pkm_n_heads": int(getattr(args, "pkm_n_heads", 4)),
                "pkm_k_dim": int(getattr(args, "pkm_k_dim", 128)),
                "pkm_top_k": int(getattr(args, "pkm_top_k", 32)),
                "pkm_value_bf16": bool(getattr(args, "pkm_value_bf16", True)),
                "pkm_score_norm": str(getattr(args, "pkm_score_norm", "layer")),
                "pkm_value_init_std": float(getattr(args, "pkm_value_init_std", 1.0)),
                "pkm_use_output_gate": bool(getattr(args, "pkm_use_output_gate", True)),
                "data_mix": args.data_mix,
                "think_burst_prob": args.think_burst_prob,
                "think_max_bursts": args.think_max_bursts,
                "think_max_burst_depth": args.think_max_burst_depth,
                "d_model": args.d_model, "n_heads": args.n_heads,
                "d_head": args.d_head, "n_layers": n_layers_actual,
                "max_T": args.T, "feedback_mode": args.feedback,
                "feedback_pairs": fb_pairs,
                "feedback_xattn_pairs": fb_xattn_pairs,
                "feedback_xattn_heads": args.feedback_xattn_heads,
                "feedback_xattn_form": args.feedback_xattn_form,
                "feedback_self_k": args.feedback_self_k,
                "feedback_alpha_mode": args.feedback_alpha_mode,
                "arch": args.arch, "layers_spec": args.layers,
                "tokenizer": args.tokenizer,
                "enable_thinking_token": bool(args.enable_thinking_token),
                "thinking_token": args.thinking_token,
                "thinking_token_id": thinking_token_id,
                "think_lambda": args.think_lambda,
                "think_lambda_start": args.think_lambda_start,
                "think_curriculum_steps": args.think_curriculum_steps,
                "think_policy": args.think_policy,
                "think_decision": args.think_decision,
                "think_gate_threshold": args.think_gate_threshold,
                "think_gate_threshold_start": args.think_gate_threshold_start,
                "think_explore_prob": args.think_explore_prob,
                "think_explore_start_prob": args.think_explore_start_prob,
                "think_safety_max_depth": args.think_safety_max_depth,
                "think_safety_max_depth_start": args.think_safety_max_depth_start,
                "think_aux_normalize": args.think_aux_normalize,
                "think_aux_loss_scale": args.think_aux_loss_scale,
                "think_queue_accum_steps": args.think_queue_accum_steps,
                "think_queue_accum_max_steps": args.think_queue_accum_max_steps,
                "think_queue_drain_target": args.think_queue_drain_target,
                "think_backpressure_target": args.think_backpressure_target,
                "think_backpressure_max": args.think_backpressure_max,
                "think_backpressure_lambda": args.think_backpressure_lambda,
                "think_backpressure_threshold": args.think_backpressure_threshold,
                "think_backpressure_explore": args.think_backpressure_explore,
                "state_readonly_at_think": bool(
                    getattr(args, "state_readonly_at_think", False)),
                "use_latent_feedback_adapter": bool(
                    getattr(args, "use_latent_feedback_adapter", False)),
                "retrieval_input_additive": False,
            },
        }
        pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save_ckpt)
        print(f"Checkpoint saved to {args.save_ckpt}")

    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
