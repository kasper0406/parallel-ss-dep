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
    DeltaNetAttention,
    SoftmaxAttention, Mamba2Attention,
)
from experiments.model import TinyLM
from experiments.aux_brackets import compute_bracket_deltas, bracket_depth
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
    mask_token_logit,
)


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
                           doc_ids=None):
    """Forward + LM loss for the non-thinking-token (pretrain) path.

    Returns (logits, ce_per_token, lm_loss, aux_loss). Factored out of the
    step loop so gradient accumulation can run it once per microbatch.
    Mirrors the inline forward + gate/plain-loss branches exactly.
    """
    if args.aux_brackets:
        logits, aux_logits = model(x, return_aux=True, doc_ids=doc_ids)
        depth = bracket_depth(x, bracket_deltas).clamp(0, args.aux_max_depth)
        aux_loss = F.cross_entropy(
            aux_logits.reshape(-1, args.aux_max_depth + 1),
            depth.reshape(-1),
        )
    else:
        logits = model(x, doc_ids=doc_ids)
        aux_loss = torch.zeros((), device="cuda")
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
        gate_terms = g_eff * ce_per_token + (1.0 - g_eff) * args.gate_lambda
        valid = (y != -100).float()
        denom = valid.sum().clamp(min=1.0)
        lm_loss = (gate_terms * valid).sum() / denom
    else:
        valid = (y != -100).float()
        denom = valid.sum().clamp(min=1.0)
        lm_loss = (ce_per_token * valid).sum() / denom
    return logits, ce_per_token, lm_loss, aux_loss


def _z_loss_term(logits, weight):
    """z-loss regulariser: weight * mean(logsumexp(logits)^2)."""
    if weight <= 0.0:
        return logits.new_zeros(())
    return weight * (torch.logsumexp(logits, dim=-1) ** 2).mean()


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
            base_seed=args.seed,
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

    # Optimizer construction — see experiments/optim_utils.py.
    from experiments.optim_utils import build_optimizer
    opts, scheds = build_optimizer(
        model, optimizer=args.optimizer, lr=args.lr, lr_muon=args.lr_muon,
        alpha_wd=args.alpha_wd, steps=args.steps, wd=args.wd,
        lr_schedule=args.lr_schedule, warmup_steps=args.warmup_steps,
        decay_frac=args.lr_decay_frac,
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

    # TensorBoard writer — no-op context when --tb_dir is not set.
    if args.tb_dir:
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
                logits, ce_per_token, lm_loss, aux_loss = _nonthink_forward_loss(
                    model, x, y, args, step, bracket_deltas, doc_ids=doc_ids)
                loss = lm_loss + args.aux_weight * aux_loss
                loss = loss + _z_loss_term(logits, args.z_loss)
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
            g_detached = model._last_gate.detach()
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
            tok_per_sec = ((step - last_log_step) * args.batch * args.T
                           * args.grad_accum / (now - last_log))
            tloss_avg = sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:]))
            line = (f"{step:>6d}  {tok_per_sec:>8.0f}  "
                    f"{tloss_avg:>8.4f}  "
                    f"{scheduler.get_last_lr()[0]:>9.2e}")
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

        if step % args.val_every == 0 or step == args.steps:
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

        # ---- Mid-training HumanEval hook (auto_stop). ----
        # tokens_seen is the rough count of tokens consumed by the train
        # loop so far. At every `mid_eval_every_tokens` boundary, save a
        # ckpt, shell out to eval_humaneval.py, log pass-rate to TB,
        # append to controller, optionally stop.
        tokens_seen = step * args.batch * args.T * args.grad_accum
        if (mid_eval_controller is not None
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
                output_gate=bool(args.output_gate
                                  or (args.enable_thinking_token
                                      and args.think_decision == "gate")),
            )
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "config": _save_cfg}, str(mid_path))
            print(f"\n[mid-eval] saved ckpt at step={step} tokens={tokens_seen:,}"
                  f" → {mid_path}")
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
            if res.humaneval_pass_rate != res.humaneval_pass_rate:  # NaN
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

    if args.save_ckpt:
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
            },
        }
        pathlib.Path(args.save_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, args.save_ckpt)
        print(f"Checkpoint saved to {args.save_ckpt}")


if __name__ == "__main__":
    sys.exit(main())
