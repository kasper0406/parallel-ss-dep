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
    DeltaNetAttention, DeltaNetNegEigAttention, GatedDeltaNetAttention,
    DeltaNetForgetGateAttention,
    GatedDeltaProductAttention,
    OrthogonalScanAttention, SymbolGroundedAttention,
    HeisenbergAttention, MultiPassAttention,
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


_NAME_TO_CLS = {
    "deltanet":   DeltaNetAttention,
    "deltanet_negeig": DeltaNetNegEigAttention,
    "deltanet_forgetgate": DeltaNetForgetGateAttention,
    "gated_deltanet": GatedDeltaNetAttention,
    "gated_deltaproduct": GatedDeltaProductAttention,
    "ortho":      OrthogonalScanAttention,
    "transformer": SoftmaxAttention,
    "mamba2":     Mamba2Attention,
}


def _make_sg_factory(vocab_size: int, n_symbols: int):
    """Factory closure for SymbolGroundedAttention with vocab params bound."""
    def _f(**kw):
        return SymbolGroundedAttention(vocab_size=vocab_size, n_symbols=n_symbols, **kw)
    return _f


def build_arch(name: str, n_layers: int, vocab_size: int = 0, n_symbols: int = 512):
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
    # SymbolGrounded variants: need vocab_size + n_symbols bound at build time.
    if name == "symgrounded":
        sg = _make_sg_factory(vocab_size, n_symbols)
        return dict(attention_cls=sg)
    if name == "hybrid_sg":
        # 50/50 alternating SymGrounded + DeltaNet (sg first).
        sg = _make_sg_factory(vocab_size, n_symbols)
        cls = [sg if i % 2 == 0 else DeltaNetAttention for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    if name == "hybrid_sg_25_75":
        # 1 sg + 3 delta pattern (sparse symbolic).
        sg = _make_sg_factory(vocab_size, n_symbols)
        cls = [sg if i % 4 == 0 else DeltaNetAttention for i in range(n_layers)]
        return dict(attention_cls_per_layer=cls)
    # Multi-pass — direction B. K cells in parallel within each layer.
    if name == "multipass_dh":
        # DeltaNet + Heisenberg in parallel at every layer.
        def _mp(**kw):
            return MultiPassAttention(
                cells=[DeltaNetAttention, HeisenbergAttention], **kw
            )
        return dict(attention_cls=_mp)
    if name == "multipass_dho":
        # DeltaNet + Heisenberg + Ortho in parallel.
        def _mp(**kw):
            return MultiPassAttention(
                cells=[DeltaNetAttention, HeisenbergAttention,
                       OrthogonalScanAttention], **kw
            )
        return dict(attention_cls=_mp)
    if name == "multipass_dd":
        # Two strong LM cells: DeltaNet + DeltaNet_negeig (allow_neg_eigval).
        # Tests whether multi-pass helps when both cells are competent LMs.
        def _mp(**kw):
            return MultiPassAttention(
                cells=[DeltaNetAttention, DeltaNetNegEigAttention], **kw
            )
        return dict(attention_cls=_mp)
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
                   choices=["deltanet", "deltanet_negeig", "deltanet_forgetgate",
                            "gated_deltanet",
                            "gated_deltaproduct",
                            "ortho", "transformer", "mamba2",
                            "hybrid", "hybrid_25_75", "hybrid_75_25",
                            "hybrid_negeig",
                            "symgrounded", "hybrid_sg", "hybrid_sg_25_75",
                            "multipass_dh", "multipass_dho", "multipass_dd"])
    p.add_argument("--n_symbols", type=int, default=512,
                   help="Hash bucket size for SymbolGrounded layer.")
    p.add_argument("--aux_brackets", action="store_true",
                   help="Add bracket-depth auxiliary loss (direction E).")
    p.add_argument("--aux_weight", type=float, default=0.1,
                   help="Weight on the aux loss term.")
    p.add_argument("--aux_max_depth", type=int, default=24,
                   help="Cap bracket depth at this value for the aux head.")
    p.add_argument("--feedback", type=str, default="none",
                   choices=["none", "additive", "film", "predictive"],
                   help="Cross-layer top-down feedback mode (Day 1).")
    p.add_argument("--surprise_weight", type=float, default=0.0,
                   help="Weight on the surprise (prediction-error) aux "
                        "loss for predictive feedback mode.")
    p.add_argument("--freeze_alpha", action="store_true",
                   help="Diagnostic: build film/additive arch with all α=0 "
                        "frozen. Tests dead-weight effect of feedback "
                        "machinery without active feedback.")
    p.add_argument("--save_ckpt", type=str, default=None,
                   help="Path to save final model checkpoint (for downstream "
                        "evals like HumanEval, bracket-structure, long-T PPL).")
    p.add_argument("--feedback_distances", type=str, default="1",
                   help="Comma-separated distances for multi-scale top-down "
                        "feedback. '1' = single-step (default), "
                        "'1,2,4,8,16' = exponential reach.")
    p.add_argument("--feedback_pairs", type=str, default="",
                   help="Sparse (target,source) feedback connections; "
                        "semicolon-separated. E.g. '2,28;3,27' means "
                        "layer 2's input gets feedback from layer 28, "
                        "and layer 3's input gets feedback from layer 27. "
                        "If non-empty, overrides --feedback_distances.")
    p.add_argument("--feedback_xattn", type=str, default="",
                   help="Cross-layer attention feedback. Each target attends "
                        "over multiple source layers' lagged hidden states. "
                        "Syntax 'target:src1,src2,...; target2:src3,src4,...'. "
                        "Example '2:14,21,28' = layer 2's input attends over "
                        "layers 14, 21, 28. 'all' = every layer attends over "
                        "every later layer (Idea 2). If non-empty, overrides "
                        "BOTH --feedback_distances and --feedback_pairs and "
                        "ignores --feedback (attention is its own form).")
    p.add_argument("--feedback_xattn_heads", type=int, default=4,
                   help="Number of heads inside the cross-layer attention.")
    p.add_argument("--feedback_lag", type=int, default=1,
                   help="Lag in tokens for the source state before feeding "
                        "to target (default 1 = t-1, parallel-scan friendly).")
    p.add_argument("--feedback_position", type=str, default="pre",
                   choices=["pre", "post"],
                   help="'pre' (default) modulates target's input; "
                        "'post' modulates target's output.")
    p.add_argument("--feedback_per_channel_alpha", action="store_true",
                   help="Use per-channel α (d_model floats) instead of scalar α "
                        "for sparse FiLM. Tests whether channel-dependent gating "
                        "can mix the negative- and positive-α basins per channel.")
    p.add_argument("--feedback_self_k", type=int, default=0,
                   choices=[0, 2, 3],
                   help="K-iteration self-feeding training for sparse FiLM. "
                        "0 (default) = standard 2-pass training. "
                        "2 = cold start + one self-feed; loss on iter 2 "
                        "(same compute as default 2-pass). "
                        "3 = cold start + two self-feeds; loss on iter 3 "
                        "(~50%% more compute). Trains the model to be "
                        "self-consistent under lagged-cached deployment, "
                        "closing the train/inference gap. Requires "
                        "--feedback_pairs to be set.")
    p.add_argument("--feedback_alpha_mode", type=str, default="scalar",
                   choices=["scalar", "surprise_modulated"],
                   help="α form for sparse-FiLM feedback. "
                        "'scalar' (default, matches Phase 21c) = single "
                        "learnable α per (target,source) pair. "
                        "'surprise_modulated' (Phase 22 / structural-surprise "
                        "PoC) = per-token α(t) = α₀·σ(scale·s_z(t)+bias) "
                        "where s_z(t) is the per-batch z-scored inter-iter "
                        "delta of source-state norms (free signal from K=3 "
                        "self-feeding). Adds 3 learnable scalars per FiLM "
                        "target. Requires --feedback_self_k >= 2.")
    p.add_argument("--semantic_loss_beta", type=float, default=0.0,
                   help="Weight on the semantic-gradient loss L_sem (Phase 22 "
                        "full PoC). 0 (default) = off. Non-zero = enable "
                        "L_sem; --encoder_ckpt and --oracle_ckpt must be "
                        "provided so per-statement target embeddings + "
                        "oracle surprises can be computed. The total loss "
                        "becomes L = L_ce + β · sum_t L_sem(s_t), where "
                        "L_sem(s_t) = surprise(s_t).detach() · "
                        "(1 - cos(W·pool(h), E(s_t).detach())).")
    p.add_argument("--encoder_ckpt", type=str, default=None,
                   help="Path to the frozen DN-baseline encoder checkpoint, "
                        "used by L_sem to produce per-statement target "
                        "embeddings E(s_t).")
    p.add_argument("--oracle_ckpt", type=str, default=None,
                   help="Path to the trained oracle predictive head checkpoint, "
                        "used by L_sem to produce per-statement surprise "
                        "weights via 1 - cos(P(prefix), E(s_t)).")
    p.add_argument("--max_stmts_per_chunk", type=int, default=64,
                   help="Maximum number of statements scored per chunk under "
                        "L_sem (caps padding cost in the collator).")
    p.add_argument("--semantic_loss_uniform_weight", action="store_true",
                   help="Ablation: set all L_sem per-statement weights to 1.0 "
                        "(i.e. ignore the oracle's surprise score). Used to "
                        "isolate whether structural-surprise weighting "
                        "contributes anything beyond the alignment loss "
                        "itself.")
    p.add_argument("--semantic_loss_granularity", type=str, default="statement",
                   choices=["statement", "token"],
                   help="Phase 22b ablation 2: granularity of L_sem alignment. "
                        "'statement' (default) = mean-pool model+encoder hidden "
                        "across each AST statement, align with cosine. 'token' "
                        "= per-token cosine alignment (no AST infrastructure, "
                        "uses standard TokenisedStream loader). Used to test "
                        "whether AST-statement segmentation does real work or "
                        "if pure per-token alignment matches it.")
    p.add_argument("--logit_kl_beta", type=float, default=0.0,
                   help="Phase 22b ablation 3: weight on a textbook KL-on-logits "
                        "distillation loss. >0 = enable; replaces L_sem "
                        "entirely. Encoder forward provides the teacher logits "
                        "(temperature `--logit_kl_temp`). Loss adds "
                        "β · T² · KL(softmax(student/T) || softmax(teacher/T)) "
                        "averaged over (B*T) positions. Requires --encoder_ckpt.")
    p.add_argument("--logit_kl_temp", type=float, default=2.0,
                   help="Temperature for the KL-on-logits ablation (default 2.0, "
                        "textbook KD).")
    p.add_argument("--feedback_scratchpad", type=str, default="",
                   help="Surprise-gated scratchpad pairs, semicolon-separated. "
                        "Each is 'target,source' where target attends causally "
                        "over source-layer pass-1 outputs with attention scores "
                        "biased by per-position surprise. E.g. '2,28' applies "
                        "the scratchpad at layer 2 reading from layer 28.")
    p.add_argument("--feedback_scratchpad_heads", type=int, default=9,
                   help="Number of heads inside the scratchpad attention.")
    p.add_argument("--feedback_scratchpad_routing", type=str, default="softmax",
                   choices=["softmax", "sigmoid", "sigmoid_uniform",
                            "uniform_surprise"],
                   help="Routing for the scratchpad attention. Disentangling: "
                        "'softmax' = content (Q·K) softmax + surprise log-bias; "
                        "'sigmoid' = content sigmoid × surprise (multiplicative); "
                        "'sigmoid_uniform' = content only, surprise IGNORED "
                        "(tests pure content retrieval); "
                        "'uniform_surprise' = surprise only, content IGNORED "
                        "(tests pure saliency-based retrieval).")
    p.add_argument("--output_gate", action="store_true",
                   help="Enable learned per-position output gate (Phase 23). "
                        "Adds a gate_head (d_model → 1) whose sigmoid output "
                        "g_t ∈ (0,1) weights the loss: "
                        "L = mean(g_t * CE_t + (1 - g_t) * gate_lambda). "
                        "When g_t → 0 the model 'thinks' (pays λ); when "
                        "g_t → 1 it emits (pays CE). Self-regulating: the "
                        "model pauses when CE > λ and emits when CE < λ.")
    p.add_argument("--gate_lambda", type=float, default=2.0,
                   help="Pause cost λ for the output gate (default 2.0). "
                        "Set relative to the model's expected CE: λ < avg_CE "
                        "→ some pausing; λ ≈ avg_CE → ~50%% pause rate; "
                        "λ > avg_CE → rarely pauses (degenerates to standard LM). "
                        "Recommended sweep: {1.0, 2.0, 3.0, 4.0} for a "
                        "codeparrot model with CE ≈ 3-4 nats.")
    p.add_argument("--gate_warmup_steps", type=int, default=2000,
                   help="Number of warmup steps over which the gate floor "
                        "linearly decays from 1.0 (always emit) to 0.0 (free "
                        "gate). Prevents early gate collapse when initial CE >> λ. "
                        "Set to 0 to disable (raw self-regulating gate). "
                        "Default: 2000 (≈25%% of a typical 8k-step run).")
    p.add_argument("--enable_thinking_token", action="store_true",
                   help="Enable discrete [THINKING] token training with an "
                        "on-policy continuation queue. A THINKING action pays "
                        "--think_lambda and requeues the original target instead "
                        "of masking it out.")
    p.add_argument("--thinking_token", type=str, default="[THINKING]",
                   help="Special token string used for internal thinking steps.")
    p.add_argument("--think_lambda", type=float, default=0.1,
                   help="Fixed ponder cost added for every emitted THINKING token.")
    p.add_argument("--think_warmup_steps", type=int, default=0,
                   help="Number of initial steps where THINKING actions are "
                        "disallowed and the model trains as a standard LM.")
    p.add_argument("--think_curriculum_steps", type=int, default=0,
                   help="After --think_warmup_steps, linearly ramp scheduled "
                        "THINKING knobs over this many optimizer steps. 0 keeps "
                        "the old fixed behavior.")
    p.add_argument("--think_explore_start_prob", type=float, default=0.0,
                   help="Exploration probability at the start of the THINKING "
                        "curriculum. It ramps to --think_explore_prob.")
    p.add_argument("--think_lambda_start", type=float, default=None,
                   help="Optional ponder cost at the start of the curriculum; "
                        "ramps to --think_lambda. If omitted, λ is constant.")
    p.add_argument("--think_policy", type=str, default="greedy",
                   choices=["greedy", "threshold", "sample"],
                   help="On-policy rule for deciding whether the model emitted "
                        "the THINKING token.")
    p.add_argument("--think_decision", type=str, default="token",
                   choices=["token", "gate"],
                   help="Decision surface for THINKING. token uses the "
                        "vocabulary [THINKING] logit; gate uses a separate "
                        "binary emit/think head and inserts [THINKING] only "
                        "into queued contexts.")
    p.add_argument("--think_gate_threshold", type=float, default=0.5,
                   help="For --think_decision gate, emit probabilities below "
                        "this threshold are treated as THINK actions.")
    p.add_argument("--think_gate_threshold_start", type=float, default=None,
                   help="Optional gate threshold at the start of the curriculum; "
                        "ramps to --think_gate_threshold. Lower values bias "
                        "against THINK early, higher final values allow more "
                        "THINK later.")
    p.add_argument("--think_threshold", type=float, default=0.5,
                   help="p([THINKING]) threshold used by --think_policy threshold.")
    p.add_argument("--think_temperature", type=float, default=1.0,
                   help="Sampling temperature used by --think_policy sample.")
    p.add_argument("--think_explore_prob", type=float, default=0.0,
                    help="Optional epsilon exploration probability that forces "
                         "eligible positions to take a THINKING action. Useful "
                         "early because a freshly-added special token may never "
                         "win greedy/threshold selection. Default 0 keeps the "
                         "policy purely model-driven.")
    p.add_argument("--think_explore_mode", type=str, default="uniform",
                   choices=["uniform", "high_ce"],
                   help="Exploration distribution. uniform samples every "
                        "eligible token equally; high_ce restricts samples to "
                        "the highest conventional-CE positions in the batch.")
    p.add_argument("--think_explore_top_frac", type=float, default=1.0,
                   help="For --think_explore_mode high_ce, only sample from "
                        "this top fraction of eligible positions by answer CE.")
    p.add_argument("--think_explore_min_ce", type=float, default=0.0,
                   help="Minimum conventional answer CE required for an "
                        "exploration candidate.")
    p.add_argument("--think_queue_max", type=int, default=262144,
                   help="Maximum number of unresolved THINKING continuations. "
                        "The queue lives in CPU memory as token-id lists; GPU "
                        "memory is controlled by --think_queue_batch.")
    p.add_argument("--think_queue_batch", type=int, default=0,
                   help="Number of queued continuations to mix into each step. "
                   "0 = one continuation microbatch (lowest peak memory).")
    p.add_argument("--think_prioritize_queue", action="store_true",
                   help="When enabled, process queued continuation obligations "
                        "first, then replay rows, and use fresh dataloader rows "
                        "only with leftover batch capacity.")
    p.add_argument("--think_max_new_per_step", type=int, default=0,
                   help="Maximum new THINKING continuations created from fresh "
                        "tokens per optimizer step. 0 disables the cap; this "
                        "is only a safety/debug backpressure knob, not part of "
                        "the model policy.")
    p.add_argument("--think_safety_max_depth", type=int, default=0,
                   help="Optional runaway safety cap. 0 disables. When a queued "
                        "item reaches this depth, THINKING is masked and an "
                        "answer is forced; frequent firing means λ/policy is "
                        "miscalibrated.")
    p.add_argument("--think_safety_max_depth_start", type=int, default=None,
                   help="Optional starting safety depth for the THINKING "
                        "curriculum; linearly ramps to "
                        "--think_safety_max_depth. Use e.g. 1 -> 5 to allow "
                        "deeper thinking only after the LM signal improves.")
    p.add_argument("--think_queue_ttl", type=int, default=0,
                   help="Optional max age in optimizer steps for queued items. "
                        "0 disables. Expired items force an answer attempt.")
    p.add_argument("--think_advantage_margin", type=float, default=0.0,
                   help="Only replay/train THINK when the resolved trajectory "
                        "beats immediate answer CE by at least this margin.")
    p.add_argument("--think_replay_batch", type=int, default=0,
                   help="Number of resolved advantage replay rows packed into "
                        "each batch. 0 = match --think_queue_batch.")
    p.add_argument("--think_checkpointing", action="store_true",
                   help="Use gradient checkpointing for the thinking continuation "
                        "passes to massively reduce memory at the cost of "
                        "extra compute.")
    p.add_argument("--think_queue_accum_steps", type=int, default=0,
                   help="Number of extra queued continuation/replay "
                        "microbatches to process with gradient accumulation "
                        "before each fresh dataloader batch. 0 keeps the old "
                        "row-packing behavior. Values >0 increase queue "
                        "processing capacity without reducing fresh LM rows.")
    p.add_argument("--think_aux_normalize", type=str, default="fresh_tokens",
                   choices=["fresh_tokens", "aux_items"],
                   help="Normalization for queued continuation/replay "
                        "microbatches processed by --think_queue_accum_steps. "
                        "fresh_tokens preserves the original behavior by "
                        "dividing each aux microbatch by batch*T; aux_items "
                        "divides by the number of queued loss items.")
    p.add_argument("--think_aux_loss_scale", type=float, default=1.0,
                   help="Multiplier applied to queued continuation/replay "
                        "microbatch losses after --think_aux_normalize.")
    p.add_argument("--think_queue_accum_max_steps", type=int, default=0,
                   help="Maximum queued microbatches per optimizer step when "
                        "--think_queue_drain_target is active. 0 means use "
                        "--think_queue_accum_steps as the fixed maximum.")
    p.add_argument("--think_queue_drain_target", type=int, default=-1,
                   help="Adaptive queue drain target. When >=0, continue "
                        "processing queued microbatches until both continuation "
                        "and replay queues are at or below this many records, "
                        "bounded by --think_queue_accum_max_steps. -1 disables "
                        "adaptive draining.")
    p.add_argument("--think_backpressure_target", type=int, default=-1,
                   help="Queue length target for dynamic THINK backpressure. "
                        "-1 reuses --think_queue_drain_target when available; "
                        "otherwise backpressure is disabled.")
    p.add_argument("--think_backpressure_max", type=float, default=4.0,
                   help="Maximum normalized queue pressure applied to dynamic "
                        "THINK backpressure.")
    p.add_argument("--think_backpressure_lambda", type=float, default=0.0,
                   help="Additional ponder cost per unit queue pressure. This "
                        "makes new/repeated THINK actions less attractive when "
                        "the continuation or replay queue is above target.")
    p.add_argument("--think_backpressure_threshold", type=float, default=0.0,
                   help="For --think_decision gate, divide the effective THINK "
                        "gate threshold by (1 + this * queue_pressure), biasing "
                        "the gate toward EMIT when backlog is high.")
    p.add_argument("--think_backpressure_explore", type=float, default=0.0,
                   help="Divide exploration probability by "
                        "(1 + this * queue_pressure), reducing artificial THINK "
                        "inflow when backlog is high.")
    p.add_argument("--think_replay_weight", type=float, default=1.0,
                   help="Multiplier for resolved advantage replay loss. This "
                        "lets sparse THINK supervision compete with dense "
                        "next-token LM loss without changing conventional CE.")
    p.add_argument("--think_gate_emit_weight", type=float, default=0.0,
                   help="For --think_decision gate, optional dense BCE weight "
                        "that trains fresh non-THINK positions to EMIT. This "
                        "keeps the binary gate calibrated while replay rows "
                        "teach the sparse THINK cases.")
    p.add_argument("--think_min_fresh_rows", type=int, default=0,
                   help="Reserve at least this many rows in each packed batch "
                        "for fresh dataloader examples. Default 0 lets "
                        "--think_prioritize_queue fully drain queued work "
                        "before adding more fresh data.")
    p.add_argument("--feedback_xattn_form", type=str, default="attn",
                   choices=["attn", "film_sum", "film_sum_mlp", "film_sum_glu",
                            "film_target_gated",
                            "film_attn", "film_sigmoid", "all_sigmoid"],
                   help="Form of the cross-layer feedback module: "
                        "'attn' = additive Q-K-V softmax residual (default); "
                        "'film_sum' = multi-source FiLM, sum (no softmax); "
                        "'film_sum_mlp' = film_sum but each W is an MLP "
                        "(Linear-GELU-Linear) — tests nonlinear expressivity; "
                        "'film_attn' = softmax routing + multiplicative FiLM "
                        "output (lets attention learn negative-α basin); "
                        "'film_sigmoid' = sigmoid (not softmax) per-source "
                        "gates + FiLM output (no 1/K dilution, per-token "
                        "routing preserved); "
                        "'all_sigmoid' = all-to-all sigmoid-gated FiLM with "
                        "shared K/W projections per source (Idea 2); pair "
                        "with --feedback_xattn 'all_above' or 'all'.")
    p.add_argument("--layers", type=str, default=None,
                   help="explicit comma-separated layer arch list, "
                        "e.g. 'ortho,deltanet,deltanet,deltanet,ortho,...'. "
                        "Overrides --arch.")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--max_T", type=int, default=0,
                   help="Max sequence length for the (optional) absolute "
                        "positional embedding. Set to >= --T (e.g. equal) "
                        "for the Transformer baseline (softmax attention "
                        "is permutation-invariant without position info). "
                        "Linear-RNN architectures (DeltaNet, Mamba2, etc.) "
                        "have implicit position via state and do not need "
                        "this. Default 0 = no positional embedding.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--d_model", type=int, default=576)
    p.add_argument("--n_heads", type=int, default=9)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "muon"],
                   help="'adamw' (default) — single AdamW for all params. "
                        "'muon' — Muon for ≥2D hidden-layer matrices, AdamW "
                        "for embeddings, lm_head, and 1D params. Typically "
                        "30-50% faster convergence per Keller Jordan / NanoGPT "
                        "speedrunning. Pair with --lr_muon ~1e-3.")
    p.add_argument("--lr_muon", type=float, default=1e-3,
                   help="Muon learning rate (used only when --optimizer muon). "
                        "AdamW LR for the remaining params is taken from --lr.")
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
    p.add_argument("--tb_dir", type=str, default=None,
                   help="Directory for TensorBoard SummaryWriter logs. "
                        "If omitted, TensorBoard logging is disabled. "
                        "Pass e.g. 'runs/exp_name' and run "
                        "'tensorboard --logdir runs' to visualize. "
                        "Scalars logged: train/loss, train/ppl, train/lr, "
                        "train/tok_per_sec, val/loss, val/ppl, "
                        "gate/think_frac, gate/mean_gate, gate/raw_ce "
                        "(gate/* only when --output_gate is set), "
                        "and alpha/* for FiLM feedback coefficients.")
    # ---- Mixed-corpus pretrain (super-coder) flags ----
    p.add_argument("--data_mix", type=str, default=None,
                   help="Path to a YAML config describing weighted multi-source "
                        "streaming. If set, overrides --dataset / --dataset_config "
                        "/ --text_field. See configs/pretrain_mix_v1.yaml.")
    p.add_argument("--use_memory", action="store_true",
                   help="Enable WorkingMemory module. Requires a thinking-token "
                        "id; auto-set when --data_mix is used.")
    p.add_argument("--mem_size", type=int, default=1024)
    p.add_argument("--mem_dim", type=int, default=0,
                   help="Memory projection dim. 0 = match d_model.")
    p.add_argument("--think_burst_prob", type=float, default=0.5,
                   help="Per-chunk probability of inserting random think-token "
                        "bursts during mixed-corpus pretrain (gives memory + "
                        "gate-head dense gradient from step 0).")
    p.add_argument("--think_max_bursts", type=int, default=2)
    p.add_argument("--think_max_burst_depth", type=int, default=6)
    p.add_argument("--mid_eval_every_tokens", type=int, default=0,
                   help="Run a HumanEval pass every N tokens of training. "
                        "0 disables. Suggested: 500_000_000.")
    p.add_argument("--mid_eval_n_problems", type=int, default=50)
    p.add_argument("--mid_eval_max_gen", type=int, default=192)
    p.add_argument("--auto_stop", action="store_true",
                   help="Stop training when HumanEval pass-rate is flat over "
                        "two consecutive mid-eval intervals (< 1pp gain each).")
    p.add_argument("--auto_stop_threshold", type=float, default=0.01)
    p.add_argument("--auto_stop_k", type=int, default=2)
    # ---- Mixed-precision / speed knobs ----
    p.add_argument("--bf16", action="store_true",
                   help="Wrap model.forward in torch.autocast(bfloat16). "
                        "Master weights stay fp32 (Muon/AdamW expect that). "
                        "Typical 1.5-2x speedup on 5090 because FLA's gated "
                        "delta-rule kernels are bf16 internally and avoid "
                        "round-trip casts. Loss + backward run normally.")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Enable TF32 for the residual fp32 matmul path. "
                        "Off by default (matches torch's 'highest' default); "
                        "set --tf32 to use TF32 — same numerics as bf16 "
                        "matmul (bf16 mantissa, fp32 exponent) but only on "
                        "matmul, so safe for training stability. Pair with "
                        "--bf16 for the full speed stack.")
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
    if args.enable_thinking_token and args.logit_kl_beta > 0:
        raise SystemExit(
            "--enable_thinking_token is not compatible with --logit_kl_beta yet "
            "because the frozen teacher checkpoint lacks the added token."
        )
    if args.enable_thinking_token and (
            args.aux_brackets or args.semantic_loss_beta > 0.0):
        raise SystemExit(
            "--enable_thinking_token currently supports the standard LM loss "
            "path only; disable aux/semantic losses for thinking experiments."
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

    # Bind alignment-loss booleans up-front (used by both data paths and the
    # model-construction block below).
    use_semantic_loss = (args.semantic_loss_beta > 0.0)
    use_logit_kl = (args.logit_kl_beta > 0.0)
    use_per_token_lsem = (use_semantic_loss
                          and args.semantic_loss_granularity == "token")
    use_stmt_lsem = (use_semantic_loss
                     and args.semantic_loss_granularity == "statement")
    needs_encoder = use_semantic_loss or use_logit_kl
    needs_ast = use_stmt_lsem

    if args.data_mix:
        print(f"Loading data mix from {args.data_mix} (streaming) ...")
        if needs_encoder or needs_ast:
            raise SystemExit(
                "--data_mix is incompatible with alignment-loss modes "
                "(semantic_loss / logit_kl / statement granularity).")
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
        )
        # Val: same sources, different seed, burst injection off so val PPL
        # reflects the clean data distribution.
        val_ds = MixedSourceStream(
            sources=sources, tokenizer=tok, block_size=args.T,
            thinking_token_id=thinking_token_id,
            think_burst_prob=0.0,
            base_seed=args.seed + 999_983,
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

        if needs_encoder and not args.encoder_ckpt:
            raise SystemExit(
                "Alignment-loss modes (--semantic_loss_beta or --logit_kl_beta) "
                "require --encoder_ckpt.")
        if (use_stmt_lsem and not args.semantic_loss_uniform_weight
                and not args.oracle_ckpt):
            raise SystemExit(
                "Statement-granularity surprise-weighted L_sem requires "
                "--oracle_ckpt (or pass --semantic_loss_uniform_weight).")
        if use_semantic_loss and use_logit_kl:
            raise SystemExit(
                "Pick at most one of --semantic_loss_beta or --logit_kl_beta.")
        if needs_ast:
            from experiments.statement_stream import (
                StatementAwareTokenisedStream, collate_with_statements,
            )
            train_ds = StatementAwareTokenisedStream(
                train_stream, tok, args.T, text_field=args.text_field,
                max_stmts_per_chunk=args.max_stmts_per_chunk,
            )
            val_ds = StatementAwareTokenisedStream(
                val_stream, tok, args.T, text_field=args.text_field,
                max_stmts_per_chunk=args.max_stmts_per_chunk,
            )
            from functools import partial
            collate_fn = partial(collate_with_statements,
                                  pad_to=args.max_stmts_per_chunk, pad_value=-1)
            train_loader = DataLoader(train_ds, batch_size=args.batch,
                                       num_workers=2, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=args.batch,
                                     num_workers=1, collate_fn=collate_fn)
        else:
            train_ds = TokenisedStream(train_stream, tok, args.T,
                                       text_field=args.text_field)
            val_ds = TokenisedStream(val_stream, tok, args.T,
                                     text_field=args.text_field)
            train_loader = DataLoader(train_ds, batch_size=args.batch,
                                       num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=1)

    # 2. Model.
    if args.layers:
        cls_list = parse_layers_arg(args.layers)
        n_layers_actual = len(cls_list)
        attn_kw = dict(attention_cls_per_layer=cls_list)
    else:
        if args.arch is None:
            raise SystemExit("specify --arch or --layers")
        attn_kw = build_arch(args.arch, args.n_layers,
                             vocab_size=model_vocab_size, n_symbols=args.n_symbols)
        n_layers_actual = args.n_layers
    aux_dim = (args.aux_max_depth + 1) if args.aux_brackets else 0
    fb_distances = tuple(int(d) for d in args.feedback_distances.split(",") if d)
    fb_pairs = ()
    if args.feedback_pairs:
        fb_pairs = tuple(
            tuple(int(x) for x in pair.split(","))
            for pair in args.feedback_pairs.split(";") if pair
        )
    # Cross-layer attention pairs.
    # 'all'  -> every layer attends over every layer above it.
    # else   -> 'tgt:src1,src2; tgt2:src3,...' explicit form.
    fb_scratchpad_pairs = ()
    if args.feedback_scratchpad:
        fb_scratchpad_pairs = tuple(
            tuple(int(x) for x in pair.split(","))
            for pair in args.feedback_scratchpad.split(";") if pair
        )
    fb_xattn_pairs = ()
    if args.feedback_xattn:
        if args.feedback_xattn.strip() == "all":
            fb_xattn_pairs = tuple(
                (tgt, tuple(s for s in range(n_layers_actual) if s != tgt))
                for tgt in range(n_layers_actual)
            )
        elif args.feedback_xattn.strip() == "all_above":
            # Every layer attends only over LATER layers (higher indices).
            # Last layer has no sources and is skipped.
            fb_xattn_pairs = tuple(
                (tgt, tuple(range(tgt + 1, n_layers_actual)))
                for tgt in range(n_layers_actual - 1)
            )
        else:
            tmp = []
            for group in args.feedback_xattn.split(";"):
                group = group.strip()
                if not group:
                    continue
                target_str, src_str = group.split(":")
                tgt = int(target_str.strip())
                srcs = tuple(int(s) for s in src_str.split(",") if s.strip())
                tmp.append((tgt, srcs))
            fb_xattn_pairs = tuple(tmp)
    mem_kwargs = {}
    if args.use_memory:
        if thinking_token_id is None:
            raise SystemExit(
                "--use_memory requires a thinking token. Set --data_mix "
                "(auto-assigns one) or --enable_thinking_token.")
        mem_kwargs = dict(
            use_memory=True,
            mem_size=int(args.mem_size),
            mem_dim=int(args.mem_dim) if args.mem_dim > 0 else int(args.d_model),
            thinking_token_id=int(thinking_token_id),
        )
    model = TinyLM(
        vocab_size=model_vocab_size, d_model=args.d_model, n_layers=n_layers_actual,
        n_heads=args.n_heads, d_head=args.d_head, aux_dim=aux_dim,
        max_T=args.max_T,
        feedback_mode=args.feedback, feedback_distances=fb_distances,
        feedback_pairs=fb_pairs,
        feedback_xattn_pairs=fb_xattn_pairs,
        feedback_xattn_heads=args.feedback_xattn_heads,
        feedback_xattn_form=args.feedback_xattn_form,
        feedback_lag=args.feedback_lag,
        feedback_position=args.feedback_position,
        feedback_per_channel_alpha=args.feedback_per_channel_alpha,
        feedback_scratchpad_pairs=fb_scratchpad_pairs,
        feedback_scratchpad_heads=args.feedback_scratchpad_heads,
        feedback_scratchpad_routing=args.feedback_scratchpad_routing,
        feedback_self_k=args.feedback_self_k,
        feedback_alpha_mode=args.feedback_alpha_mode,
        semantic_loss_dim=(args.d_model
                           if (use_semantic_loss) else 0),
        output_gate=(args.output_gate
                     or (args.enable_thinking_token
                         and args.think_decision == "gate")),
        **mem_kwargs,
        **attn_kw,
    ).to("cuda")
    # ---- Speed knobs (must run AFTER model is built but BEFORE the train
    # loop touches it).
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        print("TF32 enabled for fp32 matmul (high precision mode)")
    if args.bf16:
        # Wrap model.forward in bf16 autocast. Master weights stay fp32
        # (Muon + AdamW expect that); only forward+intermediates run in
        # bf16. Backward runs through the same autocast graph
        # (PyTorch handles the casts).
        _orig_model_forward = model.forward
        def _bf16_forward(*fwd_args, **fwd_kwargs):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return _orig_model_forward(*fwd_args, **fwd_kwargs)
        model.forward = _bf16_forward
        print("bf16 autocast wrapping model.forward")
    if args.freeze_alpha and (args.feedback != "none" or fb_xattn_pairs):
        model.freeze_alpha()
    if fb_xattn_pairs:
        n_total_pairs = sum(len(srcs) for _, srcs in fb_xattn_pairs)
        feedback_desc = (f"xattn[{args.feedback_xattn_form}]"
                         f"(targets={len(fb_xattn_pairs)},"
                         f"src-edges={n_total_pairs},"
                         f"heads={args.feedback_xattn_heads})")
    else:
        feedback_desc = f"{args.feedback}@d={args.feedback_distances}"
    print(f"  params: {model.num_params() / 1e6:.1f}M  aux_dim={aux_dim}  "
          f"feedback={feedback_desc}"
          f"{' (α frozen=0)' if args.freeze_alpha else ''}")

    if args.aux_brackets:
        print("Computing bracket-deltas table for tokenizer ...")
        bracket_deltas = compute_bracket_deltas(tok)
        print(f"  table shape: {bracket_deltas.shape}, "
              f"non-zero count: {(bracket_deltas != 0).sum().item()}")
    else:
        bracket_deltas = None

    # Phase 22 / structural-surprise full PoC: load frozen encoder and
    # oracle for L_sem (and Phase 22b ablations: KL-on-logits / per-token L_sem).
    encoder = None
    oracle_head = None
    if needs_encoder:
        from experiments.eval_statement_ppl import load_model as load_lm_ckpt
        print(f"Loading frozen encoder: {args.encoder_ckpt}")
        encoder = load_lm_ckpt(args.encoder_ckpt, device="cuda")
        for q in encoder.parameters():
            q.requires_grad_(False)
        encoder.eval()
        if (use_stmt_lsem and args.oracle_ckpt
                and not args.semantic_loss_uniform_weight):
            from experiments.oracle_train import OraclePredictor
            print(f"Loading oracle predictive head: {args.oracle_ckpt}")
            oc = torch.load(args.oracle_ckpt, map_location="cuda",
                            weights_only=False)
            oracle_head = OraclePredictor(**oc["config"]).to("cuda")
            oracle_head.load_state_dict(oc["state_dict"])
            oracle_head.eval()
            for q in oracle_head.parameters():
                q.requires_grad_(False)
            oh_params = sum(p.numel() for p in oracle_head.parameters())/1e6
            oh_str = f", oracle: {oh_params:.2f}M params"
        else:
            oracle_head = None
            oh_str = " (no oracle: uniform/per-token/KL mode)"
        print(f"  encoder: {encoder.num_params()/1e6:.1f}M params" + oh_str)
        if use_stmt_lsem:
            print(f"  semantic_loss_beta = {args.semantic_loss_beta} "
                  f"(statement-granularity"
                  f"{', uniform-weight' if args.semantic_loss_uniform_weight else ', surprise-weighted'})")
        elif use_per_token_lsem:
            print(f"  semantic_loss_beta = {args.semantic_loss_beta} "
                  f"(token-granularity per-position cosine alignment)")
        elif use_logit_kl:
            print(f"  logit_kl_beta = {args.logit_kl_beta} "
                  f"(temperature T = {args.logit_kl_temp})")

    # Optimizer construction — single AdamW (default) or Muon + AdamW split.
    if args.optimizer == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)]
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(
            opts[0], T_max=args.steps, eta_min=args.lr * 0.1)]
    else:  # muon
        # Split params: ≥2D hidden-layer matrices → Muon; embeddings, lm_head,
        # and 1D params (alphas, RMSNorm scales) → AdamW.
        embed_or_head_names = {"embed.weight", "pos_embed.weight", "lm_head.weight"}
        muon_params, adamw_params = [], []
        seen = set()
        for name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            # Muon supports strictly 2D matrices. Embeddings/lm_head + 1D
            # params + 3D+ tensors (short_conv kernels, etc.) go to AdamW.
            if name in embed_or_head_names or p.ndim != 2:
                adamw_params.append(p)
            else:
                muon_params.append(p)
        print(f"  optimizer split: {len(muon_params)} Muon params "
              f"({sum(p.numel() for p in muon_params)/1e6:.1f}M), "
              f"{len(adamw_params)} AdamW params "
              f"({sum(p.numel() for p in adamw_params)/1e6:.1f}M)")
        opts = [
            torch.optim.Muon(muon_params, lr=args.lr_muon,
                             momentum=0.95, weight_decay=0.1),
            torch.optim.AdamW(adamw_params, lr=args.lr,
                              betas=(0.9, 0.95), weight_decay=0.1),
        ]
        scheds = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opts[0], T_max=args.steps, eta_min=args.lr_muon * 0.1),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opts[1], T_max=args.steps, eta_min=args.lr * 0.1),
        ]
    # Backwards-compat aliases used elsewhere in the loop.
    opt = opts[0]
    scheduler = scheds[0]

    # 3. Train loop.
    if use_semantic_loss:
        unit = "nstmt" if use_stmt_lsem else "ntok"
        print(f"\n{'step':>6}  {'tok/s':>8}  {'tloss':>8}  {'ce':>8}  "
              f"{'lsem':>8}  {unit:>6}  {'lr':>9}")
    elif use_logit_kl:
        print(f"\n{'step':>6}  {'tok/s':>8}  {'tloss':>8}  {'ce':>8}  "
              f"{'lkl':>8}  {'lr':>9}")
    else:
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
    losses_sem_window: list[float] = []
    losses_ce_window: list[float] = []
    losses_nstmt_window: list[int] = []
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

    def compute_semantic_loss(model, x_batch, stmt_starts, stmt_ends,
                                hidden, encoder, oracle_head, beta):
        """Compute L_sem(s_t) summed over all valid statements in the batch.

        - hidden: (B, T, d) — model's final-layer hidden (out_norm output).
        - stmt_starts, stmt_ends: (B, S_max) padded with -1 for unused slots.
        - encoder: frozen DN baseline.
        - oracle_head: frozen predictive head.
        - beta: scalar weight.

        Returns:
          (l_sem_total: scalar tensor, n_stmts_total: int).
        """
        device = hidden.device
        d_model = hidden.shape[-1]
        # 1. Frozen encoder forward to get target embeddings.
        with torch.no_grad():
            _, enc_hidden = encoder(x_batch, return_hidden=True)
        B = x_batch.shape[0]
        # 2. Pool encoder hidden per statement → get target embedding seq E(s_t) per row.
        # 3. Pool model hidden per statement → get h_t. Project via W.
        # 4. Run oracle head on E sequence to get predicted next-statement embedding.
        #    Surprise(t) = 1 - cos(P(prefix), E(s_t)).
        # We process row-by-row to handle variable S per row cleanly.
        l_sem_terms: list[torch.Tensor] = []
        n_stmts_total = 0
        for b in range(B):
            ss = stmt_starts[b]
            se = stmt_ends[b]
            valid_mask = (ss >= 0)
            if not valid_mask.any():
                continue
            ss_v = ss[valid_mask].tolist()
            se_v = se[valid_mask].tolist()
            S = len(ss_v)
            # Pool encoder hidden → E(s_t) for each statement.
            with torch.no_grad():
                enc_pool = []
                for st, en in zip(ss_v, se_v):
                    enc_pool.append(enc_hidden[b, st:en].mean(dim=0))
                E_seq = torch.stack(enc_pool, dim=0)            # (S, d)
                # Oracle prediction sequence: P(prefix [0..t-1]) → predict E(s_t).
                # For statement t, surprise = 1 - cos(preds[t-1], E_seq[t]).
                # The oracle head expects (B, S, d).
                if oracle_head is not None:
                    preds = oracle_head(E_seq.unsqueeze(0))        # (1, S, d)
                    surp = torch.full((S,), 0.0, device=device,
                                       dtype=E_seq.dtype)
                    if S > 1:
                        cos = F.cosine_similarity(
                            preds[0, :-1], E_seq[1:], dim=-1,
                        )                                           # (S-1,)
                        surp[1:] = 1.0 - cos
                else:
                    # Uniform-weight path discards surp; keep dummy zeros.
                    surp = torch.zeros((S,), device=device, dtype=E_seq.dtype)
                # Per-batch z-score normalisation. We z-score across the
                # whole batch's statements (computed at the end of this
                # function before applying to L_sem, so we accumulate the
                # raw surprise here and z-score later).
            # Pool model hidden over statement → h_t. With grad.
            mod_pool = []
            for st, en in zip(ss_v, se_v):
                mod_pool.append(hidden[b, st:en].mean(dim=0))
            h_t = torch.stack(mod_pool, dim=0)                    # (S, d_model)
            h_tilde = model.W_semantic(h_t)                       # (S, d_oracle)
            # Cosine sim against E_seq (frozen).
            cos_h = F.cosine_similarity(h_tilde, E_seq.detach(), dim=-1)  # (S,)
            # L_sem per statement: surp * (1 - cos_h). Surp is detached.
            # We handle z-scoring per BATCH (across all rows) — accumulate
            # raw surprise + per-stmt loss tensors and combine outside the loop.
            l_sem_terms.append((surp.detach(), 1.0 - cos_h, S))
            n_stmts_total += S
        if n_stmts_total == 0:
            return torch.zeros((), device=device), 0
        # Concatenate all rows' (surp, loss_per_stmt) arrays.
        all_surp = torch.cat([t[0] for t in l_sem_terms])         # (N,)
        all_dist = torch.cat([t[1] for t in l_sem_terms])         # (N,)
        # Per-batch normalisation so β has portable meaning across batches.
        # We rescale raw surprise (already in [0, 2] from cosine distance) so
        # the batch mean of the weight is 1.0 — i.e., β controls the average
        # relative weight of L_sem vs L_ce. We do NOT center: a centered
        # z-score would produce negative weights for low-surprise statements,
        # which would invert the loss direction at routine code (push the
        # model AWAY from the oracle's representation). Rescaling without
        # centering preserves the design's intent — high-surprise = bigger
        # weight, low-surprise = smaller — while keeping all weights ≥ 0.
        if args.semantic_loss_uniform_weight:
            # Ablation: ignore oracle surprise; treat every statement equally.
            l_sem_total = all_dist.mean()
        else:
            mean_surp = all_surp.mean().clamp(min=1e-6)
            weights = all_surp / mean_surp                         # (N,) ≥ 0
            l_sem_total = (weights * all_dist).mean()
        return l_sem_total, n_stmts_total

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

    needs_hidden = use_semantic_loss              # both stmt + token L_sem need hidden
    needs_encoder_hidden = use_semantic_loss      # we pool encoder's hidden for cosine target
    needs_encoder_logits = use_logit_kl           # KL ablation needs encoder logits
    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if needs_ast:
            x, y, stmt_starts, stmt_ends = batch
            x, y = x.to("cuda"), y.to("cuda")
            stmt_starts = stmt_starts.to("cuda")
            stmt_ends = stmt_ends.to("cuda")
        else:
            x, y = batch
            x, y = x.to("cuda"), y.to("cuda")
            stmt_starts = None
            stmt_ends = None
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
        want_surprise = (args.feedback == "predictive" and args.surprise_weight > 0)
        if needs_hidden:
            # Need hidden for L_sem (statement or token granularity).
            if args.aux_brackets and want_surprise:
                logits, aux_logits, surprise, hidden = model(
                    x, return_aux=True, return_surprise=True, return_hidden=True
                )
            elif args.aux_brackets:
                logits, aux_logits, hidden = model(
                    x, return_aux=True, return_hidden=True
                )
                surprise = torch.zeros((), device="cuda")
            elif want_surprise:
                logits, surprise, hidden = model(
                    x, return_surprise=True, return_hidden=True
                )
                aux_logits = None
            else:
                logits, hidden = model(x, return_hidden=True)
                aux_logits = None
                surprise = torch.zeros((), device="cuda")
        else:
            if args.aux_brackets and want_surprise:
                logits, aux_logits, surprise = model(x, return_aux=True, return_surprise=True)
            elif args.aux_brackets:
                logits, aux_logits = model(x, return_aux=True)
                surprise = torch.zeros((), device="cuda")
            elif want_surprise:
                logits, surprise = model(x, return_surprise=True)
                aux_logits = None
            else:
                logits = model(x)
                aux_logits = None
                surprise = torch.zeros((), device="cuda")
            hidden = None
        if args.aux_brackets:
            depth = bracket_depth(x, bracket_deltas).clamp(0, args.aux_max_depth)
            aux_loss = F.cross_entropy(
                aux_logits.reshape(-1, args.aux_max_depth + 1),
                depth.reshape(-1),
            )
        else:
            aux_loss = torch.zeros((), device="cuda")
        # Gated loss (Phase 23): L = mean(g_t * CE_t + (1-g_t) * λ).
        # g_t is stored in model._last_gate by the forward pass (side effect).
        # When output_gate is off, fall back to standard mean CE.
        V = logits.shape[-1]
        ce_per_token = F.cross_entropy(
            logits.reshape(-1, V), y.reshape(-1),
            reduction="none",
        ).reshape(y.shape)                                               # (B, T)
        if args.enable_thinking_token:
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
        elif args.output_gate:
            g = model._last_gate                                         # (B, T)
            # Warmup floor: clamp gate from below, decaying 1→0 over
            # gate_warmup_steps. Prevents early collapse when initial CE >> λ.
            if args.gate_warmup_steps > 0:
                gate_floor = max(0.0, 1.0 - step / args.gate_warmup_steps)
                g_eff = g.clamp(min=gate_floor)
            else:
                g_eff = g
            gate_terms = g_eff * ce_per_token + (1.0 - g_eff) * args.gate_lambda
            # Mask positions where the target is -100 (e.g. positions where
            # the model is predicting a think token under mixed-corpus burst
            # injection — no loss should accrue there).
            valid = (y != -100).float()
            denom = valid.sum().clamp(min=1.0)
            lm_loss = (gate_terms * valid).sum() / denom
        else:
            valid = (y != -100).float()
            denom = valid.sum().clamp(min=1.0)
            lm_loss = (ce_per_token * valid).sum() / denom
        # Compute the alignment-loss term (one of three modes).
        l_sem = torch.zeros((), device="cuda")
        l_kl = torch.zeros((), device="cuda")
        n_stmts = 0
        n_tokens_aligned = 0
        if use_stmt_lsem:
            l_sem, n_stmts = compute_semantic_loss(
                model, x, stmt_starts, stmt_ends, hidden,
                encoder, oracle_head, args.semantic_loss_beta,
            )
        elif use_per_token_lsem:
            # Phase 22b ablation 2: per-token cosine alignment.
            with torch.no_grad():
                _, enc_hidden = encoder(x, return_hidden=True)
            h_tilde = model.W_semantic(hidden)                       # (B, T, d)
            cos_h = F.cosine_similarity(
                h_tilde, enc_hidden.detach(), dim=-1,
            )                                                          # (B, T)
            l_sem = (1.0 - cos_h).mean()
            n_tokens_aligned = int(cos_h.numel())
        elif use_logit_kl:
            # Phase 22b ablation 3: KL divergence on logits between student
            # and frozen-encoder teacher. Both detached on the teacher side.
            with torch.no_grad():
                enc_logits = encoder(x)                                # (B, T, V)
            T = float(args.logit_kl_temp)
            log_p = F.log_softmax(logits / T, dim=-1)                  # student
            log_q = F.log_softmax(enc_logits.detach() / T, dim=-1)     # teacher
            # KL(student || teacher) per textbook KD: mean over (B*T).
            # We use F.kl_div with `log_target=True`, which expects:
            #   input  = log_p (student), target = log_q (teacher).
            # The sign convention there is KL(target || input), so feed
            # teacher as `target` and student as `input` — KL(teacher || student),
            # the canonical Hinton-KD form.
            kl_pp = F.kl_div(log_p, log_q, log_target=True,
                              reduction="none").sum(dim=-1)           # (B, T)
            l_kl = (T * T) * kl_pp.mean()
            n_tokens_aligned = int(kl_pp.numel())
        loss = (lm_loss
                + args.aux_weight * aux_loss
                + args.surprise_weight * surprise
                + args.semantic_loss_beta * l_sem
                + args.logit_kl_beta * l_kl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
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
        if use_semantic_loss:
            losses_sem_window.append(float(l_sem.item()))
            losses_ce_window.append(float(lm_loss.item()))
            losses_nstmt_window.append(int(n_stmts) if use_stmt_lsem
                                        else int(n_tokens_aligned))
        elif use_logit_kl:
            losses_sem_window.append(float(l_kl.item()))
            losses_ce_window.append(float(lm_loss.item()))

        if step % args.log_every == 0 or step == args.steps:
            now = time.perf_counter()
            tok_per_sec = (step - last_log_step) * args.batch * args.T / (now - last_log)
            tloss_avg = sum(losses[-args.log_every:]) / max(1, len(losses[-args.log_every:]))
            if use_semantic_loss:
                ce_avg = sum(losses_ce_window[-args.log_every:]) / max(1, len(losses_ce_window[-args.log_every:]))
                lsem_avg = sum(losses_sem_window[-args.log_every:]) / max(1, len(losses_sem_window[-args.log_every:]))
                nstmt_avg = sum(losses_nstmt_window[-args.log_every:]) / max(1, len(losses_nstmt_window[-args.log_every:]))
                line = (f"{step:>6d}  {tok_per_sec:>8.0f}  "
                        f"{tloss_avg:>8.4f}  {ce_avg:>8.4f}  "
                        f"{lsem_avg:>8.4f}  {nstmt_avg:>6.1f}  "
                        f"{scheduler.get_last_lr()[0]:>9.2e}")
            elif use_logit_kl:
                ce_avg = sum(losses_ce_window[-args.log_every:]) / max(1, len(losses_ce_window[-args.log_every:]))
                lkl_avg = sum(losses_sem_window[-args.log_every:]) / max(1, len(losses_sem_window[-args.log_every:]))
                line = (f"{step:>6d}  {tok_per_sec:>8.0f}  "
                        f"{tloss_avg:>8.4f}  {ce_avg:>8.4f}  "
                        f"{lkl_avg:>8.4f}  "
                        f"{scheduler.get_last_lr()[0]:>9.2e}")
            else:
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
                gf = (f",floor={max(0.0, 1.0 - step/args.gate_warmup_steps):.2f}"
                      if args.gate_warmup_steps > 0 else "")
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
            print(line)
            if tb is not None:
                tb.add_scalar("train/loss", tloss_avg, step)
                tb.add_scalar("train/ppl", float(torch.tensor(tloss_avg).exp()), step)
                tb.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                tb.add_scalar("train/tok_per_sec", tok_per_sec, step)
                if use_semantic_loss:
                    tb.add_scalar("train/ce", ce_avg, step)
                    tb.add_scalar("train/l_sem", lsem_avg, step)
                elif use_logit_kl:
                    tb.add_scalar("train/ce", ce_avg, step)
                    tb.add_scalar("train/l_kl", lkl_avg, step)
                if args.output_gate and losses_gate_window:
                    tb.add_scalar("gate/think_frac", 1.0 - emit_frac, step)
                    tb.add_scalar("gate/mean_gate", mean_g, step)
                    tb.add_scalar("gate/raw_ce", raw_ce, step)
                    if emit_ce == emit_ce:  # not NaN
                        tb.add_scalar("gate/emit_ce", emit_ce, step)
                    if args.gate_warmup_steps > 0:
                        gate_floor = max(0.0, 1.0 - step / args.gate_warmup_steps)
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
                    if needs_ast:
                        vx, vy, *_ = vbatch
                    else:
                        vx, vy = vbatch
                    vx, vy = vx.to("cuda"), vy.to("cuda")
                    vlogits = model(vx)
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

        # ---- Mid-training HumanEval hook (auto_stop). ----
        # tokens_seen is the rough count of tokens consumed by the train
        # loop so far. At every `mid_eval_every_tokens` boundary, save a
        # ckpt, shell out to eval_humaneval.py, log pass-rate to TB,
        # append to controller, optionally stop.
        tokens_seen = step * args.batch * args.T
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
                feedback_distances=fb_distances,
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
            )
            model.train()
            mid_eval_controller.append(res)
            print(f"[mid-eval] {mid_eval_controller.summary_line()}")
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
                "feedback_distances": fb_distances,
                "feedback_pairs": fb_pairs,
                "feedback_xattn_pairs": fb_xattn_pairs,
                "feedback_xattn_heads": args.feedback_xattn_heads,
                "feedback_xattn_form": args.feedback_xattn_form,
                "feedback_self_k": args.feedback_self_k,
                "feedback_alpha_mode": args.feedback_alpha_mode,
                "semantic_loss_dim": (args.d_model
                                       if use_semantic_loss else 0),
                "semantic_loss_beta": args.semantic_loss_beta,
                "semantic_loss_uniform_weight":
                    bool(args.semantic_loss_uniform_weight),
                "semantic_loss_granularity": args.semantic_loss_granularity,
                "logit_kl_beta": args.logit_kl_beta,
                "logit_kl_temp": args.logit_kl_temp,
                "encoder_ckpt": args.encoder_ckpt,
                "oracle_ckpt": args.oracle_ckpt,
                "arch": args.arch, "layers_spec": args.layers,
                "n_symbols": args.n_symbols,
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
