"""CLI argument parser for experiments/train_lm.py.

Extracted into its own module so `train_lm.py`'s main loop is readable
without scrolling past ~400 lines of `add_argument` calls. The parser
is also reusable from other tooling (e.g. config-validation scripts that
want to introspect available flags).
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default=None,
                   choices=["deltanet", "transformer", "mamba2"])
    p.add_argument("--aux_brackets", action="store_true",
                   help="Add bracket-depth auxiliary loss (direction E).")
    p.add_argument("--aux_weight", type=float, default=0.1,
                   help="Weight on the aux loss term.")
    p.add_argument("--aux_max_depth", type=int, default=24,
                   help="Cap bracket depth at this value for the aux head.")
    p.add_argument("--feedback", type=str, default="none",
                   choices=["none", "additive", "film"],
                   help="Cross-layer top-down feedback mode.")
    p.add_argument("--save_ckpt", type=str, default=None,
                   help="Path to save final model checkpoint (for downstream "
                        "evals like HumanEval, bracket-structure, long-T PPL).")
    p.add_argument("--load_ckpt", type=str, default=None,
                   help="Resume training from a saved ckpt's state_dict. "
                        "Loads only model weights — optimizer / RNG state are "
                        "fresh (brief loss-spike transient expected). Pair with "
                        "--start_step to keep the LR scheduler and gate-floor "
                        "curriculum aligned with the checkpoint's training "
                        "progress.")
    p.add_argument("--start_step", type=int, default=0,
                   help="Override the training step counter (use with "
                        "--load_ckpt). Drives LR schedule, gate-floor "
                        "curriculum, and mid-eval tokens_seen budget so they "
                        "pick up where the ckpt left off.")
    p.add_argument("--feedback_pairs", type=str, default="",
                   help="Sparse (target,source) feedback connections; "
                        "semicolon-separated. E.g. '2,28;3,27' means "
                        "layer 2's input gets feedback from layer 28, "
                        "and layer 3's input gets feedback from layer 27.")
    p.add_argument("--feedback_xattn", type=str, default="",
                   help="Cross-layer attention feedback. Each target attends "
                        "over multiple source layers' lagged hidden states. "
                        "Syntax 'target:src1,src2,...; target2:src3,src4,...'. "
                        "Example '2:14,21,28' = layer 2's input attends over "
                        "layers 14, 21, 28. 'all' = every layer attends over "
                        "every later layer (Idea 2). If non-empty, overrides "
                        "--feedback_pairs and ignores --feedback (attention "
                        "is its own form).")
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
    p.add_argument("--feedback_self_k_warmup_steps", type=int, default=0,
                   help="Steps to run with FiLM bypassed (plain 1-pass "
                        "forward) before enabling the configured "
                        "--feedback_self_k. Default 0 (off).")
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
                        "linearly decays from 1.0 (always emit) toward "
                        "--gate_floor_min. Prevents early gate collapse when "
                        "initial CE >> λ. Set to 0 to disable (gate clamped "
                        "at --gate_floor_min from step 0).")
    p.add_argument("--gate_floor_min", type=float, default=0.0,
                   help="Asymptotic minimum value of the gate-floor clamp. "
                        "Default 0.0 (free gate after warmup). Set to e.g. "
                        "0.5 to prevent the maladaptive-thinking trap where "
                        "the model collapses to predict the think token "
                        "everywhere (driving down gated train loss while "
                        "real-token VAL ppl explodes). Empirically, v2 "
                        "attempt 2 collapsed to VAL ppl 940 (vs 49 at "
                        "step 2000) within 2k steps of the floor hitting 0.")
    p.add_argument("--gate_entropy_aux_weight", type=float, default=0.0,
                   help="Weight on an auxiliary BCE loss that supervises the "
                        "output-gate logit with a predictive-uncertainty target "
                        "derived from the SAME forward's logits (detached): "
                        "  target_t = exp(-H_t),  H_t = entropy of next-token p. "
                        "Confident position → target≈1 → gate trained to emit. "
                        "Uncertain position → target≈0 → gate trained to think. "
                        "Free signal — no second forward, just turns the "
                        "existing gate logit into a position-grounded "
                        "uncertainty head. Default 0.0 (off). Recommended 0.1 "
                        "as a starting point; the loss term is in nats, same "
                        "scale as CE, so 0.1 weights it about 1/10 of LM loss.")
    p.add_argument("--gate_entropy_aux_temperature", type=float, default=1.0,
                   help="Temperature for the entropy target. The raw target "
                        "exp(-H) is mostly tiny (typical CE 1-3 nats → exp(-H) "
                        "∈ (0.05, 0.37)). With T>1 we apply exp(-H/T) so the "
                        "target distribution is less compressed near 0. T=1.0 "
                        "is the unbiased entropy. T=2-4 broadens the gradient "
                        "signal at uncertain positions.")
    p.add_argument("--gate_entropy_aux_target_clamp", type=float, default=0.0,
                   help="If > 0, clip the entropy target into "
                        "[gate_entropy_aux_target_clamp, "
                        "1-gate_entropy_aux_target_clamp] before BCE. Default 0 "
                        "(no clip). Use 0.01 to prevent BCE blow-up at target=0 "
                        "(rare in practice — exp(-H) > 0 always — but useful "
                        "if the model emits a very-low-entropy collapse).")
    # --- Gate-calibration auxiliary loss (sibling of process_reward) -------
    # At sampled positions where σ ∈ [min, max] (uncertain gate), run an
    # extra forward over [prefix, K * THINK] and ask whether the K-think
    # prediction is more accurate than the no-think prediction. Use that
    # 0/1 label as a BCE target on σ. Trains the gate to PREDICT
    # whether thinking will help — symmetric (penalises both "fire but
    # didn't help" and "didn't fire but would have helped"). Default
    # 0.0 = off → byte-identical training.
    p.add_argument("--gate_calibration_weight", type=float, default=0.0,
                   help="Weight on the gate-calibration aux loss. 0 "
                        "disables (default). Try 0.1.")
    p.add_argument("--gate_calibration_K", type=int, default=4,
                   help="Number of think tokens to insert before the "
                        "sampled position for the 'think' forward.")
    p.add_argument("--gate_calibration_apply_min_sigma", type=float,
                   default=0.1,
                   help="Lower σ bound for candidate positions. Skips "
                        "positions where the gate is already saturated "
                        "low (σ ≈ 0) — calibrating those is wasted "
                        "compute.")
    p.add_argument("--gate_calibration_apply_max_sigma", type=float,
                   default=0.9,
                   help="Upper σ bound for candidate positions. Skips "
                        "positions where the gate is already saturated "
                        "high (σ ≈ 1).")
    p.add_argument("--gate_calibration_sample_frac", type=float, default=0.05,
                   help="Fraction of qualifying positions to evaluate "
                        "per batch (bounds compute).")
    p.add_argument("--gate_calibration_max_positions", type=int, default=32,
                   help="Hard cap on sampled positions per batch — extra "
                        "guard so the after-forward stays bounded even "
                        "on a huge batch.")
    p.add_argument("--gate_calibration_smooth_target_scale", type=float,
                   default=0.0,
                   help="If > 0, use a smooth target σ((logp_think - "
                        "logp_no_think) * scale) instead of the hard "
                        "{0, 1} indicator. 0 = hard target (default).")
    # --- Process-reward auxiliary loss (Phase A of THINKING_PLAN v5) -------
    # At sampled positions where σ > min_sigma (gate wants to think), run
    # an extra forward over [prefix, K * THINK_ID] and minimise
    # (log p_before - log p_after) on the true next token. Pushes thinks
    # to be productive. Default 0.0 = off → byte-identical training.
    p.add_argument("--process_reward_weight", type=float, default=0.0,
                   help="Weight on the process-reward aux loss. 0 "
                        "disables (default). Try 0.05.")
    p.add_argument("--process_reward_K", type=int, default=4,
                   help="Number of [THINKING] tokens to insert before the "
                        "sampled position for the 'after' forward.")
    p.add_argument("--process_reward_apply_min_sigma", type=float,
                   default=0.3,
                   help="Only consider positions whose post-sigmoid gate "
                        "value σ exceeds this threshold. Restricts the "
                        "aux loss to positions the gate already wants "
                        "to think on.")
    p.add_argument("--process_reward_sample_frac", type=float, default=0.05,
                   help="Fraction of qualifying positions to evaluate "
                        "per batch (bounds compute).")
    p.add_argument("--process_reward_max_positions", type=int, default=32,
                   help="Hard cap on sampled positions per batch — extra "
                        "guard so the after-forward stays bounded even "
                        "on a huge batch.")
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
    p.add_argument("--activation_checkpointing", action="store_true",
                   help="Wrap each transformer Block's loss-bearing forward "
                        "in torch.utils.checkpoint. Cuts stored activations "
                        "by ~Nlayers× at the cost of ~30% extra compute "
                        "per step. Lets us push batch significantly higher "
                        "at the same memory; net throughput usually wins.")
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
    p.add_argument("--lr", type=float, default=1.4e-3,
                   help="AdamW peak learning rate. Default 1.4e-3 — the "
                        "sqrt-batch-scaled v4 value (validated in the "
                        "batch-20 / lr_muon-5e-3 hi-LR smoke; ~4× faster "
                        "VAL-ppl convergence vs the 5e-4 / 1.5e-3 baseline).")
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "muon"],
                   help="'adamw' (default) — single AdamW for all params. "
                        "'muon' — Muon for ≥2D hidden-layer matrices, AdamW "
                        "for embeddings, lm_head, and 1D params. Typically "
                        "30-50% faster convergence per Keller Jordan / NanoGPT "
                        "speedrunning. Pair with --lr_muon ~1.5e-3.")
    p.add_argument("--lr_muon", type=float, default=5e-3,
                   help="Muon learning rate (used only when --optimizer muon). "
                        "Default 5e-3 — the sqrt-batch-scaled v4 value (was "
                        "1e-3 at v3-long's batch 7). Muon's spectrally-"
                        "normalised update is tolerant of high LR (Moonlight "
                        "ran at 6e-3 at 8M-token batches). AdamW LR for the "
                        "remaining params is taken from --lr.")
    p.add_argument("--lr_schedule", type=str, default="wsd",
                   choices=["cosine", "wsd"],
                   help="LR schedule. 'wsd' (default) = warmup-stable-decay: "
                        "constant peak LR for the bulk of training, short "
                        "decay over the last --lr_decay_frac. No wasted "
                        "low-LR tail, and the run can be stopped anywhere. "
                        "'cosine' = legacy cosine anneal to 0.1*peak.")
    p.add_argument("--warmup_steps", type=int, default=2000,
                   help="Linear LR warmup steps. Used by --lr_schedule wsd.")
    p.add_argument("--lr_decay_frac", type=float, default=0.15,
                   help="Fraction of total steps for the WSD decay phase "
                        "(cosine 1->0). Used by --lr_schedule wsd.")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--probe_humaneval_every_tokens", type=int, default=0,
                   help="Run the small in-process HumanEval probe every N "
                        "tokens (recommended: 1_000_000_000 = 1B tokens). "
                        "0 disables. Probe set at --probe_humaneval_path. "
                        "Tracks actual code-completion capability during "
                        "pretrain instead of flying blind on LM ppl.")
    p.add_argument("--probe_humaneval_path", type=str,
                   default="data/probe_humaneval_50.jsonl",
                   help="Held-out probe JSONL (built by "
                        "experiments/build_probe_dataset.py).")
    p.add_argument("--probe_humaneval_max_gen", type=int, default=192,
                   help="Max generated tokens per probe problem.")
    p.add_argument("--probe_humaneval_n_problems", type=int, default=0,
                   help="Limit probe to first N problems (0 = whole file).")
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
                        "If omitted, TensorBoard logging is disabled.")
    # ---- Mixed-corpus pretrain (super-coder) flags ----
    p.add_argument("--data_mix", type=str, default=None,
                   help="Path to a YAML config describing weighted multi-source "
                        "streaming. If set, overrides --dataset / --dataset_config "
                        "/ --text_field. See configs/pretrain_mix_v1.yaml.")
    p.add_argument("--use_memory", action="store_true",
                   help="Enable WorkingMemory module. Requires a thinking-token "
                        "id; auto-set when --data_mix is used.")
    p.add_argument("--mem_size", type=int, default=1024)
    p.add_argument("--mem_write_only_at_think", action="store_true",
                   help="FIX A (2026-05-18): force the WorkingMemory buffer "
                        "to be filled ONLY from think-position writes. "
                        "Diagnostic showed the write gate fires uniformly "
                        "across think/emit, so reads (which ARE sharp at "
                        "think positions) hit a buffer of noise. This "
                        "mechanically forces think-content into the buffer.")
    p.add_argument("--state_readonly_at_think", action="store_true",
                   help="Phase 2 thinking fix (2026-05-26): force the "
                        "DeltaNet per-token write gate β to 0 at think "
                        "positions, so think tokens can READ the recurrent "
                        "state but never WRITE to it. Preserves long-range "
                        "bindings across multi-think bursts (the documented "
                        "100% -> 20% recall-at-512 drop). Only applies to "
                        "plain DeltaNet attention blocks; other variants "
                        "silently skip. Default OFF (every existing ckpt "
                        "behaves byte-identically without the flag).")
    p.add_argument("--think_index_emb_size", type=int, default=0,
                   help="Phase 3 thinking fix (2026-05-26): per-position "
                        "think-index embedding table size. >0 enables a "
                        "small nn.Embedding(N, d_model), zero-init, added "
                        "to each think token according to its index within "
                        "its consecutive-think burst (0, 1, ..., N-1; "
                        "bursts longer than N share the last embedding). "
                        "Breaks the homogenization (8 consecutive thinks -> "
                        "8 identical inputs -> low-rank hidden states; "
                        "median pairwise cosine +0.146 vs +0.060 at emit). "
                        "0 = disabled (default, byte-identical).")
    p.add_argument("--use_think_adapter", action="store_true",
                   help="Phase B thinking-specialized adapter (2026-05-26): "
                        "every Block grows a small 2-layer MLP "
                        "(d_model -> hidden_mult*d_model -> d_model, GELU) "
                        "plus a learnable scalar alpha (init 0). At think "
                        "positions, the adapter contribution is added to "
                        "the residual stream after the standard attn + MLP: "
                        "h += alpha * think_mask * ThinkAdapter(h). With "
                        "alpha=0 a cold start is byte-identical to no "
                        "adapter; existing ckpts load with strict=False. "
                        "Gives the trunk dedicated capacity for "
                        "think-time-specialized computation (today the "
                        "trunk runs IDENTICAL ops at think vs emit; only "
                        "the input embedding differs).")
    p.add_argument("--think_adapter_hidden_mult", type=int, default=2,
                   help="Phase B think-adapter hidden width multiplier "
                        "(default 2 -> d_hidden = 2 * d_model). Only used "
                        "when --use_think_adapter is set.")
    # ----- Phase D: RefinementHead (THINKING_PLAN v5, 2026-05-27) -----
    p.add_argument("--use_refinement_head", action="store_true",
                   help="Phase D — add a RefinementHead: 1 layer of "
                        "causal local-window self-attention (n_heads, "
                        "window=N) + 2-layer MLP, alpha-gated (init 0). "
                        "Output is soft-mixed with the trunk hidden by "
                        "sigma(gate): h_final = sigma*h + (1-sigma)*h_refined. "
                        "Gives sigma a REAL job (weight two STRUCTURALLY "
                        "DIFFERENT computations) instead of just deciding "
                        "'append think token'. Default OFF; alpha init 0 "
                        "means a fresh ckpt is byte-identical at decode.")
    p.add_argument("--refinement_head_window", type=int, default=128,
                   help="Phase D refinement-head local-attention window "
                        "size (last N positions seen). Default 128.")
    p.add_argument("--refinement_head_n_heads", type=int, default=8,
                   help="Phase D refinement-head attention heads (default 8).")
    p.add_argument("--refinement_head_mlp_mult", type=int, default=2,
                   help="Phase D refinement-head MLP hidden multiplier "
                        "(default 2 -> d_hidden = 2 * d_model).")
    p.add_argument("--refinement_head_alpha_init", type=float, default=0.3,
                   help="Phase D refinement-head alpha init. 0.0 → "
                        "byte-identical at cold start BUT v10 lesson: "
                        "alpha stays stuck near 0 because the MLP gets "
                        "no real gradient when its contribution is zero. "
                        "Default 0.3 → head contributes from step 1, "
                        "MLP weights learn whether to keep or suppress.")
    p.add_argument("--mem_dim", type=int, default=0,
                   help="Memory projection dim. 0 = match d_model.")
    # ----- Persistent learned-RAG (Product-Key Memory) -----
    p.add_argument("--use_pkm", action="store_true",
                   help="Add a Product-Key Memory layer (Lample 2019 / "
                        "Memory Layers at Scale 2024) at one mid-depth "
                        "block. Sparse learned KV table — facts can live "
                        "there instead of being amortised into the dense "
                        "residual stream. See PKM_PLAN.md.")
    p.add_argument("--pkm_after_layer", type=int, default=14,
                   help="Insert PKM residual after this 0-indexed block.")
    p.add_argument("--pkm_n_keys", type=int, default=256,
                   help="Sub-key set size; total slots/head = n_keys^2.")
    p.add_argument("--pkm_n_heads", type=int, default=4)
    p.add_argument("--pkm_k_dim", type=int, default=128,
                   help="Per-side query / sub-key dim.")
    p.add_argument("--pkm_top_k", type=int, default=32,
                   help="Number of values retrieved per query.")
    p.add_argument("--pkm_value_bf16", action="store_true", default=True,
                   help="Store the value table in bf16 (math fp32). "
                        "Default on; halves persistent memory.")
    p.add_argument("--no_pkm_value_bf16", dest="pkm_value_bf16",
                   action="store_false")
    # ---------- v7 PKM-bootstrap-fix package (2026-05-17) ----------
    # The v5-pkm probe found 97 % of value rows still at random init and
    # only ~4 % of slots ever firing. These flags break the bootstrap.
    p.add_argument("--pkm_score_norm", type=str, default="layer",
                   choices=["batch", "layer"],
                   help="FIX 4: norm-on-scores kind. 'layer' is the v7 "
                        "default (no noisy running stats); 'batch' replicates "
                        "Lample/v5-pkm behaviour.")
    p.add_argument("--pkm_value_init_std", type=float, default=1.0,
                   help="FIX 3: value-row init std. v5-pkm used "
                        "1/sqrt(d_model) ≈ 0.04 (residual contribution 1 %); "
                        "v7 default 1.0 puts PKM output on residual scale "
                        "from step 0.")
    p.add_argument("--pkm_use_output_gate", action="store_true", default=True,
                   help="FIX 1: scalar α (init 0) gating PKM output. Lets "
                        "the model gradually trust PKM as gradient grows α "
                        "— mirrors the FiLM α curriculum. v7 default on.")
    p.add_argument("--no_pkm_use_output_gate", dest="pkm_use_output_gate",
                   action="store_false")
    p.add_argument("--pkm_epsilon_start", type=float, default=0.5,
                   help="FIX 2: starting ε for random-slot exploration. With "
                        "prob ε at training time, each top-k retrieval is "
                        "replaced by a uniform random slot — forces every "
                        "slot to receive gradient. 0.5 = half of retrievals "
                        "random at step 0. Anneals linearly to 0 over "
                        "--pkm_epsilon_warmup_steps. v5-pkm had ε=0 always "
                        "and ended up with only 4 % of slots active.")
    p.add_argument("--pkm_epsilon_warmup_steps", type=int, default=2000,
                   help="Anneal --pkm_epsilon_start linearly to 0 over this "
                        "many steps. 0 = no anneal (fixed ε throughout).")
    p.add_argument("--pkm_diversity_weight", type=float, default=0.01,
                   help="FIX 5: weight on slot-selection-entropy bonus. "
                        "Penalises peaky retrieval distributions (the "
                        "head_2/slot_8824 'one-slot-eats-40%%-of-the-mass' "
                        "pattern observed in v5-pkm). Encourages full table "
                        "usage. 0 disables. v7 default 0.01 = a gentle nudge.")
    # ---- v7.1 PKM bootstrap follow-ups (after v7 α-decay observation) ----
    # Step-440 trace showed α grew 0 → 0.085 (step 280) then *shrank* back
    # to 0.04 because value rows hadn't moved (v_std = 1.000 throughout)
    # so PKM was structurally noise. Two interventions below break the
    # chicken-and-egg: force α high enough that values get gradient (α-
    # floor), and run the value table on a much higher LR so it can move
    # before α can shrink (value-LR multiplier).
    p.add_argument("--pkm_alpha_floor_start", type=float, default=0.3,
                   help="FIX 1B: starting sign-preserving additive floor on "
                        "the PKM α gate. With α_floor > 0 the effective "
                        "gate is α_eff = α + sign(α)·floor — guarantees a "
                        "minimum PKM contribution magnitude during warmup "
                        "so the value table receives meaningful gradient. "
                        "Anneals linearly to 0 over --pkm_alpha_floor_"
                        "warmup_steps. 0 disables (v7.0 behaviour).")
    p.add_argument("--pkm_alpha_floor_warmup_steps", type=int, default=2000,
                   help="Anneal --pkm_alpha_floor_start to 0 over this many "
                        "steps. Default matches the ε-greedy curriculum so "
                        "both interventions retire together.")
    p.add_argument("--pkm_value_lr_mult", type=float, default=10.0,
                   help="LR multiplier for the PKM value-table parameters "
                        "(pkm_layer.values.*). The full chain α·w_k·∂loss "
                        "dampens the per-row gradient by ~10⁴×; multiplying "
                        "the per-row LR by 10× partially compensates and "
                        "lets the table actually move in early training. "
                        "Set 1.0 to disable. Default 10.0.")
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
    p.add_argument("--mid_eval_save_only", action="store_true",
                   help="Skip the HumanEval subprocess at every mid-eval "
                        "trigger; just save the ckpt and advance the counter. "
                        "Use when the only GPU is already occupied by the "
                        "trainer (the eval subprocess would OOM trying to "
                        "load its own copy of the model on the same device). "
                        "The mid-eval ckpts are the load-bearing artifact for "
                        "resume — HumanEval results during pretrain are noisy "
                        "anyway, so this is the safe default when training is "
                        "co-resident with another job.")
    p.add_argument("--mid_eval_min_free_gib", type=float, default=2.0,
                   help="If the trainer's GPU has less than this much free "
                        "memory at a mid-eval trigger, auto-skip the HumanEval "
                        "subprocess (it would OOM trying to load its own copy "
                        "of the model). Ckpt save still happens, counter still "
                        "advances. Set 0 to disable the auto-skip and always "
                        "attempt the subprocess.")
    p.add_argument("--mask_eos_in_targets", action="store_true",
                   help="In data_mix.py, set targets equal to eos_token_id "
                        "to -100 so the model never gets a gradient on "
                        "predicting EOS. Standard fix for the "
                        "halt-after-docstring artifact from small documents "
                        "in mixed-corpus pretrain. Off by default for "
                        "backwards compat with already-running v2; enable "
                        "for v3+.")
    p.add_argument("--mid_eval_min_emit_before_eos", type=int, default=30,
                   help="Suppress eos_token_id for the first N emitted tokens "
                        "during mid-eval generation. Default 30 mitigates the "
                        "halt-after-docstring artifact from EOS at small-doc "
                        "boundaries in mixed-corpus pretrain.")
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
    p.add_argument("--compile", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Apply torch.compile to model.forward (fuses the "
                        "PyTorch glue around the FLA Triton kernels). "
                        "Default on; pass --no-compile to disable — e.g. "
                        "if compile errors on the nightly-torch / FLA / "
                        "Blackwell stack or triggers recompilation storms.")
    p.add_argument("--compile_mode", type=str, default="default",
                   choices=["default", "reduce-overhead",
                            "max-autotune", "max-autotune-no-cudagraphs"],
                   help="torch.compile mode. 'reduce-overhead' wraps "
                        "compiled regions in CUDA Graphs (attacks the "
                        "'Command Buffer Full' kernel-launch stall) but "
                        "needs stable shapes — `cu_seqlens` shape varies "
                        "with doc count per batch, so test before relying.")
    p.add_argument("--wd", type=float, default=0.01,
                   help="Weight decay for non-α params (Muon + AdamW "
                        "regular groups). Default 0.01 (the validated value; "
                        "0.1 is the Moonlight-scale setting and over-brakes "
                        "at our token budget — see the residual-collapse "
                        "diagnosis).")
    p.add_argument("--layer_drop_max", type=float, default=0.0,
                   help="Stochastic Depth: per-block drop probability "
                        "ramps linearly 0 → this value across depth. "
                        "Default 0.0 (off).")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient-accumulation microbatches per optimizer "
                        "step. Effective batch = batch * grad_accum. Only "
                        "supported on the non-thinking-token (pretrain) "
                        "path. Default 1.")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Global grad-norm clip applied before each optimizer "
                        "step. 0 disables clipping. Default 1.0.")
    p.add_argument("--z_loss", type=float, default=1e-4,
                   help="Weight on the z-loss regulariser "
                        "mean(logsumexp(logits)^2), which keeps output "
                        "logits from drifting. Default 1e-4; 0 disables.")
    # Trunk multi-horizon gist loss (v7, see experiments/gist_loss.py).
    p.add_argument("--gist_loss_weight", type=float, default=0.0,
                   help="Weight on the trunk multi-horizon gist loss: "
                        "per-horizon heads predict the mean-pooled future "
                        "hidden state (windowed gist) from h[t]. A "
                        "'high-level direction' representation objective. "
                        "0 disables (default). Recommended 0.1.")
    p.add_argument("--gist_horizons", type=str, default="16,64,256",
                   help="Comma-separated future-window sizes K for the "
                        "gist loss (one prediction head per horizon).")
    p.add_argument("--bf16_optim_state", action="store_true",
                   help="Store optimizer state (AdamW exp_avg/exp_avg_sq, "
                        "Muon momentum_buffer) in bf16 instead of fp32. "
                        "Saves ~550 MB persistent on the 218 M v4 model "
                        "(see experiments/bf16_optim.py). Math runs in "
                        "fp32 (lift → step → cast back), so it's lossless "
                        "in the precision sense — validated on a small "
                        "DeltaNet (max |Δ_loss| < 0.005 vs stock AdamW "
                        "over 200 steps). Does NOT save peak GPU memory "
                        "from gradients (autograd allocates fp32 "
                        "intermediates regardless of grad_dtype).")
    p.add_argument("--alpha_wd", type=float, default=0.0,
                   help="Weight-decay applied to FiLM α scalars (matched by "
                        "name suffix '.alpha' inside any feedback container). "
                        "Default 0.0 per CLAUDE.md mandate: the WD-equilibrium "
                        "probe on the v1 500M-token ckpt showed |grad_α| ≈ "
                        "1.83 × WD·|α| — gradient consistently wanted α "
                        "higher and the Muon WD=0.1 was the brake, not a "
                        "true loss-flat ceiling. Pass --alpha_wd 0.1 to "
                        "reproduce the old behaviour.")
    return p
