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
    p.add_argument("--gate_ponder_raw", action="store_true",
                   help="Apply the (1-g)*gate_lambda THINK cost to the RAW gate "
                        "g instead of the floor-clamped g_eff. clamp(g,floor) "
                        "zeros the ponder gradient to the raw gate below the "
                        "floor, so the deploy-time gate (raw σ) is never "
                        "penalised for thinking and over-thinks at generation. "
                        "This restores that gradient so the gate learns to emit "
                        "unless CE > gate_lambda. CE term still uses g_eff.")
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
    p.add_argument("--matrix_optimizer", type=str, default="muon",
                   choices=["muon", "fused_deltanet_ns"],
                   help="Which orthogonalizing matrix optimizer to use on the "
                        "2D hidden matrices when --optimizer muon. 'muon' "
                        "(default) is BYTE-IDENTICAL to the legacy path. "
                        "'fused_deltanet_ns' uses per-head Newton-Schulz on "
                        "the DeltaNet q/k/v/b projections (head-structured "
                        "modular-norm dualizer) + whole-matrix Muon on "
                        "o_proj/MLP/other 2D matrices; the matrix-optimizer "
                        "param set and all AdamW groups are IDENTICAL to the "
                        "muon arm, so a muon-vs-fused A/B isolates only the "
                        "q/k/v/b orthogonalization. See "
                        "DELTANET_PRECONDITIONER.md. Requires --arch deltanet.")
    p.add_argument("--embed_lr_mult", type=float, default=1.0,
                   help="Multiplier on the AdamW LR for the embedding/lm_head "
                        "group (the largest non-Muon chunk) when --optimizer "
                        "muon. Default 1.0 is BYTE-IDENTICAL to the legacy "
                        "shared-LR path. >1.0 splits embed/lm_head into their "
                        "own group at lr*mult (the μP-flavoured higher "
                        "embedding LR). Requires --optimizer muon when != 1.0.")
    p.add_argument("--embed_optimizer", type=str, default="adam",
                   choices=["adam", "rownorm"],
                   help="Optimizer for the embedding/lm_head group when "
                        "--optimizer muon. 'adam' (default) keeps the legacy "
                        "AdamW (byte-identical at embed_lr_mult=1.0). 'rownorm' "
                        "routes embed/lm_head to a per-row RMS-normalized update "
                        "(the modular-norm dualizer: each token vector takes a "
                        "step of fixed RMS magnitude lr*embed_lr_mult). See "
                        "experiments/embed_optim.py. Requires --optimizer muon.")
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
    p.add_argument("--mem_dim", type=int, default=0,
                   help="Memory projection dim. 0 = match d_model.")
    # ---- WorkingMemory decoupled-key/value (DKV) addressing (2026-06-04) ----
    # These constructor kwargs already exist on TinyLM/WorkingMemory; the
    # flags below thread them through model_builder. All default to the
    # legacy (pre-DKV) behaviour so the existing pretrain path is unchanged.
    p.add_argument("--mem_decoupled_kv", action="store_true",
                   help="DKV-WM: give WorkingMemory a dedicated match-KEY "
                        "projection (W_v becomes content-only), cosine "
                        "addressing with a learned clamped temperature, and a "
                        "β-scaled write-gate log-bias. The validated reliable "
                        "config for semantic (non-token-identity) recall — the "
                        "default dot-product addressing has a magnitude "
                        "degeneracy (diffuse softmax, top_mass≈0.16). Default "
                        "OFF → byte-identical legacy WM.")
    p.add_argument("--mem_read_alpha_init", type=float, default=1.0,
                   help="Init value of the learned scalar α gate on the WM "
                        "read injection. 1.0 (default) preserves the "
                        "pre-α-gate behaviour; 0.0 = zero-init-residual boot "
                        "(FiLM-α pattern, cold start byte-identical to no-WM).")
    p.add_argument("--mem_read_alpha_floor_start", type=float, default=0.0,
                   help="Sign-preserving additive α-FLOOR on the WM read "
                        "injection (mirrors PKM FIX 1B). Holds the effective "
                        "read contribution ≥ floor during the warmup window so "
                        "W_q/W_v/W_proj keep strong gradient and the sharp "
                        "addressing locks in before α takes over. Validated "
                        "reliable value ~0.5. Anneals to 0 over "
                        "--mem_read_alpha_floor_warmup_steps. 0 disables "
                        "(default, byte-identical legacy path).")
    p.add_argument("--mem_read_alpha_floor_warmup_steps", type=int, default=0,
                   help="Anneal --mem_read_alpha_floor_start to 0 over this "
                        "many WM forwards. A few thousand is the validated "
                        "range. 0 disables the floor curriculum (default).")
    # ---- v14 WM-recall plumbing (validated embedding-key + copy readout) -----
    # All default OFF → byte-identical to the legacy WM, so an in-flight run
    # (e.g. v12) that re-imports this file on autoresume is unaffected.
    p.add_argument("--mem_key_from_embedding", action="store_true",
                   help="v14: key the WM read on a short CAUSAL input-embedding "
                        "window over the identifier (token identity) instead of "
                        "the trunk hidden. Validated top1 addressing 1.00 vs "
                        "chance — the documented fix for 'WM inert for recall'. "
                        "Requires --mem_decoupled_kv. Adds NO new params (raw "
                        "pooled embeddings + the existing DKV temperature) so a "
                        "continuation ckpt loads byte-identically. Default OFF.")
    p.add_argument("--mem_key_window", type=int, default=4,
                   help="Number of recent tokens pooled (order-sensitive) for "
                        "--mem_key_from_embedding. Default 4.")
    p.add_argument("--use_copy_head", action="store_true",
                   help="v14: add a COPY/POINTER readout. At --emit_read_mask "
                        "answer positions, mix the LM distribution with a copy "
                        "distribution over the WM-addressed source span "
                        "(p=(1-g)·p_lm + g·p_copy, g=sigmoid(Linear(h)) with a "
                        "very-negative bias so cold-start g≈0). The validated "
                        "100%-exact multi-token recall readout. Default OFF → "
                        "byte-identical; old ckpts load (no copy_head.* keys).")
    p.add_argument("--emit_read_mask", action="store_true",
                   help="v14: have the data mix emit a per-position mem_read_mask "
                        "(4th tuple element) = 1 over recall-source answer spans, "
                        "aligned through think-burst insertion. Threaded into "
                        "TinyLM.forward(mem_read_mask=) on the pretrain path so "
                        "answer-token CE flows into the WM read (the gradient "
                        "missing in v10-v13). Default OFF → 3-tuple as before.")
    # ----- v15 DISCRETE-KEY WM (validated in wm_recall_cotrain.py) ----------
    # The discrete-hash WM addressing + copy/pointer readout, wired into the
    # real pretrain trainer (2026-06-16). All default OFF / no new params →
    # byte-identical to the legacy WM, so an in-flight run that re-imports this
    # file on autoresume is unaffected and old ckpts load strict=False.
    p.add_argument("--mem_discrete_key", action="store_true",
                   help="v15: DISCRETE-HASH WM addressing — key the read on a "
                        "deterministic per-position integer code (the carried "
                        "BOUND-identifier run-hash) → onehot match → zero "
                        "cross-talk. No new params. The validated "
                        "content-addressable recall mechanism. Requires "
                        "--use_memory. Default OFF → byte-identical legacy read.")
    p.add_argument("--mem_discrete_key_vstart", action="store_true",
                   help="v15: use the task-specific `vN` value-start parser for "
                        "the discrete key INSTEAD of the GENERAL lexical "
                        "identifier-span extractor (default). Only meaningful "
                        "with --mem_discrete_key.")
    p.add_argument("--mem_discrete_key_match_window", type=int, default=32,
                   help="v15: locality window for the discrete match-existence "
                        "gate — the addressing identifier must be re-mentioned "
                        "within this many tokens for the copy to fire (rejects "
                        "stale cross-family false matches). 0 disables. "
                        "Default 32.")
    p.add_argument("--mem_always_read", action="store_true",
                   help="v15: ALWAYS-ON WM read — compute/inject the WM read at "
                        "EVERY position (not just think positions) when no "
                        "explicit read_mask is given, so WM is always on the "
                        "gradient path (PKM-style). Default OFF → byte-identical "
                        "legacy think-only read.")
    p.add_argument("--mem_copy_require_match", action="store_true", default=True,
                   help="v15: MATCH-EXISTENCE copy gating — the copy/pointer head "
                        "only fires where the discrete address matched a real "
                        "buffered binding (no-match → recurrence fallback, no "
                        "harm). Default ON; no-op on the non-discrete path.")
    p.add_argument("--mem_no_copy_require_match", dest="mem_copy_require_match",
                   action="store_false",
                   help="Disable the match-existence copy gate (let the copy head "
                        "fire wherever its learned gate opens).")
    p.add_argument("--mem_soft_namekey", action="store_true",
                   help="SOFT NAME-SPAN addressing — learned continuous key = "
                        "enc(pooled name-span input emb); cosine soft read over "
                        "binding slots. Matches the hash on exact recall and adds "
                        "surface-variant (case/camel) robustness the spelling-"
                        "locked hash cannot. Mutually exclusive with "
                        "--mem_discrete_key. Default OFF → no new params, byte-"
                        "identical.")
    p.add_argument("--mem_soft_namekey_dim", type=int, default=64,
                   help="soft name-key vector dim (default 64).")
    p.add_argument("--mem_soft_namekey_match_threshold", type=float, default=0.5,
                   help="min top-attention over binding slots for the soft "
                        "address to count as a match (gates the copy head). "
                        "Default 0.5.")
    # --- CONTEXTUAL NAME-SPAN addressing (mem_ctx_namekey, validated 2026-06-17:
    #     the FULLY-LEARNED, NO-static-hash WM addresser; key/query = the trunk's
    #     contextual hidden pooled over the identifier name-span, dot-product read,
    #     copy/pointer readout, attention-supervision aux). All default OFF / no
    #     new params (until ctx_namekey on) → byte-identical to the legacy WM, so
    #     an in-flight run that re-imports this file on autoresume is unaffected
    #     and old ckpts load strict=False. Wired into pretrain (train_lm) so it
    #     can be CO-TRAINED with the trunk (mirrors the mem_discrete_key wiring).
    p.add_argument("--mem_ctx_namekey", action="store_true",
                   help="CONTEXTUAL NAME-SPAN addressing — learned key/query = "
                        "enc(pooled name-span TRUNK HIDDEN), dot-product attention "
                        "read over binding slots, carried only over `=`-bound name "
                        "refs. The general (no static token-hash) addresser. Train "
                        "with --ctx_addr_aux_weight (attention supervision). "
                        "Mutually exclusive with --mem_discrete_key / "
                        "--mem_soft_namekey. Requires --use_memory + --use_copy_head "
                        "for the recall readout. Default OFF → byte-identical.")
    p.add_argument("--ctx_namekey_dim", type=int, default=192,
                   help="ctx name-key q/k encoder output dim (default 192, the "
                        "capacity-matched probe value).")
    p.add_argument("--ctx_namekey_match_threshold", type=float, default=0.5,
                   help="min top-attention over binding slots for the ctx address "
                        "to count as a match (gates the copy head). Default 0.5.")
    p.add_argument("--ctx_addr_aux_weight", type=float, default=0.0,
                   help="weight of the ctx_namekey ADDRESSING aux — attention-"
                        "supervision CE that trains the learned ctx read attention "
                        "to land on the binding slot the deterministic lexical code "
                        "identifies as correct, ONLY at recall answer-span "
                        "(mem_read_mask) positions. The teacher is the parameter-"
                        "free lexical hash; the ctx addresser is the general "
                        "student. RAMPED + SMALL (target ~0.1-0.3). Default 0.0 = "
                        "OFF (byte-identical: no aux term added).")
    p.add_argument("--ctx_addr_aux_warmup_steps", type=int, default=0,
                   help="linear 0->1 ramp of --ctx_addr_aux_weight over this many "
                        "steps after --ctx_addr_aux_start_step (keeps the aux "
                        "gradient negligible while the trunk/PKM settle). 0 = full "
                        "weight immediately at/after the start step.")
    p.add_argument("--ctx_addr_aux_start_step", type=int, default=0,
                   help="step at which the ctx addressing aux begins (weight 0 "
                        "before this). Default 0.")
    p.add_argument("--copy_gate_bias_init", type=float, default=-6.0,
                   help="v15: init bias of the copy-head gate Linear. Very "
                        "negative (default -6.0) → cold/closed cold-start gate "
                        "g≈0 so the mix begins ≈ the plain LM (stable). Maps to "
                        "TinyLM(copy_head_gate_bias_init=).")
    p.add_argument("--mem_freeze_read_alpha", action="store_true",
                   help="v15: pin the WM read-injection α to --mem_read_alpha_init "
                        "and freeze it (requires_grad=False) AFTER loading the "
                        "ckpt. With --mem_read_alpha_init 0.0 this turns the "
                        "additive W_proj injection fully OFF (copy-head-only — the "
                        "validated no-harm config: WM helps via the copy/pointer "
                        "at recall spans and can never corrupt general text).")
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
    # --pkm_diversity_weight REMOVED 2026-06-18: it was an inert no-op (the
    # slot-entropy loss ran on detached indices+weights → zero grad path,
    # measured 0% gradient share). PKM diversity is held by ε-greedy + LayerNorm
    # score-norm + value-LR. Stripped from launchers too. (See project_pkm_diversity_inert.)
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
    # --- v9: latent-thinking CO-TRAINING (make thinking USEFUL from day 1) ---
    p.add_argument("--latent_cotrain_weight", type=float, default=0.0,
                   help="Weight on the latent-thinking co-training loss "
                        "(grad CE on the post-R-latent-think prediction). "
                        "0 disables (default). Gives the trunk gradient to do "
                        "useful sequential computation during thinking. "
                        "Requires --state_readonly_at_think + --output_gate.")
    p.add_argument("--latent_cotrain_start_step", type=int, default=0,
                   help="Delay the latent-cotrain loss until this step (default 0 "
                        "= from step 1). v12 fix: co-training/gate aux losses + "
                        "their extra forwards destabilized the PKM α-bootstrap in "
                        "v11 (αL never committed during the floor window → v7.0 "
                        "α-decay failure). Start these AFTER the PKM α-floor window "
                        "(~pkm_alpha_floor_warmup_steps) so PKM bootstraps clean "
                        "like v7.1, then thinking engages at full weight.")
    p.add_argument("--latent_cotrain_R", type=int, default=4,
                   help="Number of state-readonly latent think steps in the "
                        "co-training loss.")
    p.add_argument("--latent_cotrain_sample_frac", type=float, default=0.05,
                   help="Fraction of clean positions co-trained per step.")
    p.add_argument("--latent_cotrain_max_positions", type=int, default=32,
                   help="Hard cap on latent-cotrain positions per step "
                        "(grad through R forwards — keep small for memory).")
    p.add_argument("--use_latent_feedback_adapter", action="store_true",
                   help="Build a LatentFeedbackAdapter (RMSNorm→zero-init "
                        "Linear, identity-residual + learnable α) that maps the "
                        "fed-back out_norm hidden into the input-embedding "
                        "manifold before it is consumed as the next latent "
                        "think-step input. Fixes the OOD-feedback failure mode "
                        "(latent Δlogp ≈ -4..-6). Identity at cold start → a "
                        "fresh ckpt is byte-identical to the no-adapter path. "
                        "Use with --latent_cotrain_weight so the adapter gets "
                        "gradient.")
    p.add_argument("--latent_cotrain_selective", action="store_true",
                   help="Sample latent-cotrain positions WEIGHTED toward where "
                        "thinking should help (high gate σ / high no-think "
                        "predictive entropy) instead of uniform-random. Keeps "
                        "the fixed (max_positions, max_prefix_len) shape the "
                        "compile path needs. The uniform path is preserved when "
                        "this flag is off.")
    # --- Depth-matched latent-REASONING co-train (the 2026-06-05 fix) ---
    # Instead of latent_cotrain_loss (R latent steps -> predict a random natural
    # next token, which learns "thinking not needed", Δlogp<0), draw from a
    # depth-bound NON-parallelizable reasoning corpus (pointer-chase) and
    # supervise the ANSWER with R = problem depth + a curriculum. Co-trained at
    # low weight alongside the general mix so the trunk keeps general ability.
    # Validated standalone (latent_arith_real.py): fair lift +0.40-0.65, the
    # autonomous gate allocates exactly n hops. Default 0.0 = OFF.
    p.add_argument("--latent_reasoning_weight", type=float, default=0.0,
                   help="Weight on the depth-matched latent-reasoning co-train "
                        "(answer-span CE on pointer-chase, R=depth). 0 disables. "
                        "Requires --state_readonly_at_think; pair with "
                        "--use_latent_feedback_adapter. Needs --no-compile.")
    p.add_argument("--latent_reasoning_train_prefix", type=str,
                   default="data/ptr10dict_train",
                   help="Prefix for the depth-bound reasoning corpus "
                        "(<prefix>_n{rung}.jsonl, each record has prompt+answer).")
    p.add_argument("--latent_reasoning_rungs", type=str, default="2,3,4,5,6,7,8",
                   help="Comma-separated depth rungs to co-train on.")
    p.add_argument("--latent_reasoning_n", type=int, default=4,
                   help="Reasoning examples per optimizer step (each is R extra "
                        "sequential forwards — keep small).")
    p.add_argument("--latent_reasoning_max_len", type=int, default=256,
                   help="Drop reasoning examples whose tokenised length exceeds "
                        "this (keeps the latent loop cheap).")
    p.add_argument("--latent_reasoning_no_ramp", action="store_true",
                   help="Skip the depth curriculum (sample rungs uniformly from "
                        "step 0). Default off = ramp 1->max over 60%% of steps.")
    p.add_argument("--latent_reasoning_gate_weight", type=float, default=0.0,
                   help="If >0, ALSO train the output gate to invoke+halt "
                        "thinking on reasoning examples (autonomous_halt BCE: "
                        "P(emit)→THINK for the first R decision positions, EMIT "
                        "at the last). Fixes 'capability baked but gate never "
                        "fires it' (avg_steps≈0.77 vs target n). 0 = capability "
                        "only.")
    p.add_argument("--latent_reasoning_start_step", type=int, default=0,
                   help="Step at which the latent-reasoning co-train begins "
                        "firing (default 0 = from scratch). Use to start it on a "
                        "warm trunk; the depth curriculum is measured relative to "
                        "this start.")
    p.add_argument("--latent_reasoning_weight_warmup_steps", type=int, default=0,
                   help="Linearly ramp the latent-reasoning weight 0->target over "
                        "this many steps after --latent_reasoning_start_step "
                        "(default 0 = full weight immediately, byte-identical to "
                        "the old path). Set to the PKM α-floor window (e.g. 3000) "
                        "so the aux gradient stays negligible while PKM bootstraps "
                        "— the v12-destabilization safety knob.")
    # --- Gate-calibration aux loss (latent "think only where helpful") ---
    # Trains the OUTPUT GATE (not the trunk) toward firing think exactly where
    # a latent think actually raises logp(true_next). Uses the shared latent
    # primitive (thinking.latent_think_logp) under no_grad to derive a
    # per-position BCE target, so it is the gate-side complement of
    # --latent_cotrain_weight (which gives the TRUNK gradient through thinking).
    # Default 0.0 = OFF (byte-identical). The latent extra forward is
    # dynamo.disable'd; pass --no-compile if a compiled run still crashes.
    p.add_argument("--gate_calibration_weight", type=float, default=0.0,
                   help="Weight on the gate-calibration BCE loss "
                        "(experiments/gate_calibration.compute_gate_calibration_"
                        "loss). 0 disables (default). Requires --output_gate. "
                        "Recommended 0.05.")
    p.add_argument("--gate_calibration_start_step", type=int, default=0,
                   help="Delay the gate-calibration loss until this step (default "
                        "0). Same v12 rationale as --latent_cotrain_start_step: "
                        "let PKM bootstrap clean before the gate aux loss + its "
                        "extra forwards engage.")
    p.add_argument("--gate_calibration_R", type=int, default=4,
                   help="Number of state-readonly LATENT think steps used to "
                        "measure 'does thinking help' for the gate target.")
    p.add_argument("--gate_calibration_sample_frac", type=float, default=0.05,
                   help="Fraction of clean positions scored per step.")
    p.add_argument("--gate_calibration_max_positions", type=int, default=32,
                   help="Hard cap on gate-calibration positions per step "
                        "(each runs R extra forwards — keep small).")
    p.add_argument("--gate_calibration_sigma_low", type=float, default=0.0,
                   help="Lower σ(gate) band for scored positions (focus on "
                        "undecided positions). 0 keeps all.")
    p.add_argument("--gate_calibration_sigma_high", type=float, default=1.0,
                   help="Upper σ(gate) band for scored positions. 1 keeps all.")
    # --- Per-feature usefulness probe (is each mechanism load-bearing yet?) ---
    p.add_argument("--feature_probe_every_tokens", type=int, default=0,
                   help="Run the per-feature usefulness probe every N tokens "
                        "of training. 0 disables (default). On a held-out "
                        "val batch, logs an ablation-delta CE for WM and PKM "
                        "(CE rises iff the feature is load-bearing), the WM "
                        "read_alpha, FiLM α per pair, and the current gate "
                        "fire-rate. Console + TensorBoard (probe/*). Cheap: a "
                        "handful of forwards on one batch.")
    p.add_argument("--feature_probe_wm_recall_path", type=str,
                   default="data/longctx_recall_heldout.jsonl",
                   help="Held-out long-context recall JSONL for the WM "
                        "load-bearing signal. The natural-text val batch has "
                        "zero think tokens, so WM (which reads only at think "
                        "positions) shows ablation-delta≈0 there; this recall "
                        "set drives think tokens and is where WM can actually "
                        "be measured. Set empty to skip the recall probe.")
    p.add_argument("--feature_probe_wm_recall_n", type=int, default=64,
                   help="Cap the number of recall tasks for the WM probe "
                        "(generation is CUDA-heavy; keep small).")
    p.add_argument("--feature_probe_code_recall_path", type=str, default="",
                   help="v14: held-out REALISTIC code/agentic recall JSONL "
                        "(data/code_recall_heldout.jsonl) for the live "
                        "WM-usefulness probe. Runs eval_code_recall teacher-"
                        "forced ON vs full_off (Δ>0 ⇒ WM load-bearing) and logs "
                        "read_alpha + addressing mass-on-binding. Empty = skip "
                        "(default) so non-v14 runs are unaffected.")
    p.add_argument("--feature_probe_code_recall_n", type=int, default=200,
                   help="Cap the number of tasks for the v14 code-recall probe "
                        "(teacher-forced is one forward each; ~200 is cheap).")
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
    p.add_argument("--ddp_no_bf16_compress", action="store_true",
                   help="Disable the bf16 gradient-compression DDP comm hook. "
                        "By default (DDP only), grads are cast fp32→bf16 "
                        "before all-reduce to halve cross-GPU bytes on a "
                        "P2P-less PCIe link (~4 GB/s on this rig). No effect "
                        "single-GPU.")
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
