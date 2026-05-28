#!/usr/bin/env bash
# D18 decisive experiment: TRAIN state-readonly thinking on arithmetic.
#
# RL (execution-grounded GRPO) from rl_grader_phase_c_v2_step300 (16/164
# HumanEval, project best) with state_readonly_at_think ON, so think
# tokens READ the DeltaNet recurrent state but never WRITE to it (beta=0).
# Hypothesis (GEMINI 1-layer probe 0.88 vs 0.41): a model TRAINED with
# read-only thinking can use the extra forward passes as latent
# computation for arithmetic chains WITHOUT corrupting the carried
# bindings (the documented "thinking corrupts recall" failure mode).
#
# Training pool = synth_arith ladder n2+n3+n4 (the competence band + one
# rung above; no-think baselines 21% / 16% / 1% at n=2/3/4).
#
# Deterministic gate (the gate already fires ~0.55 on arithmetic);
# stochastic-gate has FAILED twice (D-section "stochastic exploration
# noise drowns the reward") so it is deliberately NOT used here.
# ponder_cost 0.0 because we WANT thinking. KL anchor 0.05 to the SFT
# base for stability (v2's validated stabilizer).
#
# GPU 0 ONLY (GPU 1 is another user's).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:."

CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/train_rl_grader.py \
  --load_ckpt checkpoints/rl_grader_phase_c_v2_step300.pt \
  --save_ckpt checkpoints/rl_arith_stateread.pt \
  --dataset_jsonl data/synth_arith_train_n234.jsonl \
  --state_readonly_at_think \
  --extract_code_block \
  --steps 250 \
  --batch 4 \
  --grpo_n_group 4 \
  --lr 2e-6 \
  --max_gen 96 \
  --emit_threshold 0.5 \
  --gate_floor 0.0 \
  --temperature 0.7 \
  --kl_coef 0.05 \
  --ponder_cost 0.0 \
  --counterfactual \
  --grad_clip 1.0 \
  --save_every 50 \
  --log_every 1
