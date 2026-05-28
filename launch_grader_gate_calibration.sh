#!/bin/bash
# Grader-grounded gate calibration (2026-05-28).
#
# The synthesis of the thinking-gate debugging arc. Trains the thinking
# gate with a SUPERVISED BCE target whose label is EXECUTION-GROUNDED:
# "did thinking improve the GRADED rollout outcome?".
#
# For each (problem) we roll out a completion WITH the gate's think
# budget AND once WITHOUT thinking, grade both with code_grader, and:
#   Δ = score_with - score_without
#   Δ > 0 → gate-fire decisions in the with-think rollout get BCE target=1
#   Δ < 0 → BCE target=0
#   Δ = 0 → skipped (no signal)
# A single teacher-forced grad forward over the recorded rollout gives the
# gate logits at the fire positions; they are BCE'd against that target.
#
# Why this and not the alternatives (see THINKING_AUDIT/PLAN_FLAW_C):
#  - The OLD gate_calibration used a next-token-logp target → FATALLY wrong
#    (rewards sharpening surface tokens, blind to whether code runs).
#  - Stochastic-gate RL (Bernoulli + log-prob policy gradient) FAILED TWICE
#    (exploration noise drowns the terminal reward). NOT used here.
#  - This is the deterministic, task-grounded BCE: terminal grader reward as
#    the label, no Bernoulli noise. Fixes flaw C directly.
#
# Base: rl_grader_phase_c_v2_step300.pt — the strongest ckpt (16/164
# HumanEval) and the one with the largest probe Δ (thinking helps, pooled
# +0.079 grader). KL-to-frozen-ref anchor (ref = base) for stability.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

BASE=${BASE:-checkpoints/rl_grader_phase_c_v2_step300.pt}
SAVE=${SAVE:-checkpoints/grader_gate_calibration.pt}

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u \
    experiments/grader_gate_calibration.py \
    --load_ckpt "$BASE" \
    --save_ckpt "$SAVE" \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 200 \
    --lr 2e-6 \
    --max_problems_per_step 4 \
    --max_gate_positions_per_problem 64 \
    --max_gen 256 \
    --total_think_budget 120 \
    --max_think_per_step 8 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --min_emit_before_eos 30 \
    --prompt_style sft_comment \
    --kl_coef 0.05 \
    --grad_clip 1.0 \
    --save_every 25 \
    --log_every 1 \
    --seed 0 \
    > runs/grader_gate_calibration.log 2>&1 &
echo "Launched grader-gate-calibration on GPU ${GPU:-0} (PID $!)"
echo "  base=$BASE  save=$SAVE"
echo "Watch: tail -f runs/grader_gate_calibration.log"
