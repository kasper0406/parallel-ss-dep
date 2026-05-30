#!/bin/bash
# RL-A: validated stable grader-RL spine on the v8 SFT base (2026-05-30).
#
# Exact recipe as launch_rl_grader_phase_c_v2.sh (the run that climbed
# monotonically to 16/164 on the Phase C SFT base), repointed to the v8 600M
# SFT base. Purpose: confirm the SFT->grader-RL pipeline lifts HumanEval on the
# new 600M architecture (the core "does it work" proof) BEFORE layering the new
# machinery (RL-B: --think_budget_diversity / --stochastic_gate / --max_turns /
# --llm_judge).
#
# KL-to-reference (ref = the SFT base) is the stability mechanism; ponder_cost 0
# avoids the v1 depth-collapse trigger.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v8_combined.pt \
    --save_ckpt checkpoints/rl_grader_v8.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.05 \
    --ponder_cost 0.0 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 50 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_v8.log 2>&1 &
echo "Launched RL-A (validated spine) on v8 SFT base, GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v8.log"
