#!/bin/bash
# RL-B: gate-selectivity grader-RL on the v8 SFT base (2026-05-30).
#
# RL-A (the validated phase_c spine) got STUCK on v8: at rollout temperature
# the gate collapsed to always-think (depth_mean pinned at total_think_budget,
# think_rate 1.0), and since thinking HURTS on the v8 trunk (probe + the 3-vs-8
# thinking-on/off HumanEval split), every rollout was maximally corrupted with
# no signal to think less (ponder_cost 0, no within-group budget variance).
#
# RL-B adds the gate-selectivity machinery whose whole purpose is "think only
# where helpful / cut unproductive thinking":
#   --think_budget_diversity 0.7  : the N rollouts of each problem get a SPREAD
#       of think budgets, so cut-short (think-less) rollouts that score better
#       (they do — thinking hurts here) get positive group advantage -> the
#       policy learns to emit instead of think. The cheap RL-native "cut-short".
#   --stochastic_gate + --gate_entropy_bonus : the gate becomes an RL action
#       optimized on the sampling distribution it actually faces (counters the
#       gate's temperature-fragility), with entropy keeping it off the corners.
#
# Everything else is the stable spine (KL 0.05, lr 2e-6, clip 0.1, temp 0.7,
# counterfactual so thinking never worsens task reward). Hypothesis: think_rate
# falls, pass-rate recovers toward/past the 8/164 thinking-off ceiling.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v8_combined.pt \
    --save_ckpt checkpoints/rl_grader_v8_selective.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --think_budget_diversity 0.7 \
    --stochastic_gate \
    --gate_entropy_bonus 0.01 \
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
    > runs/rl_grader_v8_selective.log 2>&1 &
echo "Launched RL-B (gate-selectivity) on v8 SFT base, GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v8_selective.log"
