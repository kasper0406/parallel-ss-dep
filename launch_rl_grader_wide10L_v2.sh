#!/usr/bin/env bash
# Continuation grader-RL on the wide-10L coder (plain trunk, feedback=none → pure
# execution-RL, depth=0). Measures how much MORE of the confirmed pass@k exploration
# band (greedy 7 → pass@30 21/164) a second RL round converts toward greedy reliability.
# Loads the 400-step RL ckpt (7/164); KL-anchored to IT; saves to a NEW path so the
# 7/164 ckpt is preserved. Same v2 KL-stable recipe; fresh seed for exploration.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints
CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/rl_grader_wide10L.pt \
    --save_ckpt checkpoints/rl_grader_wide10L_v2.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
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
    --seed 1 \
    > runs/rl_grader_wide10L_v2.log 2>&1 &
echo "Launched RL-grader wide10L v2 on GPU ${GPU:-1} (PID $!)"
