#!/bin/bash
# GRPO Training for Thinking Head (Phase 24)
set -e
cd /home/knielsen/ml/parallel-ss-dep

# Ensure PYTHONPATH includes the current directory for 'experiments' module
export PYTHONPATH=$PYTHONPATH:.

# Reduced batch size and group size to avoid OOM
# Each token rollout: grpo_n_group * T
# batch 1 * group 8 = 8 trajectories of length 512 per step.

# 1. RL from Supervised Checkpoint
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u experiments/train_rl.py \
    --steps 5000 \
    --batch 1 \
    --grpo_n_group 8 \
    --lr 5e-6 \
    --grpo_kl_beta 0.05 \
    --grpo_ponder_cost 0.01 \
    --max_depth 10 \
    --max_T 0 \
    --load_ckpt checkpoints/think_sweep_fresh_tokens_final.pt \
    --save_ckpt checkpoints/think_rl_from_sft.pt \
    --tb_dir runs/tb/think_rl_from_sft \
    > runs/think_rl_from_sft.log 2>&1 &
echo "Launched RL from Supervised on GPU 0"

# 2. RL "From Scratch"
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/train_rl.py \
    --steps 5000 \
    --batch 1 \
    --grpo_n_group 8 \
    --lr 5e-6 \
    --grpo_kl_beta 0.05 \
    --grpo_ponder_cost 0.01 \
    --max_depth 10 \
    --max_T 0 \
    --load_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
    --save_ckpt checkpoints/think_rl_from_scratch.pt \
    --tb_dir runs/tb/think_rl_from_scratch \
    > runs/think_rl_from_scratch.log 2>&1 &
echo "Launched RL from Scratch (DN Baseline) on GPU 1"
