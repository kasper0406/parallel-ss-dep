#!/bin/bash
# Sweep aux loss normalization/scaling on the strict adaptive 4x512 setup.
set -e
cd /home/knielsen/ml/parallel-ss-dep

COMMON_ARGS=(
    --arch deltanet
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3
    --T 512 --batch 4 --steps 4000 --lr 3e-4 --seed 0
    --dataset codeparrot/codeparrot-clean --text_field content
    --log_every 200 --val_every 1000
    --enable_thinking_token --think_checkpointing
    --think_decision gate
    --think_prioritize_queue
    --think_queue_batch 4
    --think_queue_accum_steps 4
    --think_queue_accum_max_steps 32
    --think_queue_drain_target 128
    --think_lambda 0.1
    --think_lambda_start 0.5
    --think_gate_threshold 0.5
    --think_gate_threshold_start 0.05
    --think_explore_prob 0.02
    --think_explore_mode high_ce
    --think_explore_top_frac 0.1
    --think_curriculum_steps 2000
    --think_safety_max_depth 5
    --think_safety_max_depth_start 1
)

# 1. Control: fresh_tokens
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u experiments/train_lm.py \
    "${COMMON_ARGS[@]}" \
    --think_aux_normalize fresh_tokens \
    --save_ckpt checkpoints/think_sweep_fresh_tokens.pt \
    --tb_dir runs/tb/think_sweep_fresh_tokens \
    > runs/think_sweep_fresh_tokens.log 2>&1 &
echo "Launched control (fresh_tokens) on GPU 0"

# 2. aux_items, scale 0.25
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/train_lm.py \
    "${COMMON_ARGS[@]}" \
    --think_aux_normalize aux_items --think_aux_loss_scale 0.25 \
    --save_ckpt checkpoints/think_sweep_aux_items_s025.pt \
    --tb_dir runs/tb/think_sweep_aux_items_s025 \
    > runs/think_sweep_aux_items_s025.log 2>&1 &
echo "Launched aux_items scale 0.25 on GPU 1"

# Wait for one to finish before launching more? 
# Or launch all 4 if memory permits. 
# Each 217M model takes ~8-10GB. 5090 has 32GB. 
# We can easily run two per GPU.

# 3. aux_items, scale 0.5
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u experiments/train_lm.py \
    "${COMMON_ARGS[@]}" \
    --think_aux_normalize aux_items --think_aux_loss_scale 0.5 \
    --save_ckpt checkpoints/think_sweep_aux_items_s05.pt \
    --tb_dir runs/tb/think_sweep_aux_items_s05 \
    > runs/think_sweep_aux_items_s05.log 2>&1 &
echo "Launched aux_items scale 0.5 on GPU 0"

# 4. aux_items, scale 1.0
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/train_lm.py \
    "${COMMON_ARGS[@]}" \
    --think_aux_normalize aux_items --think_aux_loss_scale 1.0 \
    --save_ckpt checkpoints/think_sweep_aux_items_s10.pt \
    --tb_dir runs/tb/think_sweep_aux_items_s10 \
    > runs/think_sweep_aux_items_s10.log 2>&1 &
echo "Launched aux_items scale 1.0 on GPU 1"
