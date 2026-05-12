#!/bin/bash
# Diagnostic round: short (500-step) RL runs to characterise bounded
# working-memory dynamics on the thinking gate. Two parallel runs:
#   GPU 0 — memory ON  (verify the new path bootstraps)
#   GPU 1 — memory OFF (gate-only control)
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.

mkdir -p runs checkpoints

SHARED_ARGS=(
    --steps 500
    --batch 1
    --grpo_n_group 8
    --lr 5e-6
    --grpo_kl_beta 0.05
    --grpo_ponder_cost 0.01
    --max_depth 5
    --T 512
    --max_T 0
    --min_decision_pos 16
    --think_checkpointing
    --load_ckpt checkpoints/think_sweep_fresh_tokens_final.pt
)

# 1. Memory ON  → GPU 0
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u experiments/train_rl.py \
    "${SHARED_ARGS[@]}" \
    --use_memory \
    --mem_size 1024 \
    --save_ckpt checkpoints/diag_mem_on.pt \
    --tb_dir runs/tb/diag_mem_on \
    > runs/diag_mem_on.log 2>&1 &
echo "Launched diag_mem_on on GPU 0 (PID $!)"

# 2. Memory OFF (control)  → GPU 1
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/train_rl.py \
    "${SHARED_ARGS[@]}" \
    --save_ckpt checkpoints/diag_mem_off.pt \
    --tb_dir runs/tb/diag_mem_off \
    > runs/diag_mem_off.log 2>&1 &
echo "Launched diag_mem_off on GPU 1 (PID $!)"

echo ""
echo "Watch with:"
echo "  tail -f runs/diag_mem_on.log"
echo "  tail -f runs/diag_mem_off.log"
