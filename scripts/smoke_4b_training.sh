#!/bin/bash
# Smoke test: build the 4B student, do 5 training steps + 1 val step on the
# existing 1.5M-token teacher data. Verifies the architecture fits in 32GB
# at batch=2 grad_accum=2 and that the training loop runs end-to-end before
# committing to the long full pilot.
set -e
cd /home/knielsen/ml/parallel-ss-dep-distill

mkdir -p logs/distill_pilot_full

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=logs/distill_pilot_full/smoke_4b_training.log
echo "Smoke test: 4B build + 5 steps + 1 val. Logging to $LOG"

/home/knielsen/ml/parallel-ss-dep/.venv/bin/python -u \
    experiments/distill_pilot_full.py \
        --shards data/distill_pilot_1M \
        --mode kl_ce --alpha 0.9 --top_k 20 \
        --d_model 2048 --n_heads 32 --d_head 64 --n_layers 32 \
        --batch 2 --grad_accum 2 \
        --steps 5 \
        --optimizer muon --lr_muon 1e-3 --lr 3e-4 \
        --warmup_steps 2 \
        --log_every 1 --val_every 5 --val_chunks 8 \
        --seed 0 \
    2>&1 | tee "$LOG"
