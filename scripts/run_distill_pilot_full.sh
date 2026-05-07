#!/bin/bash
# DN-4B distillation pilot — Phase B (training).
#
# Trains a ~3B plain-DN student on the cached teacher corpus produced by
# Phase A (`scripts/run_teacher_gen_full.sh`). KL+CE α=0.9, 90% CE +
# 10% KL — the working point validated in `DISTILL_PILOT_REPORT.md`.
# Held-out PPL is evaluated separately (the codeparrot val tail).
#
# Usage:
#   ./scripts/run_distill_pilot_full.sh
#
# Pin to GPU 0 only.
set -e
cd /home/knielsen/ml/parallel-ss-dep-distill

mkdir -p logs/distill_pilot_full
mkdir -p checkpoints

# Defaults (the brief). Can be overridden via env.
SHARDS="${SHARDS:-data/distill_pilot_30M}"
STEPS="${STEPS:-30000}"
ALPHA="${ALPHA:-0.9}"
TOP_K="${TOP_K:-20}"
D_MODEL="${D_MODEL:-2048}"
N_HEADS="${N_HEADS:-32}"
D_HEAD="${D_HEAD:-64}"
N_LAYERS="${N_LAYERS:-32}"
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SEED="${SEED:-0}"
LOG="${LOG:-logs/distill_pilot_full/dn_4B_train.log}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Distillation pilot Phase B (DN-4B) — config"
echo "  shards   = $SHARDS"
echo "  d_model  = $D_MODEL  n_heads=$N_HEADS  d_head=$D_HEAD  n_layers=$N_LAYERS"
echo "  batch    = $BATCH (grad_accum=$GRAD_ACCUM, eff $((BATCH * GRAD_ACCUM)))"
echo "  steps    = $STEPS  alpha=$ALPHA  top_k=$TOP_K  seed=$SEED"
echo "  log      = $LOG"

nohup /home/knielsen/ml/parallel-ss-dep/.venv/bin/python -u \
    experiments/distill_pilot_full.py \
        --shards "$SHARDS" \
        --mode kl_ce --alpha $ALPHA --top_k $TOP_K \
        --d_model $D_MODEL --n_heads $N_HEADS \
        --d_head $D_HEAD --n_layers $N_LAYERS \
        --batch $BATCH --grad_accum $GRAD_ACCUM \
        --steps $STEPS \
        --optimizer muon --lr_muon 1e-3 --lr 3e-4 \
        --weight_decay 0.1 --grad_clip 1.0 \
        --warmup_steps 200 \
        --log_every 500 --val_every 2500 --val_chunks 128 \
        --seed $SEED \
        --save_ckpt checkpoints/dn_4B_distilled_qwen3p6.pt \
        --save_metrics logs/distill_pilot_full/dn_4B_train.json \
        --save_jsonl  logs/distill_pilot_full/dn_4B_train.jsonl \
    > "$LOG" 2>&1 &

PID=$!
echo "PID $PID logging to $LOG"
echo $PID > logs/distill_pilot_full/dn_4B_train.pid
