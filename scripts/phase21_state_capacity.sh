#!/usr/bin/env bash
# Phase 21 — state-capacity test for sparse-FiLM lift.
#
# Vary plain DeltaNet's d_head (per-layer state size) and measure the
# sparse-(2, 28) FiLM lift at each setting. d_model held fixed at 576.
#
# 4 runs total (Phase 17 d_head=64 baseline + FiLM are reused from RESULTS.md).
#
# All runs pinned to GPU 1.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=/home/knielsen/ml/parallel-ss-dep/logs/phase21
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python

COMMON="--T 512 --batch 8 --steps 5000 --log_every 200 --val_every 1000 --lr 3e-4 --seed 0"
DATA="--dataset codeparrot/codeparrot-clean --text_field content"
NL=30
DM=576

run_one() {
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo "[$(date +%H:%M:%S)] starting $label"
    "$PY" -u experiments/train_lm.py "$@" > "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] OK $label"
    else
        echo "[$(date +%H:%M:%S)] FAIL $label (exit $rc)"
    fi
    return $rc
}

# Small state: d_head=32, n_heads=18 -> per-layer state 18*32*32=18,432
run_one phase21_dhead32_baseline \
    --arch deltanet $COMMON $DATA \
    --d_model $DM --n_heads 18 --d_head 32 --n_layers $NL

run_one phase21_dhead32_film \
    --arch deltanet $COMMON $DATA \
    --d_model $DM --n_heads 18 --d_head 32 --n_layers $NL \
    --feedback film --feedback_pairs "2,28"

# Large state: d_head=144, n_heads=4 -> per-layer state 4*144*144=82,944
run_one phase21_dhead144_baseline \
    --arch deltanet $COMMON $DATA \
    --d_model $DM --n_heads 4 --d_head 144 --n_layers $NL

run_one phase21_dhead144_film \
    --arch deltanet $COMMON $DATA \
    --d_model $DM --n_heads 4 --d_head 144 --n_layers $NL \
    --feedback film --feedback_pairs "2,28"

echo "[$(date +%H:%M:%S)] all phase21 runs complete"
