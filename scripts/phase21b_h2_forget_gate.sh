#!/usr/bin/env bash
# Phase 21b — H2 (forget-gate redundancy) test.
#
# Adds a per-token learnable forget gate to plain DeltaNet (no output gate)
# and re-runs Phase 17's sparse-(2, 28) FiLM ablation. If the FiLM lift
# drops from plain DN's -3.1 % toward GDP's -1.9 %, H2 is supported:
# the cross-cell pattern is forget-gate redundancy with FiLM.
#
# 2 runs:
#   1. DN+forget-gate baseline (no FiLM)
#   2. DN+forget-gate + sparse-(2, 28) FiLM
#
# Both pinned to GPU 1. Phase 17 plain-DN reference (PPL 51.00 / 49.40,
# alpha = -0.054) is reused from RESULTS.md for comparison.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG_DIR=/home/knielsen/ml/parallel-ss-dep/logs/phase21b
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
PY=.venv/bin/python

# Match Phase 17 exactly.
COMMON="--T 512 --batch 8 --steps 5000 --log_every 200 --val_every 1000 --lr 3e-4 --seed 0"
DATA="--dataset codeparrot/codeparrot-clean --text_field content"
NL=30
DM=576
NH=9
DH=64

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

# 1. DN + forget-gate baseline (no FiLM).
run_one phase21b_dn_forgetgate_baseline \
    --arch deltanet_forgetgate $COMMON $DATA \
    --d_model $DM --n_heads $NH --d_head $DH --n_layers $NL

# 2. DN + forget-gate + sparse-(2, 28) FiLM.
run_one phase21b_dn_forgetgate_film \
    --arch deltanet_forgetgate $COMMON $DATA \
    --d_model $DM --n_heads $NH --d_head $DH --n_layers $NL \
    --feedback film --feedback_pairs "2,28"

echo "[$(date +%H:%M:%S)] all phase21b runs complete"
