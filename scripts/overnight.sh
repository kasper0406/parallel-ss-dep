#!/usr/bin/env bash
# Overnight experiment master script.
# Runs Dyck on GPU 0 and code-PPL ratio ablation on GPU 1 in parallel.
# Each track runs sequentially on its assigned GPU.
#
# Logs are written to /tmp/overnight/*.log; a summary is appended to
# OVERNIGHT_REPORT.md after each run.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p /tmp/overnight
LOG_DIR=/tmp/overnight
REPORT=$(pwd)/OVERNIGHT_REPORT.md
START=$(date +%s)

source .venv/bin/activate

echo "# Overnight run starting $(date -Iseconds)" >> "$REPORT"
echo "" >> "$REPORT"

run_with_log() {
    # $1 = label, rest = command
    local label="$1"; shift
    local logfile="$LOG_DIR/${label}.log"
    echo "[$(date +%H:%M:%S)] starting $label  (log: $logfile)"
    # `command 2>&1` and tee so we still get exit code from the command.
    if "$@" > "$logfile" 2>&1; then
        echo "[$(date +%H:%M:%S)] OK $label"
        echo "## $label  (success)" >> "$REPORT"
    else
        echo "[$(date +%H:%M:%S)] FAIL $label (exit $?)"
        echo "## $label  (FAILED)" >> "$REPORT"
    fi
    echo '' >> "$REPORT"
    echo '```' >> "$REPORT"
    tail -25 "$logfile" >> "$REPORT"
    echo '```' >> "$REPORT"
    echo '' >> "$REPORT"
}

# ============================================================================
# GPU 0 track: Dyck depth-tracking sweep.
# ============================================================================
gpu0_track() {
    export CUDA_VISIBLE_DEVICES=0

    # Dyck T=128 — both arches at small-model scale.
    run_with_log dyck_T128_deltanet \
        python -u experiments/train_dyck.py \
            --arches deltanet --T 128 --steps 5000 --batch 256 \
            --d_model 128 --n_heads 4 --d_head 32 --n_layers 4 \
            --log_every 500
    run_with_log dyck_T128_hybrid_v2 \
        python -u experiments/train_dyck.py \
            --layers ortho,deltanet,ortho,deltanet \
            --T 128 --steps 5000 --batch 256 \
            --d_model 128 --n_heads 4 --d_head 32 \
            --log_every 500

    # Dyck T=512 — long-T test, the one that distinguishes architectures.
    run_with_log dyck_T512_deltanet \
        python -u experiments/train_dyck.py \
            --arches deltanet --T 512 --steps 5000 --batch 128 \
            --d_model 128 --n_heads 4 --d_head 32 --n_layers 4 \
            --log_every 500
    run_with_log dyck_T512_hybrid_v2 \
        python -u experiments/train_dyck.py \
            --layers ortho,deltanet,ortho,deltanet \
            --T 512 --steps 5000 --batch 128 \
            --d_model 128 --n_heads 4 --d_head 32 \
            --log_every 500
}

# ============================================================================
# GPU 1 track: 135M code-PPL ratio ablation.
# Dataset: bigcode/the-stack-smol Python split (small, fast to download).
# ============================================================================
gpu1_track() {
    export CUDA_VISIBLE_DEVICES=1

    # Common args for the 135M scale.
    DM=576; NH=9; DHEAD=64; NL=30
    COMMON="--T 512 --batch 8 --steps 5000 --log_every 200 --val_every 1000 --lr 3e-4"
    SHAPE="--d_model $DM --n_heads $NH --d_head $DHEAD --n_layers $NL"
    DATA="--dataset codeparrot/codeparrot-clean --text_field content"

    # Pure DeltaNet baseline on code.
    run_with_log code_135M_deltanet \
        python -u experiments/train_lm.py --arch deltanet $COMMON $SHAPE $DATA

    # Hybrid 50/50 (the v2 we already validated on TinyStories).
    run_with_log code_135M_hybrid_50_50 \
        python -u experiments/train_lm.py --arch hybrid $COMMON $SHAPE $DATA

    # Hybrid 25/75 (Qwen3-Next-style, deltanet-heavy).
    run_with_log code_135M_hybrid_25_75 \
        python -u experiments/train_lm.py --arch hybrid_25_75 $COMMON $SHAPE $DATA

    # Hybrid 75/25 (ortho-heavy, probably worse but worth a data point).
    run_with_log code_135M_hybrid_75_25 \
        python -u experiments/train_lm.py --arch hybrid_75_25 $COMMON $SHAPE $DATA
}

gpu0_track &
PID0=$!
gpu1_track &
PID1=$!

wait "$PID0"
wait "$PID1"

END=$(date +%s)
DURATION=$((END - START))
echo "" >> "$REPORT"
echo "Total wall-clock: ${DURATION}s ($(( DURATION / 60 )) min)" >> "$REPORT"
echo "Done at $(date -Iseconds)"
