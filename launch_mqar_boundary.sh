#!/bin/bash
# MQAR memory-architecture boundary sweep.
#
# Goal: for a small DeltaNet (d_model=64, L=2, n_heads=2, d_head=32), map
# where bounded working memory starts to matter — i.e. which (T, K_pairs)
# regimes drive the baseline below saturation and let the memory module
# rescue recall.
#
# We pin model size at the size where T=512/K=128 is decisively a memory
# win, then sweep K_pairs and T around that point.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.

mkdir -p runs/mqar_sweep

SHARED=(
    --arches deltanet
    --d_model 64 --n_layers 2 --n_heads 2 --d_head 32
    --batch 128 --lr 3e-3 --steps 6000 --log_every 1000 --seed 0
)

# Pairs (T, K). Vocab needs to be >= 2*K + room for distinct query keys.
# A safe rule: vocab = 4*K (gives clean separation).
declare -a CONFIGS=(
    "256  16    64"
    "256  32   128"
    "256  64   256"
    "256 128   512"
    "512  64   256"
    "512 128   512"
    "512 256  1024"
)

run_pair() {
    local T=$1 K=$2 V=$3 gpu=$4 use_mem=$5
    local tag="T${T}_K${K}"
    local memtag="off"; [ "$use_mem" = "1" ] && memtag="on"
    local extra=""
    [ "$use_mem" = "1" ] && extra="--use_memory --mem_size $((K * 2))"
    CUDA_VISIBLE_DEVICES=$gpu nohup .venv/bin/python -u experiments/train_mqar.py \
        "${SHARED[@]}" --T "$T" --n_pairs "$K" --vocab "$V" $extra \
        > "runs/mqar_sweep/${tag}_mem${memtag}.log" 2>&1
}

# Run two configs at a time, one per GPU; mem_on on GPU 0, mem_off on GPU 1.
for cfg in "${CONFIGS[@]}"; do
    read -r T K V <<< "$cfg"
    echo "=== T=$T  K=$K  V=$V ==="
    run_pair "$T" "$K" "$V" 0 1 &
    PID0=$!
    run_pair "$T" "$K" "$V" 1 0 &
    PID1=$!
    wait "$PID0" "$PID1"
done

echo
echo "Sweep complete. Aggregating results..."
.venv/bin/python scripts/aggregate_mqar_sweep.py runs/mqar_sweep
