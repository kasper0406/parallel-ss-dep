#!/bin/bash
# Bigger-model variant of launch_longctx_recall.sh.
#
# Tests whether memory's envelope-extension at d=128 (T=512 -> T=1024)
# persists at d=256/L=8 (expected T=1024 -> T=2048 if the architecture
# scales consistently).
#
# Keeps K_pairs=128 fixed for apples-to-apples vs the d=128 sweep.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p runs/longctx_d256

SHARED=(
    --d_model 256 --n_layers 8 --n_heads 8 --d_head 32
    --n_pairs 128 --vocab 512 --steps 10000
    --lr 3e-3 --log_every 1000 --seed 0
    --arches deltanet
)

# Each row: T  batch_size
# Batch reduced relative to d=128 to fit ~4x bigger activations.
ROWS=(
    "1024  32"
    "2048  12"
    "4096  4"
)

run_one() {
    local T=$1 batch=$2 use_mem=$3 gpu=$4
    local tag="deltanet"
    local extra=""
    if [ "$use_mem" = "1" ]; then
        tag="deltanet_mem"
        extra="--use_memory --mem_size 256"
    fi
    local out="runs/longctx_d256/T${T}_${tag}.log"
    echo "[$(date +%H:%M:%S)] GPU $gpu: $tag T=$T batch=$batch"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u experiments/train_mqar.py \
        "${SHARED[@]}" --T "$T" --batch "$batch" $extra \
        > "$out" 2>&1
    echo "[$(date +%H:%M:%S)] DONE: $tag T=$T"
}

for row in "${ROWS[@]}"; do
    read -r T batch <<< "$row"
    run_one "$T" "$batch" 0 0 &
    PID0=$!
    run_one "$T" "$batch" 1 1 &
    PID1=$!
    wait "$PID0" "$PID1"
done

echo
echo "d=256 sweep complete:"
.venv/bin/python scripts/aggregate_longctx_recall.py runs/longctx_d256
