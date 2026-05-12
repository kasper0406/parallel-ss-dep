#!/bin/bash
# Long-context recall sweep — *saturated* regime.
#
# Holds K_pairs=128 (above DeltaNet's state saturation for d=128/L=4) and
# sweeps T so we measure recall at increasing context where the bounded
# state matrix is structurally challenged.  DN+memory should rescue.
#
# Drops softmax for now — at this scale + DN-tuned LR it fails to train,
# and the architectural claim is DN+mem > DN baseline (not chasing
# Transformer parity, which is a different experiment).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p runs/longctx_recall
# Clean prior partial results so the aggregator doesn't mix old runs.
rm -f runs/longctx_recall/T*_*.log

SHARED=(
    --d_model 128 --n_layers 4 --n_heads 4 --d_head 32
    --n_pairs 128 --vocab 512 --steps 6000
    --lr 3e-3 --log_every 1000 --seed 0
    --arches deltanet
)

# Each row: T  batch_size
ROWS=(
    "512   128"
    "1024  64"
    "2048  32"
    "4096  16"
    "8192  4"
)

run_one() {
    local T=$1 batch=$2 use_mem=$3 gpu=$4
    local tag="deltanet"
    local extra=""
    if [ "$use_mem" = "1" ]; then
        tag="deltanet_mem"
        extra="--use_memory --mem_size 256"
    fi
    local out="runs/longctx_recall/T${T}_${tag}.log"
    echo "[$(date +%H:%M:%S)] GPU $gpu: $tag T=$T batch=$batch"
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u experiments/train_mqar.py \
        "${SHARED[@]}" --T "$T" --batch "$batch" $extra \
        > "$out" 2>&1
    echo "[$(date +%H:%M:%S)] DONE: $tag T=$T"
}

for row in "${ROWS[@]}"; do
    read -r T batch <<< "$row"
    # GPU 0: deltanet baseline.  GPU 1: deltanet + memory.  Run in parallel.
    run_one "$T" "$batch" 0 0 &
    PID0=$!
    run_one "$T" "$batch" 1 1 &
    PID1=$!
    wait "$PID0" "$PID1"
done

echo
echo "Sweep complete. Results:"
.venv/bin/python scripts/aggregate_longctx_recall.py runs/longctx_recall
