#!/bin/bash
# Full fair grid for the bf16-regime DeltaNet-preconditioner probe.
# 2 head-configs x 2 regimes x 2 arms x 3 matrix-LRs x 3 seeds = 72 runs.
# GPU 1 ONLY.
set -u
export PYTHONPATH="${PYTHONPATH:-}:."
cd /home/knielsen/ml/parallel-ss-dep
OUT=runs/deltanet_bf16/grid
mkdir -p "$OUT/logs"
N=0
for cfg in "4 64" "8 32"; do
  set -- $cfg; NH=$1; DH=$2
  for regime in fp32 bf16; do
    for arm in muon perhead; do
      for lr in 5e-3 1e-2 2e-2; do
        for seed in 0 1 2; do
          tag=${arm}_${regime}_nh${NH}_lr${lr}_s${seed}
          if [ -f "$OUT/$tag.json" ]; then echo "skip $tag"; continue; fi
          N=$((N+1))
          echo "[$(date +%H:%M:%S)] run $N: $tag"
          CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_deltanet_bf16_lm.py \
            --arm "$arm" --regime "$regime" --n_heads "$NH" --d_head "$DH" \
            --lr_mat "$lr" --seed "$seed" --steps 600 --eval_every 20 \
            --tag "$tag" --out_dir "$OUT" > "$OUT/logs/$tag.log" 2>&1
        done
      done
    done
  done
done
echo "[$(date +%H:%M:%S)] GRID DONE ($N runs executed)"
