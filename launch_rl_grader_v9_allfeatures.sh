#!/bin/bash
# RL grader v9 — all features ON with validated defaults.
#
# All knobs in train_rl_grader.py were audited (2026-05-24) and the
# defaults were updated to the validated values from the v3..v8 arc.
# This launcher only overrides what's run-specific (path + step budget
# + the long-run KL strengthening).
#
# Features enabled (most by default):
#   ✓ DDP × 2 GPU (torchrun)
#   ✓ Adaptive curriculum (target tracks mean_p, NEW)
#   ✓ Variance-weighted EMA curriculum filter (foundation)
#   ✓ Iterative repair (re-roll failed attempts with error_text)
#   ✓ KL-to-reference anchor (kl_coef=0.15 for the long horizon)
#   ✓ PPO clip 0.1
#   ✓ Counterfactual advantage shaping
#   ✓ Ponder cost 0.001 (gate selectivity pressure, far below v1
#     collapse-zone)
#   ✓ Curriculum + repair stats logged per step
#
# Resume from v3_step25 (project best 17/164, clean slate).
#
# 500 steps × DDP × adaptive curriculum should yield the cleanest
# single-run benchmark of the full machinery. Expected wall ~10h.
#
# Best-effort prediction: adaptive curriculum + KL=0.15 + ponder=0.001
# avoid all three failure modes seen so far:
#   - v7b: variance-only sampler converged to narrow band, no real
#     learning signal (fixed by adaptive curriculum)
#   - v8: progressive ramp over-shot capability (fixed by adaptive)
#   - v1: gate collapse on excess ponder (this run uses 0.001, not 0.005)

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

nohup .venv/bin/torchrun \
    --nproc-per-node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:29500 \
    experiments/train_rl_grader.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v3_step25.pt \
    --save_ckpt checkpoints/rl_grader_phase_c_v9.pt \
    --steps 500 \
    --kl_coef 0.15 \
    --iterative_repair \
    --seed 0 \
    > runs/rl_grader_v9_allfeatures.log 2>&1 &
echo "Launched RL grader v9 all-features (DDP × 2 GPU) — PID $!"
echo "Watch: tail -f runs/rl_grader_v9_allfeatures.log"
