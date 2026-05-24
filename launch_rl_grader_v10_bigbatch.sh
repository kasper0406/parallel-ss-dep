#!/bin/bash
# RL grader v10 — activation checkpointing + bigger batch + all features.
#
# Memory optimization: activation checkpointing is now default-on in
# train_rl_grader.py. With AC, batch=4 fits comfortably alongside
# PKM + WM + KL reference forward + iterative repair on 32 GiB
# (smoke validated 2 steps at batch=4 + max_gen=384 + repair: 95 sec
# total, no OOM).
#
# Effective gradient batch per step:
#   DDP × 2 ranks × batch=4 × n_group=4 = 32 rollouts
#   vs v9: 2 × 2 × 4 = 16 rollouts (2× signal per step)
#   vs v3: 1 × 4 × 4 = 16 rollouts (2× signal per step)
#
# This should noticeably reduce the per-step reward variance (v9 had
# 8-rollout dispersion → ±0.10-0.20 swings on identical capability;
# 32-rollout dispersion should halve that).
#
# Tried but reverted: --policy_film_bypass (would skip K=3 FiLM in
# the policy forward, saving more activation memory). Rejected because
# rollouts run K=3 → bypass-during-policy creates a log-prob mismatch
# → PPO ratio drops to ~0.5 → all gradients clipped. The kill-switch
# from CLAUDE.md (_film_bypass at decode breaks T>0 sampling) blocks
# the consistent-bypass-everywhere fix. Flag kept off as default.
#
# Other defaults already validated and now baked into the parser:
#   kl_coef=0.05 (anchor), ponder_cost=0.001 (gate selectivity),
#   gate_floor=0.0 (don't disable thinking), clip_eps=0.1 (PPO stability),
#   max_gen=384, save_every=25, adaptive_curriculum=on.
#
# v10 launch overrides: only --steps + --kl_coef strengthening for the
# long horizon, plus --iterative_repair to keep the full machinery on
# (smoke showed repair occasionally produces a pass-lift from fresh
# SFT base — first time we've ever observed that).

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
    --save_ckpt checkpoints/rl_grader_phase_c_v10.pt \
    --steps 500 \
    --kl_coef 0.15 \
    --iterative_repair \
    --seed 0 \
    > runs/rl_grader_v10_bigbatch.log 2>&1 &
echo "Launched RL grader v10 bigbatch (DDP × 2 GPU + AC) — PID $!"
echo "Watch: tail -f runs/rl_grader_v10_bigbatch.log"
