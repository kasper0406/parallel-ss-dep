#!/bin/bash
# RL grader v11 — v10 continuation with parallel grading (Fix 1).
#
# v10 was killed at step 25 (best ckpt saved) to land Fix 1: parallel
# grading via ThreadPoolExecutor. With 48 rollouts/step and grading
# previously sequential at ~50-300ms per call, the per-step iteration
# wall-clock should drop noticeably (the test showed 7.9× on the
# grading phase alone; that phase is ~30% of step time → expected
# ~20-25% wall-clock improvement per step).
#
# Resumes from v10_step25 to preserve 25 steps of curriculum training
# + the curriculum_state_dict. All other defaults inherited from the
# trainer (now-validated values per the 2026-05-24 audit).
#
# 475 remaining steps × estimated 50 sec/step DDP = ~6.6h.

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
    --load_ckpt checkpoints/rl_grader_phase_c_v11b_step75.pt \
    --save_ckpt checkpoints/rl_grader_phase_c_v11c.pt \
    --steps 425 \
    --batch 3 \
    --kl_coef 0.15 \
    --iterative_repair \
    --seed 0 \
    > runs/rl_grader_v11_resume.log 2>&1 &
echo "Launched RL grader v11 resume (DDP × 2 GPU + parallel grading) — PID $!"
echo "Watch: tail -f runs/rl_grader_v11_resume.log"
