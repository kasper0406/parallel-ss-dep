#!/bin/bash
# DPO v2 — fix the over-training that killed v1 (2026-05-27).
#
# Decision D5 from AUTONOMY_DECISIONS.md.
#
# v1 problem (May 23):
#   - 3908 steps (2 epochs × 1954 pairs) at β=0.1, LR 5e-6
#   - Final winrate 0.97, log_ratio +100 to +250 (massive drift)
#   - HumanEval result: 9/164 (regressed from base 16)
#   - No intermediate ckpts → couldn't capture early sweet spot
#
# v2 fixes:
#   --beta 0.3            : 3× stronger KL anchor (was 0.1)
#                           v1 hit log_ratio +200 — way too far from ref
#   --epochs 1            : half v1's run length
#   --lr 2e-6             : 40% of v1's LR — slower drift per step
#   --save_every 250      : NEW (newly-added flag) — 7 snapshots over the
#                           run. The whole point of v2 is to be able to
#                           pick the early-sweet-spot ckpt before over-fit.
#
# Base = v2_step300 (16/164, current project best).
# Reference = same (auto-loaded by train_dpo.py).
# Data = data/rejection_v2_step300_all.jsonl (1260 problems, 1958 pairs).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_dpo.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v2_step300.pt \
    --rollouts data/rejection_v2_step300_all.jsonl \
    --save_ckpt checkpoints/dpo_v2.pt \
    --beta 0.3 \
    --epochs 1 \
    --lr 2e-6 \
    --max_pairs_per_problem 4 \
    --log_every 25 \
    --save_every 250 \
    --seed 0 \
    > runs/dpo_v2.log 2>&1 &
echo "Launched DPO v2 — PID $!"
echo "Watch: tail -f runs/dpo_v2.log"
echo
echo "EVAL STRATEGY:"
echo "  Wait for snapshots at step 250, 500, 750, 1000, 1250, 1500, 1750"
echo "  Eval each (~25min on GPU 0), pick best."
echo "  Decision: best ckpt > 16/164 → DPO works as middle-of-training picker."
echo "            all < 16/164 → DPO destabilizes regardless of duration → pivot."
