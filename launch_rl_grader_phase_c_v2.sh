#!/bin/bash
# Stable execution-grounded GRPO RL — v2 (2026-05-23).
#
# Why a v2: v1 (launch_rl_grader_phase_c.sh) hit 14/164 HumanEval at
# step-100, plateaued through step-300, then catastrophically collapsed
# around step 350 when the ponder cost finally bit and the model's
# always-on max-depth thinking dropped. Without a KL-to-reference
# constraint, the policy drifted off the SFT-distilled output format
# entirely. The full diagnosis is in the GEMINI.md update for this run.
#
# v2 stability recipe (each lever is well-known):
#  1. --kl_coef 0.05      : NEW — KL-to-reference penalty (the principled
#                            stability mechanism missing from v1). Holds
#                            the policy close to the SFT base in
#                            output-distribution space, preventing the
#                            arbitrary-drift collapse v1 hit.
#  2. --ponder_cost 0     : NO depth pressure. The proximate trigger of
#                            v1's collapse — this model's CoT-+-code
#                            output format is fragile to depth changes.
#  3. --lr 2e-6           : Half v1's LR (was 5e-6) — slower drift.
#  4. --clip_eps 0.1      : Tighter PPO clip (was 0.2) — smaller policy
#                            steps.
#  5. --temperature 0.7   : Lower than v1's 0.9 — less off-policy data,
#                            tighter rollout/learning loop.
#  6. --steps 400         : Half v1's 800 — the lift was front-loaded.
#  7. --save_every 50     : More granular snapshots so we can locate
#                            the peak HumanEval ckpt without
#                            checkpoint-overwrite races.
#
# Base: sft_phase_c_combined.pt (the v7-recipe SFT on the Phase C
# Chinchilla-completed pretrain). Reference policy = same — KL pulls
# the policy toward the SFT base while PPO surrogate pushes toward
# higher MBPP reward; the balance determines where it lands.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/rl_grader_phase_c_v2.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.05 \
    --ponder_cost 0.0 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 50 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_phase_c_v2.log 2>&1 &
echo "Launched RL-grader v2 (stable, KL+lr+clip+temp+nopond) on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/rl_grader_phase_c_v2.log"
