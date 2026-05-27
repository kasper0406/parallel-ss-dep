#!/bin/bash
# RL grader v4 — stricter LR from v3 step25 (new project best, 17/164).
#
# v3 peaked at step25 then immediately dropped back to the 14-15 band.
# The hypothesis is that lr=2e-6 at batch=4×grpo_n_group=4 (16 rollouts/
# step) is too aggressive past the v2 baseline: each gradient step has
# high variance from the tiny batch, and at this LR even one bad step
# can push past the local optimum we found at step25.
#
# Stricter settings: lr=5e-7 (4× smaller), 60 steps with save_every=10
# so we catch any further peak in fine granularity. KL/clip/temp held
# at the same KL-stable v2 recipe.
#
# This is a CONFIRMATION-style run — small budget, lots of probes.
# The REAL fix is bigger batches (see follow-on launch script).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v3_step25.pt \
    --save_ckpt checkpoints/rl_grader_phase_c_v4.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 60 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 5e-7 \
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
    --save_every 10 \
    --seed 1 \
    > runs/rl_grader_v4_strict.log 2>&1 &
echo "Launched RL grader v4 (strict-LR from v3 step25) GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v4_strict.log"
