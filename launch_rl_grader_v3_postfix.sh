#!/bin/bash
# RL grader v3 — continue from v2 step-300 (project best, 16/164) with
# the `_film_bypass=True` rollout bug FIXED (commit 0d29d94).
#
# Hypothesis: v2's collapse at step-350 (v1 path) and the step-300→400
# regression (-2 HumanEval) were partly driven by the buggy rollouts
# providing degraded gradient signal — the model was optimising against
# a sampling distribution it would never see at greedy eval. Fixed
# rollouts use the FiLM-ON forward path, matching how the model was
# trained, so the policy-gradient signal should be cleaner and the
# headroom past 16/164 may now be reachable.
#
# Recipe: same KL-stable settings as v2 (--kl_coef 0.05, --lr 2e-6,
# --clip_eps 0.1, --temperature 0.7, no ponder cost). 200 steps —
# half the v2 budget — because we're resuming from the peak and want
# to catch any further peak before drift. --save_every 25 with the
# step-suffixed save patch keeps every milestone for HumanEval probing.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v2_step300.pt \
    --save_ckpt checkpoints/rl_grader_phase_c_v3.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 200 \
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
    --save_every 25 \
    --seed 0 \
    > runs/rl_grader_v3_postfix.log 2>&1 &
echo "Launched RL grader v3 (post-fix, resume from v2 step-300) GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v3_postfix.log"
