#!/bin/bash
# RL-grader v7.1-SFT on EXPANDED dataset (super_combined = 3993 problems).
#
# Why this run:
#   The first RL-grader run (launch_rl_grader_v7_pkm_film.sh, 400 steps
#   on the 374-problem MBPP train split) produced 330 cumulative passes
#   during training but 0/164 on HumanEval — i.e. the "passes" were
#   memorized solutions, not transferred capability. Hypothesis: 374
#   problems isn't a wide enough distribution for GRPO to teach
#   generalisation; the model finds shortcuts that solve the training
#   set but don't compose.
#
# The expansion (3.6x — see experiments/code_grader.py load_mbpp_combined):
#   - mbpp_all   (974 = train+validation+test+prompt splits)
#   - mbpp_plus  (378 EvalPlus-augmented variants of MBPP)
#   = 1352 problems
#   HumanEval is NOT in the training mix (we eval on it).
#
# We initially tried adding LeetCode (super_combined = 3993 problems)
# but hit two issues: (a) OOM on the heavy-tailed LeetCode prompts
# (heavy import scaffolding pushed T_max past memory budget even at
# batch=2), and (b) the model was unable to even generate runnable code
# on the LeetCode distribution — reward stuck at the exec_error floor.
# Cleaner experiment to ask the targeted question ("does 3.6x problem
# diversity from the SAME family teach generalization?") with just MBPP
# variants where the prompt format is uniform and the base CAN at least
# produce runnable code.
#
# Step budget 800 (2x the original 400). At batch=4 problems/step that's
# 3200 problem-samples for 1352 unique problems — each problem averages
# 2.4 RL touches, vs ~4.3 on the original 374-problem MBPP run.
#
# All other knobs identical to the original v7.1 RL run for direct
# comparability. Pinned to GPU 1.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v7_pkm_film_thinking.pt \
    --save_ckpt checkpoints/rl_grader_v7_pkm_film_combined.pt \
    --dataset mbpp_combined \
    --steps 800 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 5e-6 \
    --max_gen 96 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.9 \
    --min_emit_before_eos 30 \
    --clip_eps 0.2 \
    --ponder_cost 0.005 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 50 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 100 \
    --seed 0 \
    > runs/rl_grader_v7_pkm_film_combined.log 2>&1 &
echo "Launched RL-grader v7.1 (combined dataset) on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v7_pkm_film_combined.log"
