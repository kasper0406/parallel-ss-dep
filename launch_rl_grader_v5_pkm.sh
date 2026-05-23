#!/bin/bash
# RL-grader on v5-pkm SFT base — immediate gate-behavior diagnostic.
#
# Why this run (not v7.1)?
#   The v7.1-pkm-film SFT ckpt doesn't exist yet (running in parallel on
#   GPU 1, ~3 hrs). The v5-pkm-SFT ckpt is the canonical "gate collapsed
#   to always-emit" example (post-SFT mean gate = 0.978, think_rate = 0%).
#   This makes it the perfect substrate to ask: "does GRPO with quadratic
#   ponder cost + 50-step warmup recover the gate at all?"
#
# Expected behavior we're watching for:
#   - step 1-50    (ponder warmup):  gate may stay collapsed; reward should
#                                    not move much; think_rate should stay
#                                    very low (the gate floor is what's
#                                    keeping at least some thinking).
#   - step 50-150  (ponder live):    if the gate is recoverable, depth_mean
#                                    should fall to a sensible value (<10
#                                    on average), think_rate moves above 0
#                                    on hard prompts (the ones where reward
#                                    rewards thinking).
#   - if gate STAYS collapsed:       the entropy-aux signal didn't survive
#                                    SFT, and we need a gate-bias term or
#                                    explicit "force-think" exploration.
#
# Knobs (from `train_rl_grader.py` validated defaults):
#   --grpo_n_group 4         4 rollouts per problem (B=4 problems × 4 = 16)
#   --max_gen 96             cap completion length
#   --temperature 0.9        diverse rollouts for advantage variance
#   --gate_floor 0.0         no eval-time floor — let the policy decide
#   --emit_threshold 0.5     gate decision threshold at inference
#   --ponder_cost 0.005      quadratic cost per think token
#   --ponder_warmup_steps 50 don't penalise depth until step 50
#   --counterfactual         clamp task component so thinking can't worsen
#                            reward, only the cost term applies
#   --lr 5e-6                gentle policy LR
#
# Pinned to GPU 0 (SFT v7.1 on GPU 1).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v5_pkm_thinking.pt \
    --save_ckpt checkpoints/rl_grader_v5_pkm.pt \
    --steps 400 \
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
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_v5_pkm.log 2>&1 &
echo "Launched RL-grader v5-pkm on GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v5_pkm.log"
