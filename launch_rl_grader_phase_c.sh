#!/bin/bash
# Execution-grounded GRPO RL on the Phase-C SFT base (2026-05-22).
#
# Base: checkpoints/sft_phase_c_combined.pt — the v7-recipe SFT on the
# Chinchilla-completed Phase C pretrain. HumanEval 10/164, recall 98.2%,
# PKM load-bearing (ablation pkm_off -5). RL is the one untried lever
# for the coding headline.
#
# train_rl_grader: N rollouts/problem at temperature τ, code_grader.grade
# as the dense reward, GRPO advantages, PPO-clipped update.
#
# Config notes:
#  --extract_code_block : the SFT base emits CoT + ```python``` fences;
#      the grader must pull the fenced code before running it, else the
#      reward is structurally ~0.
#  --gate_floor 0.0 < --emit_threshold 0.5 : keeps thinking ALIVE in
#      rollouts. gate_floor >= emit_threshold silently forces never-think
#      (the train_rl_grader gate-saturation bug — see GEMINI.md).
#  ponder: quadratic + counterfactual + 50-step warmup — depth cost can
#      never worsen the task reward, ramps in after cold start.
#  --max_gen 384 : the first attempt at the default 96 STALLED (9 passes
#      by step 78, then dried up — a syntax_error wall). The distilled
#      base emits CoT prose THEN a ```python``` block; 96 emit tokens
#      truncate it before a complete block, so the grader sees mostly
#      syntax_error and GRPO gets no advantage variance. The HumanEval
#      eval that scored 10/164 used max_gen 800 — the model needs room.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/rl_grader_phase_c.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 800 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 5e-6 \
    --max_gen 384 \
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
    > runs/rl_grader_phase_c.log 2>&1 &
echo "Launched RL-grader on Phase C SFT base, GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/rl_grader_phase_c.log"
