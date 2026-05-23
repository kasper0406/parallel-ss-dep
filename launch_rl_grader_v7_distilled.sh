#!/bin/bash
# RL grader on the QWEN-DISTILLED v7.1 SFT base (Phase 5).
#
# Base: checkpoints/sft_v7_pkm_film_distilled.pt — the breakthrough ckpt
# that scored 6/164 on HumanEval pass@1 (first non-zero result).
# This ckpt was distilled from Qwen 3.6 AWQ on ~14k (problem → CoT →
# code) traces, then SFT'd on that corpus with --with_thinking.
#
# Key differences from prior RL-grader runs:
#   --extract_code_block   The student emits CoT + ```python ... ```;
#                          the grader extracts only the python fence.
#                          Without this, grading would syntax_error
#                          on the CoT prose every rollout.
#   --max_gen 768          Distilled completions are ~600-800 tokens
#                          (CoT + code). Previous runs used 96 which
#                          would cut off the code block.
#   --total_think_budget 384  Cap total think tokens so we don't blow
#                          out memory with overlong rollouts.
#   --batch 2              Lower than the 4 used before — combined
#                          batch×n_group=8 rollouts at 768 tokens =
#                          6k+ tokens per step, keeps memory safe.
#   --ponder_cost 0.001    MUCH lower than the prior 0.005. The
#                          distilled student's CoT is genuinely
#                          productive (think_rate 0.27 at eval-time
#                          gives 6/164 pass@1), so we DON'T want
#                          quadratic ponder cost suppressing it.
#                          Even a tiny cost preserves "shorter is
#                          better when reward is equal" without
#                          discouraging useful thinking.
#   --ponder_warmup_steps 100  Longer warmup — give the policy time
#                          to find passing trajectories before any
#                          depth penalty kicks in.
#
# Pinned to GPU 1 (assumes era-of-experience still on GPU 0).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v7_pkm_film_distilled.pt \
    --save_ckpt checkpoints/rl_grader_v7_distilled.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch 2 \
    --grpo_n_group 4 \
    --lr 5e-6 \
    --max_gen 768 \
    --max_think_per_step 8 \
    --total_think_budget 384 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.9 \
    --min_emit_before_eos 30 \
    --clip_eps 0.2 \
    --ponder_cost 0.001 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 100 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_v7_distilled.log 2>&1 &
echo "Launched RL-grader v7-distilled on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v7_distilled.log"
