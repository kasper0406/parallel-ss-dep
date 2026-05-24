#!/bin/bash
# RL grader on the combined-SFT-v2 base (FIX A actually trained-in).
#
# Base: checkpoints/sft_v7_pkm_film_combined_v2.pt
#   - distilled from Qwen 3.6 on ~38k pairs (mbpp+leetcode+magicoder+
#     codefeedback) + 12.5k synthetic-memory tasks
#   - trained with --mem_write_only_at_think (FIX A) actually active
#   - trained with --future_emb_loss_weight 0.05 (cosine future-emb pred)
#
# Differences from the earlier failed RL attempts:
#   --gate_floor 0.5         Prevent τ=0.9 sampling from collapsing the
#                            gate to constant-think (the failure mode that
#                            killed the prior RL attempt on distilled SFT).
#                            Mirrors the SFT-time training floor.
#   --ponder_cost 0.001      LOWER than the prior 0.005. The combined SFT
#                            now has selective, productive thinking; don't
#                            suppress it with a heavy depth penalty.
#   --extract_code_block     The model emits CoT + ```python ... ```;
#                            extract the python fence and grade only the
#                            code (without this, grader syntax_errors on
#                            the CoT prose).
#   --max_gen 768            Distilled completions are ~600-800 tokens;
#                            short max_gen would cut off the code block.
#   --total_think_budget 384 Cap to keep rollouts memory-safe.
#   --batch 2                Lower than 4 — distilled prompts are longer
#                            than legacy MBPP-only prompts, so R=8 fits
#                            without OOM.
#
# Dataset: mbpp_combined (1352 problems — proven to work, fast).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v7_pkm_film_combined_v2.pt \
    --save_ckpt checkpoints/rl_grader_v7_combined_v2.pt \
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
    --gate_floor 0.5 \
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
    > runs/rl_grader_v7_combined_v2.log 2>&1 &
echo "Launched RL-grader v7-combined-v2 on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v7_combined_v2.log"
