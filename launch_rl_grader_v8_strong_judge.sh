#!/bin/bash
# "Make it STRONG" grader-RL + LLM ranker + memory optimizations (2026-05-30).
# Reward signal — not KL — was the binding constraint on the flat strong run:
# the weak base mostly emits syntax/exec errors, so GRPO groups are tied at ~0
# (zero variance => zero gradient). Two composing fixes for intra-group
# variance:
#   1. grpo_n_group 8 (was 4): MORE SHOTS per problem => higher P(a group
#      catches a partial/pass). Now free of memory cost thanks to the
#      microbatched policy update + emit-position lm_head gather.
#   2. --llm_judge: ranks STILL-tied groups (all-syntax_error) by plausibility
#      into a within-tier advantage — the only signal for groups execution
#      can't separate. Fires only when group_is_variance_bearing is False.
#   + temperature 0.9 (was 0.7): more diverse rollouts => more variance; safe
#     now that adaptive KL + the judge are in play.
#   + policy_micro_chunk 4: caps policy-update activation memory at 4 rollout
#     rows regardless of the ~32 total/step (microbatch, exact gradient).
# Keeps the strong machinery: adaptive KL (target 0.15) + progressive
# curriculum (0.7 -> 0.2). Judge = Qwen2.5-Coder-3B-AWQ at localhost:8000,
# co-resident on the trainer GPU (server started separately, capped ~6.5 GB).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

LOAD=${LOAD:-checkpoints/sft_v8_combined.pt}
SAVE=${SAVE:-checkpoints/rl_grader_v8_strong_judge.pt}
LOG=${LOG:-runs/rl_grader_v8_strong_judge.log}
EXTRA=${EXTRA:-}

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt "$LOAD" \
    --save_ckpt "$SAVE" \
    --dataset mbpp_combined --extract_code_block --activation_checkpointing \
    --steps ${STEPS:-600} --batch ${BATCH:-2} --grpo_n_group ${NGROUP:-8} \
    --policy_micro_chunk ${MICRO:-4} --lr ${LR:-2e-6} \
    --max_gen 320 --max_think_per_step 4 --total_think_budget 120 \
    --think_budget_diversity 0.7 --stochastic_gate --gate_entropy_bonus 0.01 \
    --emit_threshold 0.5 --gate_floor 0.0 --temperature ${TEMP:-0.9} --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.1 --kl_target ${KL_TARGET:-0.15} --kl_coef_min 0.02 --kl_coef_max 0.6 \
    --ponder_cost 0.0 --ponder_shape quadratic --counterfactual --ponder_warmup_steps 50 \
    --max_turns ${MAX_TURNS:-2} --no-batch_turn0 \
    --progressive_curriculum --no-adaptive_curriculum \
    --curriculum_target_start ${CTGT_START:-0.7} --curriculum_target_end ${CTGT_END:-0.2} \
    --llm_judge --judge_url http://localhost:${JUDGE_PORT:-8000} \
    --judge_model Qwen/Qwen2.5-Coder-3B-Instruct-AWQ --judge_strip_comments \
    --grad_clip 1.0 --log_every 1 --save_every ${SAVE_EVERY:-50} --seed ${SEED:-0} \
    $EXTRA \
    > "$LOG" 2>&1 &
echo "PID $!"