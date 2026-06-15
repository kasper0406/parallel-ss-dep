#!/bin/bash
# Grader-RL on v8 with the REWARD-VARIANCE UNLOCK (2026-05-31). Prior RL runs
# were flat because the base emitted mostly syntax errors -> all-0.0 GRPO groups
# -> zero gradient. Now fixed at the source:
#   - build_mbpp_prompt surfaces the function signature (syntax errors 58%->9%
#     under sampling; rollouts spread across exec/partial/pass tiers).
#   - RL grading extraction aligned with the harvester (truncate_at_stop).
#   - richer grader errors (got X, expected Y) for the repair turns.
# Settings revisited for THIS base (thinking is a confirmed net-negative on v8):
#   - thinking OFF (--total_think_budget 0 --max_think_per_step 0): removes the
#     corruption + the gate-exploration confound. No ponder cost.
#   - grpo_n_group 8, batch 2 (microbatch caps policy-update memory regardless).
#   - temp 0.7 (signature keeps it valid), tight-ish KL (coef 0.05, target 0.10).
#   - max_turns 2 (repair now has teeth), group_var_floor 0.02, gentle curriculum.
# Judge server must be up (bash launch_judge_server.sh). Watch the PASS-TIER rate,
# not just mean reward, to make sure it's solving rather than farming exec(0.05).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt ${LOAD:-checkpoints/sft_v8_combined.pt} \
    --save_ckpt ${SAVE:-checkpoints/rl_grader_v8_signal.pt} \
    --dataset mbpp_combined --extract_code_block --activation_checkpointing \
    --steps ${STEPS:-200} --batch ${BATCH:-2} --grpo_n_group ${NGROUP:-8} \
    --policy_micro_chunk ${MICRO:-4} --lr ${LR:-2e-6} \
    --max_gen 320 --max_turns ${MAX_TURNS:-2} --no-batch_turn0 \
    --max_think_per_step 0 --total_think_budget 0 \
    --emit_threshold 0.5 --gate_floor 0.0 --temperature ${TEMP:-0.7} \
    --min_emit_before_eos 30 --clip_eps 0.1 \
    --kl_coef 0.05 --kl_target ${KL_TARGET:-0.10} --kl_coef_min 0.02 --kl_coef_max 0.6 \
    --ponder_cost 0.0 \
    --group_var_floor 0.02 \
    --progressive_curriculum --no-adaptive_curriculum \
    --curriculum_target_start 0.7 --curriculum_target_end 0.4 \
    --llm_judge --judge_url http://localhost:${JUDGE_PORT:-8000} \
    --judge_model Qwen/Qwen2.5-Coder-3B-Instruct-AWQ --judge_strip_comments \
    --grad_clip 1.0 --log_every 1 --save_every 40 --seed 0 \
    > ${LOG:-runs/rl_grader_v8_signal.log} 2>&1 &
echo "PID $!  (log: ${LOG:-runs/rl_grader_v8_signal.log})"
