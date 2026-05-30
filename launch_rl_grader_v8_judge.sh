#!/bin/bash
# RL+JUDGE test (2026-05-30): does the LLM-judge ranker fix the zero-gradient
# tied-group problem that starved + collapsed RL-B?
#
# RL-B (gate-selectivity, no judge) collapsed: on the weak v8 base most groups
# tie at syntax_error -> zero reward variance -> no gradient -> drift -> collapse
# (best ckpt step-150 = 6/164). The judge exists precisely for this: it ranks
# "which broken solution is closest to working" WITHIN a tied tier and injects
# bounded reward deltas -> restores gradient on exactly the groups RL-B couldn't
# learn from. Plus --max_turns 2 (environment-feedback revision) and tighter
# --kl_coef 0.1 (anti-collapse).
#
# Judge = small co-resident ranker: Qwen2.5-Coder-7B-Instruct-AWQ served by vLLM
# (.venv-vllm) on GPU0 at capped memory; the RL trainer runs on GPU0 too. v9
# pretraining is untouched on GPU1.
#
# WATCH: `judge=N` (>0 means the ranker is firing on tied groups), var-bearing
# groups up, and whether reward/pass climbs past 6/164 instead of collapsing.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v8_combined.pt \
    --save_ckpt checkpoints/rl_grader_v8_judge.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --activation_checkpointing \
    --steps 250 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --think_budget_diversity 0.7 \
    --stochastic_gate \
    --gate_entropy_bonus 0.01 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.1 \
    --ponder_cost 0.0 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 50 \
    --max_turns ${MAX_TURNS:-1} \
    --no-batch_turn0 \
    --llm_judge \
    --judge_url http://localhost:8000 \
    --judge_model Qwen/Qwen2.5-Coder-3B-Instruct-AWQ \
    --judge_strip_comments \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_v8_judge.log 2>&1 &
echo "Launched RL+JUDGE on v8 SFT base, GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v8_judge.log  (look for judge=N)"
