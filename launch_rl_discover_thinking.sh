#!/bin/bash
# THINKING_PLAN v3 — first stochastic-gate RL run on synth_reasoning.
#
# Discovery RL: gate is Bernoulli-sampled, log-probs enter PPO ratio,
# entropy bonus prevents collapse. Reward = grader_pass on synth_reasoning
# (6-family multi-step compute tasks where thinking SHOULD help).
#
# Base ckpt: sft_phase_c_combined.pt (historical 7-8/164 baseline). This
# is the strongest base we have right now; the Phase D base regressed.
#
# Initial probe: 200 steps. Watch gate_fire_rate dynamics:
#   - Starts ~0.5 (gate entropy regularization)
#   - If pattern X wins → converges to X
#   - If gate=0 (never think) wins on these tasks → architecture can't
#     leverage thinking even with discovery; pivot decisively away
#   - If gate stays at 0.5 → exploration didn't find a useful pattern;
#     try Phase D counterfactual reward

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/rl_discover_thinking_mbpp.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 200 \
    --batch 3 \
    --grpo_n_group 4 \
    --lr 1e-6 \
    --max_gen 256 \
    --max_think_per_step 8 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.1 \
    --ponder_cost 0.0 \
    --counterfactual \
    --ponder_warmup_steps 0 \
    --grad_clip 1.0 \
    --curriculum_filter \
    --no-iterative_repair \
    --stochastic_gate \
    --gate_entropy_bonus 0.01 \
    --log_every 1 \
    --save_every 25 \
    --seed 0 \
    > runs/rl_discover_thinking_mbpp.log 2>&1 &
echo "Launched discovery-RL on synth_reasoning — PID $!"
echo "Watch: tail -f runs/rl_discover_thinking.log"
echo
echo "KEY METRICS:"
echo "  gate(fire=X, H=Y, ratio=Z) — fire is fraction electing to think"
echo "  pass_n — passing rollouts per step (out of batch*n_group=12)"
echo "  kl=+X — drift from SFT base; should stay <0.10"
