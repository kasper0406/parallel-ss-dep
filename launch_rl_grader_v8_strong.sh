#!/bin/bash
# "Make it STRONG" grader-RL (2026-05-30): the thesis that RL — not a bigger
# base — is the lever that builds capability. Same v8 SFT base as every prior
# v8 RL run (which plateaued at 6-7/164), but with the EXPLORATION machinery to
# manufacture successes the base can't one-shot, so RL can keep finding new
# reward frontiers to climb:
#
#   --max_turns 2            : multi-turn revision — find solutions the base
#                             misses in one shot (the biggest "make it strong"
#                             lever; lift~0.9 already observed).
#   --progressive_curriculum : ramp the sampled difficulty from easy (target
#                             pass 0.7) to hard (0.2) over training, so there is
#                             always a reachable-but-unsolved frontier to climb
#                             instead of saturating the easy band.
#   --kl_target 0.15         : ADAPTIVE KL (PPO/R1 controller) — auto-tune the
#                             coefficient to hold KL near 0.15 so the policy can
#                             evolve as far as the reward supports, instead of
#                             being throttled by a fixed penalty as KL grows
#                             (a fixed --kl_coef equilibrates and halts climbing
#                             on a base that COULD go further). --kl_coef 0.1 is
#                             just the starting value.
#   --steps 600              : exploration needs volume (vs the 250 that capped).
#
# No LLM judge: it never fired on this setup (partial-credit + curriculum keep
# groups variance-bearing). batch 2 fits max_turns-2 on one 5090.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v8_combined.pt \
    --save_ckpt checkpoints/rl_grader_v8_strong.pt \
    --dataset mbpp_combined --extract_code_block --activation_checkpointing \
    --steps 600 --batch 2 --grpo_n_group 4 --lr 2e-6 \
    --max_gen 320 --max_think_per_step 4 --total_think_budget 120 \
    --think_budget_diversity 0.7 --stochastic_gate --gate_entropy_bonus 0.01 \
    --emit_threshold 0.5 --gate_floor 0.0 --temperature 0.7 --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.1 --kl_target 0.15 --kl_coef_min 0.02 --kl_coef_max 0.6 \
    --ponder_cost 0.0 --ponder_shape quadratic --counterfactual --ponder_warmup_steps 50 \
    --max_turns 2 --no-batch_turn0 \
    --progressive_curriculum --curriculum_target_start 0.7 --curriculum_target_end 0.2 \
    --grad_clip 1.0 --log_every 1 --save_every 50 --seed 0 \
    > runs/rl_grader_v8_strong.log 2>&1 &
echo "Launched 'make-it-strong' RL on v8 SFT base, GPU ${GPU:-0} (PID $!)"
echo "Watch: kl=...(c=...) adapts, cur(... target ...) ramps harder, reward climbs past 7/164"
