#!/bin/bash
# SFT v8 — Phase A process reward on the Phase C SFT base.
#
# THINKING_PLAN v5 Phase A: an SFT-time aux loss that compares
# log p(true_next_token) before vs after K=4 inserted thinks on a
# sampled fraction of high-σ positions. Minimizing
# mean(log_p_before − log_p_after) pushes thinks to actually reduce
# next-token error.
#
# The Phase C SFT base shows the failure mode the loss is designed to
# fix: probe (sft_phase_c_combined.pt) reported mean Δlogp = −0.165
# with only 30.5% of high-gate positions benefiting from K=4 thinks.
# Target: after this SFT, mean Δlogp > 0 and %positive > 50.
#
# Recipe is the established v7 retrieval-as-input + trunk-gist SFT
# (the recipe that produced the historical 10/164 SFT v7), plus Phase A.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/sft_v8_process_reward.pt \
    --distilled_jsonl data/sft_cot_thinking_v1.jsonl \
    --distilled_keep_only_passing \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --state_readonly_at_think \
    --process_reward_weight 0.1 \
    --process_reward_K 4 \
    --process_reward_apply_min_sigma 0.3 \
    --process_reward_sample_frac 0.1 \
    --process_reward_max_positions 128 \
    --max_codealpaca 0 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-6 \
    --max_len 1024 \
    --log_every 50 \
    --seed 0 \
    > runs/sft_v8_process_reward.log 2>&1 &
echo "Launched SFT v8 (Phase A) — PID $!"
echo "Watch: tail -f runs/sft_v8_process_reward.log"
echo "Look for the new 'pr(n=K/N, Δlogp=±X.XXX, %pos=YY)' diagnostic"
