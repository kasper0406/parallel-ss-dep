#!/bin/bash
# SFT v9 — Phase A (process reward) + Phase B (ThinkAdapter) together.
#
# Identical recipe to SFT v8 with --use_think_adapter added. Tests
# whether dedicated think-time parameters (small 2-layer MLP fired
# only at think positions, α init 0) stack with process reward.
#
# α init 0 means at step 0 the model is byte-identical to the v8
# starting point; the adapter only contributes once the process-reward
# gradient pushes α off zero.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/sft_v9_phase_a_b.pt \
    --distilled_jsonl data/sft_cot_thinking_v1.jsonl \
    --distilled_keep_only_passing \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --state_readonly_at_think \
    --use_think_adapter \
    --think_adapter_hidden_mult 2 \
    --process_reward_weight 0.1 \
    --process_reward_K 4 \
    --process_reward_apply_min_sigma 0.3 \
    --process_reward_sample_frac 0.1 \
    --process_reward_max_positions 128 \
    --max_codealpaca 0 \
    --epochs 2 \
    --batch 2 \
    --lr 3e-6 \
    --max_len 1024 \
    --log_every 50 \
    --seed 0 \
    > runs/sft_v9_phase_a_b.log 2>&1 &
echo "Launched SFT v9 (Phase A + B) — PID $!"
echo "Watch: tail -f runs/sft_v9_phase_a_b.log"
