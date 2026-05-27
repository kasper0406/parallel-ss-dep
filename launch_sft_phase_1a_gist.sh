#!/bin/bash
# SFT Phase 1a — gist-at-think compression (THINKING_PLAN.md Phase 1a).
#
# Resumes from checkpoints/sft_phase_d_mixed.pt (the Phase 0 baseline)
# and continues SFT on the same mixed corpus (54k Qwen distill + 961
# CoT-thinking rows). The CoT-thinking rows now route through
# build_example_with_cot_compression: each row's CoT prose is compressed
# to N_think = ceil(N_cot / K) think tokens (K=5), and the trainer runs
# a no_grad teacher forward over the full (prompt + CoT prose + code)
# sequence to supervise each student think's hidden state against the
# teacher's hidden state at the K-chunk-end position via think_gist_loss.
#
# This is the LOAD-BEARING test of the architectural compression claim:
# if a single think can absorb K text-CoT tokens worth of reasoning, the
# gist loss should fall steadily and the model should retain Phase 0's
# HumanEval pass-rate at ~1/K the think budget.
#
# Lower LR than Phase 0 (3e-6 vs 5e-6): we're refining an already-SFT'd
# ckpt, the gist loss is a new supervision channel, and we want it to
# bend the model gently around the existing solution distribution.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p data runs checkpoints

MIXED_DATA="data/sft_mixed_qwen_cot.jsonl"

if [[ ! -e "$MIXED_DATA" ]]; then
    echo "ERROR: $MIXED_DATA does not exist."
    echo "Run launch_sft_phase_d_mixed.sh first (it builds the mix)."
    exit 1
fi

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/sft_phase_c_repro_historical.pt \
    --save_ckpt checkpoints/sft_phase_1a_gist.pt \
    --distilled_jsonl data/sft_cot_thinking_v1.jsonl \
    --distilled_keep_only_passing \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --state_readonly_at_think \
    --cot_compression_k 5 \
    --cot_min_thinks 4 \
    --think_gist_weight 0.1 \
    --think_gist_loss_type cosine \
    --max_codealpaca 0 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-6 \
    --max_len 1024 \
    --log_every 50 \
    --seed 0 \
    > runs/sft_phase_1a_gist.log 2>&1 &
echo "Launched Phase 1a gist-at-think SFT — PID $!"
echo "Watch: tail -f runs/sft_phase_1a_gist.log"
