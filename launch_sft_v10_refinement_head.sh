#!/bin/bash
# SFT v10 — Phase D RefinementHead on the SFT Phase C base.
#
# After v8 (Phase A process reward) regressed 6 → 4 and v9 (A+B) collapsed
# 6 → 1, the diagnosis is now clear: supervising or attaching capacity
# to a fundamentally-no-op think mechanism actively breaks the model.
#
# Phase D fixes the ROOT: adds a structurally different computation
# (1 layer windowed local self-attention + MLP) whose output is
# soft-mixed with the trunk hidden by σ(gate). Now σ has a real job:
# weight two genuinely different predictions, not "compute / don't
# compute."
#
# NO process reward this time. Standard LM loss is enough — the
# refinement head's parameters get gradient at every position; the
# model will learn to make it useful through ordinary next-token CE.
#
# α init 0 means the model starts byte-identical to the SFT base;
# the refinement head only contributes once gradient pushes α off zero.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/sft_v10_refinement_head.pt \
    --distilled_jsonl data/sft_cot_thinking_v1.jsonl \
    --distilled_keep_only_passing \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --state_readonly_at_think \
    --use_refinement_head \
    --refinement_head_window 128 \
    --refinement_head_n_heads 8 \
    --refinement_head_mlp_mult 2 \
    --max_codealpaca 0 \
    --epochs 2 \
    --batch 2 \
    --lr 3e-6 \
    --max_len 1024 \
    --log_every 50 \
    --seed 0 \
    > runs/sft_v10_refinement_head.log 2>&1 &
echo "Launched SFT v10 (Phase D RefinementHead) — PID $!"
echo "Watch: tail -f runs/sft_v10_refinement_head.log"
