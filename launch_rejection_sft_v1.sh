#!/bin/bash
# Rejection-sampling SFT (STaR/RAFT) on top of grader-RL v2 step-300.
#
# Data: data/rejection_v2_step300_all.jsonl produced by
#   gen_rejection_data.py (--keep_all) on the v2 step-300 ckpt.
#   --distilled_keep_only_passing filters to the 583 passing rollouts
#   (5.4 % of 10816). Training data is the FULL qwen_completion of each
#   pass — preserves the model's natural CoT-and-code structure.
#
# Recipe: lighter than Phase C SFT — we're refining the existing RL
# ckpt, not training from a pretrained base. lr=1e-5 (vs 3e-5 for the
# initial SFT), 1 epoch (vs 2), no synthetic data mixed in to start.
# If we see HumanEval / long-context recall regressions, follow-on
# runs can mix in the long-context recall corpus to prevent forgetting.
#
# Iterative-self-distillation order (per CLAUDE.md):
#   rejection SFT v1 → DPO on the same data (keep_all version, pair
#   passes vs fails) → optionally another rejection round from the
#   sharpened model.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v2_step300.pt \
    --save_ckpt checkpoints/rejection_sft_v1.pt \
    --distilled_jsonl data/rejection_v2_step300_all.jsonl \
    --distilled_keep_only_passing \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 1 \
    --batch 4 \
    --lr 1e-5 \
    --max_len 1024 \
    --log_every 25 \
    > runs/rejection_sft_v1.log 2>&1 &
echo "Launched rejection-SFT v1 on GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/rejection_sft_v1.log"
