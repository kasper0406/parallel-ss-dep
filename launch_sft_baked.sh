#!/bin/bash
# Combined SFT on the Phase C base (2026-05-22).
#
# Same validated SFT recipe as launch_sft_v7_combined_v7_additive.sh
# (additive α-gated retrieval-as-input + trunk multi-horizon gist loss),
# but starting from checkpoints/pretrain_phase_c.pt — the Chinchilla-
# completed base (5.3B tokens). The controlled per-source CE eval showed
# Phase C beats the v7.1 base on all 8 sources (~0.17 CE mean), so this
# SFT runs on a strictly stronger foundation than SFT v7 did.
#
# Note: Phase C's trunk-gist heads were saved inside the model
# state_dict (gist_heads.*); sft_code builds its own future_gist_heads
# fresh — they retrain in ~400 steps, so this is a non-issue.
#
# Iteration plan: train → HumanEval + long-context recall + ablation,
# compared against SFT v7 (same recipe, weaker base).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

cat data/distill_v7_phase1_with_tests.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_phase_c_combined.jsonl
echo "[sft-phase-c] corpus rows: $(wc -l < data/sft_phase_c_combined.jsonl) "\
"(distill+synthmem+longctx)"

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/latent_bake_gate2_step24415_tok3200122880.pt \
    --save_ckpt checkpoints/sft_baked.pt \
    --distilled_jsonl data/sft_phase_c_combined.jsonl \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 100 \
    > runs/sft_baked.log 2>&1 &
echo "Launched combined SFT on Phase C base, GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/sft_baked.log"
