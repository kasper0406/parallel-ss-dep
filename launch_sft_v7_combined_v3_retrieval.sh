#!/bin/bash
# Combined SFT v3 — retrieval-as-input thinking mechanism (2026-05-19).
#
# Same training recipe as v1 (the 11/164 winner), but with the new
# --retrieval_as_input_thinking flag active from step 0. The model is
# trained with a 2-pass forward where [THINKING] token input embeddings
# are REPLACED by the WM retrieval at the previous position. Each think
# step gets a unique input signal → think-position hidden states are
# diverse → WM buffer can hold diverse think-content → sharp reads
# retrieve meaningful intermediates.
#
# We do NOT set --mem_write_only_at_think (FIX A) because:
#   1. The homogeneity that motivated FIX A is solved at root by the
#      retrieval-as-input mechanism.
#   2. FIX A was empirically worse (v2: 9/164 vs v1: 11/164).
#
# We DO keep --future_emb_loss_weight 0.05 (validated future-emb-pred
# auxiliary loss — main lift from 6→11 in the v1 → combined transition).
#
# Pinned to GPU 1 (assumes era-of-experience on GPU 0).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_combined_v3.pt \
    --distilled_jsonl data/sft_v7_combined.jsonl \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.05 \
    --future_emb_T_max 8 \
    --future_emb_T_ramp_frac 0.3 \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 100 \
    > runs/sft_v7_pkm_film_combined_v3.log 2>&1 &
echo "Launched combined SFT v3 (retrieval-as-input) on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_combined_v3.log"
