#!/bin/bash
# Combined SFT on the v8-wide base (2026-05-30).
#
# Same validated recipe as launch_sft_phase_c.sh (additive α-gated
# retrieval-as-input + trunk multi-horizon gist loss), but starting from the
# v8-wide 600M base (d_model 1280, 10L) stopped early at 2.3B tokens. Purpose:
# prove the SFT->grader-RL pipeline works end-to-end on the 600M architecture
# before the v9 latent-co-training run.
#
# NO gate-calibration loss: the 2026-05-30 probe showed thinking is
# unproductive on the v8 trunk (both discrete and latent), so there is no
# useful per-position "thinking helps" target to calibrate against on v8.
# That teacher belongs to v9 (latent co-trained from day 1).
#
# mem_size 1536 matches the v8 pretrain config (modern load infers it from the
# ckpt anyway). batch 4 is the validated value; if the 600M model OOMs, drop to
# BATCH=2 (env override).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

cat data/distill_v7_phase1_with_tests.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_v8_combined.jsonl
echo "[sft-v8] corpus rows: $(wc -l < data/sft_v8_combined.jsonl) "\
"(distill+synthmem+longctx)"

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_v8_wide_step7630_tok2000158720.pt \
    --save_ckpt checkpoints/sft_v8_combined.pt \
    --distilled_jsonl data/sft_v8_combined.jsonl \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1536 \
    --epochs 2 \
    --batch ${BATCH:-4} \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 100 \
    > runs/sft_v8_combined.log 2>&1 &
echo "Launched combined SFT on v8 base, GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/sft_v8_combined.log"
