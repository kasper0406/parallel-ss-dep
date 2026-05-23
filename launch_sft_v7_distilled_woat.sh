#!/bin/bash
# Retrain the v7.1-distilled SFT base with FIX A (write-only-at-think) enabled.
#
# Background (diag_thinking_machinery.py finding, 2026-05-18):
#   - The current sft_v7_pkm_film_distilled.pt ckpt scores HumanEval 6/164
#     with thinking ON, and ALSO 6/164 with thinking OFF.
#   - The thinking machinery is decorative because the write gate fires
#     uniformly across think/emit positions — the buffer is filled with
#     random noise, so the sharp read queries at think positions hit
#     nothing useful.
#
# Hypothesis (FIX A):
#   Force the WM buffer to be filled only from think-position writes.
#   Gradient pressure on the write gate then flows ONLY to think
#   positions, so the model has to learn "which of my think positions
#   contain useful intermediate computation to store?". Read queries
#   then retrieve actual think-content.
#
# This launcher starts from the current distilled SFT base and retrains
# for 1 epoch on the same distilled JSONL with the flag on. Cheap
# (~5 min based on the prior distilled SFT timing) — if the flag has
# real signal, retraining will let the gate adapt.
#
# Note: we start from sft_v7_pkm_film_distilled.pt (already SFT'd) so
# the model has the format / code-emission skills; the retrain only
# tweaks the WM gate to honor the new mask. Starting fresh from
# pretrain would be slower and would discard the format learning.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/sft_v7_pkm_film_distilled.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_distilled_woat.pt \
    --distilled_jsonl data/distill_v7_phase1_with_tests.jsonl \
    --with_thinking \
    --mem_write_only_at_think \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 1 \
    --batch 4 \
    --lr 1e-5 \
    --max_len 1024 \
    --log_every 50 \
    > runs/sft_v7_pkm_film_distilled_woat.log 2>&1 &
echo "Launched SFT-retrain with FIX A on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_distilled_woat.log"
