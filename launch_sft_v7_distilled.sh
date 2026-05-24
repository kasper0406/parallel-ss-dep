#!/bin/bash
# SFT v7.1-pkm-film on the Qwen-3.6-distilled (problem, CoT+solution) corpus.
#
# Base: checkpoints/pretrain_mix_v7_pkm_film.pt (10L × 896d shallow-wide +
#       5 dense FiLM K=3 + PKM v7.1 + WM + entropy-gate, VAL ppl 5.83).
#       Note: we SFT directly from PRETRAIN, not from the prior
#       sft_v7_pkm_film_thinking.pt — the prior SFT was on the small
#       MBPP+CodeAlpaca corpus that didn't move HumanEval, and we want
#       to avoid baking in its docstring-echo failure mode.
#
# Data: data/distill_v7_phase1_with_tests.jsonl
#       Combined output from two distillation passes:
#         - Phase 1 (super_combined): 3993 with-tests problems × 2 samples
#           → 4468 kept (~25 MB JSONL)
#         - Phase 1b (magicoder 5000): 5000 distillation-only problems × 2
#           samples → appended to the same JSONL
#       Total expected: ~10-12k (problem, full-Qwen-completion) pairs.
#
# Key change vs the prior sft_v7_pkm_film_thinking.pt run: the SOLUTION
# field is Qwen's FULL completion (CoT prose + ```python ... ``` block),
# not just bare code. The student learns to reason before emitting code,
# which exercises the thinking gate during SFT with real reasoning
# content (instead of random think-burst injection).
#
# Pinned to GPU 0 (assumes external work has freed it; pass GPU=1 to
# override).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_distilled.pt \
    --distilled_jsonl data/distill_v7_phase1_with_tests.jsonl \
    --with_thinking \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 50 \
    > runs/sft_v7_pkm_film_distilled.log 2>&1 &
echo "Launched SFT v7.1-pkm-film distilled on GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_distilled.log"
