#!/bin/bash
# Phase 2 distillation: 20k CodeFeedback problems (Python only) via Qwen 3.6.
#
# Phase 1 + 1b gave us 14,468 distilled samples (3993 mbpp+leetcode +
# 10k magicoder). This phase adds 20k more from CodeFeedback to roughly
# triple the SFT corpus, in hopes of pushing the distilled student
# above its current 6/164 HumanEval pass@1.
#
# Appends to the SAME jsonl as Phase 1+1b so the SFT stage just sees
# a bigger pool. Each row's task_id is namespaced (codefeedback/N), so
# no collisions with phases 1 / 1b.
#
# Pinned to GPU 0 (defaults). Make sure GPU 0 is free before running.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs data

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup ~/ml/parallel-ss-dep/.venv-vllm/bin/python -u experiments/distill_solutions.py \
    --dataset codefeedback \
    --max_problems 20000 \
    --n_samples 2 \
    --max_new_tokens 1024 \
    --batch_size 64 \
    --temperature 0.7 \
    --out data/distill_v7_phase1_with_tests.jsonl \
    > runs/distill_v7_phase2_codefeedback.log 2>&1 &
echo "Launched Phase 2 (CodeFeedback 20k) on GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/distill_v7_phase2_codefeedback.log"
