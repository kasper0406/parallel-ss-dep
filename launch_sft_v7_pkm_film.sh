#!/bin/bash
# SFT v7.1-pkm-film on (problem, solution) pairs WITH thinking enabled.
#
# Base: checkpoints/pretrain_mix_v7_pkm_film.pt (10L × 896d shallow-wide +
#       5 dense FiLM K=3 + PKM v7.1 + WM + entropy-gate, VAL ppl 5.83 final).
#       Strongest pretrain we have. Designed to feed the v7.1 RL-grader run.
#
# Why with-thinking matters at SFT (same logic as v5-pkm SFT):
#   - We need the gate / WM / PKM machinery to stay active rather than be
#     de-trained by SFT-without-thinking.
#   - Random think-burst injection at training time → with target masked
#     to -100, the post-think emit gradient gets meaningful "did thinking
#     help me predict the next real token?" signal.
#
# Note: on v5-pkm SFT we observed the gate collapsing to always-emit
# (mean gate at emit = 0.978, think_rate = 0.000). The v7.1 entropy-gate
# auxiliary BCE loss that was added during pretrain *should* keep the
# gate calibrated. We're testing whether that pretrain-baked signal
# survives SFT — if not, the RL-grader stage has to do the recovery.
#
# Optimizer: stock AdamW (sft_code default — gentler than Muon for FT).
# Pinned to GPU 1 (RL grader on v5-pkm-SFT runs on GPU 0).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_thinking.pt \
    --with_thinking \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 512 \
    --max_codealpaca 10000 \
    --log_every 50 \
    > runs/sft_v7_pkm_film_thinking.log 2>&1 &
echo "Launched SFT v7.1-pkm-film on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_thinking.log"
