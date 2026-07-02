#!/usr/bin/env bash
# Linearize Qwen2.5-Coder-0.5B → our DeltaNet (MOHAWK attn-transfer + KD-heal).
# The token-inheritance pivot: transfer Qwen-Coder's trillions-of-code-tokens
# knowledge into an O(1)-decode DeltaNet instead of training from scratch.
#
# Validated prereqs (2026-06-24): copy bit-exact (COPY EXACT PASS), both stages
# run, stage3 b=2/T=1024 fits (b=4 OOMs the 152k-vocab KL). Donor = cached
# Qwen2.5-Coder-0.5B-Instruct (set in linearize_qwen.py DONOR; switch to the base
# model there for a cleaner re-SFT donor — needs a download). Single-GPU.
# Heal target: student HumanEval-solution CE → donor 0.5925 (Qwen-token).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints
CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/linearize_qwen.py \
    --out_dir checkpoints \
    --ckpt_prefix qwen_coder_05b \
    --batch 12 \
    --layerwise_tokens 200000000 \
    --e2e_batch 2 \
    --T 1024 \
    --e2e_tokens 150000000 \
    --eval_every_tokens 50000000 \
    --seed 0 \
    > runs/linearize_qwen.log 2>&1 &
echo "Launched Qwen-Coder-0.5B linearization on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/linearize_qwen.log   | ckpts: checkpoints/qwen_coder_05b_stage{2,3}*.pt"
