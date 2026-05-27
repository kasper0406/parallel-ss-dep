#!/bin/bash
# Phase 0 ISOLATION: rerun the mixed SFT recipe (Qwen distill ∪ CoT-thinking)
# from the KNOWN-WORKING Phase C base instead of Phase D. Tests whether
# the 0/164 result on Phase D is a Phase D pretrain regression or an SFT
# recipe problem.
#
# Phase C + Qwen distill historically scored 10/164 (per CLAUDE.md). If
# this run reproduces that ≈10/164, the SFT recipe is fine and the
# regression is in Phase D's pretrain (FIM + synth + self-debug may have
# hurt the base). If this also scores 0, the recipe itself is broken.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p data runs checkpoints

MIXED_DATA="data/sft_mixed_qwen_cot.jsonl"
if [[ ! -e "$MIXED_DATA" ]]; then
    echo "ERROR: $MIXED_DATA not built. Run launch_sft_phase_d_mixed.sh first."
    exit 1
fi

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --save_ckpt checkpoints/sft_phase_c_mixed.pt \
    --distilled_jsonl "$MIXED_DATA" \
    --distilled_keep_only_passing \
    --with_thinking \
    --max_codealpaca 0 \
    --epochs 1 \
    --batch 4 \
    --lr 5e-6 \
    --max_len 1536 \
    --log_every 50 \
    --seed 0 \
    > runs/sft_phase_c_mixed.log 2>&1 &
echo "Launched Phase C + mixed SFT — PID $!"
echo "Watch: tail -f runs/sft_phase_c_mixed.log"
