#!/bin/bash
# Phase D SFT with the PROVEN Phase C SFT recipe (which got 10/164 on
# the Phase C base). Earlier launch_sft_phase_d_mixed.sh used a much
# weaker recipe (1 epoch, lr=5e-6, no retrieval-as-input, no future-emb
# loss) and scored 0/164 — confirmed an isolation run from the Phase C
# base scored 0/164 too. The recipe was the bug, not the base.
#
# Proven recipe: 2 epochs, lr=3e-5, retrieval-as-input thinking,
# future-emb loss 0.1, multi-horizon WM gist, mem_size=1024.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p data runs checkpoints

# Use the same mixed corpus as before (Qwen distill + 961 CoT-thinking).
MIXED_DATA="data/sft_mixed_qwen_cot.jsonl"
if [[ ! -e "$MIXED_DATA" ]]; then
    echo "ERROR: $MIXED_DATA not built. Run launch_sft_phase_d_mixed.sh first."
    exit 1
fi

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_phase_d.pt \
    --save_ckpt checkpoints/sft_phase_d_proper.pt \
    --distilled_jsonl "$MIXED_DATA" \
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
    --seed 0 \
    > runs/sft_phase_d_proper.log 2>&1 &
echo "Launched Phase D + proven SFT recipe — PID $!"
echo "Watch: tail -f runs/sft_phase_d_proper.log"
