#!/bin/bash
# Phase C + proper SFT recipe + mixed data (Qwen distill + 961 CoT rows).
#
# This is the new working baseline for Phase 1a. Phase D + proper recipe
# scored 2/164 (vs historical Phase C + proper recipe at 10/164),
# confirming Phase D's pretrain regressed the base. So we route the
# Phase 1a compression test through the Phase C base.
#
# Comparison targets:
#   Phase C + proper recipe (historical, no CoT): 10/164
#   Phase C + proper recipe + CoT rows (this run): expect ≥ 8/164
#   Phase C + proper recipe + CoT + Phase 1a gist: TEST OF COMPRESSION

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p data runs checkpoints

MIXED_DATA="data/sft_mixed_qwen_cot.jsonl"
if [[ ! -e "$MIXED_DATA" ]]; then
    echo "ERROR: $MIXED_DATA not built. Run launch_sft_phase_d_mixed.sh first."
    exit 1
fi

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --save_ckpt checkpoints/sft_phase_c_proper.pt \
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
    > runs/sft_phase_c_proper.log 2>&1 &
echo "Launched Phase C + proper SFT + mixed data — PID $!"
echo "Watch: tail -f runs/sft_phase_c_proper.log"
