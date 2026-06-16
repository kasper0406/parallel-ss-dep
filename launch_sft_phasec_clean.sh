#!/bin/bash
# Data-hygiene A/B (2026-06-16): re-SFT the Phase C CODE base on the
# drop-broken distill corpus. The ONLY difference vs sft_phase_c_combined.pt
# (which scored 14/164 same-config) is that the 4,279 verified-broken distill
# rows (tier syntax_error/exec_error) are removed — they are active poison
# (the model demonstrably learned the inverted degrees_to_radians + infinite-
# recursion longest_chain from malformed targets). Tests the synthesis's #1
# headline lever (data hygiene) on the RIGHT (code-focused) base.
#
# A/B: clean (this) vs dirty (sft_phase_c_combined.pt = 14/164). Same base
# (pretrain_phase_c.pt), same recipe, same synth/longctx. Eval thinking-off
# with --prompt_style sft_comment --extract_code_block --max_gen 512.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

if [ ! -f data/distill_v7_phase1_clean.jsonl ]; then
  echo "ERROR: data/distill_v7_phase1_clean.jsonl missing (the drop-broken distill)."; exit 1
fi
cat data/distill_v7_phase1_clean.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_phasec_clean.jsonl
echo "[sft-phasec-clean] corpus rows: $(wc -l < data/sft_phasec_clean.jsonl) (clean-distill+synthmem+longctx)"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --save_ckpt checkpoints/sft_phasec_clean.pt \
    --distilled_jsonl data/sft_phasec_clean.jsonl \
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
    > runs/sft_phasec_clean.log 2>&1 &
echo "Launched clean-corpus SFT on Phase C base, GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_phasec_clean.log"
