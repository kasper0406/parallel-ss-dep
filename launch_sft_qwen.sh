#!/usr/bin/env bash
# PLAIN code SFT on the linearized Qwen-Coder-0.5B → DeltaNet base.
# Deliberately NO thinking/WM/gist features (the session's conclusion: features are
# net-neutral on code; inherit knowledge clean). This isolates "strong inherited
# base + clean SFT → HumanEval", directly comparable to our 14/164 and the donor's 58%.
# Base ckpt already has cfg["tokenizer"]=Qwen (patched + loader-verified, CE 0.8687).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints
BASE=${BASE:-checkpoints/qwen_coder_05b_stage3.pt}
CORPUS=data/distill_v7_phase1_astclean.jsonl   # ast-clean (parse+has-def) code distill
[ -f "$BASE" ] || { echo "ERROR: base $BASE missing"; exit 1; }
[ -f "$CORPUS" ] || { echo "ERROR: corpus $CORPUS missing"; exit 1; }
CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt "$BASE" \
    --save_ckpt checkpoints/sft_qwen_coder_05b.pt \
    --distilled_jsonl "$CORPUS" \
    --epochs 2 \
    --batch ${SFT_BATCH:-4} \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 100 \
    > runs/sft_qwen_coder_05b.log 2>&1 &
echo "Launched PLAIN SFT on Qwen-linearized base, GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_qwen_coder_05b.log  | eval: --prompt_style sft_comment --extract_code_block"
echo "If OOM at the 152k-vocab head: re-run with SFT_BATCH=2"
