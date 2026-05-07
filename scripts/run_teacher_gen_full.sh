#!/bin/bash
# DN-4B distillation pilot — Phase A (teacher corpus generation).
#
# Generates ~50M tokens of Qwen3.6-self-emitted code with top-K=20 logprobs
# captured at each generated position. Writes to data/distill_pilot_50M/.
#
# Validated throughput: ~1200 tok/s on a single 5090 in eager mode at
# max_model_len=512. 50M tokens => ~12 h wall-clock. Using nohup so the
# session can disconnect.
#
# Pin to GPU 0 only.
set -e
cd /home/knielsen/ml/parallel-ss-dep-distill

mkdir -p logs/distill_pilot_full
mkdir -p data

TOTAL_TOKENS="${TOTAL_TOKENS:-50000000}"
OUT_DIR="${OUT_DIR:-data/distill_pilot_50M}"
LOG="${LOG:-logs/distill_pilot_full/teacher_gen_50M.log}"
SEED="${SEED:-42}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Teacher data gen — Phase A"
echo "  total_tokens = $TOTAL_TOKENS"
echo "  out          = $OUT_DIR"
echo "  log          = $LOG"

nohup /home/knielsen/ml/parallel-ss-dep/.venv-vllm/bin/python -u \
    experiments/teacher_data_gen.py \
        --total_tokens $TOTAL_TOKENS \
        --out "$OUT_DIR" \
        --seed $SEED \
        --gen_batch 32 \
        --max_completion 256 \
        --top_k_logprobs 20 \
        --T 512 \
        --max_model_len 512 \
        --gpu_mem_fraction 0.92 \
    > "$LOG" 2>&1 &

PID=$!
echo "PID $PID logging to $LOG"
echo $PID > logs/distill_pilot_full/teacher_gen_50M.pid
