#!/bin/bash
# Rejection-sampling data generation, sharded across both GPUs.
#
# Each GPU runs an independent gen_rejection_data.py over half the
# problems (strided sharding, so easy/hard mix is balanced). Output
# shards are written separately and concatenated at the end.
#
# Usage:
#   ./launch_rejection_gen_dual_gpu.sh \
#       checkpoints/rl_grader_phase_c_v2_step300.pt \
#       data/rejection_v2_step300_all.jsonl \
#       --keep_all   # any extra args pass through to gen_rejection_data.py
#
# Assumes both GPUs are idle. If GPU 1 is busy (e.g. the original
# single-GPU rejection-gen is still running), use the single-GPU
# launcher pattern in launch_rejection_gen.sh or pass
# CUDA_VISIBLE_DEVICES=0 directly to gen_rejection_data.py.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <ckpt> <out.jsonl> [extra args...]" >&2
    exit 1
fi

CKPT="$1"; shift
OUT="$1"; shift
EXTRA_ARGS=("$@")

OUT_BASE="${OUT%.jsonl}"
SHARD0="${OUT_BASE}.shard0.jsonl"
SHARD1="${OUT_BASE}.shard1.jsonl"
LOG0="runs/$(basename "${OUT_BASE}").shard0.log"
LOG1="runs/$(basename "${OUT_BASE}").shard1.log"
mkdir -p runs "$(dirname "$OUT")"

echo "[dual-gpu reject-gen]"
echo "  ckpt   : $CKPT"
echo "  shard0 : $SHARD0 (GPU 0, log $LOG0)"
echo "  shard1 : $SHARD1 (GPU 1, log $LOG1)"
echo "  merged : $OUT"
echo "  extra  : ${EXTRA_ARGS[*]:-(none)}"

export PYTHONPATH=".:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/gen_rejection_data.py \
    --ckpt "$CKPT" \
    --dataset mbpp_combined \
    --n_rollouts 16 --temperature 0.9 --max_gen 384 \
    --grade_workers 16 \
    --shard_id 0 --num_shards 2 \
    --out "$SHARD0" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG0" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/gen_rejection_data.py \
    --ckpt "$CKPT" \
    --dataset mbpp_combined \
    --n_rollouts 16 --temperature 0.9 --max_gen 384 \
    --grade_workers 16 \
    --shard_id 1 --num_shards 2 \
    --out "$SHARD1" \
    "${EXTRA_ARGS[@]}" \
    > "$LOG1" 2>&1 &
PID1=$!

echo "[dual-gpu reject-gen] PIDs: shard0=$PID0  shard1=$PID1"
echo "[dual-gpu reject-gen] tail -f $LOG0 $LOG1 to watch progress"

# Wait for both; surface any failure but still try to merge what's there.
RC=0
wait "$PID0" || RC=$?
wait "$PID1" || RC=$?

if [ ! -f "$SHARD0" ] || [ ! -f "$SHARD1" ]; then
    echo "[dual-gpu reject-gen] one or both shards missing (rc=$RC) — not merging" >&2
    exit "$RC"
fi

cat "$SHARD0" "$SHARD1" > "$OUT"
N=$(wc -l < "$OUT")
echo "[dual-gpu reject-gen] merged $N total rows → $OUT (rc=$RC)"
exit "$RC"
