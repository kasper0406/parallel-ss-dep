#!/usr/bin/env bash
# Depth-via-iteration full grid: can latent-R simulate trunk depth?
# fp32, single GPU (GPU0 is broken -> CUDA_VISIBLE_DEVICES=1).
#
#   bash launch_depthiter.sh            # run grid + print report
#   bash launch_depthiter.sh report     # report only from existing jsonl
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:."
OUT=/tmp/depthiter_results.jsonl
LOPS=${LOPS:-2}          # ops for hetero / hetero_mt (homo ignores it)
STEPS=${STEPS:-5000}

if [ "${1:-run}" = "report" ]; then
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/depth_via_iteration_run.py \
    --report_only --out "$OUT"
  exit 0
fi

CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/depth_via_iteration_run.py \
  --out "$OUT" --N 8 --K 6 --eval_K_max 8 --L_ops "$LOPS" \
  --batch 256 --steps "$STEPS" --n_eval 1024 --parallel 6 --save_ckpts
