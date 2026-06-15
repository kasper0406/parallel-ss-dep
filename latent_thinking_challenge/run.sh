#!/usr/bin/env bash
# Reproduce the dichotomy: latent thinking helps the HOMOGENEOUS task but not
# the HETEROGENEOUS one. Final-answer-only supervision (the realistic regime).
# Pin to one GPU; each run is ~5-7 min on a single modern GPU.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-python}"

echo "############################################################"
echo "# HOMOGENEOUS (pointer-chase f^n(s)) — thinking SHOULD help"
echo "############################################################"
$PY train.py --task homogeneous --V 10 --n_max 6 --steps 3000 --batch 256

echo
echo "############################################################"
echo "# HETEROGENEOUS (exec-trace, diff op each step) — the OPEN problem"
echo "############################################################"
$PY train.py --task heterogeneous --V 10 --n_max 6 --steps 3000 --batch 256

echo
echo "Dichotomy: homogeneous latent R=n ~= 1.00 at every depth (big lift);"
echo "heterogeneous lift is positive shallow but DECAYS to ~0 with depth."
