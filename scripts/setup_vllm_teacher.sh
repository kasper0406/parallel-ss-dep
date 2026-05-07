#!/usr/bin/env bash
# Reproducible vLLM teacher setup for the DN-4B distillation pilot.
#
# Two roles, two venvs:
#   - .venv-vllm  -> vLLM 0.20+ + AWQ teacher inference (Qwen3.6-35B-A3B-AWQ)
#   - .venv       -> cu132 nightly torch + fla for student training
# Do NOT cross-contaminate. vLLM ships a torch that breaks `fla`.
#
# This script verifies the existing .venv-vllm works; if missing on this
# worktree, recreate it. Both venvs live in /home/knielsen/ml/parallel-ss-dep
# (the original repo); this worktree shares them.
set -euo pipefail

REPO=/home/knielsen/ml/parallel-ss-dep
VLLM_VENV="${REPO}/.venv-vllm"
WORKTREE=/home/knielsen/ml/parallel-ss-dep-distill

echo "== distillation pilot teacher setup =="
echo "  repo:     ${REPO}"
echo "  worktree: ${WORKTREE}"
echo "  vllm venv: ${VLLM_VENV}"

if [[ ! -x "${VLLM_VENV}/bin/python" ]]; then
  echo "vLLM venv not found; creating with uv ..."
  uv venv "${VLLM_VENV}" --python 3.12
  "${VLLM_VENV}/bin/pip" install --upgrade pip
  "${VLLM_VENV}/bin/pip" install "vllm>=0.20" autoawq datasets transformers
fi

echo "Python: $("${VLLM_VENV}/bin/python" --version)"
echo "vLLM:   $("${VLLM_VENV}/bin/python" -c 'import vllm; print(vllm.__version__)')"

# Quick teacher-load smoke test (load + score a 1-line prompt). Skipped if
# the user invokes with --skip-smoke (e.g., during data extraction we
# don't need to redo this).
if [[ "${1:-}" != "--skip-smoke" ]]; then
  echo "Running 1-line teacher smoke test (CUDA_VISIBLE_DEVICES=0) ..."
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${VLLM_VENV}/bin/python" "${WORKTREE}/experiments/test_qwen_logprobs.py" \
    --max_model_len 512 --top_k 20
fi

echo "== teacher OK =="
