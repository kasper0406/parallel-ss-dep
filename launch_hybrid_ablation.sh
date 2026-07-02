#!/usr/bin/env bash
# ===========================================================================
# Is the DeltaNet linear-attention tax REDUCIBLE? — stage-2 ablation launcher.
#
# Linearize SmolLM2-360M -> 32L DeltaNet (inherit non-attn weights, MOHAWK
# stage-2 attention transfer). Sweep the attention-replacement across 4 arms,
# measuring per-layer rel-MSE + stage-2 full-forward HumanEval-solution CE at
# ~45M tokens/arm (cheap, stage-2 only).
#
# REFERENCE NUMBERS (plain DeltaNet, from linearize_smollm2.py):
#   per-layer rel-MSE floor ~0.20 ; stage-2 HumanEval-solution CE floor ~1.005
#   (donor SmolLM2-360M teacher CE = 0.6142 ; our-from-scratch SFT = 0.9716)
#
# KILL-SIGNAL (the strategic decision):
#   The `hybrid` arm (some layers kept as REAL softmax attention) MUST drop the
#   stage-2 CE floor by >15-20% vs `baseline`. If it does NOT, the linear-
#   attention tax is STRUCTURAL (not fixable by a more expressive cell) and the
#   escape is a Jamba/Zamba hybrid stack, not a better linear cell.
#   The hybrid layers' per-layer rel-MSE should be << 0.20 (a real attention
#   layer can match a real attention layer — donor weights inherit byte-exact).
#
# Single GPU ONLY. GPU0 falls off the bus under load — never use it.
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PYTHONPATH:-}:."

OUT_DIR="checkpoints/hybrid_ablation"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "logging to $LOG"
# deltaproduct runs LAST: a CUDA "misaligned address" on sm_120 is sticky
# (poisons the process for all later arms), so an isolated crash there cannot
# cost the strategic `hybrid` kill-signal answer.
.venv/bin/python experiments/linearize_hybrid_ablation.py \
    --arms baseline,wide_dhead,hybrid,deltaproduct \
    --out_dir "$OUT_DIR" \
    --layerwise_tokens 45000000 \
    --T 1024 --batch 12 \
    --lr_layerwise 1e-3 \
    --eval_every_tokens 25000000 \
    --num_householder 2 \
    --wide_head_dim 128 \
    --hybrid_layers "7,15,23,31" \
    2>&1 | tee "$LOG"

echo "=== done. per-arm JSON + summary.json in $OUT_DIR ==="
