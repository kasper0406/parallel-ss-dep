#!/bin/bash
# Calibration hypothesis test: does eval-time emit_threshold alone
# account for v4's +4-5 problems over the SFT base?
#
# v4 at emit_threshold=0.5 produced think_rate=0.62 because RL+entropy
# pushed σ(gate) higher. Raising emit_threshold on the UNTRAINED SFT
# base should produce a similar think_rate and (hypothesis) a similar
# pass rate — if so, gate calibration is the entire story.
#
# Sweep: 0.5 (baseline, known 7-8/164) → 0.6 → 0.7 → 0.8
# All eval kwargs identical to the v4 eval otherwise.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p runs

CKPT="checkpoints/sft_phase_c_combined.pt"

for ET in 0.5 0.6 0.7 0.8; do
    LOG="runs/sweep_emit_${ET}.log"
    echo "=== emit_threshold=$ET → $LOG ==="
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/eval_humaneval.py \
        --ckpt "$CKPT" \
        --max_problems 164 \
        --prompt_style sft_comment \
        --extract_code_block \
        --use_thinking \
        --emit_threshold "$ET" \
        --gate_floor 0.0 \
        --min_emit_before_eos 30 \
        --max_gen 256 \
        --temperature 0.0 \
        --generator retrieval_as_input \
        > "$LOG" 2>&1
    echo "  result: $(grep -E '^pass@1' $LOG | tail -1)"
    echo "  think_rate: $(grep -oE 'think_rate=[0-9.]+' $LOG | tail -1)"
done
echo "DONE."
