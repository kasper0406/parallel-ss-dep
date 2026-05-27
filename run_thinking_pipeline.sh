#!/bin/bash
# THINKING_PLAN post-Phase-D execution pipeline.
#
# Runs the full evaluation + integration sequence once Phase D pretrain
# completes. Each step writes to a clearly-named log file so failures
# are obvious. Designed to be re-runnable: each step checks for its
# input artifact and skips if already done.
#
# Sequence:
#  1. Run Phase 1 diagnostics on the final Phase D ckpt
#     → THINKING_PROBE_RESULTS.md
#  2. Generate CoT distill data via Qwen (multi-hour GPU)
#     → data/cot_distill_v1.jsonl
#  3. Build CoT-as-thinking SFT data
#     → data/sft_cot_thinking_v1.jsonl
#  4. SFT the Phase D ckpt on the CoT data + a clean code mix
#     → checkpoints/sft_phase_d_cot_thinking.pt
#  5. HumanEval the SFT'd ckpt
#     → runs/eval_humaneval_sft_phase_d_cot_thinking.log
#  6. Long-context recall (state-readonly should preserve recall through
#     multi-think bursts)
#     → runs/eval_longctx_sft_phase_d.log
#
# Each step is gated on the previous step's artifact existing. The
# script is meant to be run AFTER Phase D's pretrain_phase_d.pt is
# saved.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

PHASE_D_CKPT="checkpoints/pretrain_phase_d.pt"
if [[ ! -e "$PHASE_D_CKPT" ]]; then
    echo "ERROR: Phase D ckpt not found: $PHASE_D_CKPT"
    echo "       Wait for pretrain_phase_d.sh to finish first."
    exit 1
fi

GPU=${GPU:-0}
echo "== Running thinking pipeline against $PHASE_D_CKPT on GPU $GPU =="

# ---------------------------------------------------------------------------
# Step 1: diagnostics
# ---------------------------------------------------------------------------
DIAG_OUT="runs/thinking_diagnostics_phase_d.json"
if [[ -e "$DIAG_OUT" ]]; then
    echo "[skip] $DIAG_OUT exists"
else
    echo "[1/6] Running Phase 1 diagnostics → $DIAG_OUT"
    # 1a. per-position CE delta
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u \
        experiments/probe_thinking_per_position_ce.py \
        --ckpt "$PHASE_D_CKPT" \
        --n_positions 200 \
        > runs/thinking_probe_ce_delta.log 2>&1 || {
            echo "[1a FAILED] see runs/thinking_probe_ce_delta.log"; }
    # 1b. counterfactual sampling
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u \
        experiments/probe_thinking_counterfactual.py \
        --ckpt "$PHASE_D_CKPT" \
        > runs/thinking_probe_counterfactual.log 2>&1 || {
            echo "[1b FAILED] see runs/thinking_probe_counterfactual.log"; }
    # 1c. RL correlation (uses real rollouts)
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u \
        experiments/probe_thinking_rl_correlation.py \
        --ckpt "$PHASE_D_CKPT" --n_problems 30 \
        > runs/thinking_probe_rl_correlation.log 2>&1 || {
            echo "[1c FAILED] see runs/thinking_probe_rl_correlation.log"; }
    touch "$DIAG_OUT"  # marker
fi

# ---------------------------------------------------------------------------
# Step 2: CoT distill data via Qwen
# ---------------------------------------------------------------------------
COT_DATA="data/cot_distill_v1.jsonl"
if [[ -e "$COT_DATA" && "$(wc -l <"$COT_DATA")" -gt 100 ]]; then
    echo "[skip] $COT_DATA exists with $(wc -l <"$COT_DATA") rows"
else
    echo "[2/6] Generating CoT distill data → $COT_DATA (multi-hour)"
    CUDA_VISIBLE_DEVICES=$GPU .venv-vllm/bin/python -u \
        experiments/gen_cot_distill_data.py \
        --dataset mbpp_combined \
        --n_per_problem 4 \
        --out "$COT_DATA" \
        > runs/gen_cot_distill.log 2>&1
fi

# ---------------------------------------------------------------------------
# Step 3: Build CoT-as-thinking SFT data
# ---------------------------------------------------------------------------
SFT_DATA="data/sft_cot_thinking_v1.jsonl"
if [[ -e "$SFT_DATA" ]]; then
    echo "[skip] $SFT_DATA exists"
else
    echo "[3/6] Building CoT-as-thinking SFT data → $SFT_DATA"
    .venv/bin/python -u experiments/build_cot_sft_data.py \
        --in "$COT_DATA" --out "$SFT_DATA"
fi

# ---------------------------------------------------------------------------
# Step 4: SFT the Phase D ckpt
# ---------------------------------------------------------------------------
SFT_CKPT="checkpoints/sft_phase_d_cot_thinking.pt"
if [[ -e "$SFT_CKPT" ]]; then
    echo "[skip] $SFT_CKPT exists"
else
    echo "[4/6] SFT-ing Phase D ckpt on CoT data → $SFT_CKPT"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/sft_code.py \
        --load_ckpt "$PHASE_D_CKPT" \
        --save_ckpt "$SFT_CKPT" \
        --distilled_jsonl "$SFT_DATA" \
        --distilled_keep_only_passing \
        --with_thinking \
        --max_codealpaca 0 \
        --epochs 1 \
        --batch 4 \
        --lr 5e-6 \
        --max_len 1536 \
        --log_every 20 \
        --seed 0 \
        > runs/sft_phase_d_cot_thinking.log 2>&1
fi

# ---------------------------------------------------------------------------
# Step 5: HumanEval the SFT'd ckpt
# ---------------------------------------------------------------------------
HE_LOG="runs/eval_humaneval_sft_phase_d_cot_thinking.log"
if [[ -e "$HE_LOG" ]]; then
    echo "[skip] $HE_LOG exists"
else
    echo "[5/6] HumanEval on $SFT_CKPT"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/eval_humaneval.py \
        --ckpt "$SFT_CKPT" \
        --use_thinking --max_think_per_step 8 --total_think_budget 400 \
        --emit_threshold 0.5 --max_gen 256 \
        --prompt_style sft_comment --extract_code_block \
        > "$HE_LOG" 2>&1
    grep "pass@1 = " "$HE_LOG" | tail -1
fi

# ---------------------------------------------------------------------------
# Step 6: long-context recall (state-readonly validation)
# ---------------------------------------------------------------------------
LR_LOG="runs/eval_longctx_sft_phase_d.log"
if [[ -e "$LR_LOG" ]]; then
    echo "[skip] $LR_LOG exists"
else
    echo "[6/6] Long-context recall on $SFT_CKPT"
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/eval_longctx_recall.py \
        --ckpt "$SFT_CKPT" \
        > "$LR_LOG" 2>&1 || {
            echo "[6 FAILED] see $LR_LOG"; }
fi

echo "== Pipeline complete =="
echo "Summary log: cat $HE_LOG | tail -10"
