#!/bin/bash
# SFT v12 — HumanEval-shaped training data (2026-05-27).
#
# Decision D7 from AUTONOMY_DECISIONS.md.
#
# The pivot after confirming user's distribution-mismatch hypothesis:
#   - DPO v2 step250 regressed 16→12 even with strong β=0.3 KL anchor
#   - Inspector showed DPO produced different code on 38/40 problems but
#     same pass rate on first 40 — the regression is in problems 40-163
#   - Two parallel research agents independently identified the cause:
#     MBPP-style "Write a function to..." rejection data trains the
#     model to emit a fresh ```python\\ndef ...``` from scratch, which
#     is structurally wrong for HumanEval's signature+docstring "complete
#     the body" task
#   - Agent B found: `data/synthetic_pyfunc.jsonl` (6501 rows) is the
#     ONLY on-disk corpus in HumanEval's exact shape. Never used for SFT.
#
# v12 = SFT v2_step300 on data/sft_synth_pyfunc.jsonl (converted from
# synthetic_pyfunc with split-at-docstring-close).
#
# Decision-gate:
#   - eval result > 16/164 → distribution hypothesis vindicated, this
#     IS the right data shape, scale it (10× synthetic_pyfunc generator)
#   - eval result = 16/164 ± 1 → SFT can't beat v2 either; ceiling real
#   - eval result < 14/164 → SFT itself regresses regardless of data
#     shape — the SFT recipe is fundamentally broken (which would
#     contradict the agent's diagnosis; investigate)

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v2_step300.pt \
    --save_ckpt checkpoints/sft_v12_synth_pyfunc.pt \
    --distilled_jsonl data/sft_synth_pyfunc.jsonl \
    --distilled_code_only \
    --with_thinking \
    --retrieval_as_input_thinking \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --max_codealpaca 0 \
    --epochs 1 \
    --batch 2 \
    --lr 3e-6 \
    --max_len 1024 \
    --log_every 100 \
    --seed 0 \
    > runs/sft_v12_synth_pyfunc.log 2>&1 &
echo "Launched SFT v12 (synth_pyfunc HumanEval-shape) — PID $!"
echo "Watch: tail -f runs/sft_v12_synth_pyfunc.log"
