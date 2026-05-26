#!/bin/bash
# SFT Phase D with MIXED data: standard Qwen distillation (anti-forgetting +
# "produce code" demonstrations) + CoT-thinking rows (gate-supervision +
# "spend think budget then emit"). The CoT-thinking-only run got 0/164
# because Phase D ckpt has never been SFT'd on standard code-completion
# at all — model jumped straight from pretrain to narrow CoT-only SFT.
#
# This is the proper experiment: Phase D + (Qwen distill ∪ CoT-thinking)
# vs the historical Phase C + Qwen distill baseline (10/164).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p data runs checkpoints

MIXED_DATA="data/sft_mixed_qwen_cot.jsonl"

# Build the mix if it doesn't exist (preserves the CoT rows verbatim;
# concatenates and shuffles with a fixed seed).
if [[ ! -e "$MIXED_DATA" ]]; then
    echo "[mix] building $MIXED_DATA"
    .venv/bin/python -c "
import json, random, sys
rng = random.Random(0)
out = []
with open('data/distill_v7_phase1_with_tests.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            out.append(line)
n_qwen = len(out)
with open('data/sft_cot_thinking_v1.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            out.append(line)
n_cot = len(out) - n_qwen
rng.shuffle(out)
with open('$MIXED_DATA', 'w') as f:
    for line in out:
        f.write(line + '\n')
print(f'[mix] wrote {len(out)} rows ({n_qwen} qwen + {n_cot} cot)')
"
fi

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_phase_d.pt \
    --save_ckpt checkpoints/sft_phase_d_mixed.pt \
    --distilled_jsonl "$MIXED_DATA" \
    --distilled_keep_only_passing \
    --with_thinking \
    --max_codealpaca 0 \
    --epochs 1 \
    --batch 4 \
    --lr 5e-6 \
    --max_len 1536 \
    --log_every 50 \
    --seed 0 \
    > runs/sft_phase_d_mixed.log 2>&1 &
echo "Launched mixed SFT on Phase D — PID $!"
echo "Watch: tail -f runs/sft_phase_d_mixed.log"
