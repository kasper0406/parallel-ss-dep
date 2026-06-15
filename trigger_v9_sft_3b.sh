#!/bin/bash
# Auto-trigger: when v9 pretrain crosses ~3B tokens, SFT that checkpoint (same
# recipe as the v8 combined SFT) and then answer the two questions that gate the
# whole direction:
#   1. Is v9 a better BASE?  -> HumanEval (thinking OFF) vs the v8 SFT's 13/164.
#   2. Does THINKING help on the stronger base? -> latent Δlogp probe (on v8 it
#      was strongly negative; we want it less-negative / positive).
# Runs on GPU0 (waits for the v8 RL smoke to finish + frees the judge first).
# v9 keeps training on GPU1, untouched.
set -u
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=/home/knielsen/ml/pytorch-release:${PYTHONPATH:-}:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/trigger_v9_sft_3b.log
STATUS=runs/trigger_v9_sft_3b.status
TARGET_TOK=3000000000
mkdir -p runs checkpoints data
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 1) Wait for a v9 checkpoint at >= 3B tokens.
log "waiting for a v9 checkpoint >= 3B tokens (current target $TARGET_TOK) ..."
V9=""
while [ -z "$V9" ]; do
    for f in $(ls -t checkpoints/pretrain_v9_step*_tok*.pt 2>/dev/null); do
        tok=$(echo "$f" | sed -E 's/.*_tok([0-9]+)\.pt/\1/')
        if [ "${tok:-0}" -ge "$TARGET_TOK" ] 2>/dev/null; then V9="$f"; break; fi
    done
    [ -z "$V9" ] && sleep 300
done
log "v9 >=3B checkpoint: $V9"
echo "ckpt=$V9" > "$STATUS"

# 2) Wait for GPU0 to be free of the v8 RL smoke, then free the judge server.
log "waiting for the v8 RL smoke (train_rl_grader) to finish ..."
while pgrep -f "train_rl_grader.py" >/dev/null; do sleep 120; done
pkill -f "vllm serve" 2>/dev/null && log "stopped judge server (free GPU0)"; sleep 8
while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    [ "${free:-0}" -ge 22000 ] 2>/dev/null && break
    sleep 60
done
log "GPU0 free (${free} MiB)"

# 3) SFT the v9 checkpoint (v8 combined recipe, pointed at v9).
cat data/distill_v7_phase1_with_tests.jsonl data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl > data/sft_v9_3b.jsonl
log "SFT v9 -> checkpoints/sft_v9_3b.pt ($(wc -l < data/sft_v9_3b.jsonl) rows)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt "$V9" --save_ckpt checkpoints/sft_v9_3b.pt \
    --distilled_jsonl data/sft_v9_3b.jsonl \
    --with_thinking --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 --mem_size 1536 \
    --epochs 2 --batch ${BATCH:-4} --lr 3e-5 --max_len 1024 --log_every 100 \
    >> "$LOG" 2>&1 || { log "SFT FAILED (see $LOG)"; exit 1; }
log "SFT done"

# 4) HumanEval, thinking OFF (the headline metric) vs the v8 SFT baseline 13/164.
log "HumanEval (thinking off) on sft_v9_3b ..."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/eval_humaneval.py \
    --ckpt checkpoints/sft_v9_3b.pt --prompt_style sft_comment --extract_code_block \
    --max_gen 320 --min_emit_before_eos 30 --gate_floor 0.0 --emit_threshold 0.0 \
    > runs/eval_sft_v9_3b.log 2>&1
HE=$(grep -oE "pass@[0-9]+ = [0-9.]+ +\([0-9]+/[0-9]+\)" runs/eval_sft_v9_3b.log | tail -1)
log "HumanEval sft_v9_3b: ${HE:-PARSE_FAIL}   (v8 SFT baseline = 13/164)"
echo "humaneval=${HE:-NA}" >> "$STATUS"

# 5) Thinking diagnostic: does latent thinking help on the stronger SFT'd base?
log "thinking probe (latent Δlogp, R=4) on sft_v9_3b ..."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/probe_gate_calibration.py \
    --ckpt checkpoints/sft_v9_3b.pt --mechanism latent --latent_R 4 \
    --n_positions 2000 --batch 2 > runs/probe_thinking_sft_v9_3b.log 2>&1 \
    && log "thinking probe done -> runs/probe_thinking_sft_v9_3b.log" \
    || log "thinking probe errored (see runs/probe_thinking_sft_v9_3b.log)"

log "=== TRIGGER COMPLETE ===  HumanEval=${HE:-NA}"
echo "done" >> "$STATUS"
