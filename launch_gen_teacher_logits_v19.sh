#!/bin/bash
# STEP 1 of the v19 distillation production run: offline-generate SmolLM2-1.7B
# top-k teacher logits over the pretraining stream, to disk (sharded safetensors).
# Then the student (STEP 2, launch_distill_v19_1b.sh) trains reading these — full
# speed, no teacher in the loop.
#
# DISK: top_k=16 → ~100 bytes/token → ~1.5B tokens ≈ 150 GB (fits the ~390 GB free).
# Bump --max_tokens for a bigger run only if disk allows (2B ≈ 200 GB).
#
# *** ALIGNMENT (load-bearing) *** — the student trainer reads these logits in
# LOCKSTEP with its data iterator and ASSERTS stored input_ids == live tokens.
# So STEP 2 MUST use the IDENTICAL stream-determining knobs as set here:
#   --data_mix, --tokenizer, --seed, --T, --num_workers, --think_burst_prob 0,
#   --mask_eos_in_targets. (batch / grad_accum may differ — the reader is a flat
#   token cursor.) Any mismatch aborts STEP 2 loudly at the first KD step.
#
# Teacher = SmolLM2-1.7B (SHARES our SmolLM2 tokenizer → exact KD, no tokenizer
# switch, no ULD). Runs teacher-forward only; put it on the free GPU.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=${GPU:-0}                       # teacher inference; GPU0 ok (inference-only is lighter)
MIX=${MIX:-configs/pretrain_mix_v18_arxiv.yaml}   # full data incl arXiv (user pref); MIX=configs/pretrain_mix_v4.yaml for code-only
OUT=${OUT:-data/teacher_logits_smollm17_v19}
MAXTOK=${MAXTOK:-1500000000}        # 1.5B tokens ≈ 150 GB at k16
TOPK=${TOPK:-16}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/gen_teacher_logits.py \
  --teacher_model HuggingFaceTB/SmolLM2-1.7B \
  --data_mix "$MIX" \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --out_dir "$OUT" \
  --top_k $TOPK --max_tokens $MAXTOK \
  --T 2048 --batch 8 --num_workers 0 --seed 0 \
  --think_burst_prob 0 --mask_eos_in_targets \
  --teacher_dtype bfloat16 --log_every 50 \
  > runs/gen_teacher_logits_v19.log 2>&1 &
echo "Launched teacher-logit gen (SmolLM2-1.7B → $OUT), GPU $GPU (PID $!)"
echo "Watch: tail -f runs/gen_teacher_logits_v19.log ; du -sh $OUT"
