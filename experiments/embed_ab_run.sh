#!/bin/bash
# A/B: does a BETTER EMBEDDING OPTIMIZER beat shared-LR AdamW on 287M DeltaNet
# pretrain? Arms vary ONLY the embedding/lm_head treatment; matrix optimizer =
# Muon in every arm (held fixed) so the embedding effect is isolated AND
# directly comparable to runs/precond_ab/{muon,fused}.log (the reused baselines).
#
# GPU 1 ONLY. v18 on GPU 0 is never touched. Sequential (one arm fully, then
# the next). Checkpoints are NOT saved (throwaway A/B, low disk).
#
# FAIRNESS: identical seed/init/data-order/batch/T/steps as the precond_ab
# baseline. The ONLY schedule difference vs the 2500-step baseline is we run
# 1500 steps with --lr_decay_frac 0.0 so the LR is warmup(200)->constant-peak
# for the WHOLE short run — which is EXACTLY what the baseline's first 1500
# steps are (its WSD decay does not start until step 2125). So matched-step
# comparison vs the baseline's first 1500 steps is apples-to-apples (no
# decay-tail confound).
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/embed_ab
mkdir -p $OUT runs/tb

STEPS=${STEPS:-1500}
BATCH=12
GA=6
KWARM=200
T=2048

cat > $OUT/meta.json <<JSON
{"batch": $BATCH, "T": $T, "grad_accum": $GA, "steps": $STEPS, "k_warmup": $KWARM}
JSON

run_arm() {
  local arm=$1; shift
  echo "[driver] $(date '+%F %T') START arm=$arm  ($*)" | tee -a $OUT/driver.log
  ( while true; do \
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1; \
      sleep 3; done ) > $OUT/$arm.mem 2>/dev/null &
  local samp=$!
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 --feedback_self_k_warmup_steps $KWARM \
    --data_mix configs/pretrain_mix_v4.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0 \
    --T $T --batch $BATCH --grad_accum $GA \
    --activation_checkpointing \
    --bf16 --tf32 --no-compile --bf16_optim_state \
    --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.0 \
    --steps $STEPS --val_every 100 --log_every 20 --seed 0 \
    --tb_dir runs/tb/embed_ab_$arm \
    "$@" \
    > $OUT/$arm.log 2>&1
  local rc=$?
  kill $samp 2>/dev/null
  echo "[driver] $(date '+%F %T') END arm=$arm rc=$rc" | tee -a $OUT/driver.log
}

# Sweep: μP-flavoured higher embedding AdamW LR (2x/5x/10x) + the rownorm
# (modular-norm) dualizer at the base LR. All Muon matrix optimizer.
run_arm lr2     --embed_lr_mult 2
run_arm lr5     --embed_lr_mult 5
run_arm lr10    --embed_lr_mult 10
run_arm rownorm --embed_optimizer rownorm

echo "[driver] $(date '+%F %T') ANALYZE" | tee -a $OUT/driver.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/embed_ab_analyze.py $OUT \
    >> $OUT/driver.log 2>&1
echo "[driver] $(date '+%F %T') ALL DONE" | tee -a $OUT/driver.log
