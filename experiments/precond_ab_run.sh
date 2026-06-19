#!/bin/bash
# Production A/B: Muon vs FUSED per-head DeltaNet-NS matrix optimizer.
# Fresh-from-scratch pretrain, SEQUENTIAL (muon arm fully, then fused arm),
# GPU 1 ONLY. v18 on GPU 0 is never touched. After both arms finish, runs the
# analyzer -> DELTANET_PRECONDITIONER_AB.md + CSVs.
#
# Fairness: identical seed/init/data-order/batch/T/schedule/steps for both arms;
# the ONLY difference is --matrix_optimizer. Clean optimizer probe: no
# thinking/WM/PKM/gate, think-burst injection OFF, --no-compile (guarantees
# bit-deterministic forwards so any loss divergence is PURELY the optimizer;
# data streaming is already single-worker-deterministic). Checkpoints are NOT
# saved (throwaway A/B, low disk); curves come from the logs + TensorBoard.
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/precond_ab
mkdir -p $OUT runs/tb

STEPS=2500
BATCH=12
GA=6
KWARM=200
T=2048

cat > $OUT/meta.json <<JSON
{"batch": $BATCH, "T": $T, "grad_accum": $GA, "steps": $STEPS, "k_warmup": $KWARM}
JSON

run_arm() {
  local arm=$1; shift
  echo "[driver] $(date '+%F %T') START arm=$arm" | tee -a $OUT/driver.log
  # peak-memory sampler for PHYSICAL GPU 1 (does not touch GPU 0 / v18)
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
    --optimizer muon "$@" --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.15 \
    --steps $STEPS --val_every 100 --log_every 20 --seed 0 \
    --tb_dir runs/tb/precond_ab_$arm \
    > $OUT/$arm.log 2>&1
  local rc=$?
  kill $samp 2>/dev/null
  echo "[driver] $(date '+%F %T') END arm=$arm rc=$rc" | tee -a $OUT/driver.log
}

run_arm muon
run_arm fused --matrix_optimizer fused_deltanet_ns

echo "[driver] $(date '+%F %T') ANALYZE" | tee -a $OUT/driver.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/precond_ab_analyze.py $OUT \
    >> $OUT/driver.log 2>&1
echo "[driver] $(date '+%F %T') ALL DONE" | tee -a $OUT/driver.log
