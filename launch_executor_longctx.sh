#!/bin/bash
# SEARCH_NATIVE_PLAN_2026_07_19.md Phase 1: Stage-A exec-trace re-attach on
# the long-context base. Recipe = original Stage-A verbatim (stageA mix,
# T=2048, isolation on, muon 6e-4/2e-3 wsd) but 2-GPU manual allreduce.
# 2,300 steps x 131,072 tok/step global = 300M tokens ≈ 1.6h.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

STEPS=${STEPS:-2300}
TAG=${TAG:-executor_longctx}
BASE=${BASE:-checkpoints/production_lean_longctx.pt}
START_STEP=${START_STEP:-0}

nohup .venv/bin/torchrun --nproc_per_node=2 experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt $BASE --keep_base_vocab 49152 \
  --feedback none --manual_allreduce --manual_drift_check_every 500 \
  --data_mix configs/pretrain_mix_stageA_executor.yaml --fim_legacy_strings \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 16 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 50 --lr_decay_frac 0.85 \
  --steps $STEPS --start_step $START_STEP --val_every 250 --log_every 20 --seed 0 \
  --mid_eval_every_tokens 100000000 --mid_eval_save_only --mid_eval_min_free_gib 2.0 \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG (torchrun 2-rank, PID $!) base=$BASE — tail -f runs/${TAG}.log"
