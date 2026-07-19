#!/bin/bash
# Long-context continued-pretrain (LONGCTX_PLAN_2026_07_19.md) — T=8192,
# --no_doc_isolation (state flows across docs), 2-GPU --manual_allreduce
# (first production use; smoke: 52.4k tok/s global = 1.9x single-GPU).
# 5,500 steps x 131,072 tok/step global = 720M tokens ≈ 4h.
# Success metric = the frozen 8-32k cartridge sanity gate flipping positive;
# guards = HE-CE / depdist vs soup3 (see the plan's decision rule).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

STEPS=${STEPS:-5500}
TAG=${TAG:-production_lean_longctx}
BASE=${BASE:-checkpoints/production_lean_soup3.pt}
# Crash-resume (GPU0 is flaky under sustained load — 4 incidents): relaunch
# with BASE=<latest periodic ckpt> START_STEP=<its step>; WSD schedule and
# token accounting continue from there (re-pass everything else unchanged,
# per the resume mandate).
START_STEP=${START_STEP:-0}

nohup .venv/bin/torchrun --nproc_per_node=2 experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt $BASE --keep_base_vocab 49152 \
  --feedback none --manual_allreduce --manual_drift_check_every 500 \
  --data_mix configs/pretrain_mix_feature_pilot.yaml --fim_legacy_strings --no_doc_isolation \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 8192 --batch 1 --grad_accum 8 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-5 --lr_muon 2e-4 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 50 --lr_decay_frac 0.3 \
  --steps $STEPS --start_step $START_STEP --val_every 250 --log_every 20 --seed 0 \
  --mid_eval_every_tokens 100000000 --mid_eval_save_only --mid_eval_min_free_gib 2.0 \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG (torchrun 2-rank, PID $!) — tail -f runs/${TAG}.log"
