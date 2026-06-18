#!/bin/bash
# v16 — FRESH from-scratch CODE-FOCUSED 287M pretrain (2026-06-18).
# GOAL: re-establish a CLEAN code base + working from-scratch -> SFT -> RL pipeline,
# after the mechanism/recall-heavy runs (v10-v15) cost the code headline
# (v12-SFT 8/164 vs code-focused Phase-C 14/164; see project_why_mechanisms_synthesis).
# = the proven v7.1-pkm-film recipe (strongest clean base, VAL 5.83) STRIPPED of the
# thinking/recall apparatus (WM, output-gate, latent, gate-calib, gist, think-burst),
# keeping only the validated PPL/knowledge lifts (FiLM K=3 + PKM-v7.1), + DDP both GPUs.
# Data = v4 (clean, code-heavy, NO recall streams).
set -euo pipefail
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH="${PYTHONPATH:-}:."
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints runs/tb

STEPS=${STEPS:-16000}        # ~5.5B tokens at 344k tok/step (2 GPU x b12 x ga7 x T2048)
BATCH=${BATCH:-12}           # 12 (not 14) for memory headroom: smoke measured 29.5GiB
GA=${GA:-7}                  # at b14+K3; b12 ~26GiB. 12*7 == 14*6 -> identical tok/step.

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 nohup .venv/bin/torchrun \
  --nproc_per_node=2 --master_port=29713 experiments/train_lm.py \
  --arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
  --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
  --feedback_self_k 3 --feedback_self_k_warmup_steps 0 \
  --use_pkm --pkm_after_layer 5 \
  --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate \
  --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 2000 \
  --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 2000 \
  --pkm_value_lr_mult 100.0 \
  --data_mix configs/pretrain_mix_v4.yaml --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0.0 \
  --T 2048 --batch "$BATCH" --grad_accum "$GA" \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
  --lr_schedule wsd --warmup_steps 600 --lr_decay_frac 0.15 \
  --steps "$STEPS" --start_step 0 \
  --val_every 400 --log_every 50 \
  --mid_eval_every_tokens 500000000 --mid_eval_save_only \
  --save_ckpt checkpoints/pretrain_v16_code.pt \
  --tb_dir runs/tb/pretrain_v16_code \
  > runs/pretrain_v16_code.log 2>&1 &
echo "launched v16 code pretrain (DDP 2-GPU) pid $! -> runs/pretrain_v16_code.log"
echo "steps=$STEPS batch=$BATCH grad_accum=$GA  (~$(python3 -c "print(round(2*$BATCH*$GA*2048*$STEPS/1e9,2))")B tokens)"
