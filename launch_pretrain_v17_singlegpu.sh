#!/bin/bash
# v17 — FRESH from-scratch ALL-FEATURES pretrain, SINGLE-GPU (2026-06-18).
# DDP path abandoned after investigation (see task #95): latent's reentrant
# multi-use needs static_graph (Error B), but static_graph (a) hits a genuine
# unfixed PyTorch bug under grad-accum no_sync (Error A, PR #103487 regression,
# present in origin/main) AND (b) is fundamentally incompatible with our VARIABLE
# latent graph (depth curriculum R=2..8 + step-2000 engagement → graph changes
# → reducer.cpp:731). Manual-allreduce DDP is possible (~30-50 lines) but yields
# <2x on PCIe-no-NVLink; single-GPU is the validated path (every latent run was
# single-GPU; smoke 2026-06-18 passed: reason-loss 10.9→1.7, WM+PKM+gist all live).
# = v13's validated recipe + the de-risk changes: latent staggered to step 2000
# (PKM bootstraps first), gate-calib OFF, WM=ctx_namekey+maskfix data.
set -uo pipefail
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH="${PYTHONPATH:-}:."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p runs checkpoints runs/tb
GPU=${GPU:-0}
STEPS=${STEPS:-19000}
LATENT_START=${LATENT_START:-2000}
TAG=${TAG:-pretrain_v17}

CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
  --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
  --feedback_self_k 3 --feedback_self_k_warmup_steps 1300 \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
  --gate_floor_min 0.5 --gate_warmup_steps 20000 --state_readonly_at_think \
  --use_memory --mem_size 2048 --mem_decoupled_kv \
  --mem_ctx_namekey --ctx_namekey_dim 192 --ctx_namekey_match_threshold 0.5 \
  --mem_always_read --use_copy_head --mem_copy_require_match \
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -6.0 \
  --mem_read_alpha_init 0.0 --mem_freeze_read_alpha --emit_read_mask \
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 3000 --ctx_addr_aux_start_step 0 \
  --use_pkm --pkm_after_layer 5 \
  --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 \
  --pkm_value_init_std 1.0 --pkm_score_norm layer --pkm_diversity_weight 0.01 \
  --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 --pkm_value_lr_mult 100.0 \
  --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
  --latent_cotrain_weight 0.0 --gate_calibration_weight 0.0 \
  --use_latent_feedback_adapter \
  --latent_reasoning_weight 0.05 --latent_reasoning_train_prefix data/ptr10dict_train \
  --latent_reasoning_rungs 2,3,4,5,6,7,8 --latent_reasoning_n 4 \
  --latent_reasoning_gate_weight 0.05 \
  --latent_reasoning_start_step "$LATENT_START" --latent_reasoning_weight_warmup_steps 3000 \
  --data_mix configs/pretrain_mix_v14_wmrecall_maskfix.yaml --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
  --T 2048 --batch 4 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
  --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
  --steps "$STEPS" --start_step 0 --val_every 200 --log_every 20 \
  --mid_eval_every_tokens 250000000 --mid_eval_save_only \
  --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
  --save_ckpt "checkpoints/${TAG}.pt" --tb_dir "runs/tb/${TAG}" \
  > "runs/${TAG}.log" 2>&1 &
echo "launched v17 single-GPU on GPU $GPU (PID $!) -> runs/${TAG}.log"
