#!/bin/bash
# Latent-thinking ADAPTER + SELECTIVE co-train continuation (2026-06-01).
#
# The core bet: make LATENT (Coconut-style) thinking actually HELP the 600M
# DeltaNet code model. Two fixes land here (see thinking.py / model.py):
#   1. LatentFeedbackAdapter (--use_latent_feedback_adapter): maps the fed-back
#      out_norm hidden into the input-embedding manifold before it drives the
#      next latent think-step. Identity at start (zero-init proj + α), so this
#      ckpt loads byte-identically to the base and the adapter learns from the
#      co-train gradient. Fixes root-cause #1 (OOD untrained feedback).
#   2. --latent_cotrain_selective: samples co-train positions WEIGHTED toward
#      where thinking should help (high gate-think / high no-think entropy)
#      instead of uniform-random. Fixes root-cause #2 (uniform co-train tanked
#      VAL by forcing post-think competence everywhere).
#
# SHORT continuation (~300 steps) from the latest v9 base. Everything else
# matches resume_v9.sh. --no-compile is required (the latent extra-forwards
# crash Inductor; documented in CLAUDE.md / thinking.py). Runs on the FREE GPU.
#
# After it finishes, re-probe latent Δlogp (success criterion below):
#   CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. .venv/bin/python \
#       experiments/probe_gate_calibration.py \
#       --ckpt checkpoints/pretrain_v9_latent_adapter.pt \
#       --mechanism latent --latent_R 2 --wm_off --n_positions 4000
set -u
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=/home/knielsen/ml/pytorch-release:${PYTHONPATH:-}:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-1}

BASE_CKPT=${BASE_CKPT:-checkpoints/pretrain_v9_step17802_tok3500015616.pt}
START_STEP=$(echo "$BASE_CKPT" | sed -E 's/.*_step([0-9]+)_.*/\1/')
# IMPORTANT: keep --steps at the SAME 31000 as the base run so WSD stays in its
# STABLE (constant peak-LR) band — a short continuation must NOT enter the
# cosine decay tail, or "VAL didn't blow up" would be an LR-decay artifact, not
# the adapter. We DON'T want to run to 31000; we want ~300 steps. So we save a
# numbered mid-ckpt every ~50M tokens (≈ step 255 at 196k tok/step) and STOP the
# run manually once a `*_latent_adapter_step*` ckpt past ~step 18100 appears.
TARGET_STEPS=${TARGET_STEPS:-31000}
STOP_AFTER_STEP=$(( START_STEP + ${ADD_STEPS:-300} ))
LOG=runs/pretrain_v9_latent_adapter.log

echo "[latent-adapter] $(date '+%F %T') base=$BASE_CKPT start=$START_STEP stable-band-steps=$TARGET_STEPS stop≈$STOP_AFTER_STEP GPU=$GPU" | tee -a "$LOG"
echo "[latent-adapter] stop the run once checkpoints/pretrain_v9_latent_adapter_step*.pt past step $STOP_AFTER_STEP exists, then probe." | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 1280 --n_layers 10 --d_head 64 --n_heads 20 \
    --feedback film \
    --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 0 \
    --output_gate \
    --gate_entropy_aux_weight 0.1 \
    --gate_entropy_aux_temperature 2.0 \
    --gate_floor_min 0.5 --gate_warmup_steps 2000 \
    --state_readonly_at_think \
    --use_memory --mem_size 1536 \
    --use_pkm --pkm_after_layer 5 \
    --pkm_n_heads 4 --pkm_n_keys 384 --pkm_k_dim 128 --pkm_top_k 32 \
    --pkm_use_output_gate --pkm_value_init_std 1.0 --pkm_score_norm layer \
    --pkm_diversity_weight 0.01 \
    --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 \
    --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 \
    --pkm_value_lr_mult 100.0 \
    --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
    --use_latent_feedback_adapter \
    --latent_cotrain_weight 0.02 \
    --latent_cotrain_R 1 \
    --latent_cotrain_selective \
    --latent_cotrain_sample_frac 0.05 \
    --latent_cotrain_max_positions 12 \
    --data_mix configs/pretrain_mix_v4.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.25 --think_max_bursts 1 --think_max_burst_depth 4 \
    --T 2048 --batch ${BATCH:-4} --grad_accum 16 \
    --activation_checkpointing \
    --bf16 --tf32 --no-compile --bf16_optim_state \
    --alpha_wd 0.0 --wd 0.01 \
    --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 0 --lr_decay_frac 0.15 \
    --steps "$TARGET_STEPS" \
    --load_ckpt "$BASE_CKPT" \
    --start_step "$START_STEP" \
    --val_every 100 --log_every 25 \
    --mid_eval_every_tokens 50000000 --mid_eval_save_only \
    --save_ckpt checkpoints/pretrain_v9_latent_adapter.pt \
    --tb_dir runs/tb/pretrain_v9_latent_adapter \
    >> "$LOG" 2>&1
echo "[latent-adapter] $(date '+%F %T') train exited code=$?" | tee -a "$LOG"
