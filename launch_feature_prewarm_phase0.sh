#!/bin/bash
# PHASE-0: frozen-trunk FEATURE PRE-WARM (2026-07-03) — the unified fix for
# the converged-base cold-start deadlock, replacing ALL forced-contribution
# floors (the pilot-B post-mortem: floors on a converged base are sabotage —
# the trunk spends the run fighting structured noise; pilot-B ended worse
# than lean at EVERY dependency stratum AND at recall).
#
# Trunk FROZEN for the whole phase -> loss pressure can only flow into the
# feature parameters -> features must become GENUINELY USEFUL (the validated
# PKM pre-warm dynamic: alpha 0.10->0.27 with VAL recovering; survives
# unfreeze with the value table growing).
#
# Pre-warms: PKM (warm alpha, no floor), WM (read_alpha warm-started 0.05,
# UNFROZEN, no floor; copy head + ctx addressing on the recall streams),
# FiLM (alpha warm 0.02; W learns useful directions frozen-trunk). Latent
# adapter attached (zero-init-safe) but NO latent aux — latent capability is
# trained post-hoc adapter-only (validated to preserve the base byte-exact).
# Gate: floor pinned 1.0 (NO CE re-weighting) + entropy-aux only.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

GPU=${GPU:-1}
STEPS=${STEPS:-600}
TAG=${TAG:-feature_prewarm_phase0}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/linearize/linearized_stage3.pt --keep_base_vocab 49152 \
  --data_mix configs/pretrain_mix_feature_pilot.yaml \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 20 --lr_decay_frac 0.0 \
  --freeze_trunk_steps $STEPS \
  --feature_lr_mult 3.0 \
  --feedback film --feedback_pairs "0,16;4,20;8,24;12,28" \
  --feedback_self_k 3 --feedback_self_k_warmup_steps 20 \
  --feedback_alpha_init 0.02 \
  --use_pkm --pkm_after_layer 16 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_out_alpha_init 0.1 \
  --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 540 \
  --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_alpha_floor_start 0.0 --pkm_alpha_floor_warmup_steps 0 \
  --pkm_value_lr_mult 233.0 \
  --use_memory --mem_size 1024 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192 \
  --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match \
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -2.0 --emit_read_mask \
  --mem_read_alpha_init 0.05 \
  --mem_read_alpha_floor_start 0.0 --mem_read_alpha_floor_warmup_steps 0 \
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 240 --ctx_addr_aux_start_step 0 \
  --use_latent_feedback_adapter --latent_reasoning_weight 0.0 \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
  --gate_floor_min 1.0 --gate_warmup_steps 1 \
  --steps $STEPS --val_every 200 --log_every 20 --seed 0 \
  --mid_eval_every_tokens 100000000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!)"
