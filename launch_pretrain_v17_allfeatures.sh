#!/bin/bash
# v17 — FRESH from-scratch ALL-FEATURES pretrain, DDP (2026-06-18).
# User: "all features, latent thinking is our core value prop." Built per the
# validation agent's de-risked "path B + DDP fix" (see task #95):
#   = v13's validated day-1 latent recipe, changed only where defensible:
#   * DDP 2-GPU (torchrun) + the model.py adapter-touch fix (latent had NEVER run
#     under DDP; the adapter is only used in the aux forward → "marked ready twice"
#     without the fix). MANDATORY: run the smoke (LATENT_START=5) first.
#   * latent_reasoning_start_step 2000 (NOT 0): v13's step-0 killed the PKM
#     bootstrap (they fight in the 0-3000 alpha-floor window). Stagger latent so
#     PKM commits first. "Latent very early" = step 2000 of ~19000 is still early.
#   * gate_calibration OFF (0.0) — v13 had it off; it destabilizes cold trunks.
#   * WM = the session-validated NO-HASH ctx_namekey addresser + copy head +
#     maskfix recall data (first-occurrence mask), not v13's decoupled-KV.
#     UNVALIDATED fresh-at-high-LR — watch the leak-free recall kill-gate @500M.
# Success metric = ptr10dict_heldout latent lift + PKM alphaL commits (+0.27..+0.35)
#   + no VAL +10% spike + reason(loss) falling. NOT HumanEval (latent is orthogonal
#   to short-context code; see project_why_mechanisms_synthesis).
set -uo pipefail
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH="${PYTHONPATH:-}:."
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60
mkdir -p runs checkpoints runs/tb

STEPS=${STEPS:-19000}
LATENT_START=${LATENT_START:-2000}          # smoke overrides to ~5 to exercise the DDP adapter fix
BATCH=${BATCH:-4}
GA=${GA:-16}                                  # 2 GPU x b4 x ga16 x T2048 = 262k tok/step (== v13)
PORT=${PORT:-29714}
TAG=${TAG:-pretrain_v17}
EXTRA=${EXTRA:-}

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 .venv/bin/torchrun \
  --nproc_per_node=2 --master_port="$PORT" experiments/train_lm.py \
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
  --T 2048 --batch "$BATCH" --grad_accum "$GA" \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
  --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
  --steps "$STEPS" --start_step 0 --val_every 200 --log_every 20 \
  --mid_eval_every_tokens 250000000 --mid_eval_save_only \
  --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
  --save_ckpt "checkpoints/${TAG}.pt" --tb_dir "runs/tb/${TAG}" $EXTRA
