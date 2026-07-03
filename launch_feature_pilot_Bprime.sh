#!/bin/bash
# Feature-pilot ARM B' (2026-07-03) — the REVISED all-features arm, applying
# every pilot-B post-mortem fix. Iso-token twin of arm A (same mix / seed /
# steps / LR); differs from the failed arm B by:
#   1. Starts from the PHASE-0 pre-warmed features (frozen-trunk phase where
#      PKM/WM/FiLM became genuinely useful) instead of no-op inits + floors.
#   2. NO forced-contribution floors anywhere (PKM alpha-floor, WM read-alpha
#      floor: gone — they made the trunk fight noise all run).
#   3. NO latent aux on the trunk (adapter-only capability training happens
#      post-hoc; the ptr10dict OOD trunk gradients were a top tax suspect).
#   4. Gate floor pinned 1.0: full LM gradient (no 0.8x CE re-weighting);
#      the gate head trains from the entropy-aux alone.
# SUCCESS GATE (vs pilot A = 0.7429 HE-CE, 1.1687 stratified total, 67%
# in-window recall): B' within ~0.03 on code metrics, recall >= A, engagement
# report green with PKM committed.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

GPU=${GPU:-1}
STEPS=${STEPS:-2500}
SEED=${SEED:-0}
TAG=${TAG:-feature_pilot_Bprime}
ENGAGE_STEP=${ENGAGE_STEP:-1000}
ENGAGE_ACTION=${ENGAGE_ACTION:-warn}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/feature_prewarm_phase0.pt --keep_base_vocab 49152 \
  --data_mix configs/pretrain_mix_feature_pilot.yaml \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.10 \
  --feature_lr_mult 3.0 \
  --feedback film --feedback_pairs "0,16;4,20;8,24;12,28" \
  --feedback_self_k 3 --feedback_self_k_warmup_steps 20 \
  --use_pkm --pkm_after_layer 16 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate \
  --pkm_epsilon_start 0.1 --pkm_epsilon_warmup_steps 1000 \
  --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_alpha_floor_start 0.0 --pkm_alpha_floor_warmup_steps 0 \
  --pkm_value_lr_mult 233.0 \
  --use_memory --mem_size 1024 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192 \
  --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match \
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -2.0 --emit_read_mask \
  --mem_read_alpha_init 0.05 \
  --mem_read_alpha_floor_start 0.0 --mem_read_alpha_floor_warmup_steps 0 \
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 100 --ctx_addr_aux_start_step 0 \
  --use_latent_feedback_adapter --latent_reasoning_weight 0.0 \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
  --gate_floor_min 1.0 --gate_warmup_steps 1 \
  --engagement_check_step $ENGAGE_STEP --engagement_check_action $ENGAGE_ACTION \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 100000000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!) — engagement gate at $ENGAGE_STEP ($ENGAGE_ACTION)"
