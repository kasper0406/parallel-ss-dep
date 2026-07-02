#!/bin/bash
# Feature-pilot Arm B (ALL FEATURES, engaged by construction) — iso-data twin
# of arm A. Implements every recipe rule from the 2026-07-02 forensics
# (SESSION_FINDINGS.md): curricula <= 40% of run (800 of 2500 steps), FiLM
# alpha WARM-STARTED (0.02, the committed scale — the alpha=0 cold-start stall
# was the arm-B killer), WM read channel UNFROZEN with a floor curriculum +
# copy gate warm-started (-2, not -6) + natural-code reuse supervision in the
# mix, PKM bootstrap inside the run, latent aux with the OOM fix + verified
# construction, gate warmup 1000 (not 20000), constant-LR plateau to step 2250
# (decay_frac 0.10), feature params at 3x LR (the converged-base toll-payer),
# and in-run ENGAGEMENT KILL-GATES that abort rather than produce another
# silent-inert "negative result".
#
# PKM value LR: --pkm_value_lr_mult 233 x base lr 6e-4 = 0.14 — the ONLY scale
# at which PKM has ever committed in this repo (v7.1/v17 ran 100 x 1.4e-3; a
# 100x mult on our conservative base LR silently halved the value LR to 0.06,
# and the first pilot-B launch reproduced the converged-base rejection: alphaL
# +0.078 -> negative by step 300 with slot coverage collapsing 43k->2.5k).
#
# Run on GPU1 (GPU0 is hardware-flaky under sustained load).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

GPU=${GPU:-1}
STEPS=${STEPS:-2500}
SEED=${SEED:-0}
TAG=${TAG:-feature_pilot_B}
ENGAGE_STEP=${ENGAGE_STEP:-1000}
# Pilot default = warn (orchestrator reads the step-1000 report and decides);
# the FULL run should use ENGAGE_ACTION=abort.
ENGAGE_ACTION=${ENGAGE_ACTION:-warn}

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
  --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.10 \
  --feature_lr_mult 3.0 \
  --feedback film --feedback_pairs "0,16;4,20;8,24;12,28" \
  --feedback_self_k 3 --feedback_self_k_warmup_steps 300 \
  --feedback_alpha_init 0.02 \
  --use_pkm --pkm_after_layer 16 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 800 \
  --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 800 --pkm_value_lr_mult 233.0 \
  --use_memory --mem_size 1024 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192 \
  --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match \
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -2.0 --emit_read_mask \
  --mem_read_alpha_init 0.0 --mem_read_alpha_floor_start 0.15 --mem_read_alpha_floor_warmup_steps 800 \
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 800 --ctx_addr_aux_start_step 0 \
  --use_latent_feedback_adapter --latent_reasoning_weight 0.05 \
  --latent_reasoning_train_prefix data/ptr10dict_train \
  --latent_reasoning_rungs 2,3,4,5,6,7,8 --latent_reasoning_n 4 --latent_reasoning_gate_weight 0.05 \
  --latent_reasoning_start_step 200 --latent_reasoning_weight_warmup_steps 600 \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
  --gate_floor_min 0.5 --gate_warmup_steps 1000 \
  --engagement_check_step $ENGAGE_STEP --engagement_check_action $ENGAGE_ACTION \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 100000000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!) — tail -f runs/${TAG}.log"
echo "Engagement kill-gate at step $ENGAGE_STEP (abort). Watch: grep -E 'pkm\\(|alpha=|copy\\(|reason\\(' runs/${TAG}.log | tail"
