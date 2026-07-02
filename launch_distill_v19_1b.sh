#!/bin/bash
# STEP 2 of the v19 distillation production run: train the 1B student on the
# offline SmolLM2-1.7B teacher logits (STEP 1 / launch_gen_teacher_logits_v19.sh).
#
# STUDENT (your design): 10L (depth-capped) × d2048 × 32 heads × d_head 64,
#   TIED embeddings, latent thinking = depth. PKM held ~constant. ~0.98B params
#   (tied). Grow width+heads, never depth — latent supplies effective depth
#   ("escape the fixed-depth circuit class").
# LOSS: combined CE + 0.5·T²·KL(teacher_topk ‖ student) from disk + latent_reasoning
#   aux (the depth mechanism). matrix optimizer = fused per-head NS (free ~3% edge).
#
# *** DESIGN TENSION (documented, not silent) ***
#   Offline-KD requires --think_burst_prob 0 (think bursts shift the token stream
#   and break logit alignment). So the main-stream DISCRETE-think gate/WM supervision
#   is OFF here. Mitigation: (a) --mem_always_read → WM trains on EMIT positions (not
#   only think), (b) PKM trains every token, (c) latent depth comes from the
#   latent_reasoning aux (separate ptr10dict data, alignment-independent). What's
#   dropped is only the discrete-[THINK]-burst gate supervision — acceptable for a
#   distilled BASE. If you want full think-burst mechanism co-training, do it as a
#   PHASE 2 (no offline-KD, think_burst_prob>0) after this base. Override: pass
#   THINK_BURST=0.5 to re-enable (will fail offline-KD alignment → only for a
#   no-KD phase-2 run).
#
# ALIGNMENT: mix/seed/T/num_workers/think_burst/mask_eos MUST equal STEP 1's.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=${GPU:-1}
MIX=${MIX:-configs/pretrain_mix_v18_arxiv.yaml}      # MUST match STEP 1
LOGITS=${LOGITS:-data/teacher_logits_smollm17_v19}   # MUST match STEP 1 --out_dir
STEPS=${STEPS:-11000}                                # ~1.5B tok at batch4·ga16·T2048 (131k tok/step)
THINK_BURST=${THINK_BURST:-0}                        # keep 0 for offline-KD (see DESIGN TENSION)

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 2048 --n_layers 10 --d_head 64 --n_heads 32 --tie_embeddings \
  --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" --feedback_self_k 3 --feedback_self_k_warmup_steps 1300 \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 --gate_floor_min 0.5 --gate_warmup_steps 20000 \
  --state_readonly_at_think \
  --use_memory --mem_size 2048 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192 \
  --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match \
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -6.0 --mem_read_alpha_init 0.0 --mem_freeze_read_alpha --emit_read_mask \
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 3000 --ctx_addr_aux_start_step 0 \
  --use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 --pkm_value_init_std 1.0 \
  --pkm_score_norm layer --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 --pkm_value_lr_mult 100.0 \
  --gist_loss_weight 0.1 --gist_horizons 16,64,256 \
  --use_latent_feedback_adapter --latent_reasoning_weight 0.05 --latent_reasoning_train_prefix data/ptr10dict_train \
  --latent_reasoning_rungs 2,3,4,5,6,7,8 --latent_reasoning_n 4 --latent_reasoning_gate_weight 0.05 \
  --latent_reasoning_start_step 2000 --latent_reasoning_weight_warmup_steps 3000 \
  --matrix_optimizer fused_deltanet_ns \
  --distill_logits_dir "$LOGITS" --distill_weight 0.5 --distill_temp 2.0 \
  --data_mix "$MIX" --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob $THINK_BURST --num_workers 0 \
  --T 2048 --batch 4 --grad_accum 16 --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
  --steps $STEPS --val_every 200 --log_every ${LOG_EVERY:-20} \
  --mid_eval_every_tokens 250000000 --mid_eval_save_only \
  --save_ckpt checkpoints/distill_v19_1b.pt --tb_dir runs/tb/distill_v19_1b \
  > runs/distill_v19_1b.log 2>&1 &
echo "Launched v19 1B distill-train, GPU $GPU (PID $!).  Watch: tail -f runs/distill_v19_1b.log"
