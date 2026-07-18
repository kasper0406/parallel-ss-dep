#!/bin/bash
# WM + FiLM gradual-attach program on the production base (task #19,
# 2026-07-18). Pre-registered decision block: IDEAS_2026_07_13.md
# "WM+FiLM attach registration".
#
# Three arms from checkpoints/production_lean_soup3.pt (HE-CE 0.6614),
# ONE launch per arm, 4,500 steps x 131,072 tok ≈ 590M tokens each:
#   ARM=control  no features, trunk trains all 4,500 steps (the extra 1,000
#                unfrozen steps vs the feature arms' frozen phase-0 is a
#                DOCUMENTED conservative bias AGAINST the feature arms).
#   ARM=wm       mem_ctx_namekey WM + copy head; --freeze_trunk_steps 1000
#                (phase-0 pre-warm: loss pressure only into WM params, NO
#                floors) then unfreeze co-train. Engagement kill-gate at
#                step 1,000 (see registration).
#   ARM=film     FiLM feedback, 32L pairs 0,16;4,20;8,24;12,28 (the phase-0
#                launcher's validated 32L mapping), K=3 self-feed with
#                warmup bypass, alpha_wd 0; same freeze/pre-warm shape.
#
# LR design: trunk LR = anneal-preserving 2e-4 muon / 6e-5 adamw (soup3 is a
# fully-annealed ckpt; a 2e-3 re-heat would undo the decay), features at
# 10x via --feature_lr_mult. WSD warmup 25, decay over the last ~2,000 steps.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

ARM=${ARM:?set ARM=control|wm|film}
GPU=${GPU:-1}
STEPS=${STEPS:-4500}
SEED=${SEED:-0}
TAG=${TAG:-attach_${ARM}}
BASE=${BASE:-checkpoints/production_lean_soup3.pt}

FEATURE_FLAGS=""
FREEZE=""
case "$ARM" in
  control) ;;
  wm)
    FREEZE="--freeze_trunk_steps 1000"
    FEATURE_FLAGS="--use_memory --mem_size 1024 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192 \
      --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match \
      --mem_discrete_key_match_window 32 --copy_gate_bias_init -2.0 --emit_read_mask \
      --mem_read_alpha_init 0.05 \
      --mem_read_alpha_floor_start 0.0 --mem_read_alpha_floor_warmup_steps 0 \
      --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 400 --ctx_addr_aux_start_step 0" ;;
  film)
    FREEZE="--freeze_trunk_steps 1000"
    # NO K-warmup bypass: with the trunk frozen, a film-only arm has no other
    # trainable param, so the bypass leaves the loss with no grad_fn (crashed
    # 2026-07-18). K=3 self-feed active from step 1 instead.
    FEATURE_FLAGS="--feedback film --feedback_pairs 0,16;4,20;8,24;12,28 \
      --feedback_self_k 3 --feedback_self_k_warmup_steps 0 \
      --feedback_alpha_init 0.02" ;;
  *) echo "ARM must be control|wm|film, got '$ARM'" >&2; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt $BASE --keep_base_vocab 49152 \
  $FREEZE --feature_lr_mult 10.0 \
  $FEATURE_FLAGS \
  --data_mix configs/pretrain_mix_feature_pilot.yaml \
  --fim_legacy_strings \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-5 --lr_muon 2e-4 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 25 --lr_decay_frac 0.44 \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 100000000 --mid_eval_save_only --mid_eval_min_free_gib 2.0 \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG ARM=$ARM on GPU $GPU (PID $!) base=$BASE — tail -f runs/${TAG}.log"
