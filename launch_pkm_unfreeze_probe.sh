#!/bin/bash
# PKM unfreeze-SURVIVAL probe (2026-07-02): does the pre-warmed PKM (alpha
# 0.27, useful values — checkpoints/pkm_prewarm_probe.pt) SURVIVE once the
# trunk is unfrozen and can fight back? 300 steps, PKM-only config.
# SUCCESS = alphaL holds >= ~0.2 and row stable/climbing; FAILURE = the
# converged trunk re-absorbs (alphaL decays toward 0 again).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-0}
TAG=pkm_unfreeze_probe
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/pkm_prewarm_probe.pt --keep_base_vocab 49152 \
  --feedback none \
  --data_mix configs/pretrain_mix_feature_pilot.yaml \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 20 --lr_decay_frac 0.0 \
  --use_pkm --pkm_after_layer 16 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate \
  --pkm_epsilon_start 0.1 --pkm_epsilon_warmup_steps 290 \
  --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_alpha_floor_start 0.0 --pkm_alpha_floor_warmup_steps 0 \
  --pkm_value_lr_mult 233.0 \
  --steps 300 --val_every 100 --log_every 10 --seed 1 \
  --mid_eval_every_tokens 100000000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!)"
