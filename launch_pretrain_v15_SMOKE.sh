#!/bin/bash
# v15 SMOKE — short discrete-key WM continuation smoke (GPU1).
# grad_accum 1 (fast steps) + short step budget. Confirms: stable (no cold-aux
# spike), discrete-WM active (copy(g,m%R,m%G) logged), step-time. NOT the full run.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

# Toggle the discrete-WM machinery via DISCRETE env (1=on default, 0=legacy WM).
if [ "${DISCRETE:-1}" = "1" ]; then
  WM_FLAGS="--mem_discrete_key --mem_always_read --use_copy_head --emit_read_mask \
            --mem_copy_require_match --mem_discrete_key_match_window 32 \
            --copy_gate_bias_init -6.0 --mem_read_alpha_init 0.0 --mem_freeze_read_alpha"
  TAG=on
else
  WM_FLAGS=""
  TAG=off
fi

CUDA_VISIBLE_DEVICES=${GPU:-1} .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 --feedback_self_k_warmup_steps 0 \
    --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
    --gate_floor_min 0.5 --gate_warmup_steps 20000 \
    --state_readonly_at_think \
    --use_memory --mem_size 1024 --mem_decoupled_kv \
    $WM_FLAGS \
    --use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
    --pkm_use_output_gate --pkm_epsilon_start 0.0 --pkm_epsilon_warmup_steps 0 \
    --pkm_value_init_std 1.0 --pkm_score_norm layer --pkm_diversity_weight 0.01 \
    --pkm_alpha_floor_start 0.0 --pkm_alpha_floor_warmup_steps 0 --pkm_value_lr_mult 100.0 \
    --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
    --data_mix configs/pretrain_mix_v15.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 4 --grad_accum 1 \
    --activation_checkpointing --bf16 --tf32 --no-compile \
    --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.15 \
    --load_ckpt checkpoints/pretrain_v12.pt \
    --start_step 0 --steps ${STEPS:-150} \
    --val_every ${VAL_EVERY:-50} --log_every 10 \
    --mid_eval_every_tokens 999999999999 \
    --save_ckpt checkpoints/_smoke_v15.pt \
    --tb_dir runs/tb/SMOKE_v15_$TAG \
    2>&1 | tee runs/SMOKE_v15_$TAG.log
