#!/bin/bash
# Resume the v9-clean pretrain (latent co-train OFF by default) from the LATEST
# pretrain_v9_step*.pt checkpoint, with AUTO-RESTART: the overnight crash was a
# HuggingFace streaming-dataset client error ("Cannot send a request, client
# closed"), a transient that can recur — so on any non-clean exit we relaunch
# from the newest checkpoint (which the trainer saves periodically). Runs on
# GPU1 (free), independent of the RL supervisor on GPU0. Stops cleanly once the
# step target is reached or the restart cap is hit.
set -u
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=/home/knielsen/ml/pytorch-release:${PYTHONPATH:-}:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-1}
TARGET_STEPS=31000
MAX_RESTARTS=${MAX_RESTARTS:-40}
LOG=runs/pretrain_v9.log

for i in $(seq 1 "$MAX_RESTARTS"); do
    CKPT=$(ls -t checkpoints/pretrain_v9_step*.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "[resume_v9] no pretrain_v9_step*.pt checkpoint found; aborting" | tee -a "$LOG"
        exit 1
    fi
    STEP=$(echo "$CKPT" | sed -E 's/.*_step([0-9]+)_.*/\1/')
    if [ "$STEP" -ge "$TARGET_STEPS" ]; then
        echo "[resume_v9] reached step $STEP >= $TARGET_STEPS — DONE" | tee -a "$LOG"
        break
    fi
    echo "[resume_v9] $(date '+%F %T') attempt $i/$MAX_RESTARTS: load=$CKPT start_step=$STEP GPU=$GPU" | tee -a "$LOG"
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
        \
        --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 \
        --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 \
        --pkm_value_lr_mult 100.0 \
        --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
        --latent_cotrain_weight "${LATENT_W:-0.0}" \
        --latent_cotrain_R 3 \
        --latent_cotrain_sample_frac 0.05 \
        --latent_cotrain_max_positions 12 \
        --data_mix configs/pretrain_mix_v4.yaml \
        --tokenizer HuggingFaceTB/SmolLM2-135M \
        --think_burst_prob 0.25 --think_max_bursts 1 --think_max_burst_depth 4 \
        --T 2048 --batch 6 --grad_accum 16 \
        --activation_checkpointing \
        --bf16 --tf32 --compile --bf16_optim_state \
        --alpha_wd 0.0 --wd 0.01 \
        --grad_clip 1.0 --z_loss 1e-4 \
        --mask_eos_in_targets \
        --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
        --lr_schedule wsd --warmup_steps 0 --lr_decay_frac 0.15 \
        --steps 31000 \
        --load_ckpt "$CKPT" \
        --start_step "$STEP" \
        --val_every 400 --log_every 25 \
        --mid_eval_every_tokens 500000000 --mid_eval_save_only \
        --save_ckpt checkpoints/pretrain_v9.pt \
        --tb_dir runs/tb/pretrain_v9 \
        >> "$LOG" 2>&1
    CODE=$?
    echo "[resume_v9] $(date '+%F %T') train exited code=$CODE" | tee -a "$LOG"
    sleep 15
done
echo "[resume_v9] loop finished" | tee -a "$LOG"
