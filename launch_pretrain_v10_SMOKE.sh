#!/bin/bash
# SMOKE for launch_pretrain_v10_all_features.sh — identical model/feature/mem
# config (batch 8 + FiLM K=3 + DKV-WM + PKM + latent-cotrain + gate-cal +
# state_readonly) so it exercises the REAL OOM/throughput path immediately
# (continuation at step 23000 is past the K=3 warmup → K=3 active from step 1).
# Short run + frequent feature-probe + throwaway output paths. ~120 steps.
# Purpose: confirm every feature FIRES, the [feature-probe]/[wm-recall] logs
# appear, and there's no OOM / Inductor / CUDA-assert before the 12h run.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film \
    --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 1300 \
    --output_gate \
    --gate_entropy_aux_weight 0.1 \
    --gate_entropy_aux_temperature 2.0 \
    --gate_floor_min 0.5 --gate_warmup_steps 20000 \
    --state_readonly_at_think \
    --use_memory --mem_size 1024 \
    --mem_decoupled_kv \
    --mem_read_alpha_init 1.0 \
    --mem_read_alpha_floor_start 0.5 --mem_read_alpha_floor_warmup_steps 3000 \
    --use_pkm --pkm_after_layer 5 \
    --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
    --pkm_use_output_gate \
    --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 \
    --pkm_value_init_std 1.0 \
    --pkm_score_norm layer \
    --pkm_diversity_weight 0.01 \
    --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 \
    --pkm_value_lr_mult 100.0 \
    --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
    --latent_cotrain_weight 0.025 \
    --latent_cotrain_R 4 \
    --latent_cotrain_sample_frac 0.05 \
    --latent_cotrain_max_positions 4 \
    --gate_calibration_weight 0.05 \
    --gate_calibration_R 4 \
    --gate_calibration_sample_frac 0.05 \
    --gate_calibration_max_positions 4 \
    --gate_calibration_sigma_low 0.1 --gate_calibration_sigma_high 0.9 \
    --feature_probe_every_tokens 20000000 \
    --feature_probe_wm_recall_path "" \
    --feature_probe_wm_recall_n 32 \
    --data_mix configs/pretrain_mix_v10.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 4 --grad_accum 32 \
    --activation_checkpointing \
    --bf16 --tf32 --no-compile \
    --alpha_wd 0.0 --wd 0.01 \
    --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --start_step 23000 \
    --steps 23120 \
    --val_every 40 --log_every 5 \
    --mid_eval_every_tokens 999999999999 \
    --save_ckpt checkpoints/SMOKE_v10_DELETEME.pt \
    --tb_dir runs/tb/SMOKE_v10 \
    > runs/SMOKE_v10.log 2>&1 &
echo "Launched v10 SMOKE on GPU ${GPU:-0} (PID $!)"
