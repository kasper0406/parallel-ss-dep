#!/bin/bash
# Auto-resuming wrapper for the v11 pretrain. The 2026-06-14 run died at step
# ~2660 on a transient HuggingFace streaming-client drop ("Cannot send a request,
# as the client has been closed") — a ~35h run is too long to babysit against
# network blips. This wrapper restarts from the LATEST mid-eval ckpt on any
# non-clean exit, until step 19000 / the final ckpt is saved.
#
# Resume granularity: mid_eval ckpts every 250M tokens (~950 steps, ~1.8h) so a
# crash loses < ~1.8h. feature-probe stays at 500M (the WM-load-bearing signal).
set -u
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH="${PYTHONPATH:-}:."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60
GPU=${GPU:-1}
LOG=runs/pretrain_v11.log
mkdir -p runs checkpoints

common_args() {
  cat <<'ARGS'
--arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14
--feedback film --feedback_pairs 0,5;1,6;2,7;3,8;4,9 --feedback_self_k 3 --feedback_self_k_warmup_steps 1300
--output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0
--gate_floor_min 0.5 --gate_warmup_steps 20000 --state_readonly_at_think
--use_memory --mem_size 1024 --mem_decoupled_kv --mem_read_alpha_init 1.0
--mem_read_alpha_floor_start 0.5 --mem_read_alpha_floor_warmup_steps 3000
--use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32
--pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 --pkm_value_init_std 1.0
--pkm_score_norm layer --pkm_diversity_weight 0.01 --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000
--pkm_value_lr_mult 100.0 --gist_loss_weight 0.1 --gist_horizons 16,64,256
--latent_cotrain_weight 0.025 --latent_cotrain_R 4 --latent_cotrain_sample_frac 0.05 --latent_cotrain_max_positions 4
--gate_calibration_weight 0.05 --gate_calibration_R 4 --gate_calibration_sample_frac 0.05 --gate_calibration_max_positions 4
--gate_calibration_sigma_low 0.1 --gate_calibration_sigma_high 0.9
--feature_probe_every_tokens 500000000 --feature_probe_wm_recall_path data/multibind_bigN_heldout.jsonl --feature_probe_wm_recall_n 64
--data_mix configs/pretrain_mix_v11.yaml --tokenizer HuggingFaceTB/SmolLM2-135M
--think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6
--T 2048 --batch 4 --grad_accum 32 --activation_checkpointing --bf16 --tf32 --no-compile
--alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets
--optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15
--steps 19000 --val_every 200 --log_every 20
--mid_eval_every_tokens 250000000 --mid_eval_save_only --mid_eval_n_problems 50 --mid_eval_max_gen 192
--save_ckpt checkpoints/pretrain_v11.pt --tb_dir runs/tb/pretrain_v11
ARGS
}

for attempt in $(seq 1 40); do
  # find latest mid-eval ckpt → resume step
  latest=$(ls -t checkpoints/pretrain_v11_step*.pt 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    rstep=$(echo "$latest" | sed -E 's/.*_step([0-9]+)_.*/\1/')
    resume="--load_ckpt $latest --start_step $rstep"
    echo "=== [autoresume $attempt] resuming from $latest (step $rstep) ===" >> "$LOG"
  else
    resume="--start_step 0"
    echo "=== [autoresume $attempt] fresh start ===" >> "$LOG"
  fi
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py $(common_args) $resume >> "$LOG" 2>&1
  rc=$?
  # NOTE: the live v11 wrapper parsed the OLD -ge 18900 into memory at launch, so
  # this edit only fixes future re-launches — the running instance will spurious-
  # resume at completion and must be killed manually (confirm pretrain_v11.pt
  # exists first). 18900 was unreachable: last mid-eval ckpt lands at ~step 18120.
  if grep -q "saved.*checkpoints/pretrain_v11.pt" "$LOG" && [ "$(ls -t checkpoints/pretrain_v11_step*.pt 2>/dev/null | head -1 | sed -E 's/.*_step([0-9]+)_.*/\1/')" -ge 18000 ] 2>/dev/null; then
    echo "=== [autoresume] COMPLETE (rc=$rc) ===" >> "$LOG"; exit 0
  fi
  echo "=== [autoresume $attempt] exited rc=$rc — will resume from latest ckpt ===" >> "$LOG"
  sleep 20
done
echo "=== [autoresume] GAVE UP after 40 attempts ===" >> "$LOG"; exit 1
