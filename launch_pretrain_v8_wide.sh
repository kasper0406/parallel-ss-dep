#!/usr/bin/env bash
# Pretrain v8 "wide" — WIDTH-scaled competence super-coder run (2026-05-29).
# THINKING_MEMORY_PLAN.md D14. Mission: small super-coder; this is a COMPETENCE
# play — scale WIDTH (d_model 896->1280) and MEMORY capacity (PKM 256->384 keys,
# WM 1024->1536), HOLD the layer count (10; the latent-thinking mechanism wants a
# shallow trunk). Thinking machinery (gate, state_readonly, think-burst) is
# co-trained but NOT the headline.
#
# Phase-0 VERIFIED (D14): builds at 600M, fits at peak ~19.5/32 GiB (b=6; b=8 has
# margin), PKM bootstraps (118-131k/147k slots), loss descends, stable past K=3.
#
# IMPORTANT FLAGS / KNOWN ISSUES (Phase-0):
#  * --no-compile is MANDATORY on the current .venv nightly torch: compile trips
#    the gist-loss AOTAutograd _unsafe_view_ViewMeta segfault. RESOLVED (D16):
#    the patched torch at ~/ml/pytorch-release (2.13.0a0+viewmetafix, the user's
#    PR #184774) fixes it. We use that build via PYTHONPATH (no .venv mutation),
#    so --compile + gist are ON. Validated end-to-end past K=3 (~27k tok/s).
#  * OMP_NUM_THREADS=8 is MANDATORY (avoids the 115-thread futex stall).
#  * think-burst injection is ON so WorkingMemory receives gradient (without
#    think positions WM is decorative).
#
# Single-GPU as written (~3.2 days w/ compile). DDP across both 5090s (NCCL is
# in the patched build) ~halves it — wire separately once GPU 1 frees.
#
# PHASE-0 GATE (already validated D16): the wide config + compile + gist crossed
# K=3 stable at peak ~21 GiB, ~27k tok/s.

set -euo pipefail
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use the patched torch (ViewMeta fix) so --compile + gist don't segfault.
export PYTHONPATH=/home/knielsen/ml/pytorch-release:${PYTHONPATH:-}
GPU=${GPU:-0}
# Optional resume: CKPT=<ckpt> START=<step> bash launch_pretrain_v8_wide.sh
RESUME_ARGS=""
if [ -n "${CKPT:-}" ]; then RESUME_ARGS="--load_ckpt ${CKPT} --start_step ${START:-0}"; fi

CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTHONPATH=$PYTHONPATH:. nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 1280 --n_layers 10 --d_head 64 --n_heads 20 \
  --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
  --feedback_self_k 3 --feedback_self_k_warmup_steps ${WARMUP:-1500} \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
  --gate_floor_min 0.5 --gate_warmup_steps 20000 \
  --state_readonly_at_think \
  --use_memory --mem_size 1536 \
  --use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 384 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 \
  --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 --pkm_value_lr_mult 100.0 \
  --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
  --data_mix configs/pretrain_mix_v4.yaml --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0.25 --think_max_bursts 1 --think_max_burst_depth 4 \
  --T 2048 --batch 8 --grad_accum 16 \
  --activation_checkpointing --bf16 --tf32 ${COMPILE:---compile} --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 600 --lr_decay_frac 0.15 \
  --steps 31000 $RESUME_ARGS --val_every 400 --log_every 50 \
  --mid_eval_every_tokens 500000000 --mid_eval_save_only \
  --save_ckpt checkpoints/pretrain_v8_wide.pt --tb_dir runs/tb/pretrain_v8_wide \
  > runs/pretrain_v8_wide.log 2>&1 &
echo "launched pretrain_v8_wide pid $! on GPU $GPU -> runs/pretrain_v8_wide.log"
