#!/bin/bash
# v19 — CODE-UPWEIGHTED fresh pretrain (the data-mix lever the ablation identified).
#
# Premise (2026-06-20): the iso-token mechanism ablation proved PKM/WM/latent are
# ~FREE on code CE; v18's HumanEval regression (6 vs 13) was MIX DILUTION (codeparrot
# 0.30->0.22 + arXiv/recall). v19 keeps the (free) mechanisms but restores code
# dominance (codeparrot 0.32, arXiv cut — see config DECISION POINT). Adds the
# validated per-head-NS matrix optimizer (~3% free convergence edge).
#
# Fresh from scratch, 287M, GPU1 (reliable card — GPU0 falls off the bus under load).
# Single-GPU (latent + DDP are incompatible, see memory). ~20000 steps ≈ 5.2B tok
# (Chinchilla-complete for 287M) at batch4/ga32 = 262k tok/step ≈ ~25h.
#
# Mechanism flags mirror v18 EXACTLY (the validated day-1 co-train recipe), so the
# ONLY deltas vs v18 are: (a) --data_mix v19 (code-upweighted), (b) fresh not resume,
# (c) --matrix_optimizer fused_deltanet_ns. NOT launched automatically — needs the
# arXiv decision (config note) + a go.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-1}
STEPS=${STEPS:-20000}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
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
  --data_mix configs/pretrain_mix_v19_codeup.yaml --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
  --T 2048 --batch 4 --grad_accum 32 --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
  --steps $STEPS --val_every 200 --log_every 20 \
  --mid_eval_every_tokens 250000000 --mid_eval_save_only --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
  --save_ckpt checkpoints/pretrain_v19.pt --tb_dir runs/tb/pretrain_v19 \
  > runs/pretrain_v19.log 2>&1 &
echo "Launched v19 code-upweighted pretrain, GPU $GPU (PID $!).  Watch: tail -f runs/pretrain_v19.log"
