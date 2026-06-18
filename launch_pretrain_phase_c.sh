#!/bin/bash
# Phase C — Chinchilla-completion continuation of the v7.1 pretrain.
#
# Why: v7.1-pkm-film stopped at 9300 steps ≈ 2.13B tokens. Chinchilla-
# optimal for a 287M model is ~5.3B tokens — so v7.1 is trained to only
# ~40% of optimal. The HumanEval plateau (~8-11/164 across every SFT
# variant) is consistent with an undertrained + small base, not a
# post-training recipe problem (see GEMINI.md). Phase C closes the
# undertraining gap.
#
# This is a CONTINUATION (--load_ckpt + --start_step 9300), not a fresh
# run. Same trunk / FiLM / PKM / WM / gate recipe as v7.1, extended:
#   * --steps 23000  → 23000 × 229,376 tok/step ≈ 5.28B tokens total
#                      (13,700 new steps ≈ 3.15B new tokens, ~20h)
#   * --gist_loss_weight 0.1  → NEW: the v7 trunk multi-horizon gist
#     loss baked into pretrain. Free at runtime (single forward), and a
#     representation-shaping "high-level direction" objective that
#     genuinely benefits from pretrain-scale tokens. gist_heads attach
#     fresh (v7.1 ckpt has none) and train over the continuation.
#
# WSD note: v7.1's ckpt ended in its LR-decay tail. Re-launching with a
# larger --steps puts the scheduler back at peak LR for the bulk of the
# continuation, decaying again over the last 15%. A brief loss-spike
# transient on resume is expected and recovers.
#
# torch.compile: kept ON. The gist loss needs model(return_hidden=True);
# returning the live hidden state (a view) across the compile boundary
# segfaulted autograd functionalization. Fix: _finalize returns
# h.clone() for return_hidden — a materialised boundary tensor. Smoke
# 2026-05-21 confirmed compile runs clean (~45.5k vs 39k tok/s eager).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film \
    --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 1300 \
    --output_gate \
    --gate_entropy_aux_weight 0.1 \
    --gate_entropy_aux_temperature 2.0 \
    --use_memory --mem_size 1024 \
    --use_pkm --pkm_after_layer 5 \
    --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
    --pkm_use_output_gate \
    --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 2000 \
    --pkm_value_init_std 1.0 \
    --pkm_score_norm layer \
    \
    --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 2000 \
    --pkm_value_lr_mult 100.0 \
    --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
    --data_mix configs/pretrain_mix_v4.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 14 --grad_accum 8 \
    --activation_checkpointing \
    --bf16 --tf32 --compile \
    --alpha_wd 0.0 --wd 0.01 \
    --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --gate_floor_min 0.5 --gate_warmup_steps 1500 \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --start_step 9300 \
    --steps 23000 \
    --val_every 200 --log_every 20 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_save_only \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_phase_c.pt \
    --tb_dir runs/tb/pretrain_phase_c \
    > runs/pretrain_phase_c.log 2>&1 &
echo "Launched Phase C pretrain on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/pretrain_phase_c.log"
