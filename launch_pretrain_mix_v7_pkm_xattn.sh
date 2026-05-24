#!/bin/bash
# v7-pkm-xattn — shallow-wide + cross-layer attention + fixed PKM.
#
# Sister run to launch_pretrain_mix_v7_pkm_xattn.sh — same shape (10L × 896d),
# same fixed PKM, only the reverse-feedback mechanism changes from dense FiLM
# to cross-layer attention with per-token sigmoid gating (film_sigmoid form).
#
# CRITICAL: this v7 run includes the no_grad-pass-1 fix landed 2026-05-17 in
# model.py xattn forward — pass 1 collects source-layer hidden states under
# no_grad (cheap, no autograd tape), pass 2 is loss-bearing. Mirrors the
# proven FiLM K=3 protocol (K-1 warmup passes no_grad). Cuts xattn step cost
# from ~4 work-units to ~3 — same per-step cost as FiLM K=3.
#
# PKM bootstrap fixes (v7):
#   FIX 1: --pkm_use_output_gate          scalar α (init 0) on PKM output.
#                                          Mirrors FiLM α curriculum so
#                                          gradient grows α as PKM proves
#                                          useful — instead of forcing a
#                                          random-init contribution that
#                                          v5-pkm's training had to fight.
#   FIX 2: --pkm_epsilon_start 0.5         ε-greedy random-slot retrieval.
#          --pkm_epsilon_warmup_steps 2000 50% of retrievals random at step 0,
#                                          annealing to 0 over 2000 steps.
#                                          Forces every slot to see gradient
#                                          (v5-pkm: only 4% of slots ever fired).
#   FIX 3: --pkm_value_init_std 1.0        Init at residual-stream magnitude
#                                          (vs v5-pkm's 1/√d_model ≈ 0.04 →
#                                          PKM contribution was 1% of resid).
#   FIX 4: --pkm_score_norm layer          LayerNorm on scores (vs BN's noisy
#                                          running stats at our token scale).
#   FIX 5: --pkm_diversity_weight 0.01     Aux loss on -H(slot distribution).
#                                          Penalises peaky retrieval pattern.
#
# All other knobs identical to v7_pkm_film. Pinned to GPU 0 (v7_film on 1).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback none \
    --feedback_xattn "0:5,6,7,8,9;1:5,6,7,8,9;2:5,6,7,8,9" \
    --feedback_xattn_form film_sigmoid \
    --feedback_xattn_heads 8 \
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
    --pkm_diversity_weight 0.01 \
    --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 2000 \
    --pkm_value_lr_mult 100.0 \
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
    --steps 9300 \
    --val_every 200 --log_every 20 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_save_only \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_mix_v7_pkm_xattn.pt \
    --tb_dir runs/tb/pretrain_mix_v7_pkm_xattn \
    > runs/pretrain_mix_v7_pkm_xattn.log 2>&1 &
echo "Launched pretrain_mix_v7_pkm_xattn on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v7_pkm_xattn.log"
