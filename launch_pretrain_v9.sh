#!/bin/bash
# v9 latent-thinking co-training — FAST VALIDATION (2026-05-30).
#
# THE QUESTION: does co-training the LATENT thinking mechanism make thinking
# USEFUL (vs v8, where the probe showed latent thinking HURTS, Δlogp ≈ -7)?
#
# Approach: warm-start from the v8 1B mid-eval ckpt (coherent enough to MEASURE
# a thinking benefit fast, not yet over-committed) and add the latent
# co-training loss (--latent_cotrain_weight): grad CE on the post-R-latent-think
# prediction, so the trunk learns to do useful sequential computation during
# thinking. The validation signal is the per-step `latent(Δlogp=...)` log — it
# should climb from ≈-7 toward 0/positive within a few hundred steps if
# co-training works.
#
# Config audit fixes vs v8 ("enable thinking + FiLM EARLY"):
#   - FiLM K=3 from step 0 (--feedback_self_k_warmup_steps 0; v8 had 1500).
#   - latent thinking co-trained from the first v9 step (v8 never had it).
#   - gate floor warmup shortened (20000 -> 2000).
# batch 4 / grad_accum 8: leaves headroom for the grad latent extra-forwards
# (no activation checkpointing on those). Drop batch if OOM.
#
# NOTE: multi-token prediction (MTP) is the next ingredient to fold in once the
# latent-usefulness signal is confirmed (it is a token-efficiency lever, not
# what makes thinking useful, so it does not gate this validation).

set -e
cd /home/knielsen/ml/parallel-ss-dep
# Patched torch (2.13.0a0+viewmetafix, PR #184774) so --compile + gist don't
# segfault. Compile is only ~5% on the FLA stack (Triton kernels are graph
# breaks) but it's free, so take it; the latent extra-forward is dynamo.disable'd
# + fixed-shape so it stays eager out of the compiled graph. Drop to --no-compile
# only if batch 8 + compile + the once/step latent forward OOMs.
export PYTHONPATH=/home/knielsen/ml/pytorch-release:$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_lm.py \
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
    --latent_cotrain_weight ${LATENT_W:-0.0} \
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
    --load_ckpt checkpoints/pretrain_v8_wide_step3816_tok1000341504.pt \
    --start_step 3816 \
    --val_every 400 --log_every 25 \
    --mid_eval_every_tokens 500000000 --mid_eval_save_only \
    --save_ckpt checkpoints/pretrain_v9.pt \
    --tb_dir runs/tb/pretrain_v9 \
    > runs/pretrain_v9.log 2>&1 &
echo "Launched v9 latent-cotrain (FAST VALIDATION) on GPU ${GPU:-1} (PID $!)"
echo "Watch the latent(Δlogp=...) field: tail -f runs/pretrain_v9.log"
