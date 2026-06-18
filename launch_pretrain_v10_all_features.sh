#!/bin/bash
# v10 — co-train ALL validated mechanisms from day 1, with per-feature
# usefulness tracking (2026-06-04).
#
# WHY: every prior attempt to add a thinking/memory mechanism POST-pretrain
# left it inert (PKM 97% dead at 2.13B tokens; WM decorative in the v1
# ablation; latent thinking HURT on v8, Δlogp≈-7). The only fix that has ever
# made a side-module load-bearing is co-training it from the FIRST step. v10
# turns ON the full validated stack together and ADDS a glanceable
# per-feature "is it load-bearing yet" probe so we can watch each mechanism
# wake up (or not) during the run instead of flying blind on VAL ppl.
#
# Mechanisms enabled (all already exist in model.py/helpers):
#   1. FiLM (5 dense reverse pairs) K=3 self-feed — validated trunk lift.
#   2. PKM v7.1 bootstrap package — the 5-fix + value-LR-mult that turned PKM
#      from "97% dead" into "always-positive per-source contribution".
#   3. WorkingMemory with the NEW decoupled-key/value (DKV) addressing +
#      α-floor curriculum (--mem_decoupled_kv, --mem_read_alpha_floor_*).
#      DKV gives reliable semantic (non-token-identity) addressing; the
#      α-floor holds the read contribution high during warmup so the sharp
#      addressing locks in before the learned α takes over.
#   4. Latent-thinking gate made USEFUL: --latent_cotrain_weight gives the
#      TRUNK gradient through R state-readonly latent thinks (so thinking does
#      useful sequential computation), and --gate_calibration_weight gives the
#      GATE a per-position teacher of WHERE a latent think actually helps.
#
# Per-feature probe (--feature_probe_every_tokens): every 500M tokens, on a
# held-out batch, logs an ablation-delta CE for WM and PKM (CE rises iff the
# feature is load-bearing — the eval_longctx_recall --wm_ablate mean / the
# probe_pkm_per_source α-toggle idea), the WM read_alpha, FiLM α per pair, and
# the gate fire-rate. Console line `[feature-probe] wm(Δce=..,α=..)
# pkm(Δce=..,α=..) film(max|α|=..) gate(fire=..)` + TensorBoard probe/*.
#
# Compile: OFF. The gate-calibration + latent-cotrain aux losses each run a
# variable-length extra forward; CLAUDE.md documents that this crashes
# Inductor (the latent primitives are dynamo.disable'd, but --no-compile is
# the documented, safe fallback for a 20h run). Compile is only ~5-10% on the
# FLA stack anyway (Triton kernels are graph breaks).
#
# This is a CONTINUATION of the Phase C base (Chinchilla-complete 287M trunk,
# strict per-source CE win) so the trunk is already strong and the NEW
# mechanisms (WM-DKV, gate-cal) attach fresh and co-train over the
# continuation. The WM/PKM/gate weights already exist in the Phase C ckpt and
# load byte-identically; the WM-DKV W_k / logit_scale / gate_bias_beta and the
# gate-cal/latent machinery are additive.
#
# Memory: batch 8 + grad_accum 16 = ~262k tok/step. The latent + gate-cal
# extra forwards fire ONCE/step (last microbatch) on a small sampled fraction;
# they are eager (no activation checkpointing on those) so keep batch modest.
# Drop batch to 6 if OOM at the K=3 self-feed boundary.

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
    \
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
    --feature_probe_every_tokens 500000000 \
    --feature_probe_wm_recall_path "" \
    --data_mix configs/pretrain_mix_v10.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 4 --grad_accum 32 \
    --activation_checkpointing \
    --bf16 --tf32 --no-compile \
    --alpha_wd 0.0 --wd 0.01 \
    --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --start_step 23000 \
    --steps 36000 \
    --val_every 200 --log_every 20 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_save_only \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_v10_all_features.pt \
    --tb_dir runs/tb/pretrain_v10_all_features \
    > runs/pretrain_v10_all_features.log 2>&1 &
echo "Launched v10 all-features pretrain on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/pretrain_v10_all_features.log"
echo "Per-feature load-bearing signal: grep '\\[feature-probe\\]' runs/pretrain_v10_all_features.log"
