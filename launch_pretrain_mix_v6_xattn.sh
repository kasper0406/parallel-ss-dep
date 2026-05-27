#!/bin/bash
# v6-xattn — shallow-wide trunk with CROSS-LAYER ATTENTION feedback.
#
# Sister run to launch_pretrain_mix_v6_xattn.sh — same shape (10L × 896d,
# iso-param to v4), but the reverse-feedback mechanism is cross-layer
# *attention* (each target layer's input attends over many lagged source-
# layer hidden states) instead of dense per-pair FiLM.
#
# Hypothesis test: at fixed depth + width + param budget, does the
# attention-style routing across source layers learn faster / reach lower
# CE than the FiLM-sum form? The two launchers should be diff'd to see
# only the feedback knobs change.
#
# Feedback config:
#   --feedback_xattn "0:5,6,7,8,9;1:5,6,7,8,9;2:5,6,7,8,9"
#     3 target layers (early: 0, 1, 2) each attend across 5 source layers
#     (late: 5..9). Same 15 connections that 5 dense FiLM pairs cover,
#     but routed via softmax(Q·K) instead of multiplicative FiLM.
#   --feedback_xattn_form film_sigmoid
#     Per-token sigmoid gates on each (target, source) pair. Tested in
#     the MQAR feedback bake-off as the "proper gating" hypothesis —
#     compared to the default scalar-α attn form whose α starts at 0 and
#     may not move at all under sparse-gradient regimes.
#   --feedback_xattn_heads 8
#     Multi-head routing over the source dim (8 heads, d_head=112).
#
# New mechanisms:
#   --gate_entropy_aux_weight 0.1  Auxiliary BCE that supervises the output
#     gate with target = exp(-H_t/T), where H_t is per-position next-token
#     entropy (detached, free from existing logits). Confident position →
#     emit; uncertain → think. Grounds the gate in predictive uncertainty
#     so it stops being random noise during pretrain (v5/RL eval showed
#     gate was permanently saturated at 0.98 → think_rate=0).
#   --gate_entropy_aux_temperature 2.0  Broadens the target (raw exp(-H)
#     is mostly tiny; T=2 spreads gradient signal at uncertain positions).
#
# Kept from v4:
#   v4 data mix (CVE-stream up-weighted), WSD schedule, compile,
#   activation_checkpointing, bf16, tf32, wd=0.01, alpha_wd=0.0,
#   z_loss=1e-4, grad_clip=1.0, mask_eos_in_targets, gate_floor_min=0.5.
#
# NOT included (yet — staged for v7 if v6 wins):
#   PKM side-table (validated in v5-pkm but doubles param load).
#   Cross-layer attention (xattn) — first MQAR feedback bake-off showed
#   scalar-α gating was the wrong knob; multi-FiLM is the natural fallback.
#
# Memory / batch / token budget:
#   --batch 14 --grad_accum 8 → 14*8*2048 = 229k tokens/step
#   ~2.1 B tokens / 229k = 9,300 steps (matches v4)
#   K=3 self-feed has 3-pass forward; activation_checkpointing makes this
#   fit at b=14 with the wider trunk. (Smoke before launch.)
#
# Pinned to GPU 0 by default — v6_shallow is on GPU 1; this run is queued
# to fire when v4 vacates GPU 0 (~step 9300 wrap-up; final ckpt save).
# Pass GPU=N to override.

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
    --feedback_self_k_warmup_steps 1300 \
    --output_gate \
    --gate_entropy_aux_weight 0.1 \
    --gate_entropy_aux_temperature 2.0 \
    --use_memory --mem_size 1024 \
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
    --save_ckpt checkpoints/pretrain_mix_v6_xattn.pt \
    --tb_dir runs/tb/pretrain_mix_v6_xattn \
    > runs/pretrain_mix_v6_xattn.log 2>&1 &
echo "Launched pretrain_mix_v6_xattn on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v6_xattn.log"
