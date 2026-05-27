#!/bin/bash
# v6 — shallow-wide trunk with dense reverse-feedback ("the brain is shallow")
#
# Hypothesis: a 10-layer trunk with wider hidden + many top-down FiLM
# connections can match or beat the 30L baseline at iso-param, AND it should
# train *faster* because gradients have shorter paths.
#
# Shape change vs v4 (30L × 576d, sparse FiLM(2,28)):
#   10L × 896d, n_heads=14 × d_head=64  (iso-param: ~225M vs v4's 217M)
#   FiLM 5 reverse pairs: (0,5),(1,6),(2,7),(3,8),(4,9)
#     — every early layer reads from a late layer (dense fan-in across depth)
#   K=3 self-feed throughout (same protocol as v4)
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
# Pinned to GPU 1 by default (v4 lives on GPU 0).

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
    --save_ckpt checkpoints/pretrain_mix_v6_shallow.pt \
    --tb_dir runs/tb/pretrain_mix_v6_shallow \
    > runs/pretrain_mix_v6_shallow.log 2>&1 &
echo "Launched pretrain_mix_v6_shallow on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v6_shallow.log"
