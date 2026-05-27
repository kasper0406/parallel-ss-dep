#!/bin/bash
# v5-pkm: same recipe as v4 + a single Product-Key Memory layer
#         after block 14. Small-scale probe (~880M tokens vs v4's 4.3B).
#
# Hypothesis: a learned KV side-table can absorb the long-tail factual
# content (Wikipedia entities, library API signatures, CVE numbers,
# algorithm references) that the dense 218M backbone cannot memorize at
# its parameter density. Primary ablation target: per-source CE on
# bigvul + cybernative *stops drifting up* across token marks. See
# PKM_PLAN.md for the full design.
#
# What's identical to v4:
#   - architecture (DeltaNet 30L × 576d), FiLM(2,28) K=3, mem 1024
#   - mix (configs/pretrain_mix_v4.yaml), schedule (WSD), all knobs
#
# What's added:
#   --use_pkm --pkm_after_layer 14
#       (defaults: 4 heads × 256² slots/head = 262k effective slots;
#        ~38.5M PKM params; bf16 value storage = ~75MB persistent)
#   --bf16_optim_state
#       (banks the 550MB optim-state saving from commit 60c9d1f)
#   --batch 12 (down from v4's 20)
#       (PKM adds ~80MB params + activations; smoke validated at 12)
#   --steps 4500 → ~880M tokens (small-scale probe)
#   tokens/step = 12 * 8 * 2048 = 196,608
#
# Pinned to GPU 1 (v4 owns GPU 0) by default.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 576 --n_layers 30 --d_head 64 --n_heads 9 \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 1300 \
    --output_gate \
    --use_memory --mem_size 1024 \
    --use_pkm --pkm_after_layer 14 \
    --bf16_optim_state \
    --data_mix configs/pretrain_mix_v4.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 12 --grad_accum 8 \
    --activation_checkpointing \
    --bf16 --tf32 --compile \
    --alpha_wd 0.0 --wd 0.01 \
    --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --gate_floor_min 0.5 --gate_warmup_steps 1500 \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
    --steps 4500 \
    --val_every 200 --log_every 20 \
    --mid_eval_every_tokens 250000000 \
    --mid_eval_save_only \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_mix_v5_pkm.pt \
    --tb_dir runs/tb/pretrain_mix_v5_pkm \
    > runs/pretrain_mix_v5_pkm.log 2>&1 &
echo "Launched pretrain_mix_v5_pkm on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v5_pkm.log"
