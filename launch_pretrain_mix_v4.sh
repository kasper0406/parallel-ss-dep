#!/bin/bash
# v4: re-weighted mix + the full staged-knobs set + the pretrain-knobs
# from the 2026-05-14 "techniques we should be using" review.
#
# Changes from v3-long:
#
#  1. --data_mix configs/pretrain_mix_v4.yaml
#     Re-weighted to fix the CVE-stream upward CE drift (bigvul/
#     cybernative up, codeparrot/jinaai down). See the YAML header.
#
#  2. Staged speed/quality knobs (GPU-validated 2026-05-14):
#       --lr_schedule wsd        no wasted low-LR tail; stop anywhere
#       --compile                +10 % (profiler)
#       --feedback_self_k_warmup_steps  K=1 FiLM bypass while early
#                                gradient is noise; +40 % over the window
#
#  3. Pretrain-knobs review (2026-05-14):
#       --activation_checkpointing  lets batch go 7 -> 14 at T=2048
#       --batch 14 --grad_accum 8   effective batch 112 seqs =
#                                   ~229k tokens/step (was 14k — the
#                                   tiny-batch noise was a real
#                                   convergence-efficiency drag)
#       --grad_clip 1.0          global grad-norm clip (was hardcoded;
#                                now an explicit flag)
#       --z_loss 1e-4            logit-drift regulariser
#
# !!! BATCH/ACCUM/STEPS ARE UNVALIDATED FOR MEMORY !!!
#     Smoke-test `--batch 14 --activation_checkpointing` fits in 32 GB
#     before the full launch; if it OOMs, drop to --batch 10 and bump
#     --grad_accum to keep tokens/step ~constant, and recompute --steps.
#
#   tokens/step = batch * grad_accum * T = 14 * 8 * 2048 = 229,376
#   2.13B tokens / 229,376  ~=  9,300 steps
#
# Pinned to GPU 0 by default.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 576 --n_layers 30 --d_head 64 --n_heads 9 \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 2000 \
    --output_gate \
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
    --gate_floor_min 0.5 --gate_warmup_steps 1200 \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.15 \
    --steps 9300 \
    --val_every 200 --log_every 20 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_mix_v4.pt \
    --tb_dir runs/tb/pretrain_mix_v4 \
    > runs/pretrain_mix_v4.log 2>&1 &
echo "Launched pretrain_mix_v4 on GPU ${GPU:-0} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v4.log"
