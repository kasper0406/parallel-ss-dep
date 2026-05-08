#!/bin/bash
# Round 3: KL β=1.0 on GPU 0 (or whichever is free).
set -e
cd /home/knielsen/ml/parallel-ss-dep

GPU="${1:-0}"
CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --logit_kl_beta 1.0 \
    --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --log_every 500 --val_every 2500 \
    --save_ckpt checkpoints/film_self_k3_kl_b10_217M.pt \
    > runs/film_self_k3_kl_b10.log 2>&1 &
echo "GPU $GPU (KL b=1.0) PID: $!"
