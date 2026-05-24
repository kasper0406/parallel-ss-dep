#!/bin/bash
# Round 2: KL β=0.5 on GPU 0 + past-ckpt encoder on GPU 1
set -e
cd /home/knielsen/ml/parallel-ss-dep

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --logit_kl_beta 0.5 \
    --encoder_ckpt checkpoints/dn_baseline_30L_217M_for_oracle.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --log_every 500 --val_every 2500 \
    --save_ckpt checkpoints/film_self_k3_kl_b05_217M.pt \
    > runs/film_self_k3_kl_b05.log 2>&1 &
echo "GPU 0 (KL b=0.5) PID: $!"

CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --semantic_loss_beta 1.0 --semantic_loss_uniform_weight \
    --encoder_ckpt checkpoints/film_self_k3_2_28_30L_217M.pt \
    --d_model 576 --n_heads 9 --d_head 64 --n_layers 30 \
    --T 512 --batch 8 --steps 5000 --lr 3e-4 --seed 0 \
    --dataset codeparrot/codeparrot-clean --text_field content \
    --log_every 500 --val_every 2500 \
    --save_ckpt checkpoints/film_self_k3_lsem_uniform_pastckpt_enc_217M.pt \
    > runs/film_self_k3_lsem_uniform_pastckpt_enc.log 2>&1 &
echo "GPU 1 (past-ckpt) PID: $!"
