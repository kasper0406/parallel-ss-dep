#!/bin/bash
# Feature-pilot Arm A (LEAN control) — iso-data/iso-token twin of arm B.
# The engaged-features A/B the repo never had (SESSION_FINDINGS.md 2026-07-02).
# Same base ckpt / mix / seed / steps / LR as B; features OFF.
#
# Run on GPU1 (GPU0 is hardware-flaky under sustained load).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

GPU=${GPU:-1}
STEPS=${STEPS:-2500}
SEED=${SEED:-0}
TAG=${TAG:-feature_pilot_A}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/linearize/linearized_stage3.pt --keep_base_vocab 49152 \
  --feedback none \
  --data_mix configs/pretrain_mix_feature_pilot.yaml \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.10 \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 100000000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!) — tail -f runs/${TAG}.log"
