#!/bin/bash
# Stage-B hop-7+ cliff FIX ARM (2026-07-13): continuation from the Stage-B ckpt
# with DEPTH-WEIGHTED curriculum sampling (P(s)~(1+s), P(K)~K) and NO ramp —
# pure consolidation aimed at the starved deep slots. Implicit control: the
# original run's last ~1,170 UNIFORM consolidation steps left hop-7/8 decode
# at ~0.09/0.00; if the cliff is exposure (not structural), this moves it.
# Everything else identical to launch_stageB_latent_trace.sh.
# Recipe = Stage-A LR/schedule/batch/arch (full plasticity, the Stage-A lesson)
# + N1' latent knobs (weight 0.1, n 4, per-hop 1.0). NO --state_readonly_at_think
# and NO gate — matching the actual N1' run / the stageA_executor base (state-
# writable; the batched growing-thread's doc_ids isolation resets the recurrent
# state at the pad boundary regardless of that flag). --no-compile is required by
# the latent path's variable-shape extra forwards.
#
# SINGLE GPU (DDP+latent is incompatible in this repo). Periodic ckpts every 50M
# tokens per the mandatory launch rule (Stage A survived a GPU-off-the-bus by it).
#
# Eval the resulting ckpt with:
#   experiments/eval_exec_trace_latent_trace.py --ckpt checkpoints/stageB_latent_trace.pt
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

GPU=${GPU:-0}
STEPS=${STEPS:-1200}
SEED=${SEED:-0}
TAG=${TAG:-stageB_depthfix}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/stageB_latent_trace.pt --keep_base_vocab 49152 \
  --feedback none \
  --data_mix configs/pretrain_mix_stageA_executor.yaml \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 50 --lr_decay_frac 0.20 \
  --use_latent_feedback_adapter \
  --latent_reasoning_weight 0.1 \
  --latent_reasoning_trace_mode \
  --latent_reasoning_no_ramp \
  --latent_reasoning_depth_weighted \
  --latent_reasoning_train_prefix data/exec_trace_train \
  --latent_reasoning_rungs 2,3,4,5,6,7,8 \
  --latent_reasoning_max_len 512 \
  --latent_reasoning_perhop_weight 1.0 \
  --latent_reasoning_n 4 \
  --latent_reasoning_aux_every 1 \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 50000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!) — tail -f runs/${TAG}.log"
