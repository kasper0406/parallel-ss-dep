#!/bin/bash
# Stage-B — Coconut text->latent replacement (EXEC_TRACE_LATENT_PLAN.md
# "Staged addendum", 2026-07-13). Full fine-tune from the Stage-A text executor
# (checkpoints/stageA_executor.pt) with a FRESH LatentFeedbackAdapter, over the
# SAME Stage-A mix (the 0.15 exec-text stream stays as the anchor keeping the
# text executor alive) + the trace-mode latent-reasoning co-train that gradually
# replaces the first `s` TEXT trace steps with `s` LATENT hidden-feedback slots
# (s ramps 0->8 over the first 55% of the run, then consolidates; rung K uniform;
# s_eff=min(s,K)). Per-hop CE supervises each latent slot to its intermediate.
#
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
STEPS=${STEPS:-2600}
SEED=${SEED:-0}
TAG=${TAG:-stageB_latent_trace}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/stageA_executor.pt --keep_base_vocab 49152 \
  --feedback none \
  --data_mix configs/pretrain_mix_stageA_executor.yaml \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.10 \
  --use_latent_feedback_adapter \
  --latent_reasoning_weight 0.1 \
  --latent_reasoning_trace_mode \
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
