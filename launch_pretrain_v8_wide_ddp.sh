#!/usr/bin/env bash
# v8-wide DDP RESUME — 2-GPU batch-parallel continuation of the single-GPU run.
# Resumes from a mid-eval ckpt and finishes the run ~1.75x faster.
#
# KEY DESIGN (keeps the WSD step-schedule semantics so the resume is seamless):
#   * batch 4 x grad_accum 16 x world 2 = 262144 tok/step = EXACT single-GPU
#     match (batch 8 x ga 16). WSD schedule is step-based so --steps/--start_step
#     carry over unchanged.
#   * --no-compile: batch 8 AND batch 5 both OOM'd under DDP — PyTorch allocated
#     ~29GB even at batch 5, i.e. the cost is fixed DDP overhead + the compile
#     first-step autotuning spike, NOT activations. Dropping compile removes the
#     spike; batch 4 halves activations. Lose compile's ~10% but DDP's ~1.75x is
#     the dominant lever (net still ~1.6x over the compiled single-GPU run).
#   * bf16 grad-compression comm hook (default ON) — GPU0<->GPU1 is PHB / no P2P
#     (~4 GB/s host-staged); compression halves the per-step all-reduce.
#
# Usage: CKPT=checkpoints/pretrain_v8_wide_stepNNNN_tokNNN.pt START=NNNN \
#        bash launch_pretrain_v8_wide_ddp.sh
set -euo pipefail
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/home/knielsen/ml/pytorch-release:.
: "${CKPT:?set CKPT to the resume checkpoint}"
: "${START:?set START to the ckpt step}"

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 nohup .venv/bin/torchrun \
  --nproc_per_node=2 --master_port=29711 experiments/train_lm.py \
  --arch deltanet --d_model 1280 --n_layers 10 --d_head 64 --n_heads 20 \
  --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
  --feedback_self_k 3 --feedback_self_k_warmup_steps 1500 \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
  --gate_floor_min 0.5 --gate_warmup_steps 20000 \
  --state_readonly_at_think \
  --use_memory --mem_size 1536 \
  --use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 384 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 \
  --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 --pkm_value_lr_mult 100.0 \
  --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
  --data_mix configs/pretrain_mix_v4.yaml --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0.25 --think_max_bursts 1 --think_max_burst_depth 4 \
  --T 2048 --batch 4 --grad_accum 16 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 600 --lr_decay_frac 0.15 \
  --steps 31000 --start_step "$START" --load_ckpt "$CKPT" \
  --val_every 400 --log_every 50 \
  --mid_eval_every_tokens 500000000 --mid_eval_save_only \
  --save_ckpt checkpoints/pretrain_v8_wide.pt --tb_dir runs/tb/pretrain_v8_wide_ddp \
  > runs/pretrain_v8_wide_ddp.log 2>&1 &
echo "launched DDP resume pid $! from $CKPT @ step $START -> runs/pretrain_v8_wide_ddp.log"
