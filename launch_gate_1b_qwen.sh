#!/bin/bash
# STEP-2 GATE (GO/NO-GO before the multi-day full base): does a 1B-Qwen-vocab
# DeltaNet train STABLY, and does KD from Qwen-Coder-7B HELP vs a no-KD control?
#
# LEAN config on purpose — trunk + FiLM + KD only (NO WM/PKM/latent/gist). The
# gate isolates the two things that are NEW at this scale (the 1B size + the Qwen
# 152k-vocab embedding + offline KD); the mechanisms are validated at 287M and go
# into the full production run only after GO.
#
# A/B: run twice ->  KD=1 (default, reads the gate store)  vs  KD=0 (CE only).
# Compare VAL CE + per-source CE trajectory at iso-token. KD arm clearly ahead +
# both stable => GO.
#
#   KD=1 bash launch_gate_1b_qwen.sh      # KD arm
#   KD=0 bash launch_gate_1b_qwen.sh      # no-KD control
#
# Single-GPU (offline KD is single-GPU only). GPU0 is flaky under sustained load
# -> default GPU1.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=${GPU:-1}
KD=${KD:-1}
MIX=${MIX:-configs/pretrain_mix_v18_arxiv.yaml}
TOK=${TOK:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}
LOGITS=${LOGITS:-data/teacher_logits_qwen_gate}
STEPS=${STEPS:-1100}                 # ~144M tok at batch2*ga32*T2048 (131k tok/step) < 150M store
KDW=${KDW:-0.5}                       # KD weight (gate A used 0.5; capacity-gap fix tries lighter)
KDT=${KDT:-2.0}                       # KD temperature (gate A used 2.0; capacity-gap fix tries softer)
TAG=${TAG:-$([ "$KD" = "1" ] && echo kd || echo noKD)}

KD_FLAGS=""
if [ "$KD" = "1" ]; then
  KD_FLAGS="--distill_logits_dir $LOGITS --distill_weight $KDW --distill_temp $KDT"
fi

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 2048 --n_layers 10 --d_head 64 --n_heads 32 --tie_embeddings \
  --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" --feedback_self_k 3 --feedback_self_k_warmup_steps 200 \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 --gate_floor_min 0.5 --gate_warmup_steps 20000 \
  --matrix_optimizer fused_deltanet_ns \
  $KD_FLAGS \
  --data_mix "$MIX" --tokenizer "$TOK" \
  --think_burst_prob 0 --num_workers 0 \
  --T 2048 --batch 2 --grad_accum 32 --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.15 \
  --steps $STEPS --val_every 100 --log_every 20 \
  --seed 0 \
  --save_ckpt checkpoints/gate_1b_qwen_${TAG}.pt --tb_dir runs/tb/gate_1b_qwen_${TAG} \
  > runs/gate_1b_qwen_${TAG}.log 2>&1 &
echo "Launched GATE 1B-Qwen [${TAG}] GPU $GPU (PID $!).  Watch: tail -f runs/gate_1b_qwen_${TAG}.log"
