#!/bin/bash
# Iso-token mechanism ablation — SEQUENTIAL, memory-safe rerun.
# The parallel/GPU0 version failed twice: GPU0 cudaErrorLaunchFailure under load
# (hardware flakiness) + MECH OOM at batch 12 (latent R=8 at full ramp + WM + PKM
# spiked past 32GB). Fixes: (1) BOTH arms on GPU1 only, SEQUENTIAL (MECH then LEAN);
# (2) batch 8 grad_accum 9 = 147,456 tok/step (SAME as the batch12/ga6 iso-token
# budget, so the comparison is unchanged) with lower peak memory.
# Everything else identical to ablation_mechanisms_run.sh. SAVE always on.
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/ablation_mech_seq
mkdir -p $OUT runs/tb
GPU=${GPU:-1}
STEPS=${STEPS:-2000}
BATCH=8
GA=9
KWARM=200
T=2048
SEED=0

COMMON="--arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
  --feedback film --feedback_pairs 0,5;1,6;2,7;3,8;4,9 --feedback_self_k 3 --feedback_self_k_warmup_steps $KWARM \
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 --gate_floor_min 0.5 --gate_warmup_steps 20000 \
  --data_mix configs/pretrain_mix_v4.yaml --tokenizer HuggingFaceTB/SmolLM2-135M \
  --T $T --batch $BATCH --grad_accum $GA --activation_checkpointing \
  --bf16 --tf32 --no-compile --bf16_optim_state \
  --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
  --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.0 \
  --steps $STEPS --val_every 200 --log_every 50 --seed $SEED"

LEAN="--think_burst_prob 0"

MECH="--think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
  --state_readonly_at_think \
  --use_memory --mem_size 2048 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192 \
  --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match \
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -6.0 --mem_read_alpha_init 0.0 \
  --mem_freeze_read_alpha --emit_read_mask \
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 800 --ctx_addr_aux_start_step 0 \
  --use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 800 --pkm_value_init_std 1.0 \
  --pkm_score_norm layer --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 800 --pkm_value_lr_mult 100.0 \
  --gist_loss_weight 0.1 --gist_horizons 16,64,256 \
  --use_latent_feedback_adapter --latent_reasoning_weight 0.05 --latent_reasoning_train_prefix data/ptr10dict_train \
  --latent_reasoning_rungs 2,3,4,5,6,7,8 --latent_reasoning_n 4 --latent_reasoning_gate_weight 0.05 \
  --latent_reasoning_start_step 400 --latent_reasoning_weight_warmup_steps 800"

echo "[ablseq] $(date '+%F %T') START steps=$STEPS batch=$BATCH ga=$GA GPU=$GPU (MECH then LEAN, sequential)" | tee $OUT/driver.log

echo "[ablseq] $(date '+%F %T') MECH start" | tee -a $OUT/driver.log
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py $COMMON $MECH \
  --save_ckpt checkpoints/ablation_mech.pt --tb_dir runs/tb/ablseq_mech > $OUT/mech.log 2>&1
RC_MECH=$?
echo "[ablseq] $(date '+%F %T') MECH end rc=$RC_MECH" | tee -a $OUT/driver.log

echo "[ablseq] $(date '+%F %T') LEAN start" | tee -a $OUT/driver.log
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py $COMMON $LEAN \
  --save_ckpt checkpoints/ablation_lean.pt --tb_dir runs/tb/ablseq_lean > $OUT/lean.log 2>&1
RC_LEAN=$?
echo "[ablseq] $(date '+%F %T') LEAN end rc=$RC_LEAN" | tee -a $OUT/driver.log

if [ $RC_MECH -eq 0 ] && [ $RC_LEAN -eq 0 ]; then
  echo "[ablseq] per-source code CE (LEAN then MECH):" | tee -a $OUT/driver.log
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u /tmp/probe_gen_vs_code.py checkpoints/ablation_lean.pt 2>/dev/null | tee -a $OUT/driver.log
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u /tmp/probe_gen_vs_code.py checkpoints/ablation_mech.pt 2>/dev/null | tee -a $OUT/driver.log
fi
echo "[ablseq] $(date '+%F %T') ALL DONE (mech_rc=$RC_MECH lean_rc=$RC_LEAN)" | tee -a $OUT/driver.log
