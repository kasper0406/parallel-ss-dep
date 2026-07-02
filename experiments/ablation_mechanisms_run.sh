#!/bin/bash
# ISO-TOKEN, ISO-MIX MECHANISM ABLATION — does the day-1 mechanism STACK
# (PKM + WM/mem_ctx_namekey + latent + gist + think bursts) cost CODE capacity,
# holding the data mix and budget FIXED? This is the experiment that disentangles
# "mechanisms eat code capacity" from "v18's mix-dilution" (the v18-vs-PhaseC
# per-source CE left them entangled because v18 ALSO changed the mix + token budget).
#
# Two arms, IDENTICAL except the mechanism flags:
#   LEAN  = pure trunk + FiLM + output gate (no PKM/WM/latent/gist, no think bursts)
#   MECH  = LEAN + the full v18 day-1 stack (PKM v7.1 + WM mem_ctx_namekey + copy head
#           + latent reasoning cotrain + gist + think-burst injection)
# Both: fresh-from-scratch 287M (10L d896 14h), SAME code-focused mix
# (pretrain_mix_v4.yaml — NO arXiv/recall dilution), SAME seed/batch/T/schedule/steps.
# So the ONLY variable is the mechanism stack → a clean read on its code-capacity cost.
#
# Mechanism warmups are SHORTENED (800 / latent-engage 400) vs v18's 3000/2000 so the
# mechanisms actually bootstrap within the ablation budget (else MECH is unfairly inert).
#
# Runs the two arms IN PARALLEL — LEAN on GPU 0, MECH on GPU 1 (both GPUs free post-v18;
# one 287M job per card fits). Throwaway (no ckpt saved); curves from logs/TB; final
# per-source code CE via probe_gen_vs_code at the end if ckpts are saved (SAVE=1).
#
# Usage:  STEPS=20 bash experiments/ablation_mechanisms_run.sh     # smoke (validate)
#         STEPS=3000 SAVE=1 bash experiments/ablation_mechanisms_run.sh   # full run
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=runs/ablation_mech
mkdir -p $OUT runs/tb

STEPS=${STEPS:-20}
SAVE=${SAVE:-0}
BATCH=12
GA=6
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

# LEAN: no mechanisms, no think bursts.
LEAN="--think_burst_prob 0"

# MECH: the full v18 day-1 stack with SHORTENED warmups (800 / latent-engage 400).
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

save_flag() { [ "$SAVE" = "1" ] && echo "--save_ckpt checkpoints/ablation_$1.pt"; }

echo "[ablation] $(date '+%F %T') START steps=$STEPS save=$SAVE  (LEAN→GPU0, MECH→GPU1, parallel)" | tee $OUT/driver.log

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/train_lm.py $COMMON $LEAN $(save_flag lean) \
  --tb_dir runs/tb/ablation_lean > $OUT/lean.log 2>&1 &
PID_LEAN=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u experiments/train_lm.py $COMMON $MECH $(save_flag mech) \
  --tb_dir runs/tb/ablation_mech > $OUT/mech.log 2>&1 &
PID_MECH=$!
echo "[ablation] LEAN pid=$PID_LEAN  MECH pid=$PID_MECH" | tee -a $OUT/driver.log
wait $PID_LEAN; RC_LEAN=$?
wait $PID_MECH; RC_MECH=$?
echo "[ablation] $(date '+%F %T') END  lean_rc=$RC_LEAN mech_rc=$RC_MECH" | tee -a $OUT/driver.log

if [ "$SAVE" = "1" ] && [ $RC_LEAN -eq 0 ] && [ $RC_MECH -eq 0 ]; then
  echo "[ablation] per-source code CE:" | tee -a $OUT/driver.log
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u /tmp/probe_gen_vs_code.py checkpoints/ablation_lean.pt 2>/dev/null | tee -a $OUT/driver.log
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u /tmp/probe_gen_vs_code.py checkpoints/ablation_mech.pt 2>/dev/null | tee -a $OUT/driver.log
fi
echo "[ablation] $(date '+%F %T') ALL DONE" | tee -a $OUT/driver.log
