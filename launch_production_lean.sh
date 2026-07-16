#!/bin/bash
# PRODUCTION lean continued-pretrain (task #5, launched 2026-07-16) — the
# 3B-token composition run on the linearized lineage, with the KT-1b-adopted
# anneal-in-decay design. Pre-registered decision block: IDEAS_2026_07_13.md
# "Tier-1 kill-tests log" -> "PRODUCTION run registration".
#
# Structure (WSD "cash-out" pattern — the decay is a separate cheap launch so
# it can be re-run from the same plateau ckpt at a different anneal strength):
#   PHASE=plateau        GPU1, 19,500 steps x 131,072 tok = 2.56B tokens,
#                        pilot mix, constant LR after warmup (decay_frac 0.0).
#   PHASE=decay_anneal   GPU1, 3,500 steps = 459M tokens from the plateau
#                        ckpt, configs/anneal_mix_v3.yaml (0.4x strength),
#                        mostly-decay LR (warmup 25, decay_frac 0.95).
#   PHASE=decay_control  GPU0, identical to decay_anneal but on the pilot mix
#                        — the guard control for the pre-registered A/B.
#
# Recipe is the KT-1 lineage verbatim (arch/optim/batch/seed from
# launch_microanneal_AB.sh; legacy-string FIM stays ON — fim_rate 0.5 on the
# code sources is part of the lineage). Per-head-NS / triage / decoupled-KD
# are EXCLUDED (no kill-test on this lineage; compose-survivors discipline).
#
#   PHASE=plateau       ./launch_production_lean.sh
#   PHASE=decay_anneal  ./launch_production_lean.sh   # after plateau finishes
#   PHASE=decay_control ./launch_production_lean.sh   # in parallel, GPU0
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

PHASE=${PHASE:?set PHASE=plateau, decay_anneal or decay_control}
PLATEAU_CKPT=${PLATEAU_CKPT:-checkpoints/production_lean_plateau.pt}
case "$PHASE" in
  plateau)
    GPU=${GPU:-1}; STEPS=${STEPS:-19500}
    MIX=${MIX:-configs/pretrain_mix_feature_pilot.yaml}
    LOAD=${LOAD:-checkpoints/stageA_executor.pt}
    WARMUP=100; DECAY_FRAC=0.0 ;;
  decay_anneal)
    GPU=${GPU:-1}; STEPS=${STEPS:-3500}
    MIX=${MIX:-configs/anneal_mix_v3.yaml}
    LOAD=${LOAD:-$PLATEAU_CKPT}
    WARMUP=25; DECAY_FRAC=0.95 ;;
  decay_control)
    GPU=${GPU:-0}; STEPS=${STEPS:-3500}
    MIX=${MIX:-configs/pretrain_mix_feature_pilot.yaml}
    LOAD=${LOAD:-$PLATEAU_CKPT}
    WARMUP=25; DECAY_FRAC=0.95 ;;
  *) echo "PHASE must be plateau|decay_anneal|decay_control, got '$PHASE'" >&2; exit 1 ;;
esac

SEED=${SEED:-0}
TAG=${TAG:-production_lean_${PHASE#decay_}}   # plateau / anneal / control

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt $LOAD --keep_base_vocab 49152 \
  --feedback none \
  --data_mix $MIX \
  --fim_legacy_strings \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps $WARMUP --lr_decay_frac $DECAY_FRAC \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 100000000 --mid_eval_save_only --mid_eval_min_free_gib 2.0 \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG PHASE=$PHASE on GPU $GPU (PID $!) mix=$MIX load=$LOAD steps=$STEPS — tail -f runs/${TAG}.log"
