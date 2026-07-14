#!/bin/bash
# Tier-1 kill-test #1 — OLMo-2-style microanneal A/B (2026-07-14).
# ideas_2026_07_13/10_small_model_recipes.md element #1 ("the most replicated
# lever": decay-phase data upgrade), IDEAS_2026_07_13.md convergence #2.
# Pre-registered decision block: IDEAS_2026_07_13.md "Tier-1 kill-tests log".
#
# ONE arm per invocation:
#   ARM=control ./launch_microanneal_AB.sh   # decay on the pilot mix (status quo)
#   ARM=anneal  ./launch_microanneal_AB.sh   # decay on configs/anneal_mix_v1.yaml
#
# Both arms: identical recipe (arch/optim/batch/seed inherited from
# launch_feature_pilot_A.sh), same base ckpt checkpoints/stageA_executor.pt,
# ~2300 steps x 131,072 tok/step (T2048 x b2 x ga32) ~= 300M tokens, single
# GPU. LR shape is the ANNEAL: wsd with --warmup_steps 50 --lr_decay_frac 0.85
# (mostly-decay from the pilot's peak LR). The ONLY inter-arm delta is
# --data_mix.
#
# Run arms on separate GPUs (GPU=0/1) or sequentially on GPU 1
# (GPU0 is hardware-flaky under sustained load).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

ARM=${ARM:?set ARM=control or ARM=anneal}
case "$ARM" in
  control) MIX=${MIX:-configs/pretrain_mix_feature_pilot.yaml} ;;
  anneal)  MIX=${MIX:-configs/anneal_mix_v1.yaml} ;;
  *) echo "ARM must be 'control' or 'anneal', got '$ARM'" >&2; exit 1 ;;
esac

GPU=${GPU:-1}
STEPS=${STEPS:-2300}          # x131,072 tok/step ~= 301M tokens
SEED=${SEED:-0}
TAG=${TAG:-microanneal_${ARM}}

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/stageA_executor.pt --keep_base_vocab 49152 \
  --feedback none \
  --data_mix $MIX \
  --fim_legacy_strings \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 50 --lr_decay_frac 0.85 \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 50000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!) mix=$MIX — tail -f runs/${TAG}.log"
