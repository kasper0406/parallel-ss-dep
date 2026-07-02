#!/bin/bash
# PKM pre-warm PROBE (2026-07-02) — can PKM learn on a CONVERGED base when the
# trunk is frozen so the loss pressure has nowhere else to go?
#
# Background: pilot-B runs 1+3 reproduced the converged-base PKM rejection
# (alphaL pinned ~0, value rows ERODING, slot coverage collapsing) at BOTH
# value-LR 0.06 and 0.14 — the trunk absorbs/fights the perturbation. Arm C
# (the earlier freeze-trunk attempt) failed differently: the alpha-floor at
# 0.3 dug an 8-nat loss hole on the frozen trunk (values still didn't learn
# inside it). This probe threads the needle:
#   - trunk FROZEN for the whole probe (--freeze_trunk_steps == steps)
#   - alpha WARM-STARTED positive (0.1) instead of floor-forced from 0
#     (no sign-flipping, real value-gradient from step 0, ~small loss hole)
#   - tiny floor (0.05) as a safety net, epsilon sustained (~whole probe)
#   - value LR 0.14 (233 x 6e-4)
# SUCCESS = value row-ratio climbs >= ~1.1 AND train loss recovers toward the
# frozen-base level (~1.2) as values learn to be useful; then the full run
# gets a Phase-0 pre-warm. FAILURE = rows flat/eroding or loss stuck high.
#
# PKM-ONLY probe (no FiLM/WM/latent/gate) — isolate the mechanism.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

GPU=${GPU:-0}
STEPS=${STEPS:-300}
TAG=${TAG:-pkm_prewarm_probe}

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
  --lr_schedule wsd --warmup_steps 20 --lr_decay_frac 0.0 \
  --freeze_trunk_steps $STEPS \
  --use_pkm --pkm_after_layer 16 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
  --pkm_use_output_gate --pkm_out_alpha_init 0.1 \
  --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 290 \
  --pkm_value_init_std 1.0 --pkm_score_norm layer \
  --pkm_alpha_floor_start 0.05 --pkm_alpha_floor_warmup_steps 290 \
  --pkm_value_lr_mult 233.0 \
  --steps $STEPS --val_every 100 --log_every 10 --seed 0 \
  --mid_eval_every_tokens 100000000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!) — tail -f runs/${TAG}.log"
