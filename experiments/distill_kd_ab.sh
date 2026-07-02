#!/bin/bash
# Pretrain logit-KD A/B (task #101).
#
# GOAL: does adding a frozen-teacher logit-KD term to normal CE pretraining let
# the 287M student reach a target CODE cross-entropy in FEWER tokens than
# CE-only? Two arms, IDENTICAL except the KD term (same seed / data / order /
# batch / budget):
#   - CEONLY: --distill_weight 0           (teacher never loaded, pure CE)
#   - KD:     SmolLM2-1.7B teacher, KL(teacher||student) on real positions,
#             weight 0.5, temperature 2.0.
#
# Both share the SmolLM2 tokenizer (vocab 49152) so token ids align; the
# student logits are sliced to the teacher vocab before the KL (drops the
# student's extra/thinking slots — the load-bearing correctness detail).
#
# Run SEQUENTIALLY on GPU1 (GPU0 is hardware-flaky under load — do NOT use it).
# Compile is OFF (--no-compile): the A/B does not need it and it keeps the run
# robust. The teacher forward is a SEPARATE uncompiled module, so KD itself does
# not require disabling compile, but we keep it off for the A/B for simplicity.
#
#   tokens/step = batch * grad_accum * T = 6 * 12 * 2048 = 147,456
#   STEPS=1500 default  ~=  221M tokens/arm.
#
# Memory (smoke-verified on a 32GB 5090): batch 8 OOMs in the KD arm — the KD
# block materializes several fp32 (B,T,49152) tensors (teacher softmax + student
# log-softmax + the kl_div elementwise output) ON TOP of the ~28 GB student
# training footprint, and the +3 GB kl_div allocation tips it over. Batch 6
# fits both arms with headroom. NOTE 6*12 == 8*9 == 72 sequences/step, so
# dropping batch to 6 (and GA to 12) keeps tokens/step IDENTICAL — the A/B
# budget is unchanged, only the per-microbatch peak shrinks. The CE-only arm
# fits at batch 8 too, but we run BOTH arms at batch 6 so the only difference
# stays the KD term.
#
# Usage:
#   ./experiments/distill_kd_ab.sh                 # full 1500-step A/B
#   STEPS=10 ./experiments/distill_kd_ab.sh        # smoke
#   BATCH=8 GA=9 ./experiments/distill_kd_ab.sh    # higher-batch (CE-only-safe)

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/distill_kd_ab checkpoints

GPU=${GPU:-1}
STEPS=${STEPS:-1500}
BATCH=${BATCH:-6}
GA=${GA:-12}
SEED=${SEED:-0}

# Common args — the ONLY difference between arms is the trailing KD flags.
COMMON=(
    --arch deltanet
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14
    --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9"
    --feedback_self_k 3 --feedback_self_k_warmup_steps 200
    --output_gate
    --data_mix configs/pretrain_mix_v4.yaml
    --tokenizer HuggingFaceTB/SmolLM2-135M
    --think_burst_prob 0
    --T 2048 --batch "$BATCH" --grad_accum "$GA"
    --activation_checkpointing
    --bf16 --tf32 --no-compile
    --bf16_optim_state
    --alpha_wd 0.0 --wd 0.01
    --grad_clip 1.0 --z_loss 1e-4
    --mask_eos_in_targets
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3
    --lr_schedule wsd --warmup_steps 200 --lr_decay_frac 0.0
    --steps "$STEPS"
    --val_every 200 --log_every 50 --seed "$SEED"
)

echo "=== Arm 1/2: CEONLY (--distill_weight 0) ==="
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u experiments/train_lm.py \
    "${COMMON[@]}" \
    --distill_weight 0 \
    --save_ckpt checkpoints/distill_ceonly.pt \
    --tb_dir runs/tb/distill_ceonly \
    2>&1 | tee runs/distill_kd_ab/ceonly.log

echo "=== Arm 2/2: KD (SmolLM2-1.7B teacher, weight 0.5, temp 2.0) ==="
CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -u experiments/train_lm.py \
    "${COMMON[@]}" \
    --distill_teacher_model HuggingFaceTB/SmolLM2-1.7B \
    --distill_weight 0.5 --distill_temp 2.0 \
    --save_ckpt checkpoints/distill_kd.pt \
    --tb_dir runs/tb/distill_kd \
    2>&1 | tee runs/distill_kd_ab/kd.log

echo "=== A/B complete. ckpts: checkpoints/distill_{ceonly,kd}.pt ==="
echo "Compare VAL CE / tokens-to-target between runs/distill_kd_ab/{ceonly,kd}.log"
