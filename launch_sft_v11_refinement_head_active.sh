#!/bin/bash
# SFT v11 — Phase D RefinementHead with α init 0.3 (active from step 1).
#
# v10 lesson: with α init 0, the head stays inert. Probe of v10 ckpt
# showed α = 2.6e-4 after 960 steps — barely moved from zero. The
# MLP weights stayed at init too. Training loss was happy with the
# trunk alone (final 0.11) so there was no pressure for α to grow.
#
# v11 fix: init α at 0.3. The head contributes meaningfully from
# step 1, MLP weights get real gradient through the loss, and the
# model learns whether to keep or suppress the refinement.
#
# Risk: at step 1 the head's MLP is random init → noise injection
# at low-σ positions → loss spike. Trade-off: short-term loss for
# real exercise of the new mechanism. The model can drive α back
# toward 0 if the head truly isn't useful.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/sft_v11_refinement_head_active.pt \
    --distilled_jsonl data/sft_cot_thinking_v1.jsonl \
    --distilled_keep_only_passing \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --state_readonly_at_think \
    --use_refinement_head \
    --refinement_head_window 128 \
    --refinement_head_n_heads 8 \
    --refinement_head_mlp_mult 2 \
    --refinement_head_alpha_init 0.3 \
    --max_codealpaca 0 \
    --epochs 2 \
    --batch 2 \
    --lr 3e-6 \
    --max_len 1024 \
    --log_every 50 \
    --seed 0 \
    > runs/sft_v11_refinement_head_active.log 2>&1 &
echo "Launched SFT v11 (Phase D RefinementHead, α init 0.3) — PID $!"
echo "Watch: tail -f runs/sft_v11_refinement_head_active.log"
