#!/bin/bash
# RL-grader on v7.1-pkm-film SFT base — the "real" run.
#
# Base: checkpoints/sft_v7_pkm_film_thinking.pt (10L × 896d shallow-wide
#       + dense FiLM + PKM v7.1 + entropy-gate, SFT'd on MBPP+CodeAlpaca
#       with thinking enabled). Final SFT loss 0.497.
#
# Companion to launch_rl_grader_v5_pkm.sh (running on GPU 0). The
# v7.1 base has the entropy-grounded gate auxiliary BCE baked into
# pretrain — if that survives SFT, we should see meaningfully lower
# think_rate from step 1 than the v5-pkm-SFT counterpart.
#
# Same RL knobs as v5 launcher for apples-to-apples gate comparison.
# Pinned to GPU 1.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v7_pkm_film_thinking.pt \
    --save_ckpt checkpoints/rl_grader_v7_pkm_film.pt \
    --steps 400 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 5e-6 \
    --max_gen 96 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.9 \
    --min_emit_before_eos 30 \
    --clip_eps 0.2 \
    --ponder_cost 0.005 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 50 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_v7_pkm_film.log 2>&1 &
echo "Launched RL-grader v7.1-pkm-film on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v7_pkm_film.log"
