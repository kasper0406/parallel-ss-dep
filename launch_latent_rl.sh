#!/bin/bash
# GATE-DRIVEN latent-thinking RL (no CoT + high-bandwidth latent thinking + exec RL).
# The thinking GATE dynamically decides whether/how-many latent ponder steps to take
# (NO fixed R). Base = pure-code SFT (no CoT) whose gate currently OVER-thinks on code
# (sigmoid(gate)<0.5 at many code positions -> reward ~0.1 vs the no-think 0.42).
#
# The test: can gate-driven RL FIX the miscalibrated gate -- climb reward back to
# >=0.42 (the no-think baseline) by suppressing harmful thinking and KEEPING it only
# where it helps? That is the user's invariant (thinking-ON never worse than no-think)
# achieved by a LEARNED gate. Watch: reward up, think_rate settling, KL bounded.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints
CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/latent_rl.py \
    --base checkpoints/sft_baked_pure.pt \
    --save checkpoints/latent_rl_pure.pt \
    --dataset mbpp_combined \
    --steps ${STEPS:-200} \
    --batch 4 --n_group 6 \
    --lr ${LR:-1e-5} \
    --max_gen 160 --min_emit_before_eos 10 \
    --temperature 0.8 \
    --gate_floor 0.05 --gate_temperature 1.0 \
    --max_think_per_step 8 --total_think_budget 400 \
    --gate_entropy_bonus 0.01 \
    --ponder_cost ${PONDER:-0.004} \
    --clip_eps 0.2 --kl_coef 0.05 \
    --save_every 50 --log_every 1 --seed 0 \
    > runs/latent_rl_pure.log 2>&1 &
echo "Launched GATE-DRIVEN latent RL on pure-code base GPU ${GPU:-1} (PID $!)"
echo "  watch: tail -f runs/latent_rl_pure.log"
