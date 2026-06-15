#!/bin/bash
# Grader-RL on the PURE-CODE base (no CoT; latent thinking is the only reasoning).
# v2-stable recipe (KL+lr+clip+temp that climbed to 16/164 on the CoT base), but
# from sft_baked_pure.pt and emit_threshold 0.3 (validated selective-thinking sweet
# spot). Execution reward forces correct entry_point/format + better logic.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints
CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_baked_pure.pt \
    --save_ckpt checkpoints/rl_grader_pure.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --emit_threshold 0.3 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.05 \
    --ponder_cost 0.0 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 50 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_pure.log 2>&1 &
echo "Launched grader-RL on pure-code base GPU ${GPU:-1} (PID $!)"
