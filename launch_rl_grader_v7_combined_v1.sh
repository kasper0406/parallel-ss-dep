#!/bin/bash
# RL grader on the v1 combined SFT base (the better of v1/v2; pass@1=11/164).
#
# Base: checkpoints/sft_v7_pkm_film_combined.pt
#   - distilled from Qwen 3.6 on ~38k pairs + 12.5k synthetic memory
#   - trained with future_emb_loss_weight=0.05
#   - FIX A was disabled-by-bug (control flow), making this the better
#     experimental base than the FIX A-active v2 (which scored 9/164)
#
# Critical knob fix from the v2 attempt: --gate_floor 0.3 (was 0.5).
# When gate_floor >= emit_threshold, the rollout's clamp(gate, floor) >=
# emit_threshold check is trivially true → model never thinks.
# See experiments/test_rl_grader_gate_floor.py for the regression test.
# With gate_floor=0.3 and emit_threshold=0.5, low gate values still
# trigger think (clamped to 0.3, below 0.5 threshold), while truly low
# values get clamped up to 0.3 (not 0.0), preventing the over-think
# pathology we saw in early RL attempts on distilled bases.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_v7_pkm_film_combined.pt \
    --save_ckpt checkpoints/rl_grader_v7_combined_v1.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch 2 \
    --grpo_n_group 4 \
    --lr 5e-6 \
    --max_gen 768 \
    --max_think_per_step 8 \
    --total_think_budget 384 \
    --emit_threshold 0.5 \
    --gate_floor 0.3 \
    --temperature 0.9 \
    --min_emit_before_eos 30 \
    --clip_eps 0.2 \
    --ponder_cost 0.001 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 100 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_v7_combined_v1.log 2>&1 &
echo "Launched RL-grader v7-combined-v1 on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/rl_grader_v7_combined_v1.log"
