#!/bin/bash
# RL grader v5 — DDP across both 5090s + larger batch + higher LR.
#
# Hardware: 2× RTX 5090, 32 GiB each, no NVLink (PCIe 5.0 sync).
# Single-rank v3/v4 used 16 rollouts/step (batch=4 × n_group=4) on
# ~24 GiB → tiny gradient, LR had to crawl at 2e-6.
#
# v5: torchrun --nproc-per-node=2, each rank with batch=4 × n_group=8 =
# 32 rollouts/step → 64 effective rollouts/step (4× v3). Variance drops
# as 1/√N, so we can use lr=5e-6 (2.5× v3) at the same stability.
#
# Other recipe tweaks:
#   - Resume from rl_grader_phase_c_v3_step25.pt (the new project best,
#     17/164).
#   - 50 steps (vs v3's 200) — bigger steps cover more ground per step.
#   - save_every=5 → 10 milestones for fine-granularity peak probing.
#   - Same KL / clip / temp as v2/v3 (the stable settings).
#
# Two structural risks to monitor:
#   1. Per-rank memory at n_group=8 — each rollout now has 2× the rows
#      of v3. If we OOM, drop to n_group=6.
#   2. Effective lr too high → drift → collapse. KL=0.05 should constrain
#      it; watch the kl metric in the log. v2 stayed at 0.05-0.10
#      throughout; if v5's KL spikes past ~0.2 we're drifting.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

# torchrun handles RANK / WORLD_SIZE / LOCAL_RANK env injection.
# Both GPUs are used; --rdzv-endpoint runs on localhost for single-node.
nohup .venv/bin/torchrun \
    --nproc-per-node=2 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:29500 \
    experiments/train_rl_grader.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v3_step25.pt \
    --save_ckpt checkpoints/rl_grader_phase_c_v5.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 50 \
    --batch 4 \
    --grpo_n_group 8 \
    --lr 5e-6 \
    --max_gen 384 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
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
    --save_every 5 \
    --seed 0 \
    > runs/rl_grader_v5_ddp.log 2>&1 &
echo "Launched RL grader v5 (DDP × 2 GPU, 4× batch, 2.5× LR) — PID $!"
echo "Watch: tail -f runs/rl_grader_v5_ddp.log"
