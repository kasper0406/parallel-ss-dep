#!/usr/bin/env bash
# grader-RL (v2 KL-stable, execution-grounded GRPO) on the SFT'd Qwen-linearized base.
# Plain model (no gate) → pure execution-RL. Conservative batch for the 152k-vocab head.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints
CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_qwen_coder_05b.pt \
    --save_ckpt checkpoints/rl_grader_qwen_coder_05b.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 400 \
    --batch ${RL_BATCH:-2} \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.05 \
    --ponder_cost 0.0 \
    --ponder_warmup_steps 50 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_grader_qwen_coder_05b.log 2>&1 &
echo "Launched grader-RL on SFT'd Qwen base, GPU ${GPU:-1} (PID $!)"
