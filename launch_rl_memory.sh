#!/bin/bash
# RL+memory experiment: does enabling working memory + deep thinking with a
# small ponder cost actually move HumanEval / per-token CE on a real base?
#
# Launches mem-on on GPU 0 and mem-off control on GPU 1 from the SFT base.
# The SFT base (sft_dn217_nomem.pt) is Qwen3.6-tokenized (vocab 248k), so we
# pass --tokenizer explicitly.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p runs checkpoints

SHARED=(
    --steps 1500
    --batch 1 --grpo_n_group 8 --lr 5e-6
    --grpo_kl_beta 0.02            # KL anchor to the SFT base — prevents drift
    --grpo_ponder_cost 0.01        # slight cost per think step, so depth must pay back
    --max_depth 16
    --T 512 --max_T 0 --min_decision_pos 16
    --think_checkpointing
    --hard_pos_sampling            # sample decision positions from the
    --hard_ce_min 1.5              # [moderate, hard] CE band where thinking
    --hard_ce_max 6.0              # can actually help
    --tokenizer QuantTrio/Qwen3.6-35B-A3B-AWQ
    --load_ckpt checkpoints/sft_dn217_nomem.pt
)

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. nohup .venv/bin/python -u experiments/train_rl.py \
    "${SHARED[@]}" \
    --use_memory --mem_size 512 \
    --save_ckpt checkpoints/rl_memon_d16.pt \
    --tb_dir runs/tb/rl_memon_d16 \
    > runs/rl_memon_d16.log 2>&1 &
echo "Launched rl_memon_d16 on GPU 0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. nohup .venv/bin/python -u experiments/train_rl.py \
    "${SHARED[@]}" \
    --save_ckpt checkpoints/rl_memoff_d16.pt \
    --tb_dir runs/tb/rl_memoff_d16 \
    > runs/rl_memoff_d16.log 2>&1 &
echo "Launched rl_memoff_d16 on GPU 1 (PID $!)"

echo
echo "Watch:"
echo "  tail -f runs/rl_memon_d16.log"
echo "  tail -f runs/rl_memoff_d16.log"
