#!/bin/bash
# Mixed-corpus pretrain v1: 217 M DN + FiLM(2,28) + memory + thinking-gate,
# on a 9-source open-license corpus (codeparrot + Python instruct datasets
# + synthetic CS textbooks + Wikipedia). Auto-stops on flat HumanEval over
# two consecutive 500 M-token intervals, or at 2 B tokens.
#
# Hardware: tested for 1× RTX 5090 (32 GB). For 2× set CUDA_VISIBLE_DEVICES
# accordingly; this script uses a single process for simplicity.
#
# Expected milestones (rough):
#   tokens   HumanEval pass@1
#   500 M     ≥ 2 %
#     1 B    ≥ 8 %
#     2 B   ≥ 13 %
#
# Total tokens consumed = steps * batch * T. Tune the smoke step count to
# match: at batch=8, T=2048: 1 step = 16 384 tokens; 2 B tokens ≈ 122 000 steps.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch gated_deltanet \
    --d_model 576 --n_layers 30 --d_head 64 --n_heads 9 \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --output_gate \
    --use_memory --mem_size 1024 \
    --data_mix configs/pretrain_mix_v1.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 8 \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --steps 130000 \
    --val_every 2000 --log_every 100 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --auto_stop --auto_stop_threshold 0.01 --auto_stop_k 2 \
    --save_ckpt checkpoints/pretrain_mix_v1.pt \
    --tb_dir runs/tb/pretrain_mix_v1 \
    > runs/pretrain_mix_v1.log 2>&1 &
echo "Launched pretrain_mix_v1 on GPU ${GPU:-0} (PID $!)"
echo "Watch:"
echo "  tail -f runs/pretrain_mix_v1.log"
echo "  tensorboard --logdir runs/tb"
