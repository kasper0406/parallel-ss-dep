#!/bin/bash
# Fast variant of launch_pretrain_mix_v1.sh:
#   * bf16 autocast on model.forward (FLA's gated-delta-rule kernels are
#     bf16 internally; this stops the fp32→bf16→fp32 round-trip per layer).
#   * TF32 enabled for the residual fp32 matmul.
#   * batch 8 (vs 4) — bf16 frees enough activation memory to fit.
#   * 130k steps (back to v1's original plan) since each step now consumes
#     8*2048 = 16384 tokens, so 130k * 16384 ≈ 2.1 B tokens.
#
# Expected throughput: ~50-60k tok/s sustained (~2.5-3x the fp32 baseline
# of 18k tok/s). Total ETA: ~10-14 hr instead of ~35.
#
# Use `GPU=1 bash launch_pretrain_mix_v1_fast.sh` to run on the second
# GPU in parallel with the live v1 run on GPU 0, so we can measure the
# speedup before deciding to kill v1.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 576 --n_layers 30 --d_head 64 --n_heads 9 \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --output_gate \
    --use_memory --mem_size 1024 \
    --data_mix configs/pretrain_mix_v1.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 8 \
    --bf16 --tf32 \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --steps 130000 \
    --val_every 2000 --log_every 100 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --auto_stop --auto_stop_threshold 0.01 --auto_stop_k 2 \
    --save_ckpt checkpoints/pretrain_mix_v1_fast.pt \
    --tb_dir runs/tb/pretrain_mix_v1_fast \
    > runs/pretrain_mix_v1_fast.log 2>&1 &
echo "Launched pretrain_mix_v1_fast on GPU ${GPU:-0} (PID $!)"
echo "Watch:"
echo "  tail -f runs/pretrain_mix_v1_fast.log"
echo "  tensorboard --logdir runs/tb"
