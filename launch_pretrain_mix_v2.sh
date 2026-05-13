#!/bin/bash
# Pretrain v2: same architecture and trainer knobs as v1_fast, but on the
# extended data mix that adds BigVul + CyberNative CVE/vulnerability data.
#
# Changes vs launch_pretrain_mix_v1.sh:
#   * Data: configs/pretrain_mix_v2_with_cve.yaml (BigVul vul=1 records
#     + CyberNative chosen-side preference pairs added at ~7% combined
#     weight; v1 sources scaled down proportionally).
#   * Precision: bf16 autocast on model.forward + TF32 enabled. FLA's
#     gated-delta-rule kernels are bf16 internally so this stops the
#     fp32→bf16→fp32 round-trip per layer (measured 2.28x speedup on
#     1× RTX 5090).
#   * Batch: 8 (vs 4). bf16 frees enough activation memory to fit.
#   * Step count: 130k (vs v1's 260k). batch*T = 8*2048 = 16384 tok/step,
#     so 130k steps ≈ 2.13B tokens — same total budget as v1, half the
#     steps, ~14 hr wall time instead of ~35 hr.
#
# Defaults to GPU 0. Use GPU=1 to point at the second card.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
# Reduce CUDA-allocator fragmentation; lets us push closer to the 32GB
# ceiling without OOM from fragmented free pages.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

# Notes on the choices below:
# --alpha_wd 0.0   : WD probe on the v1 mid-eval ckpt showed |grad_α| ≈
#                    1.83 × WD·|α|. The "saturation" was actually a slow
#                    creep regime where gradient barely beat WD. Remove
#                    WD on α and let it find its real equilibrium.
# --batch 8        : bf16 frees ~50% activation footprint vs fp32, but
#                    tried batch=10 first and OOMed at the cross-entropy
#                    allocation (~30.2 GB used on a 31.4 GB card with
#                    expandable_segments on). 8 is the proven safe ceiling.
# --steps 130000   : batch*T = 8*2048 = 16384 tok/step, so 130k steps ≈
#                    2.13 B tokens (same total budget as the original v1
#                    plan).

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 576 --n_layers 30 --d_head 64 --n_heads 9 \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --output_gate \
    --use_memory --mem_size 1024 \
    --data_mix configs/pretrain_mix_v2_with_cve.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 8 \
    --bf16 --tf32 \
    --alpha_wd 0.0 \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --steps 130000 \
    --val_every 2000 --log_every 100 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --auto_stop --auto_stop_threshold 0.01 --auto_stop_k 2 \
    --save_ckpt checkpoints/pretrain_mix_v2.pt \
    --tb_dir runs/tb/pretrain_mix_v2 \
    > runs/pretrain_mix_v2.log 2>&1 &
echo "Launched pretrain_mix_v2 on GPU ${GPU:-0} (PID $!)"
echo "Watch:"
echo "  tail -f runs/pretrain_mix_v2.log"
echo "  tensorboard --logdir runs/tb"
