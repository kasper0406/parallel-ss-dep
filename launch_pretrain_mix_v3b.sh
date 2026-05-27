#!/bin/bash
# v3b: WD intervention + Stochastic Depth.
#   --wd 0.01            (same as v3a)
#   --layer_drop_max 0.2 (linear 0 → 0.2 across L0..L29; LayerDrop convention)
#
# Stacked test against v3a: if v3a fixes ||h|| collapse on its own,
# v3b should match v3a or slightly underperform it (LayerDrop adds a
# regularizer we don't need). If v3a doesn't fix it, v3b's stochastic-
# depth signal might. The point is the comparison.
#
# Same 1.0 B token budget. Run on GPU 1 — wait for v2 to finish first.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 576 --n_layers 30 --d_head 64 --n_heads 9 \
    --feedback film --feedback_pairs "2,28" --feedback_self_k 3 \
    --output_gate \
    --use_memory --mem_size 1024 \
    --data_mix configs/pretrain_mix_v2_with_cve.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 7 \
    --bf16 --tf32 \
    --alpha_wd 0.0 --wd 0.01 \
    --layer_drop_max 0.2 \
    --mask_eos_in_targets \
    --gate_floor_min 0.5 --gate_warmup_steps 20000 \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --steps 70000 \
    --val_every 2000 --log_every 100 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_mix_v3b.pt \
    --tb_dir runs/tb/pretrain_mix_v3b \
    > runs/pretrain_mix_v3b.log 2>&1 &
echo "Launched pretrain_mix_v3b on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v3b.log"
