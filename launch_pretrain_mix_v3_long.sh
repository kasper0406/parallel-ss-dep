#!/bin/bash
# v3-long: WD-fixed setup at a proper schedule.
#
# Single-variable change from v3a: --steps 70000 → 149000. v3a's cosine
# schedule (T_max = --steps) front-loaded all the learning — by the 1B
# mark the LR was pinned at the cosine floor, so the 500M→1B VAL-ppl
# gain looked modest. v3-long stretches cosine over the full 2.13B
# token budget so there's live LR at the 1B mark and beyond.
#
#   --wd 0.01            (confirmed by v3a: residual stream un-collapsed,
#                         per-source CE beats v2 on every stream)
#   --steps 149000       batch*T = 7*2048 = 14336 tok/step → ~2.13B tokens
#   auto_stop OFF        we want the full curve regardless of HumanEval
#
# Pinned to GPU 0 by default. v3b runs concurrently on GPU 1.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
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
    --mask_eos_in_targets \
    --gate_floor_min 0.5 --gate_warmup_steps 20000 \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --steps 149000 \
    --val_every 2000 --log_every 100 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_mix_v3_long.pt \
    --tb_dir runs/tb/pretrain_mix_v3_long \
    > runs/pretrain_mix_v3_long.log 2>&1 &
echo "Launched pretrain_mix_v3_long on GPU ${GPU:-0} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v3_long.log"
