#!/bin/bash
# v3a: same arch + data + trainer knobs as v2, ONE change:
#   --wd 0.01  (was hard-coded 0.1 inside optim_utils.py)
#
# Hypothesis: the diag_ckpt run on v2 mid-eval ckpts (500M and 1B) showed:
#   - ||h||@L0 dropped 8.1 → 3.5 between 500M and 1B (residual collapse)
#   - SmolLM2-135M reference has ||h||@L0 ≈ 44 on the same data
#   - Per-source CE divergence growing (jinaai memorised, bigvul regressing)
# All consistent with weights being held too small by WD. v3a tests
# whether dropping main WD to 0.01 lets weights grow into a regime where
# the residual stream behaves like SmolLM2's.
#
# Budget: 70k steps × 14336 tok/step ≈ 1.0 B tokens (half v2's, enough
# to see the diagnostic outcome via mid-evals at 500M and 1B).
#
# Auto-stop OFF — we want full 1B regardless of HumanEval (which will
# almost certainly remain 0/50 at this budget; the metric we care about
# is the per-source CE shape via diag_ckpt).
#
# Pinned to GPU 0 by default (v2 is on GPU 1).

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
    --steps 70000 \
    --val_every 2000 --log_every 100 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/pretrain_mix_v3a.pt \
    --tb_dir runs/tb/pretrain_mix_v3a \
    > runs/pretrain_mix_v3a.log 2>&1 &
echo "Launched pretrain_mix_v3a on GPU ${GPU:-0} (PID $!)"
echo "Watch:  tail -f runs/pretrain_mix_v3a.log"
