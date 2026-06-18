#!/bin/bash
# Latent-reasoning CODE bake (2026-06-06) — user redirect: "if you trained only on
# Arithmetic, HumanEval won't be good; train thinking on our ACTUAL (code) data."
#
# Same as launch_latent_bake_probe.sh BUT the latent-reasoning co-train stream is
# swapped from data/ptr10dict_train (synthetic pointer-chase, OOD for code) to
# data/trace_train (real Python execution traces — code-distribution, depth-
# requiring, and where latent thinking is VALIDATED +0.47-0.78). The LM loss still
# trains on the actual pretrain mix (pretrain_mix_v4), so the model stays fluent on
# real code while the latent op becomes IN-DISTRIBUTION for code reasoning.
# GWEIGHT=0.5 also trains the gate to invoke+halt thinking on code-shaped problems.
#
# Go/no-go: (a) VAL tracks the phase_c base (no general regression), (b) reason(loss)
# drops + a code-trace none-vs-R lift emerges, (c) the latent op stops being
# destructive on code (HumanEval thinking-ON >= no-think after a quick SFT).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

RWEIGHT=${RWEIGHT:-0.05}
GWEIGHT=${GWEIGHT:-0.5}
TAG=${TAG:-latent_bake_code}
STEPS=${STEPS:-29000}    # probe: 23000 -> 29000 (~6000 steps)

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 0 \
    --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 \
    --use_memory --mem_size 1024 \
    --use_pkm --pkm_after_layer 5 \
    --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
    --pkm_use_output_gate \
    --pkm_epsilon_start 0.0 --pkm_value_init_std 1.0 --pkm_score_norm layer \
    --pkm_alpha_floor_start 0.0 \
    --pkm_value_lr_mult 100.0 \
    --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
    --state_readonly_at_think \
    --use_latent_feedback_adapter \
    --latent_reasoning_weight ${RWEIGHT} \
    --latent_reasoning_train_prefix data/trace_train \
    --latent_reasoning_rungs 2,3,4,5,6,7,8 \
    --latent_reasoning_n 4 \
    --latent_reasoning_gate_weight ${GWEIGHT} \
    --data_mix configs/pretrain_mix_v4.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 8 --grad_accum 8 \
    --activation_checkpointing --bf16 --tf32 --no-compile \
    --alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets \
    --gate_floor_min 0.5 --gate_warmup_steps 1500 \
    --optimizer muon --lr 3e-4 --lr_muon 1e-3 \
    --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.15 \
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --start_step 23000 \
    --steps ${STEPS} \
    --val_every 100 --log_every 25 \
    --mid_eval_every_tokens 200000000 --mid_eval_save_only \
    --mid_eval_n_problems 50 --mid_eval_max_gen 192 \
    --save_ckpt checkpoints/${TAG}.pt \
    --tb_dir runs/tb/${TAG} \
    > runs/${TAG}.log 2>&1 &
echo "Launched ${TAG} (RWEIGHT=${RWEIGHT} GWEIGHT=${GWEIGHT} trace-data) on GPU ${GPU:-0} (PID $!)"
