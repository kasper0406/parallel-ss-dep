#!/bin/bash
# v11 — IDENTICAL to launch_pretrain_v10_FRESH.sh (from-scratch, full co-trained
# stack) EXCEPT the WM recall gradient is CAPACITY-EXCEEDING multi-binding recall
# instead of single-binding (the 2026-06-13 root-cause fix).
#
# WHY (see WHY_THINKING_MARGINAL_ON_CODE.md + configs/pretrain_mix_v11.yaml):
#   v10 mixed a recall stream so WM would have a "saturating-recall gradient",
#   but it was data/longctx_recall_train.jsonl = SINGLE-binding recall, which a
#   DeltaNet solves 100% no-think (one binding fits the recurrent state). So the
#   trunk routed AROUND WM during pretrain and WM's read learned RECENCY (probe:
#   0% mass on the queried binding). The capstone principle: an auxiliary
#   mechanism stays idle whenever the recurrence can do the task; WM is forced to
#   engage ONLY when the task is capacity-exceeding AND non-memorizable. v11
#   swaps in data/multibind_recall_pretrain.jsonl (assign N in {8..32} vars,
#   print ONE queried var; 50k fresh instances) so at N=24/32 the state
#   saturates and predicting the queried value FORCES WM-addressable hiddens.
#
# WHAT TO WATCH: the live feature-probe runs WM ablation on the capacity-
# exceeding multibind heldout (--feature_probe_wm_recall_path). The win
# condition is `[feature-probe] wm(Δce=...)` going clearly POSITIVE (WM
# load-bearing) and the wm-recall addressing concentrating — UNLIKE v10 where it
# stayed ~0. If wm(Δce) stays ~0 here too, even the source fix fails and WM is
# the wrong lever for this scale (pivot to PKM/knowledge).
#
# CAVEAT: WM going load-bearing here proves the MECHANISM is fixable; it does NOT
# guarantee a HumanEval lift (real code is knowledge-bound — PKM's job). v11
# tests the mechanism at the source; transfer to code is a separate question.
#
# Cost: ~5B tokens / 19000 steps / ~20h on one 5090 (same as v10_FRESH).
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film \
    --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 1300 \
    --output_gate \
    --gate_entropy_aux_weight 0.1 \
    --gate_entropy_aux_temperature 2.0 \
    --gate_floor_min 0.5 --gate_warmup_steps 20000 \
    --state_readonly_at_think \
    --use_memory --mem_size 1024 \
    --mem_decoupled_kv \
    --mem_read_alpha_init 1.0 \
    --mem_read_alpha_floor_start 0.5 --mem_read_alpha_floor_warmup_steps 3000 \
    --use_pkm --pkm_after_layer 5 \
    --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
    --pkm_use_output_gate \
    --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 \
    --pkm_value_init_std 1.0 \
    --pkm_score_norm layer \
    --pkm_diversity_weight 0.01 \
    --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 \
    --pkm_value_lr_mult 100.0 \
    --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
    --latent_cotrain_weight 0.025 \
    --latent_cotrain_R 4 \
    --latent_cotrain_sample_frac 0.05 \
    --latent_cotrain_max_positions 4 \
    --gate_calibration_weight 0.05 \
    --gate_calibration_R 4 \
    --gate_calibration_sample_frac 0.05 \
    --gate_calibration_max_positions 4 \
    --gate_calibration_sigma_low 0.1 --gate_calibration_sigma_high 0.9 \
    --feature_probe_every_tokens 3000000 \
    --feature_probe_wm_recall_path data/multibind_bigN_heldout.jsonl \
    --feature_probe_wm_recall_n 64 \
    --data_mix configs/pretrain_mix_v11.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 4 --grad_accum 32 \
    --activation_checkpointing \
    --bf16 --tf32 --no-compile \
    --alpha_wd 0.0 --wd 0.01 \
    --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15 \
    --start_step 0 \
    --steps 30 \
    --val_every 999999 --log_every 5 \
    --mid_eval_every_tokens 999999999999 \
    --save_ckpt checkpoints/SMOKE_v11_DELETEME.pt \
    --tb_dir runs/tb/SMOKE_v11 \
    > runs/SMOKE_v11.log 2>&1 &
echo "Launched v11 SMOKE on GPU ${GPU:-0} (PID $!)"
echo "Watch:  tail -f runs/SMOKE_v11.log"
echo "WM load-bearing signal: grep -E '\\[feature-probe\\]|\\[wm-recall\\]' runs/SMOKE_v11.log"
