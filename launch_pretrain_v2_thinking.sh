#!/bin/bash
# v2 pretrain — full thinking recipe (2026-05-27).
#
# Builds on smoke (D8) validation that aux losses work. v2 adds:
#   --use_think_adapter --think_adapter_hidden_mult 2
#   --use_refinement_head --refinement_head_window 128 --refinement_head_alpha_init 0.3
#   --think_index_emb_size 16
#
# These are the new param tensors that have been inert when attached
# post-pretrain. Co-training them with the trunk + gate-calibration +
# process-reward + state_readonly_at_think gives the whole thinking
# stack a chance to actually load-bear.
#
# Continuation from `checkpoints/pretrain_phase_c.pt` step 23000.
# Run for +5000 steps = ~1.15B new tokens. At ~17k tok/s with the
# aux losses on, that's ~19 hours wall time. Save every 500 steps.
#
# Decision-gate at step 24000 (+1000 = matches smoke):
#   - gc tgt1 trending down (gate becoming WELL-CALIBRATED)
#   - σ at uncertain positions converged
#   - tloss stable or improving
#   - per-block grads healthy
# At step 25000 (+2000):
#   - HumanEval pass@1 ≥ 16/164 (matches base, validates no regression)
# At step 28000 (final):
#   - HumanEval pass@1 > 16/164 — the load-bearing answer
#
# Compute: ~19h. If running fails or shows clear degradation, ABORT
# by step 25000.

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
    --feedback_self_k_warmup_steps 0 \
    --output_gate \
    --gate_entropy_aux_weight 0.05 \
    --gate_entropy_aux_temperature 2.0 \
    --use_memory --mem_size 1024 \
    --use_pkm --pkm_after_layer 5 \
    --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32 \
    --pkm_use_output_gate \
    --pkm_value_init_std 1.0 \
    --pkm_score_norm layer \
    --pkm_diversity_weight 0.01 \
    --pkm_value_lr_mult 100.0 \
    --gist_loss_weight 0.1 --gist_horizons "16,64,256" \
    --data_mix configs/pretrain_mix_v4.yaml \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6 \
    --T 2048 --batch 8 --grad_accum 14 \
    --activation_checkpointing \
    --bf16 --tf32 --no-compile \
    --alpha_wd 0.0 --wd 0.01 \
    --grad_clip 1.0 --z_loss 1e-4 \
    --mask_eos_in_targets \
    --gate_floor_min 0.5 --gate_warmup_steps 0 \
    --optimizer muon --lr 1.4e-3 --lr_muon 5e-3 \
    --lr_schedule wsd --warmup_steps 0 --lr_decay_frac 0.15 \
    --process_reward_weight 0.05 \
    --process_reward_K 4 \
    --process_reward_apply_min_sigma 0.3 \
    --process_reward_sample_frac 0.05 \
    --process_reward_max_positions 8 \
    --gate_calibration_weight 0.05 \
    --gate_calibration_K 4 \
    --gate_calibration_apply_min_sigma 0.1 \
    --gate_calibration_apply_max_sigma 0.9 \
    --gate_calibration_sample_frac 0.05 \
    --gate_calibration_max_positions 8 \
    --state_readonly_at_think \
    --use_think_adapter \
    --think_adapter_hidden_mult 2 \
    --use_refinement_head \
    --refinement_head_window 128 \
    --refinement_head_n_heads 8 \
    --refinement_head_mlp_mult 2 \
    --refinement_head_alpha_init 0.3 \
    --think_index_emb_size 16 \
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --start_step 23000 \
    --steps 28000 \
    --val_every 200 --log_every 25 \
    --mid_eval_every_tokens 500000000 \
    --mid_eval_save_only \
    --save_ckpt checkpoints/pretrain_v2_thinking.pt \
    --tb_dir runs/tb/pretrain_v2_thinking \
    > runs/pretrain_v2_thinking.log 2>&1 &
echo "Launched v2 pretrain (full thinking stack) on GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/pretrain_v2_thinking.log"
