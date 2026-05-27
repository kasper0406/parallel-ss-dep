#!/bin/bash
# SMOKE pretrain — validate thinking-aux losses (2026-05-27).
#
# Pivot per user: "make the pre-train actually work. I think we have
# enough evidence by now." DPO / SFT continuations all regressed. The
# common thread: features were never co-trained with the trunk.
#
# Goal of this smoke run:
#   - Confirm the just-wired `compute_process_reward_loss` and
#     `compute_gate_calibration_loss` work in pretrain (no crashes).
#   - See if gate_calibration target_frac_one rises during +1000 steps
#     (it should; the gate is being supervised by an outcome signal it
#     has never seen before).
#   - See if process_reward mean Δlogp becomes positive (thinking
#     becomes productive — currently negative per the v8 probe).
#
# Continuation from `checkpoints/pretrain_phase_c.pt` (the trained
# 287M base at step 23000 / 5.3B tokens). 1000 new steps = ~5 min on a
# 5090.
#
# What's NEW vs Phase C:
#   --process_reward_weight 0.05    : trunk-side, makes thinks productive
#   --gate_calibration_weight 0.05  : gate-side, BCE on whether think-helps
#   --state_readonly_at_think       : prevents thinks from corrupting
#                                      the DeltaNet recurrent state (the
#                                      "thinks corrupt recall" finding,
#                                      validated on the 1-layer probe)
#
# What's NOT here (deferred to a second smoke / full run):
#   - --use_think_adapter, --use_refinement_head, --think_index_emb_size
#     These add new param tensors; 1000 steps isn't enough for them to
#     train from init. First validate aux losses on the EXISTING model.
#
# Known issue (FIXED 2026-05-27): torch.compile used to crash on the
# extra forward inside process_reward / gate_calibration (Inductor
# symbolic-shape assertion in tiling_utils because the aux forward has
# shape (N, L_after) ≠ the compiled main (B, T) graph). Fix:
# `apply_speed_knobs(..., compile_model=True)` now stashes the
# pre-compile forward as `model._eager_forward`, and
# `process_reward._call_model_eager` uses it so the aux forwards run
# eager while the hot main forward stays compiled. The next long run
# can drop `--no-compile`. See experiments/test_process_reward.py
# `test_eager_forward_used_when_present` for the regression test.
#
# Decision-gate (after this smoke):
#   (a) target_frac_one trends up + mean_log_ratio positive → aux losses
#       work; move to Smoke B with new modules + longer (5000 steps).
#   (b) metrics flat or worse → debug the aux loss recipe (weight, K,
#       sample_frac) before scaling.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: do NOT set CUDA_LAUNCH_BLOCKING=1 here. It forces synchronous
# kernel launches and tanks throughput ~2x (was the dominant cause of
# this launcher's 17k tok/s vs Phase C's 45k tok/s — see the
# "Throughput regression analysis" report 2026-05-27). Only enable when
# actively debugging a CUDA-side assert.
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_lm.py \
    --arch deltanet \
    --d_model 896 --n_layers 10 --d_head 64 --n_heads 14 \
    --feedback film \
    --feedback_pairs "0,5;1,6;2,7;3,8;4,9" \
    --feedback_self_k 3 \
    --feedback_self_k_warmup_steps 0 \
    --output_gate \
    --gate_entropy_aux_weight 0.1 \
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
    --lr_schedule wsd --warmup_steps 0 --lr_decay_frac 0.0 \
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
    --load_ckpt checkpoints/pretrain_phase_c.pt \
    --start_step 23000 \
    --steps 24000 \
    --val_every 100 --log_every 5 \
    --save_ckpt checkpoints/pretrain_smoke_thinking.pt \
    --tb_dir runs/tb/pretrain_smoke_thinking \
    > runs/pretrain_smoke_thinking.log 2>&1 &
echo "Launched SMOKE pretrain (thinking-aux losses) on GPU ${GPU:-0} (PID $!)"
echo "Watch:  tail -f runs/pretrain_smoke_thinking.log"
echo
echo "KEY METRICS TO WATCH (every --log_every step):"
echo "  pr(n=., Δlogp=.)  — process-reward. Want Δlogp > 0 by end."
echo "  gc(n=., tgt1=., σ=., Δlogp=.)  — gate-cal. Want tgt1 > 0.3 by end."
echo "  Existing: gate-fire, ppl, gnorm, layer_grad_norm/L*."
