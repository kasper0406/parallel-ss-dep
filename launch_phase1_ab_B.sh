#!/bin/bash
# PHASE-1 A/B  —  ARM B (TREATMENT): SAME base / data / tokens / schedule /
# optimizer as ARM A, but with OUR FEATURES ON (adapted to the 960-wide / 32L base):
#   * FiLM feedback (reverse fan-in, 4 pairs spanning 32 layers) + K=3 self-feed
#   * WorkingMemory: learned ctx_namekey addresser (no hash) + mem_always_read
#     + copy head + ctx-addr attention-supervision aux  (mem_size 1024, dim 192)
#   * PKM side-table after layer 16  (v7.1 bootstrap-fix package)
#   * latent thinking: LatentFeedbackAdapter + latent_reasoning depth aux (ptr10dict)
#   * output_gate + gate_entropy_aux  + trunk multi-horizon gist loss
# Every feature module inits near-no-op (FiLM alpha=0, latent proj zero-init,
# PKM alpha-floor curriculum, WM read_alpha frozen 0 + copy-gate bias -6) so the
# inherited base is PRESERVED at step 0.
#
# vocab: --keep_base_vocab 49152 -> NO embedding resize; discrete think token
#        aliased to EOS (in-range), never emitted (--think_burst_prob 0 +
#        --mem_always_read). A load-manifest PRE-FLIGHT asserts 0 shape-mismatch.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-1}

# Idempotent ATOMIC guard: two launchers (my orchestrator + the agent's watcher)
# may both fire after Arm A finishes — even concurrently during the manifest
# pre-flight. mkdir is atomic, so exactly one wins the lock; the rest no-op.
# (Two Arm-B trainers on one GPU would OOM + corrupt the shared log/ckpt.)
if ! mkdir runs/.phase1_ab_B.launch.lock 2>/dev/null; then
  echo "Another Arm-B launcher holds the lock — skipping duplicate launch."; exit 0; fi
if grep -q "Done in" runs/phase1_ab_B.log 2>/dev/null; then
  echo "Arm B already COMPLETED — skipping."; exit 0; fi

# Full arg list shared by the manifest pre-flight and the trainer (no divergence).
ARGS=(
  --arch deltanet --d_model 960 --n_layers 32 --d_head 64 --n_heads 15 --tie_embeddings --d_ff 2560
  --load_ckpt checkpoints/linearize/linearized_stage3.pt --keep_base_vocab 49152
  --data_mix configs/pretrain_mix_v18_arxiv.yaml --tokenizer HuggingFaceTB/SmolLM2-135M
  --think_burst_prob 0 --num_workers 0 --mask_eos_in_targets
  --T 2048 --batch 2 --grad_accum 32 --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state
  --wd 0.01 --alpha_wd 0.0 --grad_clip 1.0 --z_loss 1e-4
  --optimizer muon --lr 3e-4 --lr_muon 1e-3 --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.15
  --steps 1900 --val_every 200 --log_every 20 --seed 1234
  --mid_eval_every_tokens 100000000 --mid_eval_save_only
  # ---- FEATURES ON ----
  --feedback film --feedback_pairs "0,16;4,20;8,24;12,28" --feedback_self_k 3 --feedback_self_k_warmup_steps 300
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 --gate_floor_min 0.5 --gate_warmup_steps 20000
  --use_memory --mem_size 1024 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192
  --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -6.0 --mem_read_alpha_init 0.0 --mem_freeze_read_alpha --emit_read_mask
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 3000 --ctx_addr_aux_start_step 0
  --use_pkm --pkm_after_layer 16 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32
  --pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 --pkm_value_init_std 1.0
  --pkm_score_norm layer --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 --pkm_value_lr_mult 100.0
  --gist_loss_weight 0.1 --gist_horizons 16,64,256
  --use_latent_feedback_adapter --latent_reasoning_weight 0.05 --latent_reasoning_train_prefix data/ptr10dict_train
  --latent_reasoning_rungs 2,3,4,5,6,7,8 --latent_reasoning_n 4 --latent_reasoning_gate_weight 0.05
  --latent_reasoning_start_step 200 --latent_reasoning_weight_warmup_steps 600
  --save_ckpt checkpoints/phase1_ab_B.pt --tb_dir runs/tb/phase1_ab_B
)

if [ "${SKIP_MANIFEST:-0}" != "1" ]; then
  echo "=== ARM B load-manifest pre-flight (asserts 0 shape-mismatch) ==="
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python /tmp/phase1_manifest.py "${ARGS[@]}"
fi

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py "${ARGS[@]}" \
  > runs/phase1_ab_B.log 2>&1 &
echo "Launched PHASE1 ARM B (treatment), GPU $GPU (PID $!).  Watch: tail -f runs/phase1_ab_B.log"
