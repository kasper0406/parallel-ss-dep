#!/bin/bash
# 10L shallow-wide feature A/B on the linearized_10L base (CE 1.193).
# STRICTLY SEQUENTIAL on GPU1 (no concurrency): Arm A (bare) -> Arm B (features w/
# validated dense shallow-wide FiLM 0,5;1,6;2,7;3,8;4,9 K=3) -> CE evals -> pass@1.
# Tests the bet: does FiLM-unroll recover the depth dropped by 32L->10L?
exec >/tmp/phase1_10L.log 2>&1
set -x
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
GPU=1
BASE=checkpoints/linearize/linearized_10L_stage3.pt

COMMON=(
  --arch deltanet --d_model 960 --n_layers 10 --d_head 64 --n_heads 15 --tie_embeddings --d_ff 2560
  --load_ckpt "$BASE" --keep_base_vocab 49152
  --data_mix configs/pretrain_mix_v18_arxiv.yaml --tokenizer HuggingFaceTB/SmolLM2-135M
  --think_burst_prob 0 --num_workers 0 --mask_eos_in_targets
  --T 2048 --batch 4 --grad_accum 16 --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state
  --wd 0.01 --alpha_wd 0.0 --grad_clip 1.0 --z_loss 1e-4
  --optimizer muon --lr 3e-4 --lr_muon 1e-3 --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.15
  --steps 1900 --val_every 200 --log_every 20 --seed 1234
  --mid_eval_every_tokens 40000000 --mid_eval_save_only
)
FEATURES=(
  --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" --feedback_self_k 3 --feedback_self_k_warmup_steps 300
  --output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0 --gate_floor_min 0.5 --gate_warmup_steps 20000
  --use_memory --mem_size 1024 --mem_decoupled_kv --mem_ctx_namekey --ctx_namekey_dim 192
  --ctx_namekey_match_threshold 0.5 --mem_always_read --use_copy_head --mem_copy_require_match
  --mem_discrete_key_match_window 32 --copy_gate_bias_init -6.0 --mem_read_alpha_init 0.0 --mem_freeze_read_alpha --emit_read_mask
  --ctx_addr_aux_weight 0.2 --ctx_addr_aux_warmup_steps 3000 --ctx_addr_aux_start_step 0
  --use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32
  --pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 --pkm_value_init_std 1.0
  --pkm_score_norm layer --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000 --pkm_value_lr_mult 100.0
  --gist_loss_weight 0.1 --gist_horizons 16,64,256
  --use_latent_feedback_adapter
)

echo "[10L] Arm A (bare, features OFF) start $(date)"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py "${COMMON[@]}" \
  --feedback none --save_ckpt checkpoints/phase1_10L_A.pt --tb_dir runs/tb/phase1_10L_A
echo "[10L] Arm A done $(date)"

echo "[10L] Arm B (features ON, shallow-wide FiLM-unroll) start $(date)"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py "${COMMON[@]}" \
  "${FEATURES[@]}" --save_ckpt checkpoints/phase1_10L_B.pt --tb_dir runs/tb/phase1_10L_B
echo "[10L] Arm B done $(date)"

echo "[10L] CE evals $(date)"
: > /tmp/phase1_10L_ce.txt
echo "ANCHORS: 10Lbase=1.1928  32Lbase=0.7585  32L-ArmA=0.7523  32L-ArmB=0.7573  donor=0.6142" | tee -a /tmp/phase1_10L_ce.txt
for CK in checkpoints/linearize/linearized_10L_stage3.pt checkpoints/phase1_10L_A.pt checkpoints/phase1_10L_B.pt; do
  [ -f "$CK" ] && CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python /tmp/probe_he_ce.py "$CK" 2>>/tmp/phase1_10L_ce.err | tee -a /tmp/phase1_10L_ce.txt
done

echo "[10L] pass@1 $(date)"
: > /tmp/phase1_10L_pass.txt
for CK in checkpoints/phase1_10L_A.pt checkpoints/phase1_10L_B.pt; do
  echo "=== $CK ===" >> /tmp/phase1_10L_pass.txt
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python experiments/eval_humaneval.py \
    --ckpt "$CK" --n_samples 1 --temperature 0.0 --min_emit_before_eos 30 \
    >> /tmp/phase1_10L_pass.txt 2>>/tmp/phase1_10L_pass.err || echo "pass@1 FAILED $CK" >> /tmp/phase1_10L_pass.txt
done
echo "[10L] ALL DONE $(date)"; touch /tmp/phase1_10L_DONE
