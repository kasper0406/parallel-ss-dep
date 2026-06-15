#!/bin/bash
# v13 — latent thinking from DAY 1, done RIGHT (2026-06-14).
#
# WHY v13 (the day-1-latent decision):
#   v12 destabilized when the BROKEN general-text `latent_cotrain` engaged cold at
#   step 3500 (VAL +10%, gnorm 20×, PKM top-slot 0.008→0.17). Root cause: that loss
#   is a full-trunk R=4 unroll with a RANDOM natural next-token target → not a depth
#   task → Δlogp stuck negative, huge cold gradient. We disabled it.
#
#   THE KEY FINDING (design agent, 2026-06-14): task #62 already BUILT the correct
#   replacement — the DEPTH-MATCHED `LatentReasoningCotrain` (answer-span CE on the
#   pointer-chase corpus at R=depth, easy-first depth curriculum, clean latent thread
#   = WM off + FiLM bypass) wired behind `--latent_reasoning_weight`. It was never
#   put in any pretrain launcher; v10/v11/v12 inherited the OLD `--latent_cotrain_*`.
#   This loss is SOLVABLE (reason-loss falls as the trunk learns) and co-evolves from
#   step 0, so it does NOT shock the trunk the way the general-text loss did. A 12-step
#   from-scratch smoke confirmed it runs clean alongside the PKM bootstrap.
#
#   v13 = v12 stack, BUT swap the broken loss for the depth-matched bake at weight
#   0.05, full-trunk-plastic (from scratch the trunk MUST learn to reason; adapter-only
#   freeze is only the post-hoc HARMLESSNESS fix on an already-reasoning trunk), with
#   the autonomous-gate BCE (--latent_reasoning_gate_weight), PLUS the new 0→target
#   weight RAMP over the PKM α-floor window (--latent_reasoning_weight_warmup_steps
#   3000) as belt-and-suspenders so the aux gradient stays negligible while PKM commits.
#   general-text latent_cotrain AND gate_calibration stay at 0 (both v12 destabilizers).
#
# DATA: NO data-mix change. The depth benefit comes from the SEPARATE ptr10dict corpus
#   loaded directly by LatentReasoningCotrain (latent loss applies ONLY to pointer-chase,
#   never to recall-bound code). Mix stays configs/pretrain_mix_v11.yaml.
#
# SUCCESS = net-positive on ptr10dict heldout (latent_arith_real.py, R=n vs a FAIR
#   100%-no-think control, lift ≥ +0.4 at n≥3) at ZERO VAL/PKM cost. NOT a HumanEval
#   lift (code per-token is recall, not iteration — wrong probe at 287M).
#
# Runs on GPU 0 (default) so the v12 WM run on GPU 1 is untouched.
# Same auto-resume + 250M-ckpt + HF-timeout robustness as run_v12_autoresume.sh.
set -u
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH="${PYTHONPATH:-}:."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60
GPU=${GPU:-0}
LOG=runs/pretrain_v13.log
mkdir -p runs checkpoints

common_args() {
  cat <<'ARGS'
--arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14
--feedback film --feedback_pairs 0,5;1,6;2,7;3,8;4,9 --feedback_self_k 3 --feedback_self_k_warmup_steps 1300
--output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0
--gate_floor_min 0.5 --gate_warmup_steps 20000 --state_readonly_at_think
--use_memory --mem_size 1024 --mem_decoupled_kv --mem_read_alpha_init 1.0
--mem_read_alpha_floor_start 0.5 --mem_read_alpha_floor_warmup_steps 3000
--use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32
--pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 --pkm_value_init_std 1.0
--pkm_score_norm layer --pkm_diversity_weight 0.01 --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000
--pkm_value_lr_mult 100.0 --gist_loss_weight 0.1 --gist_horizons 16,64,256
--latent_cotrain_weight 0.0
--gate_calibration_weight 0.0
--use_latent_feedback_adapter
--latent_reasoning_weight 0.05 --latent_reasoning_train_prefix data/ptr10dict_train
--latent_reasoning_rungs 2,3,4,5,6,7,8 --latent_reasoning_n 4
--latent_reasoning_gate_weight 0.05
--latent_reasoning_start_step 0 --latent_reasoning_weight_warmup_steps 3000
--feature_probe_every_tokens 500000000 --feature_probe_wm_recall_path data/multibind_recall_heldout.jsonl --feature_probe_wm_recall_n 64
--data_mix configs/pretrain_mix_v11.yaml --tokenizer HuggingFaceTB/SmolLM2-135M
--think_burst_prob 0.5 --think_max_bursts 2 --think_max_burst_depth 6
--T 2048 --batch 4 --grad_accum 32 --activation_checkpointing --bf16 --tf32 --no-compile
--alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets
--optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15
--steps 19000 --val_every 200 --log_every 20
--mid_eval_every_tokens 250000000 --mid_eval_save_only --mid_eval_n_problems 50 --mid_eval_max_gen 192
--save_ckpt checkpoints/pretrain_v13.pt --tb_dir runs/tb/pretrain_v13
ARGS
}

for attempt in $(seq 1 40); do
  latest=$(ls -t checkpoints/pretrain_v13_step*.pt 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    rstep=$(echo "$latest" | sed -E 's/.*_step([0-9]+)_.*/\1/')
    resume="--load_ckpt $latest --start_step $rstep"
    echo "=== [autoresume $attempt] resuming from $latest (step $rstep) ===" >> "$LOG"
  else
    resume="--start_step 0"
    echo "=== [autoresume $attempt] fresh start ===" >> "$LOG"
  fi
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py $(common_args) $resume >> "$LOG" 2>&1
  rc=$?
  # Completion gate: the FINAL-save line is printed only at --steps; the last
  # mid-eval ckpt (250M cadence ≈ 953.7 steps) lands ~step 18120 in a clean
  # 19000-step run, so 18000 is reachable yet never satisfied by the 2nd-to-last.
  if grep -q "saved.*checkpoints/pretrain_v13.pt" "$LOG" && [ "$(ls -t checkpoints/pretrain_v13_step*.pt 2>/dev/null | head -1 | sed -E 's/.*_step([0-9]+)_.*/\1/')" -ge 18000 ] 2>/dev/null; then
    echo "=== [autoresume] COMPLETE (rc=$rc) ===" >> "$LOG"; exit 0
  fi
  echo "=== [autoresume $attempt] exited rc=$rc — will resume from latest ckpt ===" >> "$LOG"
  sleep 20
done
echo "=== [autoresume] GAVE UP after 40 attempts ===" >> "$LOG"; exit 1
