#!/bin/bash
# v14 — make the VALIDATED WM recall mechanism LOAD-BEARING.
#
# WHY v14 (see memory project_recall_discrete_key_direction +
#   project_recall_stream_no_answer_supervision):
#   The WM recall mechanism is validated end-to-end on a SATURATING probe:
#     - ADDRESSING: key the read on the identifier's INPUT-EMBEDDING window
#       (wm_namekey_probe.py) → top1=1.00 vs chance for cosine-on-hidden.
#     - READOUT: copy/pointer over the addressed source span
#       (wm_multitok_readout.py) → 100% exact multi-token recall vs ~0 base.
#   v10–v13 never made WM load-bearing because the pretrain recall stream gave
#   the WM read NO answer-token gradient (root cause: no mem_read_mask over the
#   answer span). v14 fixes all three legs:
#     (1) --mem_key_from_embedding : embedding-key cosine addressing (adds NO
#         new params → a v12 ckpt loads byte-identically; effective immediately).
#     (2) --use_copy_head          : pointer/copy readout at the answer span.
#     (3) --emit_read_mask         : data_mix emits a per-position mem_read_mask
#         = 1 over recall-source answer spans (configs/pretrain_mix_v14_wmrecall
#         flags multibind/code_recall/agentic_recall) → answer CE flows into the
#         WM read W_proj/W_v + the cosine temperature + the embeddings.
#   --mem_read_alpha_floor (0.5 over 3000 fwds) holds the read contribution up
#   while the addressing locks in (anti-starvation; the floor counter is per-
#   module and resets each process start, so it re-bootstraps on continuation).
#
# CONTINUATION, NOT SCRATCH (recommended): seed from the latest v12 ckpt.
#   - inherits general code competence + the trained DKV WM (W_v/W_proj/temp);
#   - embedding-key addressing adds NO params → byte-identical load, and the
#     cosine address on raw embeddings is effective with no warmup;
#   - copy_head is the only new params (a d_model→1 gate, cold-start g≈0 → base
#     preserved exactly at step 0);
#   - start_step = the v12 step so the STEP-based PKM bootstrap curricula
#     (epsilon/alpha-floor, warmup 3000) are already PAST → PKM stays in its
#     clean trained state (no re-randomization). The read_alpha floor is fwd-
#     count based and resets, so the WM read re-bootstraps on the NEW answer
#     gradient — exactly what we want.
#   - anti-forgetting: the v14 mix keeps recall a minority (~16%) + the full
#     code/instruction/wiki bulk.
#   To cold-start instead, drop the v12 seed and pass --start_step 0 (slower,
#   loses code competence, NOT recommended).
#
# THINK BURSTS OFF (--think_burst_prob 0.0): the WM read is now driven by the
#   answer-span mem_read_mask, not think tokens, so random think-burst injection
#   is pure noise here AND could land between a predicting position and its
#   answer token (mask misalignment). The gate was already trained in v12. To
#   re-enable thinking later, raise --think_burst_prob.
#
# LATENT OFF for the first v14 (isolate WM): --latent_cotrain_weight 0.0 and
#   --gate_calibration_weight 0.0 (as v12). To add latent later, raise
#   --latent_cotrain_weight with --latent_cotrain_start_step well past the WM
#   bootstrap, per the v12 notes on cold-latent instability.
#
# MEASURE WM usefulness (live + post-run):
#   - live: --feature_probe_code_recall_path data/code_recall_heldout.jsonl runs
#     eval_code_recall teacher-forced ON vs full_off (Δ>0 ⇒ WM load-bearing) and
#     logs read_alpha (+ copy gate) every --feature_probe_every_tokens.
#   - post-run: experiments/eval_code_recall.py on code+agentic heldout, arms
#     {on, full_off, no_think}, modes {teacher_forced, generate} (see tail).
#
# SAFETY: do NOT co-locate with v12 (GPU1). Pick a FREE GPU via GPU=<idx>.
set -u
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH="${PYTHONPATH:-}:."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=60
GPU=${GPU:-0}                 # MUST be a free GPU (v12 is on GPU1).
LOG=runs/pretrain_v14.log
mkdir -p runs checkpoints

common_args() {
  cat <<'ARGS'
--arch deltanet --d_model 896 --n_layers 10 --d_head 64 --n_heads 14
--feedback film --feedback_pairs 0,5;1,6;2,7;3,8;4,9 --feedback_self_k 3 --feedback_self_k_warmup_steps 1300
--output_gate --gate_entropy_aux_weight 0.1 --gate_entropy_aux_temperature 2.0
--gate_floor_min 0.5 --gate_warmup_steps 20000 --state_readonly_at_think
--use_memory --mem_size 1024 --mem_decoupled_kv --mem_read_alpha_init 1.0
--mem_read_alpha_floor_start 0.5 --mem_read_alpha_floor_warmup_steps 3000
--mem_key_from_embedding --mem_key_window 4 --use_copy_head --emit_read_mask
--use_pkm --pkm_after_layer 5 --pkm_n_heads 4 --pkm_n_keys 256 --pkm_k_dim 128 --pkm_top_k 32
--pkm_use_output_gate --pkm_epsilon_start 0.5 --pkm_epsilon_warmup_steps 3000 --pkm_value_init_std 1.0
--pkm_score_norm layer --pkm_diversity_weight 0.01 --pkm_alpha_floor_start 0.3 --pkm_alpha_floor_warmup_steps 3000
--pkm_value_lr_mult 100.0 --gist_loss_weight 0.1 --gist_horizons 16,64,256
--latent_cotrain_weight 0.0 --latent_cotrain_R 4 --latent_cotrain_sample_frac 0.05 --latent_cotrain_max_positions 4
--latent_cotrain_start_step 99999999
--gate_calibration_weight 0.0 --gate_calibration_R 4 --gate_calibration_sample_frac 0.05 --gate_calibration_max_positions 4
--gate_calibration_sigma_low 0.1 --gate_calibration_sigma_high 0.9 --gate_calibration_start_step 99999999
--feature_probe_every_tokens 250000000
--feature_probe_wm_recall_path data/multibind_recall_heldout.jsonl --feature_probe_wm_recall_n 64
--feature_probe_code_recall_path data/code_recall_heldout.jsonl --feature_probe_code_recall_n 200
--data_mix configs/pretrain_mix_v14_wmrecall.yaml --tokenizer HuggingFaceTB/SmolLM2-135M
--think_burst_prob 0.0 --think_max_bursts 2 --think_max_burst_depth 6
--T 2048 --batch 4 --grad_accum 32 --activation_checkpointing --bf16 --tf32 --no-compile
--alpha_wd 0.0 --wd 0.01 --grad_clip 1.0 --z_loss 1e-4 --mask_eos_in_targets
--optimizer muon --lr 1.4e-3 --lr_muon 5e-3 --lr_schedule wsd --warmup_steps 400 --lr_decay_frac 0.15
--steps 30000 --val_every 200 --log_every 20
--mid_eval_every_tokens 250000000 --mid_eval_save_only --mid_eval_n_problems 50 --mid_eval_max_gen 192
--save_ckpt checkpoints/pretrain_v14.pt --tb_dir runs/tb/pretrain_v14
ARGS
}

for attempt in $(seq 1 40); do
  # Prefer a v14 resume point; else seed from the latest v12 ckpt (continuation).
  latest_v14=$(ls -t checkpoints/pretrain_v14_step*.pt 2>/dev/null | head -1)
  if [ -n "$latest_v14" ]; then
    rstep=$(echo "$latest_v14" | sed -E 's/.*_step([0-9]+)_.*/\1/')
    resume="--load_ckpt $latest_v14 --start_step $rstep"
    echo "=== [autoresume $attempt] resuming v14 from $latest_v14 (step $rstep) ===" >> "$LOG"
  else
    seed_v12=$(ls -t checkpoints/pretrain_v12_step*.pt 2>/dev/null | head -1)
    if [ -z "$seed_v12" ]; then
      echo "=== [autoresume] NO v12 seed ckpt found — aborting (set up the base first) ===" >> "$LOG"
      exit 1
    fi
    sstep=$(echo "$seed_v12" | sed -E 's/.*_step([0-9]+)_.*/\1/')
    resume="--load_ckpt $seed_v12 --start_step $sstep"
    echo "=== [autoresume $attempt] CONTINUATION seed from $seed_v12 (start_step $sstep) ===" >> "$LOG"
  fi
  CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u experiments/train_lm.py $(common_args) $resume >> "$LOG" 2>&1
  rc=$?
  if grep -q "saved.*checkpoints/pretrain_v14.pt" "$LOG" && [ "$(ls -t checkpoints/pretrain_v14_step*.pt 2>/dev/null | head -1 | sed -E 's/.*_step([0-9]+)_.*/\1/')" -ge 29000 ] 2>/dev/null; then
    echo "=== [autoresume] COMPLETE (rc=$rc) ===" >> "$LOG"; exit 0
  fi
  echo "=== [autoresume $attempt] exited rc=$rc — will resume from latest ckpt ===" >> "$LOG"
  sleep 20
done
echo "=== [autoresume] GAVE UP after 40 attempts ===" >> "$LOG"; exit 1

# ---------------------------------------------------------------------------
# POST-RUN WM-usefulness eval (run by hand on the finished base; pick a free GPU):
#
#   CKPT=$(ls -t checkpoints/pretrain_v14_step*.pt | head -1)
#   for arm in on full_off no_think; do
#     CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     PYTHONPATH=. .venv/bin/python experiments/eval_code_recall.py \
#         --ckpt "$CKPT" --tasks data/code_recall_heldout.jsonl \
#         --mode teacher_forced --wm_arm $arm --max_problems 400
#   done
#   # repeat with --tasks data/agentic_recall_heldout.jsonl, and --mode generate.
# WM is load-bearing iff (on) recall >> (full_off) and >> (no_think).
