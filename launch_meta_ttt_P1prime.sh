#!/bin/bash
# Meta-TTT Phase P1 — repo-adaptive coder pilot (META_TTT_PLAN_2026_07_13.md).
# Meta-train DeltaNet's recurrent state into a deliberate test-time learner over
# full-repo ingestion: ~300M tokens co-training the standard code mix (retention
# anchor) with the meta-TTT episode aux (read a repo 4-32k tok at O(1) cost →
# CE ONLY on the cross-file usage task span; state carried across the whole
# episode; truncated BPTT over the last --meta_ttt_grad_chunks chunks).
#
# BASE = checkpoints/stageA_executor.pt.  WHY this base (per META_TTT_PLAN P0/P1,
# "run the kill-test on the current base FIRST"):
#   * it is the most recent / canonical base and the one eval_repo_adaptive.py
#     defaults to (--ckpt checkpoints/stageA_executor.pt), so P1 trains exactly
#     the base P0 measured the incidental-learning baseline on;
#   * it is a PLAIN DeltaNet (no memory / PKM / gate) — the recurrent state IS
#     the whole learner, which is precisely the meta-TTT thesis, and keeps the
#     chunked state-carrying ingest free of WM/PKM confounds.
# (feature_pilot_A carries mostly-inert features; port to the stronger
# linearized base only in P3 if meta-TTT passes the P2 kill-test.)
#
# Recipe = Stage-A/B LR / schedule / batch / arch (full plasticity, the Stage
# lesson) + the meta-TTT aux. NO thinking token, NO latent adapter — plain
# deltanet + episode co-train. --no-compile: the variable-shape long-ingest
# forwards crash Inductor (auto-disabled anyway; explicit here).
#
# SINGLE GPU (episode ingest state-carry is a sequential per-episode path; DDP is
# not wired for it). Periodic ckpts every 50M tokens per the MANDATORY launch
# rule (Stage A survived a GPU-off-the-bus by exactly this).
#
# Engagement diagnostics: the per-step log prints mttt(ce=..,ctx=..,bkt=..,gt=..)
# — watch that `ce` FALLS on the ingested episodes. Per-arm eval every ~50M tok:
#   CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. .venv/bin/python \
#     experiments/eval_repo_adaptive.py --ckpt checkpoints/${TAG}_step*.pt \
#     --eval data/repo_episodes/eval.jsonl \
#     --controls data/repo_episodes/eval_controls.jsonl \
#     --out results/repo_adaptive_${TAG}.json
# The P2 kill-line reads lift(real - shuffled) vs the SAME on stageA_executor.
set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints results

GPU=${GPU:-0}
STEPS=${STEPS:-2300}      # 2 x 32 x 2048 x 2300 = ~301M tokens
SEED=${SEED:-0}
TAG=${TAG:-meta_ttt_P1prime}
MTTT_WEIGHT=${MTTT_WEIGHT:-0.1}
GRAD_CHUNKS=${GRAD_CHUNKS:-2}   # ~4k grad tokens; lower to 1 if the episode
                                # grad-window forward tightens memory margin.

CUDA_VISIBLE_DEVICES=$GPU nohup .venv/bin/python -u experiments/train_lm.py \
  --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64 --d_ff 2560 --tie_embeddings \
  --load_ckpt checkpoints/stageA_executor.pt --keep_base_vocab 49152 \
  --feedback none \
  --data_mix configs/pretrain_mix_stageA_executor.yaml \
  --fim_legacy_strings \
  --tokenizer HuggingFaceTB/SmolLM2-135M \
  --think_burst_prob 0 --mask_eos_in_targets \
  --T 2048 --batch 2 --grad_accum 32 \
  --activation_checkpointing --bf16 --tf32 --no-compile --bf16_optim_state \
  --optimizer muon --lr 6e-4 --lr_muon 2e-3 --alpha_wd 0.0 --wd 0.01 \
  --grad_clip 1.0 --z_loss 1e-4 \
  --lr_schedule wsd --warmup_steps 100 --lr_decay_frac 0.10 \
  --meta_ttt_weight $MTTT_WEIGHT \
  --meta_ttt_train_prefix data/repo_episodes_v2/train.jsonl \
  --meta_ttt_grad_chunks $GRAD_CHUNKS \
  --meta_ttt_chunk_size 2048 \
  --meta_ttt_every 1 \
  --meta_ttt_prefix_supervise_m 0 \
  --meta_ttt_max_ctx_tokens 32000 \
  --steps $STEPS --val_every 250 --log_every 20 --seed $SEED \
  --mid_eval_every_tokens 50000000 --mid_eval_save_only \
  --save_ckpt checkpoints/${TAG}.pt --tb_dir runs/tb/${TAG} \
  > runs/${TAG}.log 2>&1 &
echo "Launched $TAG on GPU $GPU (PID $!) — tail -f runs/${TAG}.log"
