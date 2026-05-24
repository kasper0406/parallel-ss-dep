#!/bin/bash
# Combined SFT v5 — WM-load-bearing experiment (2026-05-19).
#
# Goal: get WM to MEASURABLY contribute to the headline metric (see
# MILESTONE_ARCH.md). The architectural changes from v1 (the 11/164
# winner) → v5:
#
#   1. --retrieval_as_input_thinking (the v3 mechanism). Forces the
#      model to USE the WM retrieval (it replaces the [THINKING]
#      input embedding), so the read can't be ignored. Solves the
#      think-position homogeneity issue that motivated FIX A.
#
#   2. --wm_future_pred_weight 0.1 + --wm_future_pred_T 4 (Option A).
#      Direct supervision through the WM read: at think positions, an
#      aux head reads model.memory._last_injection_grad and is
#      supervised to predict embed(input_ids[t+4]) via cosine loss.
#      Gradient flows to W_v / W_q / W_proj / W_write — finally giving
#      the WM weights a clean, non-noisy signal. The injection is
#      autograd-tracked thanks to the _last_injection_grad stash
#      added 2026-05-19.
#
#   3. Long-context recall data MIXED IN. data/longctx_recall.jsonl has
#      5000 examples where a variable is defined early, hidden behind
#      ~512-1024 tokens of distractor, then asked for. DeltaNet's
#      recurrent state will plausibly lose the binding, so WM is the
#      escape hatch. The corpus has 38792 distill + 12500 synthetic
#      memory + 5000 long-context = 56k examples (long-context ~9%).
#
# Iteration plan: train → eval HumanEval + (re-run) ablation. If the
# WM ablation (wm_off) on v5 drops pass@1 by ≥10%, WM is now
# load-bearing.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

# Combined corpus = previous combined + long-context.
cat data/distill_v7_phase1_with_tests.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_v7_combined_v5.jsonl
echo "[v5] corpus rows: $(wc -l < data/sft_v7_combined_v5.jsonl) "\
"(distill+synthmem+longctx)"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_combined_v5.pt \
    --distilled_jsonl data/sft_v7_combined_v5.jsonl \
    --with_thinking \
    --retrieval_as_input_thinking \
    --wm_future_pred_weight 0.1 \
    --wm_future_pred_T 4 \
    --future_emb_loss_weight 0.05 \
    --future_emb_T_max 8 \
    --future_emb_T_ramp_frac 0.3 \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 100 \
    > runs/sft_v7_pkm_film_combined_v5.log 2>&1 &
echo "Launched combined SFT v5 (WM-load-bearing) on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_combined_v5.log"
