#!/bin/bash
# Combined SFT v6 — WM multi-horizon GIST supervision (2026-05-20).
#
# The v5 WM-load-bearing experiment did not move the needle (HumanEval
# ~10/80, indistinguishable from v1; the wm_off ablation showed WM was
# still decorative). Root cause diagnosed: v5's Option-A supervision
# trained the WM read to predict embed(input_ids[t+4]) — the INPUT
# EMBEDDING of one token 4 positions ahead. That is a context-free,
# lexical target ("the token `range`"). It contains no "high-level
# direction", so WM had nothing of a mental picture to learn.
#
# v6 fix (Options 2+3 from the WM-target analysis):
#
#   * Predict the GIST of the upcoming window, not one lexical token.
#     Target = stop_grad( mean(h[t+1 : t+1+K]) ) — the mean-pooled
#     HIDDEN STATE over the next K positions. Because the trunk is
#     causal, each hidden state is a running contextualised summary;
#     the windowed mean is a genuine "where this is going" vector.
#
#   * Multi-horizon K = {16, 64, 256} — local tactic, mid plan, global
#     direction. One head per horizon. (--wm_gist_horizons)
#
# Everything else is held identical to v5 (retrieval-as-input,
# long-context data, future-emb-pred) so the eval + re-ablation
# isolate the gist-target change.
#
# Iteration plan: train → eval HumanEval + re-run ablation. If wm_off
# on v6 drops pass@1 by >=10%, the gist target was the bug and WM is
# now load-bearing. If it STILL doesn't, that is strong evidence the
# 217M trunk does not form rich enough plans for WM to store — a
# capacity wall, not a target bug (see GEMINI.md).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

# Combined corpus = previous combined + long-context (same as v5).
cat data/distill_v7_phase1_with_tests.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_v7_combined_v6.jsonl
echo "[v6] corpus rows: $(wc -l < data/sft_v7_combined_v6.jsonl) "\
"(distill+synthmem+longctx)"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_combined_v6.pt \
    --distilled_jsonl data/sft_v7_combined_v6.jsonl \
    --with_thinking \
    --retrieval_as_input_thinking \
    --wm_future_pred_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
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
    > runs/sft_v7_pkm_film_combined_v6.log 2>&1 &
echo "Launched combined SFT v6 (WM gist supervision) on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_combined_v6.log"
