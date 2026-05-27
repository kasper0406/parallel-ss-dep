#!/bin/bash
# Combined SFT v7 — additive α-gated retrieval + trunk gist (2026-05-20).
#
# The v6 long-context recall eval was a clean negative: recall fell
# 99%→61% as distance grew, while v5 (same recipe, lexical target) held
# 99.8%. The diagnosis (see GEMINI.md):
#
#   * Fix B — retrieval-as-input REPLACED the think-token embedding with
#     the WM retrieval. A blurry retrieval overwrote the precise binding
#     the DeltaNet trunk carried. v7 makes the injection ADDITIVE and
#     α-gated: input[think] = think_embed + α·retrieval, α a learned
#     scalar (model.retrieval_input_alpha, init 0.1, no weight decay).
#     A useless retrieval contributes ≈0; the think_embed baseline
#     always survives, so a bad retrieval can never corrupt.
#
#   * Fix C — v6 supervised the WM read to predict the windowed
#     hidden-state GIST. Routing a blurry gist through the recall path
#     is self-defeating. v7 moves the multi-horizon gist target to the
#     TRUNK's heads (predict mean-pooled h[t+1:t+1+K] from h[t]). WM is
#     left to learn precise retrieval via the LM loss alone; "direction"
#     lives in the trunk. The v6 --wm_future_pred_weight loss is gone.
#
# Open finding this run also tests: the v1 recall eval (plain [THINKING]
# token, no retrieval-as-input) was the WORST of all (74%, 20% at
# distance 512, think_rate 0.36) — think VOLUME corrupts recall
# regardless of mechanism. v7 removes the WM-gist loss that plausibly
# drove v6's think_rate up to 0.23; if v7's think_rate reverts toward
# v5's 0.012, recall is preserved. If v7 STILL over-thinks and recall
# degrades, the next step is an architectural fix (think tokens
# state-read-only in the DeltaNet recurrence).
#
# Iteration plan: train → eval_longctx_recall (recall must hold ~99%)
# + HumanEval + ablation.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

cat data/distill_v7_phase1_with_tests.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_v7_combined_v7.jsonl
echo "[v7] corpus rows: $(wc -l < data/sft_v7_combined_v7.jsonl) "\
"(distill+synthmem+longctx)"

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_combined_v7.pt \
    --distilled_jsonl data/sft_v7_combined_v7.jsonl \
    --with_thinking \
    --retrieval_as_input_thinking \
    --future_emb_loss_weight 0.1 \
    --wm_gist_horizons "16,64,256" \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 1024 \
    --log_every 100 \
    > runs/sft_v7_pkm_film_combined_v7.log 2>&1 &
echo "Launched combined SFT v7 (additive + trunk gist) on GPU ${GPU:-0} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_combined_v7.log"
