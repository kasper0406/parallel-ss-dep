#!/bin/bash
# Track A — Combined SFT on the v12 base (2026-06-15).
#
# v12 is the project's "keeper" pretrain base: the clean WM+PKM trunk
# (aux-free, PKM bootstrapped properly — αL≈+0.41, row≈4.14 grown, vs
# v11's dead PKM). ~5B tokens (19000 steps), Chinchilla-complete for 287M.
#
# This is an EXACT mirror of the validated launch_sft_phase_c.sh recipe
# (additive α-gated retrieval-as-input + trunk multi-horizon gist loss)
# — the recipe that took the Phase C base to HumanEval 16/164 after
# grader-RL. Only the base ckpt changes (phase_c.pt -> v12.pt), so this
# is a clean base-vs-base comparison: does v12's stronger PKM trunk beat
# the Phase C base on identical post-training?
#
# WM verdict note (working-memory-recall-saga.md): WM is dead weight for
# realistic code recall (recurrence absorbs it). We KEEP the retrieval-as-
# input mechanism here only because it is part of the validated 16/164
# recipe and is harmless on short HumanEval prompts; the headline mover is
# the trunk + PKM, not WM. Latent thinking is added LATER as an adapter-
# only co-train (latent_code_cotrain.py) with the Pareto-safe structural
# floor, so it can only help — never in pretrain (latent-vs-PKM tension).
#
# HIGHEST-LEVERAGE ENHANCEMENT (do this version if time allows — task #84):
# the bottleneck on real code is base KNOWLEDGE, not mechanisms
# (project_code_thinking_ceiling: ~6% of distill targets are verified-broken,
# ~70% are no_tests/unverified, rare algorithm families are under-exposed
# <100 grad exposures). So the real SFT win is DATA HYGIENE:
#   (1) execution-filter distill_v7_phase1_with_tests.jsonl with code_grader —
#       drop score-0 (broken) targets, prefer with_tests (verified) over no_tests;
#   (2) tail up-sample rare algorithm families past ~100 exposures.
# That attacks the dominant failure tier (syntax/wrong-logic) that no inference
# mechanism can touch. This launcher runs the UNFILTERED corpus as the baseline;
# the cleaned-corpus run is the A/B that should actually move the headline.
#
# Next after this: launch_rl_grader_phase_c_v2.sh style grader-RL on
# checkpoints/sft_v12_combined.pt -> HumanEval.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

BASE=${BASE:-checkpoints/pretrain_v12.pt}
if [ ! -f "$BASE" ]; then
  echo "ERROR: v12 base ckpt $BASE not found — wait for v12 pretrain to finish."
  echo "(latest mid-eval ckpts: $(ls -t checkpoints/pretrain_v12_step*.pt 2>/dev/null | head -1))"
  exit 1
fi

cat data/distill_v7_phase1_with_tests.jsonl \
    data/synthetic_memory.jsonl \
    data/longctx_recall.jsonl \
    > data/sft_v12_combined.jsonl
echo "[sft-v12] base=$BASE  corpus rows: $(wc -l < data/sft_v12_combined.jsonl) "\
"(distill+synthmem+longctx)"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt "$BASE" \
    --save_ckpt checkpoints/sft_v12_combined.pt \
    --distilled_jsonl data/sft_v12_combined.jsonl \
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
    > runs/sft_v12_combined.log 2>&1 &
echo "Launched combined SFT on v12 base, GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v12_combined.log"
echo "After: HumanEval with  --prompt_style sft_comment --extract_code_block"
