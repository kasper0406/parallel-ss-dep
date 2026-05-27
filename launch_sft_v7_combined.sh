#!/bin/bash
# TRACK C: Combined SFT from pretrain with FIX A + future-emb-pred + the
# union of (distillation phases 1+1b+2) + synthetic-memory tasks.
#
# Inputs:
#   Base: checkpoints/pretrain_mix_v7_pkm_film.pt (raw pretrain, VAL ppl 5.83)
#   Data:
#     data/distill_v7_phase1_with_tests.jsonl   (4468 with-tests + 10k
#                                                 magicoder + 20k codefeedback
#                                                 = ~34k Qwen completions —
#                                                 phases 1/1b/2 all append)
#     data/synthetic_memory.jsonl                (12.5k synthetic memory-
#                                                 required tasks — variable
#                                                 binding, list-recall, dict-
#                                                 lookup, multi-step arithmetic)
#   Combined: ~46k examples = ~50-80M tokens of training.
#
# Architectural changes vs prior distilled SFT:
#   --mem_write_only_at_think  Forces WorkingMemory buffer to be filled
#                              only from think-position writes. Trains the
#                              write gate to discriminate at think positions
#                              from step 0 (vs the earlier "decorative
#                              thinking" failure mode).
#   --future_emb_loss_weight 0.05
#   --future_emb_T_max 8       At each position t, also predict the input
#                              embedding at t+T_eff via a 1-1 cosine loss.
#                              T_eff ramps 1 → 8 over first 30% of training.
#                              Rationale: forces position-t representations
#                              to encode the high-level structure of what
#                              comes next ('this is a graph problem → use
#                              Dijkstra'), attacking early-commitment errors
#                              where the model emits `return 0` because the
#                              docstring example showed 0.
#
# Combined corpus prep done by the queue daemon /tmp/queue_combined_sft.sh:
#   cat data/distill_v7_phase1_with_tests.jsonl data/synthetic_memory.jsonl \\
#       > data/sft_v7_combined.jsonl
#
# Pinned to GPU 1 (assumes era-of-experience still on GPU 0).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints data

# Build the combined JSONL (idempotent).
cat data/distill_v7_phase1_with_tests.jsonl data/synthetic_memory.jsonl \
    > data/sft_v7_combined.jsonl
echo "[combined-sft] corpus: $(wc -l < data/sft_v7_combined.jsonl) examples in data/sft_v7_combined.jsonl"

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v7_pkm_film.pt \
    --save_ckpt checkpoints/sft_v7_pkm_film_combined.pt \
    --distilled_jsonl data/sft_v7_combined.jsonl \
    --with_thinking \
    --mem_write_only_at_think \
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
    > runs/sft_v7_pkm_film_combined.log 2>&1 &
echo "Launched combined SFT on GPU ${GPU:-1} (PID $!)"
echo "Watch: tail -f runs/sft_v7_pkm_film_combined.log"
