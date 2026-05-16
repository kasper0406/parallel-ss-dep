#!/bin/bash
# SFT v5-pkm on (problem, solution) pairs WITH thinking enabled.
#
# Goal: take the v5-pkm pretrain base (884 M tokens, VAL ppl 7.02 but
#       HumanEval 0/50 — fluent code-shaped output, no problem solving)
#       and teach it to emit working solutions in response to natural-
#       language problem statements. Thinking is enabled during SFT so
#       the gate, WorkingMemory and PKM lookup heads remain active
#       (rather than being de-trained by SFT-without-thinking).
#
# Why with-thinking matters at SFT:
#   - The pretrain inspection (runs/inspect_v5_pkm_unclamped.json) showed
#     the thinking infra is wired and *responsive* — gate fires
#     meaningfully on hard prompts, WM reads sharply, PKM has strong
#     per-head specialization — but the model never learned to use
#     thinking *productively* during pretrain (target-mask kills the
#     signal).
#   - If SFT runs WITHOUT thinking, the model would settle into an
#     "ignore the gate / never inject memory reads" basin, and Phase-C
#     RL would have no thinking activations to amplify.
#   - With thinking ON during SFT, the gate decides when to think on
#     real (problem, solution) data and the model gets dense signal on
#     "did thinking help me emit the right next token?".
#
# Data: CodeAlpaca (20k) ∪ MBPP — already loaded by sft_code.load_pairs.
# Random think-burst insertion at training time:
#   --think_max_bursts 3 --think_max_depth 8 (~12-24 extra think tokens
#   per example, with targets masked to -100 so they don't contribute
#   to the loss directly — but the post-think emit DOES, which is
#   exactly the productive-thinking signal pretrain couldn't give us).
#
# Optimizer: stock AdamW (sft_code.py uses it — gentler than Muon for
# fine-tuning, no need for Newton-Schulz here).
#
# Pinned to GPU 1 (v4 is finishing on GPU 0).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-1} nohup .venv/bin/python -u experiments/sft_code.py \
    --load_ckpt checkpoints/pretrain_mix_v5_pkm.pt \
    --save_ckpt checkpoints/sft_v5_pkm_thinking.pt \
    --with_thinking \
    --think_max_bursts 3 --think_max_depth 8 \
    --mem_size 1024 \
    --epochs 2 \
    --batch 4 \
    --lr 3e-5 \
    --max_len 512 \
    --max_codealpaca 10000 \
    --log_every 50 \
    > runs/sft_v5_pkm_thinking.log 2>&1 &
echo "Launched SFT v5-pkm on GPU ${GPU:-1} (PID $!)"
echo "Watch:  tail -f runs/sft_v5_pkm_thinking.log"
