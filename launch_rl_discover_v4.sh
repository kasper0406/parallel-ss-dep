#!/bin/bash
# THINKING_PLAN v4 — Stability-first discovery RL.
#
# Yesterday: stochastic gate on a deterministic-gate-trained base
# destabilized outputs at sampled uncertain positions; reward signal
# washed out; gate fire didn't evolve.
#
# v4 fixes:
#   PHASE A (selective sampling): only sample when σ(gate) ∈ [0.1, 0.9].
#     Decisive positions stay deterministic → preserves trained quality.
#   PHASE B (entropy curriculum): start with strong exploration push
#     (0.05), decay to weak (0.001) over 200 steps. Bypasses chicken-
#     and-egg of needing exploration to learn but having exploration
#     destabilize.
#
# Base: sft_phase_c_combined.pt (historical 7-8/164 baseline).
# Data: mbpp_combined (reward signal exists there; synth_reasoning was
#   too OOD yesterday — confirmed reward collapse to 0).
#
# Decision gates:
#   step 50:  is the σ histogram showing enough uncertain positions
#             (sampling actually fires)? If samp < 0.05 → gate too
#             decisive, range needs widening
#   step 100: has gate fire EVOLVED from initial value? (Δ > 0.05)
#             If not → reward signal still not reaching gate
#   step 200: pass_n trend; gate-fire convergence

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/sft_phase_c_combined.pt \
    --save_ckpt checkpoints/rl_discover_v4.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 200 \
    --batch 3 \
    --grpo_n_group 4 \
    --lr 1e-6 \
    --max_gen 256 \
    --max_think_per_step 8 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.1 \
    --ponder_cost 0.0 \
    --counterfactual \
    --ponder_warmup_steps 0 \
    --grad_clip 1.0 \
    --curriculum_filter \
    --no-iterative_repair \
    --stochastic_gate \
    --gate_sample_range_low 0.1 \
    --gate_sample_range_high 0.9 \
    --gate_entropy_bonus_start 0.05 \
    --gate_entropy_bonus_end 0.001 \
    --gate_entropy_curriculum_steps 200 \
    --log_every 1 \
    --save_every 25 \
    --seed 0 \
    > runs/rl_discover_v4.log 2>&1 &
echo "Launched discovery-RL v4 — PID $!"
echo "Watch: tail -f runs/rl_discover_v4.log"
echo
echo "KEY METRICS TO WATCH:"
echo "  gate(fire=X, H=Y, ratio=Z, samp=A, dec=B, ent_bonus=C)"
echo "    fire — fraction of recorded decisions electing think"
echo "    samp — fraction of gate positions actually sampled (in range)"
echo "    dec  — fraction decisive (outside range) — should be 0.5-0.8"
echo "    if samp very small → gate too decisive; widen range"
echo "  sigma_hist (every 20 steps): the σ distribution itself"
echo "  pass_n   — passing rollouts; want this to trend up"
echo "  kl=+X    — drift; should stay <0.10"
