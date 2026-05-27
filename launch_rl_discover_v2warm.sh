#!/bin/bash
# Conservative composition: RL v2 recipe + ONLY stochastic gate
# (2026-05-27, after seeing discovery-v4 = 12/164 vs RL v2 = 16/164).
#
# Decision D1 / D3 from AUTONOMY_DECISIONS.md.
#
# Hypothesis: RL v2 plateaued at step300 (16/164) because the gate
# became deterministic — once σ(gate) settled around emit_threshold,
# the rollout-time threshold decision stopped exploring. The gate
# itself is the LAST untouched policy degree of freedom.
#
# Conservative design: this is the v2 recipe (same LR, KL, clip,
# temperature, KL coef, ponder=0) with EXACTLY ONE change:
# --stochastic_gate + selective sampling. This isolates the gate-
# exploration lever from any other knob movement.
#
# Reference policy = the WARM START (rl_grader_phase_c_v2_step300.pt).
# KL pulls policy back toward the 16/164 state, not the weaker SFT base.
#
# Knob deltas vs launch_rl_grader_phase_c_v2.sh:
#   --load_ckpt           : v2_step300 (was sft_phase_c_combined)
#   --steps 200           : extension run (was 400 from scratch)
#   --stochastic_gate     : NEW — gate as RL action
#   --gate_sample_range_low/high : 0.1 / 0.9 — only sample uncertain positions
#   --gate_entropy_bonus_*: 0.02 → 0.001 over 200 steps — mild,
#                           not the aggressive 0.05 from discovery v4
#                           (we want gentle exploration on top of v2's
#                            already-good policy, not destabilization)
#
# Decision-gates (from AUTONOMY_DECISIONS.md D1):
#   step  25: 'samp' fraction > 0.02 → sampling is firing at all
#   step  50: pass_n stable at v2 level or trending up
#   step 100: gate fire EVOLVED from start (Δ > 0.05)? eval ckpt ≥ 16?
#   step 200: eval ckpt pass@1 > 16/164? if not → pivot.
# ABORT EARLY if step50 pass_n drops > 30% below v2 baseline (recipe
# destabilizing the warm start despite KL anchor).

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

CUDA_VISIBLE_DEVICES=${GPU:-0} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt checkpoints/rl_grader_phase_c_v2_step300.pt \
    --save_ckpt checkpoints/rl_discover_v2warm.pt \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps 200 \
    --batch 4 \
    --grpo_n_group 4 \
    --lr 2e-6 \
    --max_gen 384 \
    --max_think_per_step 4 \
    --total_think_budget 120 \
    --emit_threshold 0.5 \
    --gate_floor 0.0 \
    --temperature 0.7 \
    --min_emit_before_eos 30 \
    --clip_eps 0.1 \
    --kl_coef 0.05 \
    --ponder_cost 0.0 \
    --ponder_shape quadratic \
    --counterfactual \
    --ponder_warmup_steps 50 \
    --grad_clip 1.0 \
    --stochastic_gate \
    --gate_sample_range_low 0.1 \
    --gate_sample_range_high 0.9 \
    --gate_entropy_bonus_start 0.02 \
    --gate_entropy_bonus_end 0.001 \
    --gate_entropy_curriculum_steps 200 \
    --log_every 1 \
    --save_every 25 \
    --seed 0 \
    > runs/rl_discover_v2warm.log 2>&1 &
echo "Launched discovery-RL on v2 warm start (PID $!)"
echo "Watch: tail -f runs/rl_discover_v2warm.log"
echo
echo "KEY METRICS:"
echo "  gate(fire=X, H=Y, samp=A, dec=B, ent_bonus=C)"
echo "    samp very small (<0.05) → gate too decisive, widen range"
echo "    fire change > 0.05 by step100 → exploration working"
echo "  pass_n trending up over baseline"
echo "  kl=+X should stay <0.10"
