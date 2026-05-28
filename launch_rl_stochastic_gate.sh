#!/bin/bash
# Stochastic-gate execution-grounded GRPO RL — the real fix for the
# thinking gate (2026-05-28). Implements PLAN_FLAW_C §1-2 (the FATAL flaw
# in THINKING_AUDIT_2026_05_28.md).
#
# WHY THIS RUN EXISTS
# -------------------
# The per-token-logp aux losses (process_reward / gate_calibration) were
# audited as fatally mis-designed: they reward a think iff it raises
# logp(next surface token), which is near-free extra compute, biased by
# self-fulfilling candidate selection, and STRUCTURALLY BLIND to whether
# the emitted function passes tests (audit flaw C). The fix is to make
# "did thinking help" a TERMINAL grader-reward delta decided as an RL
# policy action — exactly what train_rl_grader.py --stochastic_gate does:
# the gate's emit-vs-think Bernoulli draw becomes a policy action that
# receives the SAME group-relative advantage as the emitted tokens, so a
# rollout that thought-and-passed beats a sibling that didn't-think-and-
# failed. No logp anywhere.
#
# DO NOT resurrect or extend the logp aux-loss path (process_reward.py /
# compute_gate_calibration_loss). They remain in-tree (with tests) only as
# next-token-logp proxies — see their docstrings.
#
# GO/NO-GO (PRIORITY-1 diagnostic, runs/probe_think_grader_reward.log):
#   probe_think_grader_reward.py measured Δ(grader score) = with_think −
#   no_think on the ACTUAL deploy generator (generate_with_retrieval_as_
#   input, iterative re-decode):
#     sft_phase_c_combined.pt      : Δ = +0.083  (5 vs 3 pass / 40)
#     rl_grader_phase_c_v2_step300 : Δ = +0.143  (8 vs 5 pass / 40)
#   BOTH POSITIVE → thinking already helps the task. This is a clear GO
#   (the audit had predicted ≤ 0 on the logp proxy; the honest terminal
#   measurement disagrees). RL refines WHICH uncertain positions to think
#   on, with the terminal reward as the only signal.
#
# RECIPE
# ------
# Validated v2 stability recipe (launch_rl_grader_phase_c_v2.sh: monotonic
# climb to 16/164) PLUS the stochastic gate as a policy variable:
#   --stochastic_gate                       : gate emit/think = Bernoulli action
#   --gate_sample_range_low 0.1 / _high 0.9 : only EXPLORE uncertain gates;
#                                             decisive σ uses the threshold
#   entropy curriculum 0.03 -> 0.0 / 200    : anneal exploration (anti-collapse)
#   --kl_coef 0.05 --lr 2e-6 --clip_eps 0.1 --temperature 0.7 : v2 stability
#   --ponder_cost 0.0                       : v2 lesson — depth pressure was
#                                             the v1 collapse trigger; keep
#                                             the FIRST test of the terminal
#                                             signal alone unconfounded.
#   --gate_floor 0.0 --emit_threshold 0.5   : gate_floor < emit_threshold is
#                                             the PINNED footgun rule (else the
#                                             gate is silently never-think).
#
# BASE CKPT: sft_phase_c_combined.pt (10/164), NOT rl_grader_phase_c_v2_
# step300 (16/164). Rationale (PLAN_FLAW_C §"Base ckpt choice"): v2-step300's
# gate was shaped by 300 RL steps that NEVER exercised thinking
# (deterministic never-think gate), a poor starting policy. The SFT base
# still has its SFT-distilled think_rate ≈ 0.33 — a non-degenerate Bernoulli
# prior. Use v2-step300 only as a SECONDARY run (set BASE / SAVE / KL ref).
#
# state_readonly_at_think CONSISTENCY (audit flaw A/B/E, PLAN_FLAW_B §2):
#   The SFT base + v2 ckpts have cfg["state_readonly_at_think"]=None → OFF.
#   build_model_from_ckpt reads cfg, so SFT, this RL run, AND the eval
#   generator ALL run state-WRITING thinks (World A: OFF end-to-end =
#   train==deploy). This is CONSISTENT today. NOTE the divergence the audit
#   flags: the PRETRAIN launchers (launch_pretrain_v2_thinking.sh,
#   launch_pretrain_smoke_thinking.sh) DO pass --state_readonly_at_think, so
#   pretrain calibrated state-PRESERVING thinks while SFT/RL/eval run state-
#   writing thinks. We do NOT flip a default here (risky mid-arc); we make
#   the RL setting EXPLICIT (it inherits OFF from the loaded cfg) and document
#   the inconsistency. World B (ON end-to-end, incl. the forward_step β-mask)
#   is the recall-safe long-term move but is gated on decode-path work.
#
# MECHANISM NOTE (audit flaw A/B): train_rl_grader.py's rollout think step
# appends the discrete [THINKING] token, whereas the eval generator
# (generate_with_retrieval_as_input) injects the additive-α WM retrieval.
# Unifying these is the larger flaw-B refactor (shared _run_think_burst) and
# is intentionally out of scope for this launcher. Track gate_fire_rate +
# the Δ(grader) probe on each --save_every ckpt, not pass@1 alone.

set -e
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=$PYTHONPATH:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs checkpoints

# Defaults: primary run from the SFT base. Override BASE/SAVE/REF_TAG for
# the secondary run from v2-step300.
BASE=${BASE:-checkpoints/sft_phase_c_combined.pt}
SAVE=${SAVE:-checkpoints/rl_stochastic_gate.pt}
STEPS=${STEPS:-400}
# GPU 0 is the free card on this rig; GPU 1 is reserved/busy. Override with
# GPU=N if needed, but NEVER point this at a busy card.
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES=${GPU} nohup .venv/bin/python -u experiments/train_rl_grader.py \
    --load_ckpt "${BASE}" \
    --save_ckpt "${SAVE}" \
    --dataset mbpp_combined \
    --extract_code_block \
    --steps "${STEPS}" \
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
    --stochastic_gate \
    --gate_sample_range_low 0.1 \
    --gate_sample_range_high 0.9 \
    --gate_entropy_bonus 0.02 \
    --gate_entropy_bonus_start 0.03 \
    --gate_entropy_bonus_end 0.0 \
    --gate_entropy_curriculum_steps 200 \
    --grad_clip 1.0 \
    --log_every 1 \
    --save_every 50 \
    --seed 0 \
    > runs/rl_stochastic_gate.log 2>&1 &
echo "Launched stochastic-gate RL on GPU ${GPU} (PID $!)"
echo "  base=${BASE}  save=${SAVE}  steps=${STEPS}"
echo "Watch: tail -f runs/rl_stochastic_gate.log"
