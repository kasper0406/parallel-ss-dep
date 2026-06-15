# Output / thinking gate

## Summary
A **per-position sigmoid head** deciding emit-vs-think (σ = P(EMIT)). It decides *where* to spend [[latent-thinking]] compute and can learn adaptive halting. Its central problems: it is **temperature-fragile** (selective at greedy, collapses to bimodal under RL sampling), it **mis-routes on code** (fires think where thinking hurts), and "will thinking help here" is **not linearly decodable** from the hidden state. The working fix on code is hard routing-to-emit + a low inference threshold (selective thinking). Source: `train_lm.py`, `train_rl_grader.py`, `thinking.py`; `project_gate_selectivity.md`, `THINKING_HUMANEVAL_2026_06_06.md`.

## Conventions (get the polarity right)
- `σ = sigmoid(gate_logit) = P(EMIT)`; emit iff `σ ≥ threshold`.
- The **gate-calibration loss** trains `P(think) = σ(−gate_logit) → 1{Δlogp>0}` (think where a latent think *helps*). An earlier version had this backwards — regression-guarded now. See `compute_gate_calibration_loss` and [[broken-probe-lessons]].
- **Don't let `--gate_floor_min` reach 0.0** during BPTT pretrain with gate-terms loss: the maladaptive-thinking trap (VAL ppl 49 → 940) where the model routes all probability mass into the think token. Use `--gate_floor_min 0.5 --gate_warmup_steps 20000`.

## The three failure modes
1. **Temperature-fragility.** Healthy `think_rate ≈ 0.30` at τ=0 (greedy); at τ=0.7 (RL rollout) collapses to bimodal (~0 or ~1) depending on whether `gate_floor` is below or ≥ `emit_threshold`. Operational rule: for RL rollouts that should preserve thinking use **`gate_floor < emit_threshold`** (e.g. 0.3 vs 0.5). Don't expect train-time selectivity to survive RL sampling without robustification.
2. **Anti-aimed on code.** `corr(P_think, Δlogp) ≈ −0.10` at every R — the gate fires think where thinking HURTS most. A gate-head-only BCE calibration did NOT flip it (−0.10 → −0.17): "will thinking help" is **not linearly decodable from `out_norm(h_t)`** by the ~897-param head (the value is *recall*, a property of what's retrievable, not of the current hidden). See [[code-is-recall-not-iteration]].
3. **Unproductive base.** On a trunk that never co-trained the latent mechanism, thinking is OOD and Δlogp is net-negative — calibration has no good target. Productivity must be *trained in*, not gated in. See [[route-around-principle]].

## What actually works on code
- **Route-to-emit** (`route_emit_finetune.py`: freeze all but the 897-param gate head, BCE→emit on code, 400 steps) drives σ(emit) 0.49→1.00, eliminating the compulsive-firing collapse (0/164 → 6/164).
- Then **low inference `--emit_threshold` (~0.1–0.3)** → thinks at only ~2 % of positions (strongest signal) → 7→11/164. Selectivity is the lever. See [[humaneval-trajectory]].

## Related
[[latent-thinking]] · [[code-is-recall-not-iteration]] · [[pareto-safe-thinking]] · [[broken-probe-lessons]] · #architecture #thinking
