# Principle: the route-around principle

## Summary
**The model routes around an auxiliary mechanism whenever the primary path (trunk/recurrence) can do the task — even under co-training.** A side mechanism stays idle unless TWO things hold: (1) the task forces the primary path to genuinely FAIL, and (2) the training gradient actually reaches the mechanism. This single principle explains the project's entire "post-hoc features are inert" history. Source: `project_wm_addressing_root_cause.md` (capstone), `WHY_THINKING_MARGINAL_ON_CODE.md` (capstone).

## The capstone experiment
Training trunk+WM+mem_alpha jointly on multibind recall was meant to prove "co-adapt the trunk → WM becomes addressable." It did the OPPOSITE: train loss → 0 (the trunk *memorized* the set), heldout recall 5 % (worse than frozen), WM read still diffuse, Δ(WM-on − off) = +0.0, mem_alpha pushed DOWN. **Why**: the whole program fits in the local context window, so the trunk can solve recall directly → gradient took the trunk path and left WM idle.

## The two necessary conditions for a mechanism to be load-bearing
1. **The primary path must provably FAIL.** WM was load-bearing on MQAR K=128 only because there the trunk *cannot*: (a) keys are random per batch → **non-memorizable**, AND (b) K=128 > recurrent-state capacity → **capacity-exceeding**. Both conditions are required.
2. **The gradient must reach the mechanism.** WM injects only at think positions, think targets are −100, the injection feeds lm_head not the trunk → the recall target's gradient flows through the recurrence, never through WM (the gradient-disconnect — see [[working-memory-recall-saga]]).

## Corollaries (use these when adding a mechanism)
- **Adding a module is never enough.** It must be *conditioned* so the optimizer uses it AND *probed* so you can prove it contributes.
- **Co-training is necessary but not sufficient** — the co-training LOSS must contain the bottleneck (this is [[objective-function-alignment]]). v10 co-trained WM for 7 B tokens and it still went recency, because the loss never required content recall.
- **To make a mechanism load-bearing, design a sub-task that is BOTH capacity-exceeding AND non-memorizable** (fresh random content per instance) so the mechanism is the ONLY low-loss path.
- On the deployment side: a mechanism's marginal benefit on a workload is *correct, not a bug*, when that workload doesn't have the matching bottleneck.

## Related
[[objective-function-alignment]] · [[working-memory-recall-saga]] · [[mechanism-verdicts-overview]] · [[key-separability]] · #principle
