# Small Super-Coder — the goal

## Summary
The top-level goal is a **small (~217–360 M) code model that punches above its weight on coding benchmarks** (HumanEval / MBPP / long-context recall) under a tight 2× RTX 5090 budget. Every architectural experiment in this repo is meant to ladder up to that target — not to be an architecture-vs-architecture ablation for its own sake. See [[thesis]] for the bet behind it and [[hardware-and-compute]] for the constraints.

## The target
- A **deployable, efficient** code model, not just research ablations. The user wants outputs that yield a real model.
- Compete per-parameter with much larger models on HumanEval/MBPP. The thesis-confirming bar (stated in `THESIS.md`): **HumanEval ≥ 25 % at 217 M after SFT, ≥ 30 % after RL** — would beat StarCoderBase-1B per-parameter and approach Qwen2.5-Coder-0.5B at <½ the params.
- Current best is **16/164 = 9.8 %** (`rl_grader_phase_c_v2_step300`). See [[humaneval-trajectory]].

## Operating rules that follow from the goal
- **Validate on synthetic recall/reasoning tasks before scaling** (MQAR for memory, pointer-chase for depth). A mechanism must be shown load-bearing on a task whose bottleneck it matches before it's worth pretrain compute. See [[objective-function-alignment]].
- **Prefer experiments that move toward "actually competitive on HumanEval at small scale"** over pure architecture comparisons.
- **Compute budget is real.** Chinchilla-optimal for 287 M is ~5.3 B tokens; don't propose 4 B+ runs without a clear reasoning chain.

## The uncomfortable truth this project keeps rediscovering
At 287 M, the coding headline is **bottlenecked by base knowledge/composition**, not by any inference-time mechanism (see [[thinking-on-code-verdict]] and [[code-is-recall-not-iteration]]). The validated mechanisms ([[product-key-memory]], [[working-memory]], [[latent-thinking]]) are each load-bearing **on their matched bottleneck** but marginal on short knowledge-bound MBPP. The honest levers for the coding headline are base capability (params + clean data) and execution-grounded RL — not more thinking machinery.

## Related
[[thesis]] · [[mechanism-verdicts-overview]] · [[route-around-principle]] · #goal
