# HumanEval trajectory

## Summary
The headline coding number over the project's arc: **0 → 8 → 10 → 14 → 16/164 (9.8 %)**, current best `rl_grader_phase_c_v2_step300`. The big levers were distillation + SFT and **execution-grounded RL with a KL anchor**; latent thinking contributed a modest, noise-band selective lift (7→11) on a side branch. Source: `CLAUDE.md`, `THINKING_HUMANEVAL_2026_06_06.md`, the `project_*` SFT/RL memory files.

## The trajectory
| stage | HumanEval | note |
|---|---|---|
| undertrained base (8.2 M tok) | 0/20 | not usable for RL |
| distill (Qwen) over 10 M tok | 0/20 | code-shaped, no problem-solving |
| Combined SFT v1 (distill + future-emb + synthetic memory) | **11/164** | first non-zero |
| SFT v7 (v7.1 base) | 8/164 | — |
| Phase C SFT (Chinchilla base + additive + trunk gist) | **10/164** | +2; PKM load-bearing here (−5 ablation) |
| RL v1 step-100 (peak, then collapse ~step 350) | **14/164** | first RL lift; collapsed (no KL anchor) |
| **RL v2 step-300 (KL-stable GRPO)** | **16/164 (9.8 %)** | **current best**, monotonic climb |

## What worked
- **Distillation** (Qwen 3.6 AWQ, ~38 k (problem, CoT, code) pairs) replaced the tiny codeparrot distill — the student learns reasoning prose around the code.
- **Execution-grounded grader-RL** (`train_rl_grader.py`, GRPO with `code_grader.grade` dense reward) — the only signal that teaches productive thinking. RL v1 reached 14 then **catastrophically collapsed** when the ponder cost bit (depth 120→30 took the output format down → reward 0). Diagnosis: no KL-to-reference.
- **RL v2 added the KL term** (`--kl_coef 0.05`, frozen reference = starting SFT ckpt), dropped ponder cost, halved LR, tightened PPO clip, lowered temperature → **monotonic 14→15→16**, KL bounded 0.05–0.10.
- **Pure-code SFT beats CoT SFT**: training on `--distilled_code_only` (no CoT) collapsed syntax_error 96→1; fairly graded, pure-code 11 vs CoT 8. (Grade standalone funcs with imports+alias+`_run_test_in_subprocess` or eval undercounts ~3×.)

## The latent-thinking side branch (modest)
On `route_emit_code.pt`: route the gate to emit on code (σ 0.49→1.00), then a **low inference `--emit_threshold` (0.1–0.3)** → thinks at ~2 % of positions → **7→11** (matched 0-think control = 7 rules out a decode artifact; reproduced at two thresholds). The collapse fix (0/164 → ~no-think) is solid; the **+3–4 magnitude is within the greedy ±2–3 noise band** and needs temperature-sampled confirmation. The insight: thinking helps only when **rare and selective**; compulsive firing (46 %) collapses generation. See [[thinking-gate]], [[humaneval-trajectory]]'s parent [[latent-thinking-verdict]].

## Dead ends documented (don't repeat)
Discrete-token thinking (never amplifies), gate aux loss on the wrong mechanism, latent thinking as a post-hoc bolt-on (regresses VAL), continuation SFT/DPO on off-distribution data (regresses 9/12 not beating 16). SFT v8–v11: 1–5/164. DPO v1/v2: 9 and 12. Discovery RL on v2_step300 regressed 16→13.

## Why 287 M caps here
MBPP/HumanEval are knowledge/composition-bound at 287 M — see [[thinking-on-code-verdict]]. The ~12–15 % band is typical for this size class; the headline lever is base scale + post-training, not more thinking machinery.

## Related
[[thinking-on-code-verdict]] · [[latent-thinking-verdict]] · [[thinking-gate]] · [[pretrain-run-history]] · #history #code
