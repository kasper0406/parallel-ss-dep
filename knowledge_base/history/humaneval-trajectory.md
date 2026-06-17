# HumanEval trajectory

## Summary
The headline coding number was historically reported as **0 → 8 → 10 → 14 → 16/164**, but a 2026-06-16 SAME-CONFIG re-measurement (see the section below + memory `project_humaneval_config_artifact`) shows the **robust best is ~14/164**, reached by BOTH Phase C SFT (no thinking) and RL v2_step300 (with thinking) — the "16" was a noisy peak and the SFT→RL gain ≈0 at consistent config. The real lever was **distillation + SFT onto a code-focused base** (the data); execution-grounded RL and latent thinking did **not** reliably add on top once eval config + greedy noise are controlled. HumanEval-164 greedy is too noisy as the dev signal. Source: `project_humaneval_config_artifact`, `CLAUDE.md`, the `project_*` SFT/RL memory files.

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

## Same-config re-measurement (2026-06-16) — eval config matters; v12 is a worse code base
Re-evaluating with **fixed flags** (`--prompt_style sft_comment --extract_code_block --min_emit_before_eos 30 --max_gen 512`, thinking-off, greedy, full 164):
- **Phase C SFT = 14/164** (not the earlier "10" — the original likely truncated at a shorter `max_gen`; HumanEval numbers are eval-config-sensitive, so only compare same-config).
- **v12 SFT = 8/164.** v12 (recall-heavy WM-experiment pretrain mix) is a **materially worse CODE base** than Phase C (code-focused mix), confirming [[thinking-on-code-verdict]] / the mechanisms-synthesis: the code headline tracks code-data content, and pouring pretrain capacity into recall/WM/PKM *costs* code (data-mix opportunity cost).
- **RL v2_step300 (the documented "16/164") = 13 thinking-off / 14 thinking-ON (native)** — ≈ its own SFT base (14). And RL **v11c = 8**. The `eval_rl_*.log` numbers cluster 7–14; **"16/164" is not robustly reproducible**. ⇒ the celebrated **+6 SFT→RL gain compresses to ≈0** at consistent config: **robust best = 14/164**, reached by BOTH Phase C SFT (no thinking) and RL (with thinking); thinking on-vs-off ≈ +1 (noise). The trajectory over-read eval-config changes as capability.
- **Drop-broken data hygiene = 9/164 (−5 vs dirty 14)** — pruning verified-broken targets HURTS (exec_error rows are mostly-valid code + MBPP coverage; only 632 true syntax garbage). Naive data hygiene ≠ pruning.
- **Load-bearing lesson** ([[fair-baselines]]): fix ONE eval harness before quoting any delta; HumanEval-164 greedy is too noisy to be the dev signal (use temp pass@k / a bigger bench). See memory `project_humaneval_config_artifact`. The headline path needs real data-regeneration or scale, not the cheap levers (all negative tonight).

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
