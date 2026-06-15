# Pretrain run history (v4 → v14)

## Summary
The lineage of mixed-corpus pretrain runs. The trunk shape settled at **shallow-wide 10L×896d** by v6; the production base is the **Phase C** Chinchilla-completion (5.28 B tokens). v8–v14 are the "co-train every mechanism from day 1" line, which surfaced most of the [[mechanism-verdicts-overview]] findings. Sources: `CLAUDE.md`, `project_v11_pkm_bootstrap_fail.md`, `project_cold_latent_cotrain_destabilizes_pretrain.md`, `project_v13_day1_latent_reasoning.md`, `run_v14_autoresume.sh`.

## The deep-trunk baselines
- **v4** — 30L×576d + FiLM(2,28) K=3 + WM + gate. 9300 steps / 2.13 B tokens. **Final VAL ppl 5.89** (the deep-trunk baseline).
- **v5-pkm** — 30L×576d + PKM-v5. Looked like a token-efficiency win but the probe found **97 % of PKM value rows still at random init** → PKM inert, the "win" was the trunk. Triggered the v7.1 bootstrap fixes. Don't reuse this PKM recipe.

## The shallow-wide swap (validated)
- **v6-shallow** — 10L×896d + 5 dense FiLM. Killed early but tracked 5–7 % ahead of v4 on VAL ppl → shallow-wide hypothesis validated. See [[shallow-wide-trunk]].
- **v7.1-pkm-film** — 10L×896d + dense FiLM + **PKM-v7.1 bootstrap** + WM + entropy-aux gate. 9300 steps. **Final VAL ppl 5.83** (strongest pretrain), beats v4 in 18 % less wall time. PKM live: αL +0.346, row 2.54. Per-source PKM always positive (wikipedia +0.099). This is the canonical "PKM works" run. See [[pkm-verdict]].
- **v7.1-pkm-xattn** — sister, cross-layer attention instead of FiLM. 5.94, within noise; FiLM is cheaper-per-step.
- **Phase C** — continuation of v7.1-film 2.13 B → **5.28 B tokens** (Chinchilla-optimal for 287 M) + trunk gist loss. Strict win on all 8 per-source CEs; VAL 5.83 → **4.90**. The production architectural base.
- **Persistent fact**: pretrain-only HumanEval is **0/50 on every ckpt** — the capability bottleneck is post-pretrain (SFT + RL).

## The co-train-everything line (v8–v14)
- **v8** (wide, thinking trunk) — the decisive probe verdict: **thinking is unproductive on the v8 trunk** (both discrete and latent Δlogp net-negative) because it never co-trained the latent mechanism. → latent thinking must be co-trained from day 1, not bolted on. See [[thinking-gate]] / [[latent-thinking-arc]].
- **v9** (601 M) — the base for the "latent thinking on the real model" work (+0.65–0.80 on pointer-chase). See [[latent-thinking-verdict]].
- **v10** — all-features co-train, but the recall stream was **single-binding** (no headroom) → WM routed around. See [[working-memory-recall-saga]].
- **v11** — swapped to capacity-exceeding multibind recall, BUT (a) the stream had **no answer supervision** (fed `problem_prompt` only) → WM still no recall gradient, and (b) the new aux losses **starved the PKM α-bootstrap** (αL stuck ~0, the v7.0 decay failure). v11 futile for its WM purpose as built.
- **v12** — clean WM+PKM base (aux losses off / deferred). PKM bootstrapped clean (+0.39). **The keeper base.** Recall improves strongly with training but **100 % via the recurrence** (think_frac=0, WM contributes nothing) — the route-around verdict. Currently running (GPU1).
- **v13** — day-1 latent reasoning via the **depth-matched `LatentReasoningCotrain`** loss (NOT the broken general-text `latent_cotrain`). **DAY-1 LATENT WORKS**: +0.45–0.60 across n=2–8 at 2.75 B, gate depth-calibrated. But day-1 latent **costs the PKM bootstrap** (they compete in the 0–3000 step window) → PKM dead. Retired ~57 % after banking the latent verdict (v12 is the PKM keeper).
- **v14** — WM-recall continuation from v12 with the validated **embedding-key addressing + copy head + answer-span `mem_read_mask`**, testing whether WM is load-bearing on **agentic** recall (the code probe has no headroom). Currently running (GPU0). See [[open-questions-and-roadmap]].

## Key lesson threaded through v11–v13
You **cannot cleanly co-train day-1 latent AND bootstrap PKM in the same 0–3000-step window** — they compete. For a unified model, stagger latent past the PKM α-commit window and/or strengthen the PKM α-floor. And **never co-train a cold latent path in a from-scratch pretrain** (v12 destabilized: VAL +10 %, gnorm 20×, PKM collateral damage) — use the depth-matched solvable loss + a weight ramp, or do it post-hoc adapter-only.

## Related
[[shallow-wide-trunk]] · [[product-key-memory]] · [[pkm-verdict]] · [[working-memory-recall-saga]] · [[latent-thinking-verdict]] · [[launchers-and-metrics-to-watch]] · #history
