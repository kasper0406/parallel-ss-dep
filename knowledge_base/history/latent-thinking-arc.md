# The latent-thinking arc

## Summary
Chronology of the thinking work: discrete-token "thinking" never helped (0/80 on every rung) → the redirect to **latent-space, maximum-bandwidth** thinking, which finally worked on synthetic depth tasks → port to the real 601 M model (+0.65–0.80 on pointer-chase) → mapping the heterogeneity wall → the long, mostly-negative attempt to make it help HumanEval → day-1 co-train validated (v13). The verdict is in [[latent-thinking-verdict]]; this is the timeline. Sources: `THINKING_LATENT_2026_05_28.md`, `project_latent_thinking_*`, `project_v13_day1_latent_reasoning.md`.

## Timeline
1. **Discrete thinking fails** — `[THINK]` token append never amplified on this trunk (0/80 arithmetic). In a linear RNN the think slot already reads the whole state, so a discrete token's input adds nothing.
2. **(2026-05-28) latent thinking WORKS (synthetic)** — feed the trunk's own hidden back as the next input (Coconut), state-readonly. Pointer-chase floor→1.00 at exactly R=K; token-mode 0.09 (bandwidth essential); learns from final-answer-only + depth curriculum + consolidation; length-generalizes; gate learns adaptive halting. Real arithmetic chains 0.15–0.52 → ~1.00. See [[latent-thinking]].
3. **(2026-06-01..04) real 601 M model** — one forward has a ~2-hop budget; latent extends to ≥8 hops, +0.65–0.80 at n≥3 vs a fair no-think control. Two blockers fixed first: WM contaminates the fed-back latent (run WM-off on the homogeneous probe), and arithmetic is the wrong task (parallelizable; use pointer-chase). Autonomous + implicit-depth (fixed-point) halting both work.
4. **(2026-06-03) the heterogeneity wall** — a latent thread = one value + one fixed transform iterated. Generalizes over op *presentation* (framing/syntax, +0.2–0.5) but breaks on per-step op *variety* (multi-op composition, code-execution traces). Supplies depth, never per-step width. (Part of this was later found to be an eval artifact — see step 6.)
5. **(2026-06-05) over-step cliff + fixed-point fix** — the latent op is a non-convergent iterated map; overshoot by 2 = chance. Training the answer as a **fixed point** makes over-think harmless; `--emit_after_R` bounds compute. See [[pareto-safe-thinking]].
6. **(2026-06-04, correcting step 4) heterogeneity wall was partly an EVAL BUG** — the digit-run parser scored correct deep traces as collapse. Fixed → latent thinking SOLVES heterogeneous code-execution traces at depth (+0.47–0.78). See [[broken-probe-lessons]].
7. **(2026-06-06..07) the HumanEval saga (mostly negative)** — latent thinking is net-negative on code generation (adapter OOD for code, Δlogp −3.7). Many attempts: gate-cal SFT (too weak), think-then-write prefix (hurts), depth-cap (wrong band-aid), trace-trained op (doesn't transfer to generation). The ONE positive: **route-to-emit + selective low-threshold thinking → 7→11** (see [[humaneval-trajectory]]). The clean fix for net-≥0 on code = **adapter-only freeze-trunk on-code co-train** (base byte-identical).
8. **(2026-06-14/15) day-1 co-train validated (v13)** — the depth-matched `LatentReasoningCotrain` loss (NOT the broken general-text one) co-trained from day 1 gives +0.45–0.60 across n=2–8, gate depth-calibrated. Latent-from-day-1 is a validated win — but competes with the PKM bootstrap. See [[pretrain-run-history]].

## The throughline
Latent thinking is a **validated depth-extender for homogeneous iterated computation** and the **wrong lever for code generation at 287 M** ([[code-is-recall-not-iteration]]). Its invariant-safety is solved structurally ([[pareto-safe-thinking]]). Co-train from day 1 with a *solvable depth-matched* loss; never bolt it on or cold-start it in pretrain.

## Related
[[latent-thinking-verdict]] · [[latent-thinking]] · [[pareto-safe-thinking]] · [[code-is-recall-not-iteration]] · #history #thinking
