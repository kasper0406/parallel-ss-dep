# Verdict: latent thinking works on DEPTH, not on code

## Summary
[[latent-thinking]] is a **validated, robust depth-extender for homogeneous iterated computation** — pointer-chase / arithmetic-chain / execution-trace tasks where the answer requires N sequential dependent hops. There it gives **+0.45 to +0.80** over a fair no-think control, growing with depth, with the gate learning to allocate exactly the right depth. On **code generation (HumanEval/MBPP) it is net-neutral-to-negative** because per-token code is recall, not iteration ([[code-is-recall-not-iteration]]). Source: `THINKING_LATENT_2026_05_28.md`, `project_latent_thinking_real_model.md`, `project_v13_day1_latent_reasoning.md`, `project_latent_thinking_code_gen.md`.

## Where it WORKS (the wins)
- **Synthetic pointer-chase** (`f^K(s)`): floor 0.12–0.23 → **1.00** at R=K; needs *exactly* R=K (under/overshoot fail → genuine per-hop computation); length-generalizes (K=4→10 hops, K=8→12). Token-mode (discrete feedback) = 0.09 → **bandwidth is essential**.
- **Real arithmetic chains** (the task discrete thinking scored 0/80): n=1–6 go 0.15–0.52 → **~1.00**.
- **Real 601 M model, fair control**: one forward has a **~2 dependent-hop budget**; latent thinking extends effective depth to ≥8 hops, **+0.65–0.80 at n≥3** vs a fully-trained 100 %-no-think control. (The n=2 "lift" was partly an undertraining artifact — caught by the fair control, see [[fair-baselines]].)
- **Day-1 co-train (v13)**: with the depth-matched `LatentReasoningCotrain` loss, lift grows with training to **+0.45–0.60 across the full n=2–8 range** at 2.75 B tokens; gate `avg_steps` becomes depth-calibrated (tracks n). This validated "latent thinking from day 1" works. See [[pretrain-run-history]].
- **Heterogeneous code-execution traces solve at depth too** (+0.47–0.78) — the earlier "heterogeneity is a hard wall" was partly an eval artifact (digit-run parsing bug, see [[broken-probe-lessons]]); plain per-hop latent traces deep programs ~95–100 %.

## Where it DOESN'T (code generation)
- **MBPP net-negative at every gating threshold** (Δlogp −3.7 at R=1, −6.4 at R=4): the reasoning-trained adapter is **OOD for code** → feeding the hidden back maps the state to the reasoning manifold → prediction collapses, degenerate loops.
- The fix flips it to net-≥0: **adapter-only freeze-trunk on-code co-training** (`latent_code_cotrain.py --freeze_trunk`) — the no-think forward never uses the adapter so the base is preserved **byte-identical** (zero forgetting) while the adapter learns code. But the ceiling is only **neutral-to-slightly-positive** because code-token prediction is recall, not iteration.
- **HumanEval is the wrong probe** — short, fits in the recurrent state, no depth bottleneck. The one real HumanEval lift (7→11) came from *selective routing*, not from depth (see [[humaneval-trajectory]]).

## The unifying mechanism + caveats
- A latent thread = **one value + one fixed transform, iterated**. It supplies **depth, never per-step width** — breaks when the per-step op varies ([[latent-thinking]] "the wall: heterogeneity").
- **Over-thinking degrades on a cliff** (the iterated map is non-convergent; overshoot by 2 = chance) unless the answer is trained as a **fixed point** → over-think harmless. Pairs with `--emit_after_R` to bound compute. See [[pareto-safe-thinking]].
- **Don't co-train a cold latent path inside a from-scratch pretrain** (it destabilizes — v12). Either day-1 with the *depth-matched solvable* loss + a weight ramp, or post-hoc adapter-only. See [[pretrain-run-history]].

## Related
[[latent-thinking]] · [[code-is-recall-not-iteration]] · [[pareto-safe-thinking]] · [[fair-baselines]] · [[mechanism-verdicts-overview]] · #verdict #thinking
