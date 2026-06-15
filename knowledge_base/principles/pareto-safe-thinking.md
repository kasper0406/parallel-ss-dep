# Principle: Pareto-safe thinking (upside without downside)

## Summary
The project invariant **"thinking must never be worse than no-think"** is solved by **structure + a verifier**, NOT by a smarter gate. Two pieces: (1) a **structural floor** — the latent adapter is applied only on the think-path, so a no-think forward is byte-identical to base (proven Δ=0 even at α=999), plus `state_readonly_at_think` (β=0) means a think can't corrupt the recurrent state/recall → thinking is *safe to leave on*; (2) a **verifier best-of arbiter** that picks max(score_nothink, score_think) → strictly ≥ no-think by construction. Label-free per-token arbiters (confidence-veto, raw gate) do NOT work. Source: `project_pareto_safe_thinking.md`.

## (1) Structural floor — makes thinking safe to leave on (no training needed)
- The `LatentFeedbackAdapter` is applied ONLY at think slots → if the gate never fires, the model is **byte-identical to base** (verified: α forced to 999, no-think `max|Δlogits| = 0.000`).
- `--state_readonly_at_think` (DeltaNet β=0 at think positions) → a think can change only the one emit it precedes, never the recurrent state → **cannot corrupt downstream long-range recall**. The historic "thinking corrupts recall −0.047" was the state-*writable* / WM-injection path; state-readonly removes that failure mode.
- So the surviving real downside is **code** (the ptr-trained adapter is OOD for code: always-think = −0.52 acc / −3.8 logp).

## (2) Verifier best-of — makes it strictly ≥ no-think
- `best_of_think.py::best_of_think(gen_nothink, gen_think, grade)` returns the higher-scoring candidate, tie → no-think → **strictly ≥ no-think by construction** (42 CPU tests assert the invariant across all grade-orderings).
- `skip_think_if_passed=True` runs think only when no-think fails the verifier (amortized cost = fail_rate × think-gen).
- The verifier for code **exists**: `code_grader.grade`.

## (3) What does NOT work — label-free per-token arbiters
- **Confidence-veto** (adopt think iff more confident) is net-negative on code AND discards the recall upside — because **confidence is anti-correlated with correctness exactly where it matters** (no-think on hard recall: conf 0.59 at 0 % acc).
- The **raw gate** fires think on ~49 % of code positions (should be ~0). The think-benefit sign is not linearly readable from `h_t` ([[code-is-recall-not-iteration]]). These reduce harm but do NOT deliver the invariant — don't ship them as the guarantee.

## Two complementary stability tricks (for the latent op itself)
- **Fixed-point answer-stability**: train the answer as a fixed point (gold on slots R+1..R+k) so over-stepping is a harmless no-op → thinking-on ≥ no-think by construction, halt-precision stops mattering.
- **`--emit_after_R`** bounds compute (gate → emit past R). See [[latent-thinking]].

## The gap / next step
Best-of only finds a LIFT if the think branch is actually good; today the code oracle ceiling is only +0.005 because the adapter is OOD. Fix = adapter-only frozen-trunk on-code co-train (structurally safe, base byte-identical), THEN best-of finds real code lifts. See [[latent-thinking-verdict]].

## Related
[[latent-thinking]] · [[thinking-gate]] · [[code-is-recall-not-iteration]] · #principle #thinking
