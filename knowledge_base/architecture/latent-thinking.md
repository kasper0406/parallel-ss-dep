# Latent (Coconut-style) thinking

## Summary
The thinking primitive that **finally works**: at a "think" slot, feed the trunk's OWN continuous hidden state back as the next input embedding (Coconut-style) for R iterations, in **state-readonly** mode (DeltaNet β=0) so the recurrent state is never corrupted. Each step feeds a full `d_model` vector — **maximum bandwidth, no discrete token emitted**. This performs genuine sequential computation and extends the model's effective reasoning depth. It is a **depth-extender for homogeneous iterated computation** — not a general code lever. See [[latent-thinking-verdict]] for the verdict, [[code-is-recall-not-iteration]] for why it's marginal on code. Source: `experiments/latent_think.py`, `latent_arith_real.py`, `thinking.py`; `model.py::LatentFeedbackAdapter` (line 93).

## Why it works where discrete tokens didn't
In a linear RNN the think slot already READS the whole state, so a discrete `[THINK]` token's input adds nothing — only a continuous fed-back latent carries the intermediate result. **Bandwidth is essential**: token-mode (constant `[THINK]` embedding) scores 0.09 on pointer-chase; latent scores 1.00. Discrete-token thinking was 0/80 on every arithmetic rung; latent solves them ~1.00.

## The recipe (non-obvious, took many iterations)
1. **Genuinely-sequential task** — pointer-chase `f^K(s)` (compose an arbitrary in-context table), NOT arithmetic (which collapses to an affine map a single forward can shortcut — see [[fair-baselines]]).
2. **Depth curriculum** (easy→hard ramp of R=depth) to bootstrap — no-curriculum control fails.
3. **Consolidation phase** (uniform-depth sampling after the ramp) or it forgets shallow rungs.
4. **Final-answer-only supervision** suffices (per-hop reasoning emerges unsupervised); per-hop supervision (`--per_hop`) helps push past ~2 hops but isn't always needed.
5. **Co-train from day 1** — bolted onto a converged trunk it is inert/OOD (see [[route-around-principle]]). The depth-matched `LatentReasoningCotrain` (`--latent_reasoning_weight`) on the `ptr10dict` corpus is the validated day-1 loss; the OLD `--latent_cotrain_weight` (general-text random-next-token target) is broken and destabilizes pretrain — see [[pretrain-run-history]] (v12/v13).
6. **Answer-stability / fixed-point** training makes over-thinking harmless (the latent op is a non-convergent iterated map; overshoot by 2 = chance unless the answer is trained as a fixed point). Pairs with `--emit_after_R` to also bound compute. See [[pareto-safe-thinking]].

## Adaptive halting
The existing `output_gate` (see [[thinking-gate]]) can learn to halt the latent loop at exactly the right depth (synthetic `halt_exact 1.00`; on the real v13 base `avg_steps` tracks n: 2.1/3.0/3.6/4.1/5.4/5.9 for n=2–7). Implicit-depth halting (fixed-point recognition, depth never stated) also works.

## The wall: heterogeneity
A latent thread carries **one value + one fixed transform**, iterated. It generalizes over the *presentation* of that op (framing/syntax/operand, +0.2–0.5) but **breaks when the per-step op VARIES** — multi-op composition or real code-execution traces (different op per line) fail (loss plateaus, lift negative). Code execution needs value + program-counter + per-line op-selector, which a single iterated readout can't represent. So latent thinking supplies **depth, never per-step width**. See [[latent-thinking-verdict]].

## Related
[[latent-thinking-verdict]] · [[thinking-gate]] · [[pareto-safe-thinking]] · [[code-is-recall-not-iteration]] · [[deltanet-backbone]] · #architecture #thinking
