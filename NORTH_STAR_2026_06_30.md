# North-star decision (2026-06-30)

> ⚠️ **CORRECTIONS (audited 2026-07-01, `SESSION_FINDINGS.md`).** This doc's confident framing outran
> the evidence in places; these override where they conflict. (1) **EVIDENCED:** constant-memory decode
> (lean config) + synthetic recall is cheaply trainable at low code-CE cost. **ASPIRATIONAL/UNMEASURED:**
> a *competent* bounded-state coding base (coding pass@k unchanged this session), natural-code recall,
> and the ENTIRE adaptivity / "learns-from-deployment" pillar (zero experiments; the surprise-gate may
> collapse to a no-op). (2) **"Beats transformers at recall" is retracted** — train-on-eval-distribution
> vs zero-shot. (3) Bounded memory is **not** unique to recurrent backbones (windowed/streaming/paged
> transformers get it too); the honest moat is "constant memory + high-concurrency throughput vs a
> FULL-KV baseline", and single-stream latency is ~2.3× worse until ~131k. (4) `forward_step` is
> argmax≥14/16-tested on the RL WM ckpt, **not** logit-tested on the lean config; `prefill_state_only`
> is untested. (5) The hybrid −29/−32% is a stage-2, undertrained, construction-monotone screen on a
> GLOBAL-attention config that BREAKS the O(1) moat — do **not** bake the hybrid into the base run on
> its strength. The dependency-gated build order below stays valid.

> Consolidates a multi-experiment session that started as "evaluate the best
> future direction" and converged, with evidence, on a different north-star than
> the project began with. Supersedes the benchmark framing in
> `STRATEGY_2026_06_28.md` (kept for the ranked-lever detail). Decision-ready.

## The pivot, in one paragraph

The original goal — *a small model that beats big models on coding benchmarks* — is,
on our own evidence, **structurally unreachable** at this budget on this backbone
(token-poverty: half-size SmolLM2 beats our 287M on code CE; plus a linear-attention
tax that *grows* with donor strength). Chasing it lands in the 13–21 greedy-noise band,
below a size-matched instruct model. **But the session also proved the project has a
real, large, hard-to-copy moat that has nothing to do with benchmark rank: cost.** So
the north-star changes to **a cheap, bounded-state, long-context / high-concurrency
coding agent that learns from its own deployment** — where every bounded-state design
choice (DeltaNet, O(1) decode, the fast/slow memory tiers) becomes the moat instead of a tax.

## What the session established (evidence)

1. **Benchmark goal unwinnable here** — token-bound + linear-attn tax; HumanEval-164
   greedy is ±2–3 noise (the "0→10→14→16" arc was a config artifact; headlines corrected
   in AGENTS.md/README this session).
2. **The linear-attn tax IS reducible — only by reintroducing real softmax attention**
   (hybrid/Jamba), monotone & iso-param (−14.9% at 4/32 attn layers → −32% at 12), NOT by
   a fancier linear cell (DeltaProduct/wide were weak/param-confounded). **But full
   attention breaks the O(1) moat** → off the table per the no-full-attention mandate;
   sliding-window hybrid is the only moat-preserving variant (untested, can't inherit
   donor weights byte-exact). (`linearize_hybrid_ablation.py`, `checkpoints/hybrid_*`.)
3. **The cost moat is REAL, large, and realized in code** (`DECODE_COST_BENCH.md`):
   constant memory (8.2 MiB recurrent state, *L-independent*) vs the transformer's B×L KV
   that explodes; in **batched serving** our per-seq latency is batch-invariant and
   throughput beats the transformer by **1.7–1.8×** at long context / high concurrency,
   on **5–11× less memory**, and we serve whole (B,L) regions that OOM it. True O(1)
   `forward_step` decode is wired (the docs were stale that it wasn't) — argmax≥14/16-tested on the
   RL WM ckpt, NOT logit-tested on the lean config; `prefill_state_only` (used for batched numbers) is
   untested. Caveat: no raw single-stream short-context latency win; production WM stack reintroduces
   an O(T) buffer. The batched "1.7–1.8×" is eager-vs-eager (fragile vs vLLM); the memory scaling is
   the robust win.
4. **Latent thinking = salvage-narrow** — kill as a code/agentic booster (net-negative,
   refuted lifts, heterogeneous-composition wall); keep as a publishable science result.
5. **Ingraining / "learn what to remember" / LoRA-sleep** = a real, well-supported line
   (CLS / Wake-Sleep / SEAL / sparse-memory-FT). DeltaNet *is already* the fast-weight
   inner loop (delta rule = Widrow-Hoff GD step, β = meta-learned write gate). It's the
   **agentic/lifelong differentiator** — a *step-3* lever, not a benchmark mover.

## The committed thesis

**A cheap bounded-state coding agent whose moat is cost + adaptivity:**
- **Cost:** O(1)/constant-memory decode → dominates batched, high-concurrency, long-context serving
  vs a FULL-KV baseline (proven for the lean config). NB: a windowed/streaming/paged transformer is
  also bounded-memory (at the same long-range-recall loss), so the edge is "constant memory + batched
  throughput vs full-KV", not a backbone others can't have; single-stream latency is ~2.3× worse
  until ~131k.
- **Adaptivity:** learns from its own deployment (fast DeltaNet state → experiential LoRA →
  sleep-consolidated base) — the lifelong-learning differentiator. **UNMEASURED this session — an
  aspirational step-3 bet, not evidenced.**

## Build order (dependency-gated)

1. **[DONE] Prove the moat.** ✓ Constant-memory + batched-throughput win validated.
2. **Strengthen + competent base (the next commitment):**
   - *(cheap, free win)* **optimize the decode latency constant** — compile / CUDA-graph
     `forward_step` (currently un-optimized eager; the flatness is architectural, the bad
     constant is fixable) → turn the per-token latency from a blemish into a win earlier.
   - *(the real step-2)* a **competent BOUNDED-STATE base** — the validated, moat-preserving
     pure-DeltaNet linearization (CE 0.759, still descending at 370M tokens) pushed further
     toward the donor; sliding-window hybrid only if more quality is needed without full attn.
   - stand up a **long-horizon / batched agentic eval** where the moat is the metric (not
     HumanEval-164 greedy).
3. **The differentiator:** lifelong/ingraining stack (fast-weight hierarchy + LoRA-sleep),
   on a base that's competent and cheap. Entry probes: P1 (surprise-β vs forget-α),
   P4 (sparse-PKM one-shot). (`project_ingraining_fastweights_2026_06_29`.)

## Honest concessions

- This concedes the benchmark headline. If "win HumanEval" is the actual goal, the only
  path is inherit-a-strong-base + concede the architecture — a different project.
- No raw single-stream short-context latency win; the moat is batched/long-context/memory.
- The production WM decode buffer is O(T) — the moat as proven holds for the *lean* config;
  keeping it means keeping decode bounded-state (WM needs a bounded-buffer rework or stays off).
- Ingraining's payoff is agentic, not benchmark; the surprise-gate may meta-train to ~no-op
  (γ→0) — run the cheap probes before betting on it.

## The decision in front of us

Step 1 passed. Step 2 (decode-kernel optimization + a competent bounded-state base + an
agentic eval) is the next real, multi-day investment. Committing to it = committing to the
cost-moat agent thesis over the benchmark thesis. That's the call.
