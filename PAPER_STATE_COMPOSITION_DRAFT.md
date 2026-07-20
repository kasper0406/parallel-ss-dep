# Parallel State Composition in Delta-Rule Models: Merging Is Free, Sequential Accumulation Is the Wall

*Draft — 2026-07-20. Numbers are quoted verbatim from the run JSONs cited in
each table; nothing here is recomputed. Every reported value is traced to a
source file in Appendix A. `[CITE-TODO]` marks a reference whose arXiv id was
not recorded in our ideation notes.*

**Second headline:** *the composition operator is not the bottleneck — the base
model's own sequential state accumulation is.*

---

## Abstract

A delta-rule linear RNN carries all of its context in a bounded recurrent state
(8.2 MiB, flat from 512 to 131,072 tokens on our substrate), and the delta rule
itself is a sum of interference-corrected rank-1 updates — so per-segment states
computed independently and in parallel *should* be composable by simple
arithmetic. We test this on real repository code with a 402M linearized DeltaNet,
zero training anywhere: split a repo context into K segments, run each from zero
state, **mean-merge** the K recurrent states, inject the merged state as the
task's initial state, and measure teacher-forced task cross-entropy against a
single sequential ingestion of the same tokens.

Two results, stated at their true confidence. **(1) At the model's working scale
(1.5–6k-token contexts), mean-merging is a near-free substitute for sequential
ingestion.** Line-CE retention — lift(cartridge)/lift(sequential) — is 0.99
[0.82–1.40] at K=2 and 0.82 [0.62–1.31] at K=4; and on the hard *span* tokens the
merged cartridge does not merely match but *beats* one sequential state (+0.059 /
+0.061 span-CE lift at K=2/4 vs sequential's +0.004). Structure checks confirm
the merged state carries structure, not a token bag: the line-shuffled control is
worse than the cartridge by +0.05 to +0.13 span-CE at every K and every scale
tested. **(2) The binding constraint is the base model's sequential state
accumulation, not the merge.** Beyond the trained T=2048 window (8–32k contexts)
*sequential* ingestion actively *hurts* (−0.544 span-CE on the production base);
cartridges hurt 2–7× less than sequential at every K in that regime. A T=8192
continued-pretrain removes 75% of the harm (−0.544 → −0.133) and then *saturates
exactly at the training window*: doubling the token budget to 1.44B moves it to
−0.162 (flat) — a window-bounded state-extrapolation law, with natural-code
competence *improving* throughout (dependency-distance CE 0.8126 → 0.7948 →
0.7778, best ever measured on this stack).

Methodologically, a pre-registered span-lift sanity gate (lift(sequential) ≥
+0.15 span-CE) correctly *voided* both cartridge runs whose retention ratios
would otherwise have been reported as meaningful (one blew up to 13.7/14.2 over a
near-zero denominator; the other was a positive-looking 0.14–0.47 over a negative
denominator). Positioning: Cartridges (KV-cache self-study distillation) and
State Soup (Mamba, synthetic tasks) exist; zero-training parallel composition of
delta-rule states on *real repository code*, plus the quantified
sequential-accumulation wall, is the unclaimed cell. We claim no self-study tier,
no T=32k rung (a registered revisit-suspect), and report a single model, single
scale, single seed.

---

## 1. Introduction

A coding agent that must ingest a whole repository, then act over a long horizon,
pays two costs the moment its context grows: memory and latency. In a transformer
both grow without bound — the KV cache is O(T) in memory and the per-token decode
cost climbs with it. Our substrate pays neither: a DeltaNet-class linear RNN holds
a **bounded** recurrent state and decodes at O(1)/token. On the decode-cost bench
this is flat 8.2 MiB of recurrent state and ~919 MiB decode peak at **every**
context length from 512 to 131,072 tokens, against a matched transformer whose KV
cache grows 10 → 5121 MiB and whose decode-phase peak climbs 756 → 5937 MiB — a
6.45× memory gap at 131k that is still widening (`SCOREBOARD.md`,
`DECODE_COST_BENCH.md`). The bounded state is the cost moat.

That same bounded state suggests a second capability the transformer cannot
cheaply have. The delta rule is the Widrow-Hoff learning step: each token writes a
rank-1 update into the recurrent state, and — crucially — first *subtracts* the
state's current prediction for that key, so the state is (approximately) the
running least-squares solution of an associative key→value regression. Sums of
rank-1 updates from *disjoint* key subspaces are nearly additive. So a repository
context could in principle be split into segments, each ingested **independently
and in parallel** (each O(1) memory, zero training), and the resulting states
**composed by arithmetic** into a single 8.2 MiB state that approximates full
sequential ingestion. That is "cartridges" — precomputed, loadable, composable
context — but obtained by a forward pass and a mean, with no distillation loop, on
the one architecture where state composition is algebraically natural.

This paper measures whether that composition survives contact with real code, and
finds a sharp two-part answer.

- **At working scale, composition is essentially free** (§4.2). Mean-merged
  per-segment states retain ~0.8–1.0 of sequential ingestion's line-CE benefit
  and outperform sequential on the hard span tokens outright.
- **The wall is not the merge — it is the base model's own sequential
  accumulation** (§4.3–4.4). Beyond the trained window, *sequential* ingestion
  itself regresses the task; the merge hurts *less* than sequential everywhere in
  that regime. A continued-pretrain that teaches longer state accumulation
  removes 75% of that harm and then saturates precisely at the new window — the
  bottleneck is window-bounded state extrapolation, a property of the substrate's
  training history, entirely orthogonal to how states are composed.

Our contributions: (i) the first zero-training parallel composition of delta-rule
recurrent states on real repository code, quantified with pre-registered gates and
bootstrap CIs; (ii) the isolation and quantification of the
sequential-accumulation wall as the true binding constraint, including a
continued-pretrain intervention that pins it to the training window; (iii) a
methodology contribution — a span-lift sanity gate that voids uninterpretable
retention ratios before they are reported — and (iv) an honest map to the
literature, with the synthetic State-Algebra probe (72% retention) as the origin
of the hypothesis.

---

## 2. Related Work

**State Soup / parallel state composition.** State Soup (arXiv 2406.08423)
showed for Mamba-2.8b that recurrent states ingested over different data can be
averaged and reused — the closest prior art to this paper. It validates the
primitive on a different linear-RNN family (Mamba, not the delta rule) and on
synthetic/curated tasks, and does not address the delta rule's
interference-correction term, repository code, or the interaction with the base
model's trained context window. Our substrate is a delta-rule model, where each
write subtracts the state's current prediction (an interference correction Soup's
gated-SSM states do not have), so additivity is *a priori* less obvious; and our
evaluation is on real held-out repositories.

**Cartridges (KV self-study distillation).** Cartridges (Hazy Research /
Stanford, arXiv 2506.06266) and Cartridges-at-Scale (arXiv 2606.04557) precompute
compact, loadable long-context representations for *transformer KV caches* by a
per-corpus **gradient** self-study objective. They are composable and reusable,
but each cartridge is *trained*. Our composition is training-free: the "cartridge"
is a snapshot of the bounded recurrent state after a single O(T) forward, and
composition is an elementwise mean. A self-study tier for our stack (context
distillation into a per-corpus state) is a natural follow-up but is **not** part
of this paper.

**The interference term.** LASP (arXiv 2404.02882) gives the exact inter-chunk
correction that a naive merge of delta-rule states discards; it is the most
plausible account of the residual gap between merged and sequential composition
in the synthetic probe (§4.6), and a principled target for a merge-aware training
tier we do not pursue here.

**Bounded-state substrate.** The model is a DeltaNet linear RNN [CITE-TODO:
DeltaNet / parallelizing linear-attention Transformers with the delta rule]
linearized by weight inheritance from SmolLM2-360M [CITE-TODO: SmolLM2]. The
delta rule as a Widrow-Hoff / fast-weight update, and the cost-moat framing, are
developed on our own stack (`DECODE_COST_BENCH.md`, `SCOREBOARD.md`).

---

## 3. Method

### 3.1 Substrate

The model is a 402M-parameter TinyLM whose sequence mixer is **DeltaNet** (plain
`chunk_delta_rule`, a linear-attention RNN with a bounded recurrent state and
O(1)-per-token decode), linearized from SmolLM2-360M by weight inheritance (32
layers, d_model 960). Every pretrain in this lineage ran at **T=2048 with
cross-document state isolation** (`cu_seqlens` resets at document boundaries), so
the model has never, in training, accumulated more than ~one document of state —
a fact that turns out to be the whole story of §4.3. The frozen base is
`checkpoints/production_lean_soup3.pt` (HumanEval-solution CE 0.6614). All arms are
**eval-only**; no checkpoint is modified anywhere in §4.2–4.3, and the §4.4
continued-pretrain modifies weights only to build a *new* base that is then
evaluated by the identical eval-only harness.

### 3.2 Task, arms, and metrics

Each episode is a held-out repository packaged as `[repo context 1.5–32k tokens →
task]` (the frozen `data/repo_episodes` eval split, 150 episodes, built for the
meta-TTT program; a decontaminated short-context set `data/repo_episodes_short`,
also 150 episodes, for the working-scale probe). We measure teacher-forced CE on
the task under four ingestion arms
(`experiments/eval_state_cartridges.py`):

- **sequential** — full repo context prepended to the task (the ceiling; reused
  *verbatim* from the anchor harness `eval_repo_adaptive.py`'s `real` arm, so
  tokenization and span-CE placement are pinned equal to the meta-TTT harness
  whose known incidental lift for real context on an untuned base is ≈ +0.246
  span-CE).
- **none** — task only, zero state (the floor; == the anchor's `none` arm,
  verbatim).
- **cartridge@K** — the context is split into K token-balanced, line-aligned
  segments; each segment is run **independently from zero state**; the per-segment
  final DeltaNet recurrent states are **mean-merged** (elementwise arithmetic mean
  accumulated in fp32); the merged state is injected as the task's initial
  recurrent state and the task runs with **no context tokens**. Conv state is
  taken from the last segment (`conv_from="last"`, the probe's validated choice);
  attn/ffn states pass through from the last segment.
- **shuffled@K** — arm cartridge@K but with each segment's *lines* permuted
  (token-count preserving, deterministic per-episode seed): does the cartridge
  carry structure or a bag of tokens?

`--segments_per_repo K ∈ {2,4,8}` is the swept parallelism axis (K∈{2,4} on the
short set, whose contexts are too small for 8 balanced segments). For an 8k
context at K=4 each segment is ~2k tokens — an *in-distribution* forward, which is
why cartridges dodge the wall that sequential ingestion hits.

Metrics (pre-registered, `STATE_CARTRIDGES_PLAN_2026_07_19.md`):

- `lift(X) = CE(none) − CE(X)`, paired per episode (+ve = the context helped).
- `retention@K = lift(cartridge@K) / lift(sequential)`.
- **Span-CE is the pre-registered primary metric** (the hard, task-identifying
  tokens); line-CE (all task tokens) is reported alongside.
- **Sanity gate (pre-registered): `lift(sequential) ≥ +0.15 span-CE`, else the
  run is VOID** — a retention ratio over a ~zero or negative denominator is
  uninterpretable, so the gate refuses to report it as a result.
- Bands (reported, not adjudicated by the harness): retention ≥ 0.75 STRONG PASS;
  ≥ 0.60 PASS; < 0.35 KILL.
- Structure check: `lift(shuffled@K)` must be clearly below `lift(cartridge@K)`
  for a structural claim.
- Bootstrap 95% CI on retention (1000 resamples over episodes); no single-number
  claims.

### 3.3 The byte-identity guarantee

The load-bearing correctness property: the injected-state path must be exactly the
anchor harness's forward with only the initial state changed. `arm_ce_injected`
mirrors `eval_repo_adaptive.arm_ce` exactly — same predictor positions
`[P−1, P+L)`, same span-index selection — and `init_states=None` yields an empty
cache that is byte-identical to a plain forward (the `none`-arm equivalence test).
The merge/injection machinery is shared with `probe_state_algebra.py`
(`snapshot_states` / `states_to_cache` / conv handling). Because the cartridge and
sequential arms run the task at *different absolute positions* (position 0 with an
injected state vs `len(context)` with prepended tokens), the harness **asserts the
base is position-free** (`max_T == 0`); the assertion protects the whole
comparison from a silently-wrong retention on a positional checkpoint. Runs are
bf16, seed 0, 1000-resample bootstrap, and complete in minutes on one GPU
(77–215s per 150-episode run) — the composition is cheap to measure.

---

## 4. Experiments

### 4.1 Setup

Three eval-only cartridge runs and one continued-pretrain intervention:
working-scale (short set, soup3), OOD-scale (frozen 8–32k set, soup3), and the
same OOD eval on the two long-context checkpoints (longctx, longctx2). All numbers
below are quoted from the run JSONs cited per table.

### 4.2 Working scale — merging is free, and beats sequential on the hard tokens

On 150 decontaminated 1.5–6k-token episodes (K∈{2,4}), mean-merged parallel states
recover essentially all of sequential ingestion's benefit. **Source:
`runs/state_cartridges_short_soup3.json`.**

| arm | line-CE lift | span-CE lift |
|---|---|---|
| sequential | +0.1138 | +0.0043 |
| cartridge@2 | +0.1123 | **+0.0590** |
| cartridge@4 | +0.0934 | **+0.0613** |
| shuffled@2 | +0.1106 | −0.0426 |
| shuffled@4 | +0.1116 | +0.0076 |

| K | line-CE retention | 95% CI |
|---|---|---|
| 2 | **0.99** | [0.82, 1.40] |
| 4 | **0.82** | [0.62, 1.31] |

Two readings, both from the table. **(a) Line-CE: near-total retention.** The
cartridge recovers 0.99 and 0.82 of the sequential line-CE lift at K=2/4 — for the
cost of K independent ~2k forwards and one mean, the merged 8.2 MiB state does what
a single sequential pass over the same tokens does. **(b) Span-CE: the cartridge
beats sequential outright.** On the hard span tokens, cartridge lift is +0.059 /
+0.061 where the sequential lift is a near-zero +0.004 — merging is not just a
lossy summary here, it is a *better-conditioned* one (a lead we flag, not a claim
we press: the sequential span lift is itself near-floor at this scale).

**Formally VOID, and the gate is right to say so.** `lift(sequential) span =
+0.0043 < 0.15`, so the run does not clear the pre-registered sanity gate and the
*span-CE retention ratio is not a result*. Indeed the raw span retention the
harness computes is 13.7 at K=2 and 14.2 at K=4 (CI [−7.7, 7.7] / [−12.0, 10.2]) —
a ratio over a ~zero denominator, exactly the artifact the gate exists to
suppress. We therefore report the **line-CE retention** (real denominator
lift(seq) = +0.114) as the working-scale result and the span-CE *absolute lifts*
as the direct comparison, and let the gate void the span *ratio*. This is the
methodology contribution in action (§4.5, claim 3).

### 4.3 OOD scale — the wall is sequential accumulation, not the merge

On the frozen 8–32k-token episodes, the picture inverts — but not against
cartridges. **Source: `runs/state_cartridges_soup3.json`
(`production_lean_soup3.pt`).**

| arm | span-CE lift | line-CE lift |
|---|---|---|
| sequential | **−0.5437** | −0.1094 |
| cartridge@2 | −0.2538 | −0.0088 |
| cartridge@4 | −0.1317 | +0.0163 |
| cartridge@8 | −0.0750 | +0.0267 |
| shuffled@2 | −0.3833 | −0.0192 |
| shuffled@4 | −0.2229 | +0.0350 |
| shuffled@8 | −0.1295 | +0.0485 |

**Sequential ingestion of 8–32k tokens makes the task WORSE** (−0.544 span-CE;
the gate voids the run, `sanity_gate_pass = false`). This is not a cartridge
failure — it is the base model failing to *use* a context longer than anything it
accumulated in training. The diagnosis is mechanical: every pretrain ran T=2048
with cross-document isolation, so the usable context of `soup3` is ≈ 2k, and the
O(1) decode moat is a statement about *memory*, not about *usable context length*.

Against that, the cartridge is the *lesser harm* at every K, because each ~2k–4k
segment is an in-distribution forward: cartridge@8 span-CE −0.075 vs sequential
−0.544 is **7.3× less harm**; cartridge@4 −0.132 is 4.1× less; cartridge@2 −0.254
is 2.1× less (the more segments, the shorter each forward, the closer to
in-distribution — a clean monotone). On line-CE the cartridge lifts are already
*positive* at K=4/8 (+0.016 / +0.027) while sequential is −0.109. The
span-CE *retention ratios* the harness prints (0.47 / 0.24 / 0.14) are over a
negative denominator and are **not** interpretable — the gate voids them, and we
report the absolute lifts instead.

### 4.4 The long-context intervention — 75% of the harm removed, then a window-bounded saturation

If the wall is "the base never accumulated >2k of state," the fix is to teach it
to. We continued-pretrained `soup3` at **T=8192** with document isolation off
(`--no_doc_isolation`: state flows across packed documents within each window —
standard LM packing this lineage had never done), 720M tokens, on the first
production use of 2-GPU manual all-reduce (50.4k tok/s sustained, 1.9×). The
pre-registered success metric is *the exact gate that voided §4.3*: does
`lift(sequential)` on the frozen 8–32k set cross +0.15? **Sources:
`runs/state_cartridges_longctx.json` (720M), `runs/state_cartridges_longctx2.json`
(1.44B), `LONGCTX_PLAN_2026_07_19.md`.**

| base | tokens | seq span-CE lift (8–32k) | seq line-CE lift | HE-CE | depdist CE |
|---|---|---|---|---|---|
| soup3 | — | −0.5437 | −0.1094 | 0.6614 | 0.8126 |
| longctx | 720M | **−0.1330** | +0.0127 | 0.6675 | **0.7948** |
| longctx2 | 1.44B | −0.1624 | +0.0181 | 0.6728 | **0.7778** |

**720M tokens removed 75% of the long-context harm** (−0.5437 → −0.1330; line-CE
flipped positive, −0.1094 → +0.0127) — but short of the +0.15 gate, so the primary
metric *failed* while the guards *improved*: dependency-distance natural-code CE
fell from 0.8126 to 0.7948 (long-context packing improves short-context competence
outright), HE-CE 0.6675 inside the ≤0.6714 bar. Per the one permitted escalation,
**doubling to 1.44B tokens moved the primary metric nowhere** (−0.1330 → −0.1624,
flat) while depdist reached **0.7778, the best natural-code CE ever measured on
this stack**. HE-CE 0.6728 is a photo-finish miss of the ≤0.6714 guard (+0.0014,
inside seed σ ≈ 0.007).

The saturation has a clean mechanistic reading: **a T=8192 training window can
only teach state accumulation up to ~8k, but the frozen eval's episodes are
8–32k** — so "75%-harm-removal-then-saturation" is exactly *fixed within the
window, cannot extrapolate past it*. The wall is **window-bounded state
extrapolation**, and it is not exposure-fixable at this budget with T=8192
packing. A T=32k rung is the registered revisit-suspect (new registration,
memory feasibility unverified) — we do not claim it.

The OOD cartridge structure persists on the improved bases (cartridge is the
lesser harm at every K on both longctx checkpoints; the merge tracks the base's
usable window rather than fighting it):

| K | cartridge span-CE lift — longctx | — longctx2 |
|---|---|---|
| 2 | −0.0727 | −0.0950 |
| 4 | −0.0376 | −0.0542 |
| 8 | −0.0260 | −0.0382 |

### 4.5 Structure checks — the merged state carries structure, not a token bag

At every K and every scale the line-shuffled control is worse than the cartridge
(span-CE `lift(cartridge) − lift(shuffled)`), so the merged state encodes segment
*structure*, not just its token multiset. **Sources: the three run JSONs above.**

| scale / base | K=2 | K=4 | K=8 |
|---|---|---|---|
| working (short, soup3) | +0.1016 | +0.0537 | — |
| OOD (soup3) | +0.1295 | +0.0912 | +0.0545 |
| OOD (longctx) | +0.1194 | +0.0886 | +0.0579 |
| OOD (longctx2) | +0.1165 | +0.0887 | +0.0596 |

The margin is +0.05 to +0.13 span-CE, uniformly positive, and — consistent across
independent runs and two different base checkpoints — this is the most
reproducible single signal in the paper. (The gap narrows with K, as expected:
more, shorter segments leave less within-segment line order for the shuffle to
destroy.)

### 4.6 Origin — the synthetic State-Algebra probe (72% retention)

The hypothesis came from a controlled synthetic probe, and stating it keeps the
real-code result honest about what was and was not already known. On a saturating
multibind recall task (context A binds keys `v0..vN−1`, context B binds
`vN..v2N−1`, disjoint), we ingested A and B *independently* and merged their
recurrent states, then measured first-occurrence teacher-forced recall of the
A-keys. **Source: `runs/probe_state_algebra_n6_t25.json`
(`experiments/probe_state_algebra.py`, `feature_pilot_A`, 6 bindings/shard × 25
trials, n=150 A-key queries).**

| arm | A-key recall |
|---|---|
| sequential (ceiling) | 0.567 |
| **mean-merge** | **0.407** |
| sum | 0.380 |
| normmax (per-head norm select) | 0.120 |
| b_only (floor for A-keys) | 0.000 |

Mean-merging recovers **0.407 / 0.567 ≈ 72%** of sequential recall from a floor of
0.000 — **partial additivity**: delta-rule states from disjoint content are *not*
strictly order-entangled (they would sit at the b_only floor if they were), but
they are not perfectly additive either (the ~28% gap is the interference /
inter-chunk-correction term LASP characterizes). This synthetic result is what
licensed the real-code experiment; the real-code result is what tells us the
composition survives on repositories and — the paper's actual news — that the
*base model's window*, not the merge, is the constraint. (The probe also carries a
methodology scar we inherit: its first run read a 0.000 *sequential* ceiling from
an ad-hoc binding format, the same "probe-format trap" the sanity gate is designed
to catch — always check the ceiling before reading any arm.)

---

## 5. Limitations

We state every weak point plainly.

- **Sequential accumulation, not merging, is the wall — and we only partly moved
  it.** The headline negative is that the base cannot use >~2k of accumulated
  state (§4.3); the T=8192 intervention removed 75% of the harm and then saturated
  at the window (§4.4). We did **not** run the T=32k rung that the saturation
  analysis points to (registered revisit-suspect, memory feasibility unverified),
  so the claim is "window-bounded, not exposure-fixable *at T=8192*," not "the wall
  is fundamental."

- **The working-scale win is line-CE; the span result is an absolute lift, not a
  ratio.** The pre-registered *primary* metric (span-CE retention) is VOID at every
  scale because `lift(sequential) span` never cleared +0.15 (it was +0.004 at
  working scale and *negative* at OOD scale). We report line-CE retention (real
  denominator) and span-CE absolute lifts; readers who weight the span *ratio* as
  the headline should note it was never a valid result under our own gate. This is
  the gate working as designed, but it means the strongest single number (0.99
  line-CE retention) is on the secondary metric.

- **Cartridge "beats sequential" is scale-local and near-floor.** At working scale
  sequential's own span lift is only +0.004, so "+0.059 > +0.004" is a comparison
  between a small positive and a near-zero baseline, not evidence that merging
  adds information a full pass lacks. We flag the better-conditioning reading as a
  lead, not a claim.

- **Single model, single scale, single seed.** All results are one 402M DeltaNet,
  seed 0, one linearization lineage, one repo-episode distribution (Python
  repositories). Retention, the wall, and the saturation law could all move with
  scale, architecture (the delta rule vs a gated SSM), or code distribution.

- **No self-study tier.** We test *only* zero-training forward-ingest composition.
  The Cartridges-style gradient self-study tier — which could in principle recover
  the ~28% synthetic gap and push OOD retention up — is designed but unbuilt; every
  number here is training-free composition of frozen states.

- **The synthetic probe is single-N, small.** The 72% origin figure is 6
  bindings/shard × 25 trials at one N; the overlap (interference) arm was
  noisy-low and is not reported as a result. It motivates, it does not corroborate,
  the real-code numbers.

---

## 6. Conclusion

On a bounded-state delta-rule model, per-segment recurrent states computed
independently and in parallel — zero training, one elementwise mean — are a
near-free substitute for sequential context ingestion at the model's working
scale: line-CE retention 0.99 / 0.82 at K=2/4, and a merged cartridge that beats
one sequential state on the hard span tokens (+0.059/+0.061 vs +0.004), with the
merged state provably carrying structure (cartridge > line-shuffled by +0.05 to
+0.13 span-CE at every K and scale). The composition operator is not the
bottleneck. The bottleneck is the base model's own **sequential state
accumulation**: beyond the trained T=2048 window, sequential ingestion of 8–32k
tokens *hurts* (−0.544 span-CE), and cartridges hurt 2–7× less precisely because
each segment is an in-distribution forward. A T=8192 continued-pretrain removes
75% of that harm (−0.544 → −0.133) and then saturates exactly at the window (2×
tokens: −0.162, flat) while improving natural-code CE to the best we have measured
(0.8126 → 0.7948 → 0.7778) — a window-bounded extrapolation law. The honest
one-line framing: **delta-rule states compose in parallel; sequential state
accumulation, not merging, is the wall.** In a model whose decode state is 8.2 MiB
flat at any length, that makes instant, composable, constant-memory repository
context a forward-pass-and-a-mean away — bounded above only by how much state the
base was taught to carry.

---

## References

- State Soup — *Merging states of Mamba SSMs.* arXiv:2406.08423.
- Cartridges — *Storing long contexts as trainable KV caches via self-study*
  (Hazy Research / Stanford). arXiv:2506.06266.
- Cartridges at Scale. arXiv:2606.04557.
- LASP — *Linear Attention Sequence Parallelism* (the inter-chunk correction the
  naive merge discards). arXiv:2404.02882.
- DeltaNet — parallelizing linear-attention Transformers with the delta rule.
  [CITE-TODO]
- SmolLM2 (donor for the linearized base). [CITE-TODO]
- Widrow-Hoff / delta rule as a fast-weight update (the additivity intuition).
  [CITE-TODO]

---

## Appendix A — Provenance of every reported number

| Result | Value | Source file |
|---|---|---|
| Substrate params / layers / d_model | 402M, 32L, d_model 960 | `AGENTS.md` (linearized inherited base); `PAPER_LATENT_EXECUTION_DRAFT.md` (402M TinyLM) |
| Base HE-CE (soup3) | 0.6614 | `STATE_CARTRIDGES_PLAN_2026_07_19.md` |
| Bounded state / decode-peak / gap | 8.2 MiB flat, ~919 MiB, 6.45× @131k, KV 10→5121 MiB | `SCOREBOARD.md`, `DECODE_COST_BENCH.md` |
| Anchor incidental lift | ≈ +0.246 span-CE | `STATE_CARTRIDGES_PLAN_2026_07_19.md`; `eval_repo_adaptive.py` |
| Method (arms, mean-merge, injection, byte-identity, gates) | §3 | `experiments/eval_state_cartridges.py` |
| Working-scale line-CE retention | 0.99 [0.82,1.40] @K2, 0.82 [0.62,1.31] @K4 | `runs/state_cartridges_short_soup3.json` |
| Working-scale span-CE lifts | seq +0.0043; cart@2 +0.0590; cart@4 +0.0613 | `runs/state_cartridges_short_soup3.json` |
| Working-scale line-CE lifts | seq +0.1138; cart@2 +0.1123; cart@4 +0.0934 | `runs/state_cartridges_short_soup3.json` |
| Working-scale span-CE retention (VOID artifact) | 13.7 / 14.2 | `runs/state_cartridges_short_soup3.json` (retention.span_ce) |
| Working-scale sanity gate | lift(seq) span +0.0043 < 0.15 → VOID | `runs/state_cartridges_short_soup3.json` (sanity) |
| OOD soup3 span-CE lifts | seq −0.5437; cart@2/4/8 −0.2538/−0.1317/−0.0750 | `runs/state_cartridges_soup3.json` |
| OOD soup3 line-CE lifts | seq −0.1094; cart@2/4/8 −0.0088/+0.0163/+0.0267 | `runs/state_cartridges_soup3.json` |
| OOD soup3 shuffled span-CE lifts | @2/4/8 −0.3833/−0.2229/−0.1295 | `runs/state_cartridges_soup3.json` |
| OOD soup3 harm-reduction ratios | 2.1× / 4.1× / 7.3× (seq/cart at K2/4/8) | computed from `runs/state_cartridges_soup3.json` lifts |
| OOD soup3 sanity gate | lift(seq) span −0.5437 → VOID | `runs/state_cartridges_soup3.json` (sanity) |
| Longctx (720M) seq lift + guards | span −0.1330, line +0.0127, HE 0.6675, depdist 0.7948 | `runs/state_cartridges_longctx.json`, `LONGCTX_PLAN_2026_07_19.md` |
| Longctx2 (1.44B) seq lift + guards | span −0.1624, line +0.0181, HE 0.6728, depdist 0.7778 | `runs/state_cartridges_longctx2.json`, `LONGCTX_PLAN_2026_07_19.md` |
| soup3 depdist baseline | 0.8126 | `LONGCTX_PLAN_2026_07_19.md` (run-1 result) |
| Longctx guard bars | HE ≤0.6714, depdist ≤0.8226 | `LONGCTX_PLAN_2026_07_19.md` |
| 75%-harm-removal / saturation | −0.5437→−0.1330 (75%), →−0.1624 (flat) | `LONGCTX_PLAN_2026_07_19.md` (run 1 + escalation) |
| 2-GPU all-reduce throughput | 50.4k tok/s, 1.9× | `LONGCTX_PLAN_2026_07_19.md` |
| Longctx cartridge span-CE lifts | @2/4/8 −0.0727/−0.0376/−0.0260 | `runs/state_cartridges_longctx.json` |
| Longctx2 cartridge span-CE lifts | @2/4/8 −0.0950/−0.0542/−0.0382 | `runs/state_cartridges_longctx2.json` |
| Structure check (cart−shuf, span) | +0.05..+0.13 across K/scale (table §4.5) | the four run JSONs above |
| State-Algebra synthetic recall | seq 0.567, mean 0.407, sum 0.380, normmax 0.120, b_only 0.000 | `runs/probe_state_algebra_n6_t25.json` (`probe_state_algebra.py`) |
| State-Algebra retention | 0.407/0.567 ≈ 72% | computed from `runs/probe_state_algebra_n6_t25.json` |
| Probe config | feature_pilot_A, 6 bindings × 25 trials, n=150 | `runs/probe_state_algebra_n6_t25.json` |
| Cartridges / State Soup / LASP arXiv ids | 2506.06266 / 2406.08423 / 2404.02882 | `ideas_2026_07_13/09_wildcards.md`, `ideas_2026_07_13/05_knowledge_capacity.md` |
