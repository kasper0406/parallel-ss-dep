# WHY PKM / WM / latent thinking give only a marginal benefit on MBPP code

Intuition (correct): a small model SHOULD gain hugely from parametric memory
(PKM), context recall (WM), and extra sequential compute (latent thinking).
Reality: on MBPP/HumanEval the combined benefit is marginal (+1/+3). This
documents the experiments that found WHY — it is NOT that the mechanisms are
broken; it is that MBPP at 287M is bottlenecked by something none of them
address.

## The experiments (all on `latent_code_adapteronly.pt`, the co-trained base)

### 1. Per-token thinking ceiling is low and depth-independent
`probe_gate_placement.py` / `probe_retrieval_channels.py`: a latent burst helps
only ~10-13% of code positions, by ~+0.3 logp, FLAT across R=1/2/4/8. (Arithmetic
chains, by contrast, scale strongly with R.) → code-token prediction is mostly
recall/pattern; iterating the trunk adds little.

### 2. No addressable retrieval content for code (WM ruled out, cheaply)
`probe_oracle_retrieval.py`: ORACLE retrieval (best-of-16 buffer slots, peeking
at the label) scored frac-helpful 0.249 — but a matched-norm RANDOM-injection
control scored 0.298 (HIGHER). oracle−random ≈ 0. The "oracle lift" was pure
max-of-K selection noise. The WM buffer has no addressable code signal → no
addressing scheme (trained DKV included) can help. PKM is the only positive
channel (+0.052 frac), modest and already exploited by the base.

### 3. Failures are runnable but fundamentally wrong (not near-misses, not broken)
`probe_failure_bottleneck.py` (120 MBPP, no-think greedy): 27 pass, 81 partial,
7 exec_error, 5 syntax_error. 87% of failures RUN, so it is not a code-formation
gap. But the partial pass-fraction is 0.06 — those runnable functions pass ~6%
of tests → plausible-but-fundamentally-wrong logic, not off-by-one.

Concrete examples: `lucid_number` returns `n` (should return a list of ludic
numbers); `longest_chain` calls itself with no base case (infinite recursion);
`degrees_to_radians` uses `180/π` instead of `π/180`. Also a non-trivial rate of
BAD TESTS (`first_repeated_char` is correct but the test asserts `== "None"` the
string) — so the benchmark even undercounts the model.

### 4. THE CRUX — failures are KNOWLEDGE-bound, not SEARCH-bound
`probe_knowledge_vs_search.py` (120 MBPP): for each FAILED problem, compare under
the model the mean per-token logp of the GOLD solution vs the model's own wrong
greedy output:

| | failures |
|---|---|
| SEARCH-bound (logp_gold > logp_own — knowledge latent, decoding misses it) | **0 (0%)** |
| KNOWLEDGE-bound (logp_gold < logp_own — model prefers its wrong answer) | **93 (100%)** |

mean logp_gold = −2.26, mean logp_own = −0.73 → the model rates its OWN WRONG
answer ~1.53 nats/token (~4.6×) MORE likely than the correct solution, on EVERY
failure. (Style confound rebutted by §3: the model emits a different *algorithm*,
e.g. returns an int where a list is required — not a restyled gold.)

## Conclusion

**MBPP failures at 287M are a base-KNOWLEDGE gap, not a thinking/memory/search
gap.** The model doesn't know the algorithm and confidently confabulates a wrong
one. No inference-time mechanism can supply knowledge the weights don't contain:
- latent thinking refines a distribution centered on the wrong answer;
- WM retrieves from a context that doesn't contain the algorithm;
- PKM would need the algorithm in its 262k slots, but the 287M model trained on
  limited data never learned it.

This is a scale × data (Chinchilla) problem. It also explains every prior result:
the mechanisms ARE load-bearing where the bottleneck matches their function
(WM +11pp on MQAR recall; latent +0.65-0.80 on arithmetic chains; PKM −5 on
HumanEval where parametric knowledge IS the lever), and marginal where it does
not (MBPP general code).

## Implications (where the benefit actually lives)

1. **To SHOW the mechanisms' huge benefit, use a matched-bottleneck task.** The
   "small super-coder with PKM/WM" wins on tasks bottlenecked by memory/recall:
   long-context code (recall a symbol defined 1000s of tokens back), repo/agentic
   settings, long dependency chains. MBPP is the wrong probe — it's short and
   knowledge-bound. Build/borrow a long-context-code recall eval and show WM is
   load-bearing there.
2. **To raise the MBPP headline, inject knowledge, not thinking**: more/better
   pretrain + distillation (put the algorithms in the weights / PKM), or a
   larger model. The cheapest knowledge-injection lever is distillation from a
   strong teacher on the failing problem families.
3. **Stop spending inference-mechanism effort on MBPP** — the addressable surface
   is ~0 there (0/93 search-bound). Reserve thinking/WM for matched tasks.

Probes (reusable): `probe_gate_placement`, `probe_retrieval_channels`,
`probe_oracle_retrieval`, `probe_failure_bottleneck`, `probe_knowledge_vs_search`,
`probe_thinking_passk`.

## ADDENDUM — "why didn't it learn it in training?" (capacity vs exposure)

The model SAW the failing facts (degrees_to_radians 22× with the correct
`math.pi/180` in-target; longest_chain is training row 1) and still fails. Two
controlled experiments (`probe_pkm_capacity.py`, param-matched dense vs PKM
fact-memorization) decompose why:

**Part 1 — raw capacity is NOT the bottleneck.** A 76k-param model memorizes
~10k random facts at 100%; the cliff is 10k–20k facts. Scaled to 287M that's
~tens of millions of facts — MBPP needs a few thousand. So the model has orders
of magnitude more storage than required. At matched params PKM ≈ dense for
storage (no magic capacity advantage).

**Part 2 — EXPOSURE/data-efficiency IS the bottleneck.** Holding facts well under
capacity (N=4000) and varying exposures-per-fact:

| exposures/fact | DENSE acc | PKM acc |
|---|---|---|
| 2  | 0.05 | 0.07 |
| 10 | 0.05 | 0.07 |
| 22 | 0.07 | **0.12** |
| 50 | 0.13 | **0.29** |
| 150| 0.95 | 0.93 |

A precise fact needs **~100+ gradient exposures** to lock in. At the 22 the model
actually got, retention is ~7–12%. Rare coding algorithms appear far fewer times
than that in the training mix → the long tail is structurally under-exposed.
**This is the root cause: not capacity, not thinking — under-exposure of
long-tail facts** (compounded by the ~6% broken / 70% unverified SFT targets).

**PKM's real role:** at low exposure PKM is 1.7–2.2× more sample-efficient than
dense (22 exp: 0.12 vs 0.07; 50 exp: 0.29 vs 0.13) — it acquires rare facts
faster. So "it should be a huge benefit to have PKM" is partly right: PKM helps
the long tail acquire, but the benefit is bounded because EVERYTHING still needs
many exposures and PKM doesn't change that.

**The real, cheap levers (boost real coding, use the insight):**
1. **Targeted repetition / tail up-sampling** — over-sample the rare/failing
   problem families so their facts cross the ~100-exposure threshold. Cheap,
   directly attacks the root cause.
2. **Data cleaning** — drop the ~6% verified-broken targets, verify the ~70%
   no-test ones (the model learned degrees_to_radians from malformed examples).
3. **Route tail facts through PKM** (2× sample-efficient) + give them exposure.
Dense scale also works but is the expensive path; the under-exposure levers are
much cheaper and are the honest first move.

Probe: `probe_pkm_capacity.py`.

## ADDENDUM 2 — the deepest layer: stability–plasticity, and why PKM doesn't escape it

Validating the exposure lever ON THE REAL MODEL (`probe_exposure_lever.py`):
continue-train `sft_baked_pure` on CLEAN gold solutions of 40 failing problems
(~80 exposures each), eval the trained problems, a disjoint failing CONTROL set,
and a previously-SOLVED set (forgetting check):

| training mode | TRAIN (fixed) | failing-CONTROL | SOLVED-control (forgetting) |
|---|---|---|---|
| full fine-tune (all params)        | 0→18 | 0→0 | 30→**8** |
| PKM only (addressing + values)     | 0→15 | 0→0 | 30→**6** |
| PKM values only, addressing FROZEN | 0→**0** | 0→0 | 30→**30** |

Findings:
1. **Learnable, but zero generalization.** Clean repeated exposure DOES fix the
   specific trained problems (0→18, train-loss→0.001) — so it's not raw capacity
   — but failing-CONTROL stays 0→0: pure memorization, no transfer to unseen
   problems.
2. **Catastrophic forgetting is the real wall.** Cramming 40 facts destroyed
   ~22/30 previously-solved problems (full AND PKM). The binding limit is not
   storage (ample) but **plasticity-without-interference**.
3. **PKM does NOT escape the dilemma.** PKM-only forgets as badly as full FT,
   because PKM is read at EVERY token through SHARED learned addressing — storing
   a new fact shifts the query/sub-keys and re-routes all other facts. Freeze the
   addressing (values-only) and forgetting vanishes (30→30) but so does learning
   (0→0). Learning a fact REQUIRES the addressing plasticity that CAUSES the
   forgetting. Same stability-plasticity dilemma as dense weights.

**This is the mechanistic answer to "it should be a huge benefit to have PKM/WM,
why isn't it":** the current PKM is not an interference-free, growable memory —
it's a densely-read, shared-addressing layer with the same forgetting limit as
dense weights. The "huge benefit" requires a memory where adding a fact
ALLOCATES A FRESH/DEDICATED slot without shifting existing routes — i.e. a
growable sparse store with per-fact slot allocation, or a NON-PARAMETRIC kNN
retrieval (append the fact, no gradient, no interference). That is the concrete
architectural direction the evidence points to for a small super-coder.

Probe: `probe_exposure_lever.py` (modes: full | pkm | pkmval).

## ADDENDUM 3 — non-parametric retrieval: clears forgetting + consumption, but only for near-EXACT matches

`probe_knn_oracle.py` (kNN-LM: key = last pre-lm_head hidden, value = next token;
interp p = λ·p_kNN + (1-λ)·p_LM with retrieval-CONFIDENCE-gated λ). On 40 failing
MBPP problems (sft_baked_pure, baseline 0/40):

| datastore | gate | pass |
|---|---|---|
| ORACLE (gold of the eval problems IN store, 2.5k keys) | conf_scale 0.15 | **23/40** |
| REALISTIC (1000 DISJOINT problems' gold, 62k keys) | conf_scale 0.15 | 1/40 |
| REALISTIC (disjoint) | conf_scale 1.0, λ_max 0.5 (loose) | 0/40 |

- **The consumption barrier is NOT fatal** (oracle 0→23): a confidently-wrong
  model CAN be overridden by retrieval when the answer is retrievable and λ is
  gated on retrieval confidence. Non-parametric retrieval also structurally
  escapes the forgetting wall. Both of the agent's risks cleared.
- **BUT cross-problem near-neighbor retrieval does NOT transfer** (realistic
  0-1/40, both gates). Loosening the gate to force distant neighbors in only
  added noise. So kNN-LM helps ONLY when the datastore holds the (near-)exact
  continuation — it is retrieval-as-memorization-at-scale, NOT
  compose-a-novel-solution-from-similar-ones.

## ADDENDUM 4 — coverage-datastore test: the apparent win was LEAKAGE (fair-baseline catch)

Tested the "small model + LARGE datastore" lever (`probe_knn_oracle.py` mode
`corpus` / `corpus_clean`), datastore from distill_corpus (147k magicoder+
codefeedback solutions), eval = the same 40 failing MBPP, d_min coverage logged:

| datastore (20k solutions) | pass | near-exact step coverage (d_min<0.1) |
|---|---|---|
| `corpus` (raw distill_corpus) | 11/40 | 0.50 |
| `corpus_clean` (exact eval problems REMOVED) | **1/40** | 0.10 |

The raw 11/40 was **pure leakage**: distill_corpus (magicoder/codefeedback)
*contains the exact MBPP problems* — every checked flipped problem had its exact
prompt + gold in the store (verified). De-leaking (drop entries whose normalized
prompt matches an eval prompt or whose gold contains the eval gold prefix)
collapses it to 1/40 — identical to the disjoint-MBPP realistic test, and
near-exact coverage drops 0.50→0.10. So "scale the datastore" helps ONLY by
containing the exact problem (memorization/leakage), NOT by generalizing from
similar solutions. The coverage lever is refuted as a path to NOVEL-problem gains.

## VALIDATION & CORRECTIONS (2026-06-13, adversarial agent + B1 run) — read this before trusting the strong-form claims below

A background validation agent + a gentle-update experiment (`probe_exposure_lever`
with replay+KL) found the strong-form synthesis OVER-CLAIMS. Corrections:

1. **"0/93 search-bound, model rates wrong answer 4.6× over gold" is a
   GREEDY-ARGMAX ARTIFACT.** `probe_knowledge_vs_search` computes `logp_own` on
   the model's OWN argmax decode (near per-token max by construction) vs
   `logp_gold` on an arbitrary alternative → `logp_gold < logp_own` is nearly
   tautological. Proper rank test: gold tokens are **62% top-1 / 82% top-5 /
   88% top-10** reachable; pass@8 recovers **~9%** of greedy failures. So failures
   are **predominantly — NOT exclusively (not 100%)** — knowledge-bound; there is
   a ~10–15% search-recoverable tail. [Re-measure with a pass@k≥50 sweep.]
2. **"Can't add knowledge without catastrophic forgetting" is RECIPE-bound, not
   fundamental.** Naive continue-train forgot 30→8. Gentle update (full FT +
   replay 1.0 + KL 0.5, `probe_exposure_lever ... full 1.0 0.5`) → 30→**16**
   (forgetting HALVED, −22→−14) while TRAIN still 0→18. So anti-forgetting works
   partially already; LoRA / EWC / lower-lr / higher-replay are untried and
   likely improve it further. The forgetting wall is softer than ADDENDUM 2
   stated.
3. **"Fundamentally composition-bound at 287M" conflates scale-vs-data and
   over-reaches.** ADDENDUM 2's own "clean exposure flips 0→18" shows these
   failures are LEARNABLE (under-exposed / mis-learned from the ~6%-broken,
   ~70%-unverified SFT), not a hard synthesis ceiling. The honest framing:
   **under-exposed + dirty-SFT + decoding-fragile, predominantly knowledge-bound,
   ON THIS BASE/EVAL** — distinguish from a true 287M wall by re-running the
   bottleneck probes on a cleaner-SFT and on the 708M base.

**DEFENSIBLE CORE (survives scrutiny):** on short, knowledge-bound MBPP code-gen
with this SFT base, inference-time depth/recall/retrieval have little addressable
surface; non-parametric retrieval helps only near-exact (the de-leaked kNN
11→1/40 is the best-controlled result and stands); each mechanism is load-bearing
on its matched bottleneck. Good for routing effort. The BINARY/strong rhetoric
below (0/93, 4.6×, "every feature bounded by the same wall, none can manufacture
composition") outruns the data — read it as the weaker, base/recipe-scoped claim.

## FINAL SYNTHESIS — the one root cause under everything

At 287M the model is **composition-bound**: it can reproduce knowledge that is
DIRECTLY PRESENT (in weights via enough clean exposure, or in a datastore via
near-exact retrieval) but cannot SYNTHESIZE a novel algorithm it doesn't already
know and can't retrieve verbatim. Every mechanism is bounded by this same wall:
- latent thinking → helps only the iterated-computation slice (arithmetic), flat
  on code composition;
- PKM/WM (parametric) → can't add knowledge without catastrophic forgetting;
- kNN-LM (non-parametric) → adds knowledge without forgetting, but only the
  near-exact kind; doesn't compose.

So "it should be a huge benefit to have PKM/WM/latent thinking" is true only on
the bottlenecks those mechanisms match (recall, exact retrieval, iterated depth),
which the composition-bound coding benchmarks don't primarily stress. The levers
for REAL coding gains, in honest order:
1. **Base capability** (more params / more+cleaner training) — the only thing
   that buys composition. Expensive but it's the real bottleneck.
2. **Datastore that COVERS the task distribution** — a 287M model + a LARGE code
   datastore can match a bigger model on problems whose solutions resemble seen
   ones (the oracle 23/40 shows consumption works at coverage). Viable real-world
   strategy where most tasks are variations of solved ones; won't solve genuinely
   novel problems. (This is where to take the non-parametric memory next, if
   pursued: scale the datastore, measure coverage vs pass.)
3. **Reserve latent thinking / WM for their matched bottlenecks** (multi-step
   computation, long-context recall) — real but narrow on general coding.

Probe: `probe_knn_oracle.py` (modes: oracle | realistic).
