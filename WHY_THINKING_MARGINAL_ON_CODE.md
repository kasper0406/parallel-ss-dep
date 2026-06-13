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
   stated. SWEEP (added after): replay3.0+KL1.0+lr5e-6 → 30→19 but TRAIN only
   0→5 (plasticity tax) — so full-param continue-train has a stability-plasticity
   TRADE-OFF; no recipe cleanly adds facts AND keeps all old. **The clean fix is
   NOT a better continue-train recipe — it's a FULL RETRAIN on the combined
   (cleaned-old + new-verified) corpus** (old data present → nothing forgotten),
   i.e. E1 rejection-sampling self-distillation as a full SFT pass. Continue-
   training was the wrong frame for knowledge addition.
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

---

## ADDENDUM (2026-06-13): WM on its IDEAL niche — addressing is the wall

The WM×latent "cooperation" build (M0 plumbing + Stage A: attach fresh DKV +
mem_alpha, freeze trunk, train ONLY WM addressing + mem_alpha) was tested on the
regime WM is SUPPOSED to own: long-context recall. Findings:

1. **Single-binding recall has ZERO headroom.** Base `sft_baked_pure` = 100%
   no-think recall at every in-context distance (64→512); think_rate 0. A delta-
   rule state trivially holds ONE never-overwritten binding. WM literally can't
   improve 100%. (The "thinking corrupts recall" v1 finding was a different,
   over-thinking ckpt.)

2. **Multi-binding recall (the realistic MQAR analog) HAS headroom.** "Assign N
   vars, print the queried one" — base no-think recall N=8 → 15%, N=24 → 5% vs
   100% single-binding. The delta-rule state saturates holding many simultaneous
   bindings. This is where WM should win. (`gen_multibind_recall.py`.)

3. **Trained WM addressing barely helps, and the cooperation channel is inert.**
   Stage A (600 steps, fresh DKV + mem_alpha, trunk frozen) on N∈{8,12}:
   recall 8% → 10.7% (**+2.7pp**, vs the +15pp kill-gate → FAIL). coop-off
   (mem_alpha→0) == WM-on (Δ +0.0pp): ALL of the lift is the direct WM→logits
   injection; the latent-feedback cooperation (mem_alpha·wm_inj) adds NOTHING
   (mem_alpha grew only 0.10→0.13).

4. **Localized to the READ/addressing side (`probe_wm_recall_addressing.py`).**
   At a forced think at the recall position: the binding's value is in the buffer
   (WRITE 100%), but the read query lands on the correct slot only **1.7%** of
   the time, with **1.6%** of attention mass on it and a diffuse read overall
   (top-1 mass 5%). The trained W_q/W_k cosine addressing on a FROZEN trunk
   cannot find the right slot.

**WHY (the deep one):** the frozen trunk's hidden states do NOT encode
content-addressable `(which-variable, its-value)` structure. A thin post-hoc
addressing layer (2 matrices, 600 steps) can't conjure addressability the
underlying representations don't have. This is the project-wide "post-hoc
features are inert because the trunk fit the data WITHOUT them" pattern —
now MEASURED on WM's ideal niche, not inferred. It also explains why PKM/WM/
latent modules added after pretrain stay decorative: the trunk never learned to
emit the representations they need.

**Implied fix (next test):** WM must be load-bearing DURING pretrain so the trunk
learns to write addressable keys/values — i.e. co-train the trunk (the #55–58
pretrain-wiring direction), not bolt WM on after. The constructive test: unfreeze
the trunk in Stage A and check whether the read sharpens (mass-on-binding ↑) and
recall jumps well above +2.7pp.

Probes: `eval_stage_a_killgate.py` (WM-on vs full_off, forced think),
`probe_wm_recall_addressing.py` (write/read/readout localization),
`gen_multibind_recall.py` (the headroom-bearing probe).

### Fix-test (2026-06-13): fine-tuning addressing does NOT install content recall

Ran Stage A (train WM addressing + mem_alpha on multibind, trunk frozen) on the
CO-TRAINED v10 base (DKV baked in over 7B tokens) to test "does a co-trained
trunk make the hiddens addressable":
- Addressing probe: top-1-on-binding 0.0% (unchanged), peakiness 0.70 → 0.90 —
  the fine-tune made the read MORE confidently wrong (sharper onto the recency/
  positional slot), not redirected to the queried binding.
- Kill-gate: recall ~0.8% — CONFOUNDED (v10 is a raw pretrain ckpt, can't follow
  the "# instr\n program → answer" format; recall floor ~0). The kill-gate on v10
  is uninformative; the addressing probe is the clean, format-independent signal.

Conclusion: neither (frozen non-co-trained trunk + addressing FT) nor (co-trained
trunk + addressing FT) yields content-addressable retrieval of the queried
binding. A short addressing fine-tune reinforces the query-independent prior; it
cannot install an objective the representations were never shaped for. The fix is
NOT post-hoc addressing training — it is making content-recall (multibind/MQAR)
a PRETRAIN objective so the trunk co-adapts its hiddens to be content-addressable
AND the read query is shaped by a loss that rewards finding the queried key.
That is a fresh multi-hour pretrain (resource decision), not a quick fine-tune.

---

## UNIFYING SYNTHESIS (2026-06-13): a mechanism helps iff training demanded its function

Pulling PKM + WM + latent thinking together, one principle explains all three:

> **Each mechanism helps exactly to the degree its TRAINING OBJECTIVE demanded
> its FUNCTION — and the deployment workload actually has that bottleneck.**

| mechanism | its function | was that the training objective? | is it the deployment bottleneck? | result |
|---|---|---|---|---|
| **PKM** (product-key parametric store) | store/recall facts | YES — facts reduce next-token CE directly, everywhere in text | YES — the 287M model is knowledge-bound | **load-bearing** (pkm_off −5/−50% on HumanEval) ✓ |
| **WM** (content-addressable recall) | retrieve a specific bound value | NO — general text only needs RECENCY, which the delta-rule recurrence already gives | rarely (real code seldom holds many live bindings to recall) | **inert** — read collapses to recency, redundant with the recurrence ✗ |
| **latent thinking** (sequential computation) | multi-hop iterated reasoning | PARTIALLY — only on the arith/reasoning co-train, OOD for code | rarely (per-token code is recall, not iteration; ~10% of positions, flat in R) | **marginal** ~ |

**The mechanisms are not broken.** Gradient flows, the plumbing is correct
(M0 review confirmed it). They are inert because the *training never created the
bottleneck each one relieves* — except PKM, whose bottleneck (knowledge) the LM
loss creates automatically because facts are everywhere. WM's bottleneck
(many-simultaneous-binding recall) and latent thinking's bottleneck (multi-hop
iteration) are rare in general text AND rare in the deployment task, so the loss
never pressured the trunk to build the representations those mechanisms need
(measured: WM's read is recency/positional, never content-addressable — see the
addressing addendum above).

**Why the user's intuition ("should be a HUGE benefit") and the result both
hold:** WM/latent thinking WOULD be huge on a workload bottlenecked on
memory/iteration. The actual workload (general code at 287M) is bottlenecked on
KNOWLEDGE, which PKM already covers. So their marginal benefit is *correct given
this workload*, not a failure of the idea.

**Actionable corollary (the real fix):** to make WM or latent thinking
load-bearing, the bottleneck must exist in TRAINING — bake tasks whose function
IS the mechanism (multibind/MQAR content-recall for WM; multi-hop chains for
latent thinking) into PRETRAIN, so the loss demands the representations. This is
*why* PKM works out-of-the-box and the others don't, and it is a stronger claim
than "co-train the modules": co-training is necessary but not sufficient — the
co-training LOSS must contain the bottleneck (v10 co-trained WM still went
recency because the pretrain loss never required content recall).

CONSTRUCTIVE CHECK (in flight): a small DeltaNet+WM trained FROM SCRATCH on MQAR
(where retrieving the queried key IS the objective) — does its read become
content-addressable (val_hit/either_hit high)? If yes, it proves the mechanism is
sound and the only missing ingredient is the objective. (The project already has
the recall-side result: +11.1pp WM recall on saturated MQAR.)

### CAPSTONE (2026-06-13): the model routes AROUND WM unless WM is the ONLY way

The unfreeze fix-test (train trunk+WM+mem_alpha together on multibind) was meant
to prove "co-adapt the trunk → WM becomes addressable." It did the OPPOSITE, and
the failure is the deepest finding of the whole investigation:

- Train loss → 0.000 (the trunk MEMORIZED the training set).
- Heldout recall 5% (WORSE than frozen Stage A's 10.7% — overfit), read still
  diffuse (2.5% on-binding), Δ(WM-on − full_off) = +0.0pp, mem_alpha pushed DOWN
  0.10→0.088.

**Why:** the whole multibind program (≤626 tokens, ≤12 bindings) fits in the
local context window, so the TRUNK can solve recall directly (memorize / attend
within the recurrent state). Given any path that works, gradient descent took the
trunk path and left WM idle. **The model routes around an auxiliary mechanism
whenever the primary path (trunk/recurrence) can do the task — even under
co-training.**

**This unifies the entire project history.** Every post-hoc feature was inert
because the trunk could already fit the data. The ONE place WM was load-bearing
(MQAR K=128, +11.1pp) is the ONE place the trunk provably COULD NOT: (a) keys are
random every batch → NOT memorizable, and (b) K=128 > the recurrent state's
capacity → the recurrence CANNOT hold them. Both conditions are necessary.

**Sharpened fix (supersedes "bake content-recall into pretrain"):** for WM to
become load-bearing, training must contain a sub-task that is BOTH
(1) capacity-exceeding (more simultaneous bindings than the linear-RNN state
holds) AND (2) non-memorizable (fresh random content per instance), so WM is the
ONLY path to low loss. Plain "mix recall in" is insufficient if the trunk can
memorize or fit it. And on the DEPLOYMENT side: WM helps real code only where the
task has a sub-problem the recurrent state genuinely can't hold AND that varies
per instance — rare in general code at 287M, which is why WM's benefit there is
correctly marginal, not a bug.

Probes/ckpts: `latent_code_cotrain.py --wm_on [--unfreeze_trunk]`,
`probe_wm_recall_addressing.py`, `eval_stage_a_killgate.py` (full_off arm).

### ROOT CAUSE AT THE SOURCE (2026-06-13): v10's pretrain recall stream was SINGLE-binding

The decisive discovery: v10 (the co-trained pretrain) ALREADY mixed a recall
stream into the loss at 13% (`configs/pretrain_mix_v10.yaml`: longctx_recall_train
0.09 + synthetic_memory 0.04) specifically "so the co-trained WorkingMemory has a
saturating-recall gradient." But that stream is `data/longctx_recall_train.jsonl`
— SINGLE-binding recall, which a DeltaNet solves at 100% no-think (the binding
fits in the recurrent state). So v10's recall task was NOT capacity-exceeding →
the trunk routed around WM during pretrain → v10's read learned recency (0% on
the queried binding). The "make WM load-bearing" data-side fix was in place but
used the WRONG task.

THE FIX (`configs/pretrain_mix_v11.yaml`): swap the single-binding recall stream
for CAPACITY-EXCEEDING multi-binding recall (`data/multibind_recall_pretrain.jsonl`,
N∈{8..32}, 50k fresh instances; `experiments/gen_multibind_recall.py`). At N=24/32
the recurrent state saturates → predicting the queried value FORCES the trunk to
make its hiddens WM-addressable (WM is the only low-loss path). This is the one
genuinely-untested lever; every retrofit/fine-tune onto an already-trained trunk
failed (a 600-step fine-tune can't re-shape committed recency representations).

Status: v11 config + 50k-example stream are BUILT and committed. Launching the
fresh v11 pretrain is ~20 GPU-h (v10 was 7B tok / ~20 h) — a deliberate resource
decision. Open caveat: WM going load-bearing in v11 would prove the MECHANISM is
fixable at the source, but transfer to HumanEval is separate (real code is
knowledge-bound; PKM already covers that). v11 tests the mechanism, not
necessarily the coding headline.
