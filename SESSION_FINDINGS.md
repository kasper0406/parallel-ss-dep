# SESSION FINDINGS — audited empirical log

> Chronological log of empirical results, **after adversarial metrology audit**.
> Every claim below is labelled PROVEN / CAVEAT / CORRECTED-OVERCLAIM against what
> the actual scripts + result JSONs support (audit: workflow `wf_d5ba5b87-0e2`,
> 2026-07-01). Numbers reproduce faithfully (no fabrication); the issues audited
> were baseline fairness, proxy metrics, undertrained operating points, and framing.

---

## 2026-06-28 .. 07-01 — cost-moat + recall arc (audited 2026-07-01)

Context: the session pivoted from "beat coding benchmarks" (judged unreachable —
token-poverty + linear-attn tax) to a **cost + adaptivity** thesis
(`NORTH_STAR_2026_06_30.md`). Below is what the experiments actually established.

### PROVEN (fair baseline / adequate measurement, on their proper axis)

1. **Constant decode memory (architecture-determined).** The lean linearized
   DeltaNet (`linearized_stage3.pt`, 32L×960d, **449M** params) decodes with a
   flat **8.2 MiB recurrent state** and ~919 MiB peak from L=256→131072, via true
   state-passing `prefill`+`forward_step`. A full-KV SmolLM2-360M's KV grows
   linearly to 5.1 GiB (peak → ~5.9 GiB). Peak-memory crossover ≈ L=8192; **6.45×
   peak advantage at 131k** (≈624× on the state itself), widening without bound.
   Batched: our peak scales with B only, the transformer's with B×L, so there are
   (batch, context) regions that OOM the transformer while ours serves.
   Source: `checkpoints/decode_bench/results.json`, `DECODE_COST_BENCH.md`.
   *Caveats:* holds ONLY for the lean config — the production WM stack has an O(T)
   decode buffer and is **not** constant-memory. Below ~4k tokens ours uses MORE
   peak memory (more weights). A **windowed / streaming / paged** transformer is
   also bounded-memory (see corrected overclaim #4).

2. **The linearized base retains coding-CE but has zero out-of-the-box long-range
   recall.** Measured this session on this base: HumanEval-solution CE **0.759**
   with **0% generated recall at every distance > 0** (100% at distance-0 — genuine
   floor, not a format artifact). Source: `eval_before_1key.json`,
   `results_summary.json`. *Caveat:* the DESCRIPTIVE fact is measured; the causal
   story ("linearization transferred knowledge but not recall because the donor
   used attention") is a plausible **inference**, not an ablation.

3. **Recall is cheaply teachable into the recurrent state without wrecking code-CE.**
   A ~197M-token recall cotrain lifted synthetic-needle recall **0%→100% in-window
   (L≤4096, single & 6-binding)**, leak-free / first-occurrence scored on generated
   output, at equal-or-fewer params. Length-generalizes robustly to **2× the train
   window (L=4096: 100%/96%)**. Source: `RECALL_COTRAIN_PROBE.md`,
   `checkpoints/recall_cotrain/`. *Caveats:* train stream and eval share the same
   `_build_body` generator (within-format); n=30/90 (100%@n=30 has a ~88% lower 95%
   CI bound); code-CE retention (+0.021) was **non-monotonic** (rose to +0.043 at
   160M, recovered only by the WSD decay) — interference is real, retention holds
   for the decayed ckpt only; CE is a next-token proxy, **not pass@k**.

4. **The recall eval protocol is sound** — leak-free, first-occurrence, identical
   elicitation across arms, scored on generated (not teacher-forced) output.

### CAVEAT (real but narrower / undertrained / proxy than first stated)

- **Hybrid dose-response (tax reducibility screen):** swapping 4/8/12 of 32
  DeltaNet layers for byte-inherited donor softmax attention lowered stage-2 CE by
  **14.9 / 29.0 / 32.1%** at equal-or-fewer params (`checkpoints/hybrid_sweep/`).
  BUT: stage-2-only, **undertrained** (baseline CE 1.349 vs its own ~250M floor
  1.005, vs e2e proof 0.759), and the monotone trend is **guaranteed by
  construction** (retaining N donor-attention layers → closer to the donor; N=32 IS
  the donor at 0.614; `ce_init` already ranks the arms before training). Absolute %s
  will shrink after stage-3 e2e (untested). Report against headroom, not raw %.
- **Batched throughput "1.7–1.8× win":** **eager-vs-eager** (both HF/naive, no
  compile / paged-attention / continuous-batching). Throughput is exactly what
  vLLM optimizes for the transformer → this number is unlikely to survive vs a
  production serving stack. The **memory / OOM-frontier** advantage (B vs B×L
  scaling) is the robust part; the throughput number is fragile.
- **`forward_step` "logit-tested":** cost-O(1) is empirically shown, but the
  equivalence test (`test_incremental_decode.py`) runs on the **wrong ckpt**
  (production 10L×896d use_memory RL model, not the benched lean config), threshold
  is **argmax ≥14/16 under ~0.19 bf16 drift** (not exact logit equivalence, not
  "≥15/16"), and **`prefill_state_only()`** (used for all batched numbers) is
  **untested** in-repo. Cost claims are unaffected (cost is state-value-independent);
  the argmax-equivalence of the exact benched config is asserted, not demonstrated.

### CORRECTED OVERCLAIMS (do not repeat these)

1. **"OURS beats both transformer configs at recall (L=4096 45% vs 38%, L=8192 35%
   vs 0%)."** — NOT a fair comparison. OURS was continue-trained on a stream from
   the **same `_build_body` the eval uses**; SmolLM2-360M is **zero-shot** and 24%
   smaller. full=0% @8192 is a **RoPE/training-window artifact** (max_pos=8192);
   window=0% is **by construction** (binding truncated out). OURS also **loses** to
   the zero-shot transformer at L=512 (62 vs 70) and 1024 (57 vs 62), and collapses
   to ~0–2% by 16k. **Accurate:** *at flat O(1) cost, a recall-cotrained bounded-
   state model retains 35–45% synthetic-needle recall past a 2048-window
   transformer's blind spot and past the full transformer's trained-length wall — a
   COST/REACH result, not a quality win over transformers. A fine-tuned or
   context-extended transformer would very likely match or beat it (it gets 100%
   single-key @4096 zero-shot).*
2. **"The tax is reducible ONLY by real attention, not a fancier linear cell."** —
   under-tested. The only param-matched comparison is baseline vs hybrid. wide_dhead
   was **not weak — it gave the LARGEST raw drop (−25.2%)**, just at +29% params;
   DeltaProduct was only tested at the weakest `num_householder=2`. No iso-param
   linear-cell arm and no higher-order DeltaProduct were run. **Accurate:** *at
   iso-param only the hybrid beat baseline; the fancier-linear-cell arms were param-
   inflated and confounded, so "a fancier linear cell can't help at iso-param" is
   untested, not established.*
3. **"Recall generalizes to natural code (strict cross-distribution held-out)."** —
   partly mislabeled. Only the arrow/eq **cue** variants and the **codeparrot
   cross-source** distractor row are genuinely held-out (codeparrot **decays to 48%
   @4096**); the int4/string value rows reused trained forms. "Natural" code is
   **filtered to remove conflicting literals / reassignment / shadowing** — the hard
   real-code recall cases are **untested**. **Accurate:** *recall generalizes across
   cue/value surface forms and to non-conflicting real-code filler (100/92%→2048,
   48%@4096 cross-source); conflicting/reassigning real code, string values, and
   distances >4k are open.*
4. **"A transformer lab would have to switch backbones to match the memory moat."**
   — false. Sliding-window / streaming / GQA / paged-KV transformers achieve
   bounded memory **without switching backbones** (our own `control_window2048`
   demonstrates it), at the same long-range-recall loss ours also has. **Accurate:**
   *the moat is "constant memory + high-concurrency throughput vs a FULL-KV
   baseline"; single-stream latency is a ~2.3× loss until ~131k.*
5. **"Cost moat + recall ⇒ a feasible competent bounded-state coding agent."** —
   two separate narrow results don't compose to that. Coding **pass@k is unchanged**
   this session, natural-code recall is untested, and the **adaptivity / learns-
   from-deployment pillar was not measured at all** (the doc concedes the surprise-
   gate may collapse to a no-op). **Accurate:** *the cost substrate is proven and
   synthetic recall is trainable; the competent-base and adaptivity halves are
   unbuilt bets (step-2 / step-3).*

### STILL UNPROVEN / UNTESTED (the honest gap list)

Iso-param linear-cell tax reduction; hybrid quality after stage-3 e2e; any FAIR
(task-matched / context-extended) transformer recall baseline; natural-code recall
with conflicts/reassignment; genuine cross-distribution generalization beyond
surface variants; stable ≥4× window recall (L=8192 noisy 0–83%, >16k collapses);
pass@k impact of the recall cotrain; argmax-equivalence of the exact benched config
+ `prefill_state_only`; production-serving throughput vs vLLM; the moat-preserving
sliding-window hybrid's quality; the entire adaptivity thesis.

### Metrology lessons (carry forward)

- When "we beat X", check X had **equal task exposure** and is run **in its trained
  regime** — train-on-eval-distribution vs zero-shot, and OOD-context foils, are the
  recurring trap here.
- Stage-2-only screens are **undertrained**; report vs headroom and don't quote raw
  %s as converged. A dose-response that is monotone **by construction** is not a
  discovery.
- CE retention ≠ capability retention (measure pass@k); check the **trajectory**
  (interference can spike mid-run and only decay away).
- A synthetic train/eval-matched task proves **teachability**, not **generality**;
  "natural code" that filters out the hard cases isn't the natural-code test.
- Equivalence/regression tests must run on the **exact config** being benched.

### Multi-day base-run verdict (audit)

**QUALIFIED YES** — justified as a feasibility-gated **build** of the bounded-state,
knowledge-inherited base (the two make-or-break risks — cost moat, recall
teachability — are the ones actually retired, on their proper axes). NOT justified
by the quality/generality claims, and **do not commit the global-attention hybrid**
(its −32% is a stage-2, construction-monotone screen and its config breaks the O(1)
moat). Before/alongside the run: (1) if any hybrid is in scope, first screen the
**moat-preserving sliding-window** variant + an **iso-param linear-cell** arm; (2)
add a **fine-tuned / context-extended transformer** recall baseline before any
quality-vs-transformer claim; (3) add a **pass@k** check + a **natural-code
(conflicting-literal)** recall probe. Scope the run strictly as "build the
bounded-state knowledge-inherited base + teach recall."

---

## 2026-07-01 — CODE-level audit (workflow `wf_06cd3923-be4`): 6-slice bug hunt over the uncommitted load-bearing experiments + adversarial verification + two corrective re-measurements

The earlier 2026-07-01 audit was CLAIMS-level; this one hunted for defects in the
code that produced the numbers. 6 parallel reviewers (cost-bench, recall-probes,
linearization, KD-pipeline, uncommitted trainer diff, cross-model CE harness) +
adversarial verification of every major finding + full pytest (779/781 pass; the
2 failures are a stale `test_line_selector` α-init assert [fixed] and a flaky
0.824-vs-0.85 convergence threshold in `test_state_readonly_thinking`).

### CONFIRMED + RE-MEASURED (numbers corrected in the source docs)

1. **6-binding recall row was mislabeled + inflated** (`scoreboard_longctx_cost.py::run_ours_quality`
   queries `keys[:keys_per_task]`; the "query each of the 6 keys" row ran with
   `keys_per_task=3` → only the earliest-bound v0–v2, n=90). Re-run with ALL 6
   keys (n=180): L4096 **96%→88.9%**, L8192 **47%→30.6%** gen (top1 63%→43.9%);
   in-window ~100% unchanged. `RECALL_COTRAIN_PROBE.md` corrected; raw:
   `checkpoints/recall_cotrain/eval_after_final_6key_all6.json` (rerun on
   `recall_cotrain.pt`).
2. **Transformer OOM-frontier was partly a prefill-harness artifact**
   (`decode_bench.py::bench_transformer_batched` did a ONE-SHOT (B,L) prefill while
   ours got chunked prefill; single try/except couldn't attribute the phase).
   Fixed (`chunk=` arg) + phase-attributed re-run: **(B=64,L=8192) NOT fundamental** —
   xf serves 1,313 tok/s @ 21,730 MiB (corrected cell: ours 4,244 tok/s @
   1,457 MiB = 3.2× tput at ~15× less mem — a STRONGER honest claim than
   "xf-OOM");
   **(B=256,L=2048) genuine decode-phase OOM** (eager HF DynamicCache; paged-KV
   would likely serve it); KV≥40 GiB cells remain arithmetically fundamental.
   `DECODE_COST_BENCH.md` corrected.

### CONFIRMED, fix applied (future-run protection; no reported number affected)

3. **Live-KD think-burst corruption**: the live-KD path clamps think ids to a
   stand-in token for the teacher forward and `valid_kd` masks only the burst
   positions, NOT downstream positions whose teacher context was corrupted; the
   trainer default `--think_burst_prob 0.5` would silently poison any default-flag
   live-KD launch (the committed A/B passed 0, so reported numbers are clean).
   Guard added in `train_lm.py` (mirrors the offline-KD force-to-0).
4. **`--keep_base_vocab` aliases think→EOS(=pad)**, which silently disables
   gate_calibration / latent_cotrain via their
   `thinking_token_id != pad_token_id` guards; `train_lm.py` now hard-errors
   on those two combinations. NOTE (post-review correction): latent_reasoning
   is NOT pad-guarded in HEAD code so it is excluded from the guard.
   **CORRECTION (2026-07-02, forensics):** an earlier version here said arm B
   "legitimately combined" latent_reasoning with keep_base_vocab — true of the
   LAUNCHER, false of the RUN-OF-RECORD: the final arm-B run never trained the
   latent aux (adapter proj bit-exact 0.0; no startup banner / reason() diags /
   TB tags; two earlier attempts ran it and died at step 620). The audit error
   was checking HEAD code instead of the run's own log — see the 2026-07-02
   forensics section below.

### CONFIRMED, open — **ALL FIXED later the same day** (5 Sonnet-5 subagents,
### each fix reviewed by the orchestrator + full pytest; see the fix list below)

5. **KD teacher-context mismatch (the big one for the planned KD base run):** the
   student trains with cross-document state isolation (doc_ids→cu_seqlens reset)
   but the live teacher forward AND both offline generators
   (`gen_teacher_logits{,_vllm}.py`) condition on the full un-isolated packed
   block — KD targets after every in-block doc boundary are conditioned on
   context the student structurally cannot see. This diluted the completed #101
   A/B (KD 2.076 vs CE 2.139 — the true KD lift is understated) and is baked
   into any offline store generated so far.
   **FIXED (2026-07-01):** doc isolation is now default-ON end-to-end —
   per-doc segment teacher forwards in BOTH generators (`--no_doc_isolation`
   opt-out; store format unchanged; e2e smoke: input_ids byte-identical,
   ~71% of top-k rows change at post-boundary positions) AND in the live
   trainer path (`_kd_teacher_forward_doc_isolated`), plus the shared
   consumer-side convention `_kd_valid_mask` masking KD at doc-final
   positions (target crosses the boundary) for both live and offline paths.
   Tests: `test_gen_teacher_doc_isolation.py` (20), `test_kd_doc_isolation.py`
   (17). **Existing offline stores predate the fix — regenerate before the
   base run.** The #101 A/B number stands as recorded (understates KD).
6. **Cross-tokenizer tax comparison is unit-mixed**: "+0.145 (SmolLM2) vs +0.276
   (Qwen) — tax GROWS with donor" compares nats/token under DIFFERENT tokenizers.
   Per-byte (measured 3.118 vs 3.343 bytes/tok on the HumanEval solutions):
   ratio 1.90→**1.78**. Direction survives; quote per-byte next time.
   **FIXED (2026-07-01):** `humaneval_solution_ce.py` now reports `ce_per_byte`
   (nats/UTF-8-byte, the cross-tokenizer-comparable number) alongside per-token
   CE, warns loudly on a non-cosmo2 tokenizer, and has `--dtype {bf16,fp32}`
   applied to BOTH arms (default bf16; historical ours numbers were fp32).
   Verified: reproduces the documented SmolLM2-135M 0.798 per-token CE exactly
   (0.7982; 0.2560 nats/byte).

### Notable minors — status after the fix pass

- ~~`top1` first-digit-only~~ **FIXED**: scoreboard prints `top1(1st-digit)` and
  adds a strict teacher-forced `tf_exact` (every gold token argmax-correct) to
  BOTH arms; string-value top1 space-merge **FIXED** in `eval_recall_variants.py`
  (in-context gold-token encoding; verified ' z' id 1892 case).
- ~~`run_xf_quality` blanket except~~ **FIXED**: failures counted as `n_failed`,
  EXCLUDED from the accuracy denominator, loud warnings + `nF` table column.
- ~~`--n_hybrid` silently ignored~~ **FIXED**: `--hybrid_layers` default now "",
  layers derived from `--n_hybrid` (default still resolves to [7,15,23,31];
  launcher-compatible), both-explicit mismatch hard-errors.
- ~~`linearize_qwen.py` stale 10L layer-select~~ **FIXED**: computed from donor
  depth (24L → [0,3,5,8,10,13,15,18,20,23]; the old list is exactly the 32L
  spacing, confirming it was stale not wrong).
- ~~Ours' redundant bench autocast~~ **FIXED**: removed from all 6 OURS timed
  paths after argmax-equivalence verification; re-measured ms/tok
  **14.5 → 13.3–13.5 (−7–8%)**, still flat (single-stream deficit ~2.3×→~2.1×).
  Note: raw logits DO shift (RMSNorm fp32-under-autocast → bf16, max|Δ|~0.44) —
  argmax sequences identical in the benched regime. See DECODE_COST_BENCH.md note.
- ~~`d_ff`/`tie_embeddings` not persisted~~ **FIXED**: both saved in ckpt cfg
  (mid-eval + final), consumed by `build_model_from_ckpt` (tie_embeddings) and
  the `--load_ckpt` continuation path (d_ff auto-fill; tie mismatch warns —
  `store_true` has no "unset" sentinel). Old ckpts load unchanged.
- ~~pass@k top_p mismatch~~ **FIXED**: `eval_hf_humaneval.py` defaults
  top_p=1.0 (off) matching our full-distribution sampler; `--top_p` flag kept.
- Also from the fix pass: HF generator `--num_workers` default 2→0 with
  hard-error (the old default produced trainer-unconsumable stores); vLLM
  vocab-filter boundary now `len(tok)` not padded student_vocab (teacher
  padding rows incl. the student [THINKING] slot id 151665 can no longer leak
  into stores); `gen_recall_variants.build_natural_pool` hard-fails on load
  failure (`--allow_synth_fallback` opt-in, rows then labeled
  synthetic_fallback); flaky `test_state_readonly_thinking` threshold 0.85→0.60
  with rationale (FLA atomic-add nondeterminism; observed ON range 0.668–0.824
  vs OFF 0.094–0.285; NOTE the probe is seed-sensitive across DIFFERENT seeds —
  3/5 alternative seeds show weak ON/OFF margins at 500 steps, worth knowing
  before citing it as load-bearing).

### What did NOT survive review

- "keep_base_vocab corrupted Phase-1 arm B" — refuted (documented deliberate
  alias; no aux losses enabled there).
- Everything else in the cost/recall/linearize slices held up: timers, memory
  metrology, leak-freeness, seed separation, KD off-by-one conventions, CE
  target alignment, weight-copy correctness all verified clean.

**Net metrology verdict:** the strategic conclusions (cost moat real, recall
teachable, token-poverty, tax-grows-with-donor) ALL SURVIVE, but three headline
numbers were inflated in our favor (6-binding tail, one OOM-frontier cell, the
tax ratio) and one pipeline defect (KD doc-isolation) understates our own next
lever. Bug-class to watch: OUR arm gets an accommodation the baseline doesn't
(chunked prefill, early-bound keys) — the recurring asymmetry trap.

---

## 2026-07-02 — FORENSICS: why don't the features help on code? (workflow `wf_7b7baabc-011`, 46 agents: log/ckpt autopsies + impl review + detection-power analysis + a NEW stratified probe; all high-confidence findings adversarially verified)

The owner challenged the "features are orthogonal to code" verdict. The forensic
answer: **the verdict rests mostly on runs where the mechanisms were inert,
absent, or structurally invisible to the eval — plus one genuinely-measured
null (latent-on-code).** A new probe shows the first statistically-real
iso-token feature WIN on natural code, in exactly the strata every prior eval
was blind to.

### The negative experiments, autopsied

- **Phase-1 32L arm B (the "net-tax on a competent base" headline): all four
  mechanisms numerically inert at eval.** Ckpt `phase1_ab_B.pt`: PKM αL=−0.0005,
  value rows 1.004× init (dead; ε=0.18/φ=0.11 still active — 3000-step windows
  in a 1900-step run; WSD LR hit literally 0.00 at step 1900); FiLM α≤1.5e-4
  (100–1000× below committed scale; zero-grad first 300 steps by K-warmup);
  WM read_alpha frozen 0 by flag + copy gate σ≈0.0026 (moved −6.00→−5.95);
  latent adapter proj EXACTLY 0.0 — **the latent aux never ran in the
  run-of-record** (two earlier attempts died at step 620; final run has no
  banner/diags/TB tags). Arm B paid the taxes (K=3 at 2.4× wall-clock with α≈0,
  gist 0.1, addr 0.2, gate-entropy with floor≥0.95) for zero function → "B
  slightly worse than A" measures taxes, not mechanisms.
- **10L arm B nuance:** PKM α committed (+0.461, monotone — shallow IS more
  receptive) but the VALUE TABLE was only ~13% drifted (row 1.13 vs 2.5–3.9
  committed refs) = amplified near-random values; latent never configured
  (launcher lacks the weight flag). The recorded "even COMMITTED features don't
  help" overstates commitment. HE-CE delta B−A=+0.0030 is a statistical null
  (paired bootstrap CI [−0.0009,+0.0070]).
- **Iso-token ablation (2000 steps):** valid COST result (mechanisms ~free),
  weak BENEFIT result: WM structurally inert (v4 mix has NO recall streams; 0
  addr-aux firings; W_proj purely weight-decaying), gate floored 0.95+ all run
  (warmup 20000), latent at full weight only the last 40%, PKM mid-trajectory
  and still rising at cutoff, eval capped at T=512.
- **v18 (the one genuinely-engaged run):** mechanisms committed (PKM row
  3.70–3.94, latent proj fro 110, copy path trained) and PKM/FiLM are strongly
  load-bearing WITHIN-MODEL (see probe below). Its HumanEval regression was
  mix-dilution (already documented). But within-model kill-gates measure
  co-adaptation, not counterfactual value → **a clean engaged-features
  iso-data A/B at real length still does not exist anywhere in the repo.**
- **No A/B in the record has a seed error bar (all n=1)** — every ±0.003–0.018
  CE feature delta, in either direction, is unverdictable at unknown variance.

### Detection power: the evals could not have seen success

- Every code eval behind the verdicts lives at ≤512-token context (per-source
  probe T=512; HE-solution CE median ~175 tokens, max identifier reuse distance
  ~540, ZERO >1024) — entirely inside the recurrence-is-perfect regime.
- WM's validated niche (N≥48 bindings / >512 distance) occurs in **0 tokens of
  HumanEval** and ~0 of the SFT corpus, but in ~1.5–2% of ALL tokens (11–12% of
  identifier reuses) of natural ≥800-line Python files, where 97.5% of NAME
  positions sit in the ≥48-live-identifier saturation regime. A perfect WM
  cannot move HumanEval.
- HumanEval-164 greedy noise (±3–4) exceeds the expected effect of everything
  except a fully-engaged PKM at the optimistic end (+2–4 passes via the repo's
  own CE→pass slope) — which is why the +0.04..0.14 CE PKM effect kept reading
  as zero there. PKM was detected EVERY time it actually engaged (v7.1 toggles,
  v18 claw-back, Phase-C pkm_off −5).
- **Latent-on-code is the one genuine, well-measured null**: the per-position
  probe had power, found the ceiling (~10–13% of positions × +0.3 logp, flat in
  R, gate anti-aimed) → aggregate ≤0.03–0.04 nats even oracle-placed. Stands.

### NEW probe (experiments/probe_depdist_stratified_ce.py, runs/depdist_probe/)

Per-token CE on 600 natural codeparrot files (≥3072 SmolLM2 tokens; 1.84M
scored tokens/arm), stratified by identifier reuse distance ×
identifier-vs-other, with paired mechanism toggles:

- **First statistically-real iso-token feature win on natural code:** 10L
  features-vs-lean pair (same base/data/seed/steps): B−A = **−0.0045 CE overall
  (paired bootstrap p<1e-4)**, monotone with dependency distance (−0.0016
  first-occurrence → −0.0102 at 1024–2047), features better in 11/12 strata —
  while TIED on HE-solution CE. The standard evals are blind to exactly the
  strata where the features act. (And 10L_B was itself mid-bootstrap → likely
  an underestimate.)
- **v18 within-model:** pkm_off +0.1198 CE total, concentrated at
  first-occurrence (+0.26) and far identifier strata (+0.25 at 1024–2047);
  film_off +0.70; all_off +0.98. Fully-engaged mechanisms carry major function.
- **WM confirmed structurally zero on code CE in every deployed ckpt** (read
  frozen; copy head requires a data mask evals never pass; force-engaging the
  untrained gate HURTS +0.007..+0.12). WM-on-code = untested, not refuted.
- **Honest cap (verifier's full-attention control):** SmolLM2-135M shows nearly
  the same CE-vs-distance curve (0.68→3.26 vs ours 0.67→3.42) → ~93% of the
  far-reuse hardness is intrinsic token rarity, NOT bounded-state forgetting.
  The recall-feature upside on natural code is real but ~1% relative at these
  budgets. The features' larger payoff remains the >2k-context / agentic regime
  (the north-star axis), not aggregate code CE.

### Verdict per mechanism

| mechanism | verdict | basis |
|---|---|---|
| PKM | **works when engaged** (+0.04..0.14 CE, sample-efficiency lever); every "no effect" came from mid-bootstrap runs | v7.1/v18/Phase-C detections; arm-B autopsy |
| FiLM | engaged=load-bearing within-model (film_off +0.70 on v18); iso-token effect small-positive at far strata; arm-B null = cold-start stall (no toll-payer), NOT the LineSelector init trap (CPU test: α=0×random-W opts in fine when first-order signal exists) | probe + CPU gradient test |
| WM | **never tested on code** — output channel structurally zero in every negative run (frozen read + mask-gated copy); niche absent from HumanEval by census, present in natural long files | wiring analysis + census |
| latent | genuine null ON CODE (well-measured ceiling); arm-B "test" trained it at exactly zero | probe_gate_placement + autopsy |

### Recipe rules (so the next feature run is a real test)

1. **Engagement kill-gates in-run:** assert at step ~N/3 that αL/row/copy-fire/
   reason-loss have moved (and the latent banner printed at step 1) or ABORT —
   a feature run that ends inert is a wasted launch, not a negative result.
2. **Curricula ≤ ~40% of run length; never let WSD hit 0 at curriculum end**
   (leave a constant-LR plateau after bootstrap completes).
3. **Converged-base attach needs toll-payers per feature:** PKM has one (floor/
   ε/value-LR — schedule it INSIDE the run); FiLM needs α warm-start or an
   α-floor; WM needs read_alpha unfrozen with its floor curriculum (or a
   trained, eval-time copy trigger — supervise the copy gate at natural-code
   reuse positions ≥256 distance, the 2–3.4-nat regime the probe exposed);
   latent needs its OOM at step 620 root-caused first.
4. **≥2 seeds (or a variance estimate) for any ≤0.02-nat claim.**
5. **Adopt dependency-distance-stratified CE as the standing feature dev
   signal** (the probe is now in-repo); stop deciding feature questions on
   HumanEval greedy or ≤512-token CE.
