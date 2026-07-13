# AGENTS_HISTORY.md — archival appendix to AGENTS.md

> This file holds everything removed from `AGENTS.md` during the 2026-07-13
> consolidation that still has archival value: the chronological findings
> sections, superseded claims **with their correction notes attached**, the full
> pretrain/SFT/RL run history, and per-date engineering logs. It is organized
> newest-relevant-first by theme, then dated.
>
> **These are HISTORY, not current instructions.** Where any claim here conflicts
> with `AGENTS.md`, `AGENTS.md` wins. Read this for the "why" behind a resolved
> decision, or to avoid re-running a dead experiment.

---

## Superseded headlines (corrections attached — the corrected claim is in AGENTS.md)

### The "16/164 = 9.8% project best" HumanEval arc — SUPERSEDED

> **SUPERSEDED — re-measurement correction (2026-06-28).** The "16/164 = 9.8%
> project best" headline does NOT reproduce under a same-config eval. A
> matched-harness re-run clusters the headline ckpts at **~13–15/164 with SFT ≈ RL**
> (the SFT→RL gain ≈ 0); "16" is one draw from a 13–17 greedy-noise band. Greedy
> HumanEval-164 is too noisy as a dev signal at this scale — use temp pass@k / a
> bootstrap CI. See `project_humaneval_config_artifact` and `STRATEGY_2026_06_28.md`.

Original log (2026-05-23): Headline (full arc): SFT v7 8/164 → Phase C SFT
10/164 → grader-RL v2 step-300 **16/164 = 9.8% HumanEval pass@1**, project best.
Phase C pretrain is the new architectural baseline (5.28 B tokens,
Chinchilla-complete for 287 M, strict per-source CE win over v7.1 pretrain). RL v2
added the KL-to-reference penalty to `train_rl_grader.py`.

Headline trajectory table (as originally claimed, now corrected to ~13–15
SFT≈RL):

| stage | HumanEval | Δ |
|---|---|---|
| SFT v7 (v7.1 base, replacement retrieval, v6 WM-gist) | 8/164 | — |
| Phase C SFT (Chinchilla base + additive + trunk gist) | 10/164 | +2 |
| RL v1 step-100 (peak, then collapse) | 14/164 | +4 |
| RL v2 step-300 (KL-stable, then "best") | 16/164 (9.8%) | +2 |

> **CORRECTION (2026-06-28):** this trajectory conflates eval-config changes with
> capability. Same-config re-measurement collapses it to a ~13–15 cluster with
> SFT ≈ RL. Don't treat ≥1-point deltas as real. (2026-06-16 same-session sweep:
> v12 SFT 8, **Phase C SFT (dirty) 14 = robust best**, Phase C SFT drop-broken 9,
> RL v11c 8, RL v2_step300 13 off / 14 on. Naive data hygiene backfires: dropping
> "verified-broken" distill rows HURT 14→9.)

### "Latent thinking lifts code" / feature-mechanisms-as-headline — SUPERSEDED

- "8→11/164 first thinking lift on code" — within ±3 greedy noise, never
  temp-confirmed.
- Latent-CoT distillation "+0.12 logp" — n=963 stratified probe found zero
  code-CE benefit in every difficulty bucket (the n=64 edge was self-caught noise).
- "Fix = a bolt-on learned op-selector" — a learned answer-only selector FAILS
  (0/8 at K6, α decays); only fixed-gather (= oracle) or process supervision works.
- Feature mechanisms (FiLM/PKM/WM) as a code-capability/headline lever — two
  controls agree the effect is ±0.005 CE = noise. Keep as harmless free-riders on
  from-scratch runs / synthetic probes only. (Phase-1 A/B headline also
  over-reached: 32L arm-B mechanisms were inert/absent; deltas n=1 statistically
  null — void as orthogonality evidence.)

### Cost-moat / "beats transformers at recall" — SCOPED DOWN

> **CORRECTIONS (2026-07-01).** (1) "Beats transformers at recall" retracted —
> that number is train-on-eval-distribution vs zero-shot. (2) Bounded memory is
> NOT unique to recurrent backbones (windowed/streaming/paged transformers get it
> too); honest moat = "constant memory + high-concurrency throughput vs a FULL-KV
> baseline"; single-stream latency is ~2.3× worse until ~131k. (3) The lean base
> reads 0% gen_acc at every distance on `SCOREBOARD.md` — the quality half of the
> moat is the OPEN target; the COST half is fully demonstrated.

---

## Deprecation cleanup (2026-05-30) — the mechanisms that were removed

A cleanup pass removed several deprecated/confusing mechanisms:

- **Thinking measurement standardized on LATENT thinking.** Canonical primitive
  `experiments/thinking.py::latent_think_logp`. The previous DISCRETE-token append
  (`[THINK]*K`) is the validated `mode="token"` baseline that does NOT help (0.09
  vs 1.00) — calibrating against it taught the gate to suppress thinking. Don't
  reintroduce discrete-token thinking measurement.
- **Gate-calibration polarity fix (2026-06-13):** was training `σ→1{Δlogp>0}`
  (EMIT where a think helps — backwards); fixed to `P(think)=σ(−gate_logit)→1{Δlogp
  >0}`, and baseline lp0 fixed to the SAME left-padded prefix as lpR. This bug was
  live in the v10 all-features pretrain.
- **Process-reward aux loss REMOVED** — `experiments/process_reward.py` deleted.
  (Original mechanism: SFT-only aux giving trunk+WM gradient through the
  consequence of thinking; extra forward over `[prefix, K*THINK_ID]`; loss
  `mean(log p_before − log p_after)`. Probe on `sft_phase_c_combined.pt` confirmed
  the "thinks aren't productive" baseline: mean Δlogp −0.165, only 30.5% of
  high-gate positions benefit.)
- **`--mem_write_only_at_think` (FIX A) REMOVED end-to-end** — empirically
  rejected (see the FIX-A section below).
- Deprecated no-op SFT flags removed: `--wm_future_pred_weight`,
  `--wm_future_pred_T`, `--future_emb_T_max`, `--future_emb_T_ramp_frac`.
- `launch_pretrain_v2_thinking.sh` removed (referenced flags `train_lm.py` never
  exposed).

Also removed/superseded think-modules that were once bulleted as "validated" and
are now either off-path or subsumed by the exec-trace latent program:
`--use_think_adapter` (per-Block 2-layer MLP + α at think positions, tests
`test_think_adapter.py`), `--gate_mode soft` soft-mixture decode
(`generate_soft_mixture`, ~2× per step, `test_soft_mixture_decode.py`),
`--think_index_emb_size N` per-position think-index embedding
(`test_think_index_emb.py`), `--stochastic_gate` gate-as-policy-variable in
`train_rl_grader.py` (`test_stochastic_gate.py`). All were inert when attached
post-pretrain; co-training from day 1 was the only path, and the working thinking
primitive is latent, not these discrete-token modules.

---

## Pretrain run history (2026-05-17..18) and Phase C (2026-05-21..23)

- **v4** (30L×576d + FiLM(2,28) K=3 + WM + gate, `launch_pretrain_mix_v4.sh`, 9300
  steps / 2.13 B tokens) — final VAL ppl **5.89**, the deep-trunk baseline.
- **v5-pkm** (30L×576d + PKM-v5 + FiLM K=3) — ppl 7.02 at ~step 4500; PKM-
  utilization probe revealed **97% of value rows at random init** (4% slot
  coverage, 1% residual magnitude). The "win" was the trunk; PKM inert. Triggered
  the v7.1 bootstrap-fix package. Don't reuse this PKM recipe.
- **v6-shallow** (10L×896d + 5 dense FiLM) — killed ~step 2800; at iso-step
  tracking 5–7% ahead of v4. Shallow-wide hypothesis validated.
- **v7.1-pkm-film** (10L×896d + 5 dense FiLM K=3 + PKM-v7.1 + WM + entropy-aux
  gate, 9300 steps) — final VAL ppl **5.83**, strongest pretrain then; beat v4 in
  18% less wall time (13.3 h vs 16.3 h). PKM end-of-run αL=+0.346, row=2.542.
  Per-source PKM-toggle Δ always positive (wikipedia +0.099, cybernative +0.063,
  code_exercises +0.046, bigvul +0.044). Ckpt `pretrain_mix_v7_pkm_film.pt`.
- **v7.1-pkm-xattn** (sister, cross-layer attention instead of FiLM) — final VAL
  ppl **5.94**; within noise of film, film cheaper-per-step. (xattn no_grad-pass-1
  fix equalised step cost with FiLM K=3.)
- **Persistent unsolved (then):** pretrain-only HumanEval was 0/50 on every ckpt.
  The capability bottleneck was diagnosed as post-pretrain (SFT+RL) and, later,
  as token-budget.
- **Phase C** (`launch_pretrain_phase_c.sh`, 20.6 h) — continuation of v7.1-pkm-
  film 2.13 B → **5.28 B tokens** (Chinchilla-optimal for 287 M) + v7 trunk
  multi-horizon gist loss. Strict win over v7.1 on all 8 per-source CEs (~0.17 CE
  mean; cybernative −0.29, python_codes −0.24, bigvul −0.21). In-run VAL ppl
  5.83→4.90. PKM value-row norm 2.54→3.58. The gist loss's backward tripped an
  AOTAutograd ViewMeta-replay SymInt bug; workaround
  `torch._functorch.config.view_replay_for_aliased_outputs=False` in
  `speed_knobs.py`; upstream PyTorch fix root-caused at `~/ml/pytorch` branch
  `fix-viewmeta-replay-garbage-shape` (commit 78e6245).
- **v18** (2026-06-19, data-hygiene fix + arXiv + day-1 PKM/WM/latent, 4.75 B tok,
  VAL ppl 5.21) → SFT → HumanEval **6/164** (< v12's 8 < Phase C's 13). Root cause:
  finite-budget MIX DILUTION (magicoder −0.25 better, but codeparrot +0.20 and
  wiki +0.23 worse — smaller budget spread across more sources/objectives), NOT
  "mechanisms are dead weight". HE-solution CE v18 1.056 vs Phase C 0.969.

Shallow-wide validated recipe (kept in AGENTS.md): `--n_layers 10 --d_model 896
--n_heads 14 --d_head 64 --feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9"
--feedback_self_k 3`. Sister xattn form:
`--feedback_xattn "0:5,6,7,8,9;1:5,6,7,8,9;2:5,6,7,8,9" --feedback_xattn_form
film_sigmoid --feedback_xattn_heads 8`.

---

## Distillation + thinking-token arc (2026-05-19..27)

### What worked: distillation + future-emb + synthetic memory (first non-zero)
**Combined SFT v1** (`sft_v7_pkm_film_combined.pt`) → HumanEval **11/164**, first
non-zero on the codebase. Three stacked changes vs the prior 0/164 ckpt:
1. **Qwen 3.6 AWQ distillation** (`distill_solutions.py`, vLLM, ~38k (problem,
   CoT, code) pairs from MBPP/LeetCode/Magicoder/CodeFeedback). New grader loaders:
   `mbpp_all`, `mbpp_plus`, `mbpp_combined`, `leetcode`, `super_combined`,
   `magicoder_oss`, `codefeedback`, `distill_corpus`. **Critical eval flags for
   distilled ckpts:** `--prompt_style sft_comment --extract_code_block`.
2. **Future-embedding prediction aux** (`--future_emb_loss_weight`): position t
   predicts input embedding at t+T_eff (1→8 ramp) via `1−cosine`. Attacks
   early-commitment.
3. **Synthetic memory tasks** (`gen_synthetic_memory_tasks.py`, 12.5k, 5 families:
   var_binding, chain_arithmetic, list_index_recall, dict_lookup,
   multi_step_arithmetic).

### FIX A (write-only-at-think) — rejected
`WorkingMemory(write_only_at_think=True)` masks the write-gate to large-negative
at non-think positions. Inference A/B favored it (1 vs 3), but SFT'd from pretrain
WITH FIX A (combined v2) scored **11→9 (loss of 2)**. Root cause: the `[THINKING]`
token has ONE learned input embedding → a burst of 8 thinks gives 8 correlated
hidden states (median cos +0.146 vs +0.060 at emit; effective rank ~210 vs ~560);
FIX A fills the buffer with this low-rank pool. REMOVED 2026-05-30.

### Retrieval-as-input + WM-gist supervision (2026-05-20)
The gate's think decision triggers a WM lookup whose result IS the next input
embedding (`inputs_embeds`, `WorkingMemory._last_injection`,
`generate_with_retrieval_as_input`). Ablation (retrieval-as-input generator):

| ckpt | baseline | wm_off | pkm_off | both_off |
|---|---|---|---|---|
| v1 (no retrieval-as-input) | 11 | +1 (decorative) | −6 | — |
| v5 (lexical target) | 10 | 5 (−5) | 3 (−7) | 2 (−8) |
| v6 (gist target) | 9 | 4 (−5) | 1 (−8) | 3 (−6) |

Finding: **retrieval-as-input — not the supervision target — made WM
load-bearing** (gist-vs-lexical flat). Two reasons HumanEval can't see a gist
effect: (1) ablation zeroes the injection so `wm_off` partly measures "think
mechanism broken" — fix: ablate by mean vector (`eval_longctx_recall.py
--wm_ablate mean`); (2) HumanEval is short-context so WM's long-range value is
structurally invisible — right probe is `eval_longctx_recall.py`.

### Long-context recall — thinking corrupts recall (2026-05-20)
`eval_longctx_recall.py`, 600 tasks, var_binding, per-distance:

| distance | v1 (plain think) | v5 (lexical) | v6 (gist) |
|---|---|---|---|
| 512 | 20% | 100% | 61% |
| overall | 74.0% | 99.8% | 84.8% |
| think_rate | 0.36 | 0.012 | 0.23 |

Finding: **thinking corrupts long-range recall; damage scales with think volume**
— any think token stepping the DeltaNet recurrence perturbs the binding the linear
RNN state carries. A gist and a precise pointer are in tension. Motivated
state-readonly thinking (β=0 at think, now standard).

### v7 — additive α-gated injection + trunk gist (2026-05-20)
Fix B: `input[think] = think_embed + α·retrieval` (learned scalar
`retrieval_input_alpha` init 0.1, no WD) so a bad retrieval can't overwrite trunk
state. Fix C: multi-horizon windowed gist (`--wm_gist_horizons "16,64,256"`) moved
to TRUNK heads, WM precision-only. Tests `test_trunk_gist_loss.py`,
`test_retrieval_as_input.py`.

### Grader-RL trajectory (2026-05-21..23)
- **v1** (800-step GRPO): 14/164 at step-100 (+4), plateaued step-200 (13), still
  14 at step-300, **catastrophic collapse ~step-350** when ponder cost bit and
  depth dropped 120→30, taking the output format down (reward→0, all
  syntax_error). Diagnosis: no KL penalty.
- **v2** (KL-stable): `--kl_coef 0.05`, frozen ref, dropped ponder, LR 5e-6→2e-6,
  PPO clip 0.2→0.1, τ 0.9→0.7, capped 400 steps. Climbed 14→15→16 at
  step-100/200/300, KL bounded 0.05–0.10. (The "16" is the superseded headline
  above.)
- Decode-speedup: `model._film_bypass=True` around generate loops (~2×).

### Gate-calibration aux + pretrain wiring (2026-05-27)
Premise: every post-pretrain attempt to make thinking productive regressed (SFT
v8–v11: 1–5/164; DPO v1/v2: 9 and 12/164; discovery RL on v2_step300: 16→13). Fix
has to be in pretrain. New aux `compute_gate_calibration_loss` wired into
`sft_code.py` and `train_lm.py` (`--gate_calibration_weight`). Smoke
(`launch_pretrain_smoke_thinking.sh`, +1000 steps): gate IS miscalibrated
(gc(tgt1=0.88, σ=0.29, Δlogp=+2.9) — thinking would help at 88% of uncertain
positions but σ=0.29); gate learns the right direction (σ 0.29→0.76); thinking is
productive at high-σ (Δlogp +3..+8) but under-used; cost VAL ppl +3–5% (trunk
competes for capacity). Five latent bugs hit+fixed during the smoke: compile+extra-
forward shape mismatch (→ `--no-compile`), b=14 OOM (→ b=8 grad_accum=14), forward
returns (logits, gist_loss) tuple, pretrain y contains thinking_token_id (→ mask
to −100), `_last_gate_logits` overwritten by extra forwards (→ snapshot before).

### Engineering bugs caught & fixed (2026-05-19 audit)
- **sft_code control-flow bug**: modern-load path's flags silently overwritten by
  a re-running `else: build_model_from_ckpt`. Combined-SFT-v1 trained WITHOUT FIX
  A despite the launcher passing it. Fixed; test
  `test_sft_code_loading.py::test_modern_path_honors_mem_write_only_at_think_flag`.
- **eval_bracket_structure reload bug**: `mem_write_only_at_think` not passed from
  cfg → re-eval with FIX A off. Fixed (now moot, FIX A removed).
- **gate_floor/emit_threshold saturation** in `train_rl_grader` rollout:
  `gate.clamp_min(gate_floor) ≥ emit_threshold` trivially True when gate_floor ≥
  emit_threshold → never-think. Rule: `gate_floor < emit_threshold`. Test
  `test_rl_grader_gate_floor.py`.
- **future_head_state_dict saved but never loaded** — broke continuation. Fixed.

### Open architectural finding: thinking gate is temperature-fragile
At τ=0 greedy, think_rate ≈ 0.30 on HumanEval; at τ=0.9 RL sampling it collapses
bimodal (≈0 or ≈1 depending on gate_floor vs emit_threshold). Don't expect
train-time selectivity to survive RL rollouts without robustification.

---

## LATENT-SPACE THINKING WORKS (2026-05-28) — the original breakthrough

After discrete-token thinking never helped (0/80 on every arithmetic rung), the
user redirected to latent-space thinking, max bandwidth, not CoT. Mechanism
(`latent_think.py`): at an appended think slot feed the trunk's own continuous
hidden back as the next input for R iterations, state-readonly (β=0). Validated on
pointer-chase `f^K(s)`: floor → 1.00; needs exactly R=K steps (per-hop 100%);
token feedback FAILS (0.09) → bandwidth essential; learns from final-answer-only +
depth curriculum; length-generalizes (K=4→10, K=8→12); gate learns adaptive
halting (halt_exact 1.00). Transfer (`latent_arith.py`): real DeltaNet on real
arithmetic-chain text solves n=1–6 (~1.00) vs no-think 0.15–0.52 — the exact task
discrete thinking scored 0/80. Needs a consolidation phase (uniform-depth sampling
after the ramp). Writeup `THINKING_LATENT_2026_05_28.md`; tests
`test_latent_think.py`; ckpts `latent_think_{curriculum_k4,halt_k6}.pt`,
`latent_arith_n6_mixed.pt`.

Follow-on results (2026-06): latent on the real 601M model — one forward has a ~2
dependent-hop budget; latent extends to ≥8 hops (+0.65–0.80 at n≥3 vs a
fully-trained no-think control). On CODE GEN, inference-time latent is net-NEGATIVE
at every gating threshold (Δlogp −3.7, adapter OOD for code); fix = adapter-only
freeze-trunk on-code co-train (`latent_code_cotrain.py`) flips it to net-≥0 with
base preserved exactly. Code-thinking ceiling is low because per-token code is
recall not iteration (only ~10% of positions help by +0.3 logp, flat across R=1–8);
fix direction = RETRIEVAL, not depth (`WHY_THINKING_MARGINAL_ON_CODE.md`,
`probe_gate_placement.py`). Depth-via-iteration grid: latent simulates depth for
HOMOGENEOUS iteration but COLLAPSES on HETEROGENEOUS composition (per-step
op-selection is a learnability wall, not a compute/width limit).

`THINKING_HUMANEVAL_2026_06_06.md` (selective gating): the collapse (gate firing
an OOD op mid-gen → degenerate loops, 0/164) was a gating bug; fix = route
gate→emit-on-code (`route_emit_finetune.py`) + low inference emit_threshold →
think ~2% selectively → thinking-ON 11/164 vs matched no-think control 7/164.
Modest lift, collapse fix solid.

---

## WM arc (2026-06) — made to work, then scoped to non-code

- **Discrete lexical-hash WM** revived WM (leak-free, Pareto-safe, deployable;
  `wm_discrete_v12.pt`; free-gen const 100% / setvar 92.8% strict, WM-OFF 0%) but
  is spelling-locked.
- **KEY-DESIGN INVERSION:** the WM-addressing mistake was never softmax — it was
  the non-separable KEY. A 3-arm probe (`wm_vqkey_probe.py`) showed a continuous
  SOFT attention read over a SEPARABLE name-span key beats both hash and VQ. ⟹ the
  right addresser is a **learned continuous name-span key**, not a discrete hash.
- **`mem_ctx_namekey`** (the no-hash learned addresser) co-trained into pretrain is
  load-bearing leak-free at 500M–1B tokens (const 0.03→0.98, syn64 +0.59, syn128
  +0.41). The WM-pretrain run is a v12-continuation at lower LR (lr 6e-4 / muon
  2e-3; the 5e-3 peak degraded the base); cost ~+0.2 ppl general VAL.
- **const recall mask train/eval mismatch (root cause of the dead code-const
  gate):** `code_recall_train`'s read mask supervised the RESTATED "Answer:NNNN"
  (recurrence wins there) but the leak-free eval scores the FIRST occurrence
  (recurrence fails) → gate trained where not needed, tested where it is → +0.00.
  Fix = first-occurrence masking (multibind used it → +0.76). Supervise recall
  where recurrence FAILS.
- **WM verdict for the code model (2026-06-16):** DROP it. Recurrence absorbs
  trained recall (copy_g→0, read_alpha→0.028, code-recall off=1.000). Realistic
  recall tops at N≤24, below WM's N≥48 niche. One untested niche: a genuine N≥48
  long-horizon agentic eval. `knowledge_base/verdicts/working-memory-recall-saga.md`.
- **WM recall probe was broken + routed-around (2026-06-14):** the [wm-recall]
  probe used the SFT `# {flatten}` format, OOD for pretrain → same v12 1B ckpt read
  0.000 vs 43.8% training-matched; "recall=0" was an artifact. Deeper: correct
  format → gate NEVER thinks (recurrence solves recall on the emit path) → WM
  (think-only) never exercised. Fix probe + run forced-thinking kill-gate at
  higher N.

---

## The WHY-loop capstone (2026-06-16..24) — token-limited, not capacity-limited

- **Iso-token mechanism ablation:** lean vs full v18 stack, identical mix/budget/
  seed, 2000 steps → per-source CE Δ(MECH−LEAN) ≈ ±0.02, VAL tied. "Mechanisms
  displace code capacity" REFUTED; v18's HumanEval regression was mix-dilution +
  token-budget.
- **Capacity-vs-data reference probe:** HumanEval-solution CE SmolLM2-135M=0.798,
  SmolLM2-360M=0.614 vs our Phase-C-SFT=0.969, v18-SFT=1.056. A HALF-size
  well-trained base beats our 287M because SmolLM2 saw ~2–4T tokens vs our ~5B
  (400–800×). The binding "capacity" is the TRAINING BUDGET.
- **Unifying answer** (`project_why_mechanisms_synthesis`): an add-on helps only
  where (a) recurrence fails AND (b) the add-on has a learnable mechanism there;
  on real code the dominant bottleneck is base knowledge/data, orthogonal to all
  three add-ons. Headline moves via data + scale.
- **pass@k two-component wall (2026-06-24):** wide-10L SFT base 3 greedy / 18
  pass@30; grader-RL 7 greedy / 21 pass@30. The ~18–21 envelope is SFT/base-set,
  not RL-created (RL = +4 greedy sharpening, near noise); 2nd RL round saturated.
  Wall = hard knowledge wall (back ~64 = 0/30) + a reliability gap greedy→envelope
  that best-of-N+verifier converts for free.
- **SmolLM2 goalpost + Qwen-donor pivot (2026-06-24):** we don't beat Smol at our
  size (CE 0.61 vs 0.92). DeltaNet linear-attn tax +0.145 CE even at 32L → can't
  beat the softmax donor. Decision: pivot to linearize a donor + KD-heal, goal =
  match a strong small coder AT O(1) decode.

### Features-null forensics (2026-07-02, corrects Phase-1 A/B)
WHY features showed no code effect: disengaged runs + blind evals (all code evals
≤512-token ctx; WM's niche absent from HumanEval), NOT orthogonality. A new
dep-distance-stratified probe = the first real iso-token feature WIN on natural
code (p<1e-4, monotone with reuse distance). PKM works when engaged (+0.04..0.14
CE); latent-on-code null genuine; WM untested-not-refuted. Recipe rules distilled
into AGENTS.md (engagement kill-gates, curricula ≤40%, frozen-trunk pre-warm
instead of forced floors, ≥2 seeds, stratified CE as dev signal).

### Feature-pilot A/B (2026-07-03)
First engaged-features iso-token A/B on the linearized 32L base (2×2500 steps,
327M tok). Arm A (lean) HE-CE **0.7429** (beat base 0.7585), stratified natural-
code CE 1.1687, leak-free 6-bind recall 67–68%, VAL falls 1.05→1.00, 27.9k tok/s.
Arm B (FiLM+WM+PKM+gate+latent, all engaged with floors) HE-CE 0.9549 (+0.196),
stratified +0.37 worse at every stratum, recall ~20–24% flat, VAL drifts up, 14k
tok/s — a **decisive NEGATIVE**. Post-mortem: **forced-contribution floors on a
converged base are sabotage** (WM read-α floor injected 15%-RMS noise every
position for 800 steps; PKM sign-flipping floor; latent full-trunk gradient from
OOD ptr10dict; gate at 0.8× LM weight). Also found: `prefill`/`forward_step` never
apply FiLM; WM copy head can't fire at inference — both being wired (58c7e70). B′
recipe = frozen-trunk pre-warm (no floors), then joint train, latent adapter-only.

---

## Exec-trace "neural interpreter" arc (2026-07-04..13) — full detail

The current-state summary is in AGENTS.md. Full arc:

- **N0** — pipeline sanity on state-track passed but silently trained rungs [2,3]
  only (`--latent_reasoning_max_len` default 256 dropped K≥4). LESSON: always pass
  `--latent_reasoning_max_len 512` for exec traces.
- **N1** — FAIL, but a methodology bug: `latent_reasoning_cotrain` never consumed
  the `intermediates` field (loss was answer-only) → merely replicated the
  answer-only negative. Per-hop supervision wired in fc6a833 (slot j at position
  P_max+j−1, unshifted logits decode intermediates[j−1]; 44 tests).
- **N1′** — REAL FAIL (2,671 steps / 350M tok full FT from feature_pilot_A;
  `runs/n3_killtest_N1prime.json`): all four arms indistinguishable (no_think
  8.7–13.0%, latent(R=K) 8.0–11.3%, lift −2.3..+2.3pp; needed ≥5pp @K≥4). Per-hop
  decode 9–14% ≈ digit prior; hop CE plateaued ~2.2 ≈ ln 10. Three stacked issues:
  inverted capability gradient (base can't do it in text either), skipped
  Scratchpad→Coconut staging, marginal-stats attractor under CE.
- **Stage A** (text-scratchpad executor) — DECISIVE PASS (2026-07-11). 0.15-mix
  stream over pilot ×0.85 (`configs/pretrain_mix_stageA_executor.yaml`), full FT,
  2,120 steps (survived GPU-off-bus, resumed from step-382 periodic ckpt). Heldout
  K=2–8: trace step-acc 0.963–0.988, answer-with-trace 0.937–0.970 vs direct
  0.117–0.160 = **+80–84pp**; base control 0% format compliance; **HE-CE 0.7343**
  (best code ckpt, `stageA_executor.pt`; `eval_exec_trace_text.py`). Length-gen
  degrades gracefully past trained depth (step-acc 0.79/0.73/0.60 at K=9/10/12,
  answers collapse — curriculum artifact).
- **Stage B** (Coconut text→latent, 2,600 steps / ~340M tok full FT from
  stageA_executor.pt) — NOT KILLED (`launch_stageB_latent_trace.sh`). Curriculum:
  stage s replaces the first s text-trace steps with s latent hidden-feedback
  steps; loss = CE on remaining text+final + per-hop CE decoding latent slot j →
  intermediates[j−1]; rides `_answer_span_latent_loss_batched` (R=s). s_max ramps
  0→8 over ~55% then consolidation s~uniform{0..min(K,8)}. Per-hop CE broke the
  N1′ digit-prior plateau (~2.30 flat → 0.43–0.89) at step ~750. Pre-registered
  lines: KILL not tripped (latent(R=K) beats same-ckpt direct by
  +50.7/+51.7/+45.7/+36.3/+19.7pp at K=4..8); mechanism gate PASS (per-hop decode
  0.844 @K4, bar 0.50, N1′ 0.11 — NOTE a units bug read 0.000 first, fixed+
  regression-tested); success bar PARTIAL (answer 0.63/0.63/0.59 at K=4/5/6 pass,
  K=7 0.497 K=8 0.333 fail); code guard PASS (HE-CE 0.7491). Depth-true signature:
  R=1 collapses AND R=K+4 collapses, only R=K works. **Limit = a ~6-hop latent
  horizon** (hops 1–5 at 0.78–0.96, hop 6 ~0.60, hop 7+ cliff) — likely training-
  exposure of deep slots, not a wall (fix arm: `--latent_reasoning_depth_weighted`,
  8482023). Open anomaly: K=2 latent answer 0.14 despite 0.925 per-hop (emission
  fails at the shallowest rung). Text skill cost real but bounded (heldout text
  answer 0.95 → 0.73–0.83). `EXEC_TRACE_LATENT_PLAN.md`.
- **CRUXEval-O transfer** (2026-07-13, commit ef52951): mechanism does NOT
  transfer, but direct-answer internalization signal +53% rel (z=2.58, n=800,
  token-count confound flagged).
- **Durable methodology finding:** externalize before you internalize
  (Scratchpad→Coconut). Same data, same aux machinery, opposite outcome vs N1′.
- Novelty map: `LITERATURE_LATENT_EXEC_2026_07_13.md`. Claimable (if it holds):
  "latent execution" (C∩B), the staging cell, the depth-true signature, the
  bounded-state angle. Not claimable: Coconut mechanism, per-step-supervision in
  general, "LMs simulate Python" (CWM at 32B), curriculum-necessity as a bare claim.

---

## Historical config bullets (kept for exact values)

- **Sparse FiLM (2,28)**: −3.1% to −5.4% PPL at 217M/360M/708M. K=3 self-feed
  closes train/inference gap.
- **State-readonly thinking** (`--state_readonly_at_think`): β=0 at think via a
  forward-hook on the inner FLA `b_proj` (pre-sigmoid logits clamped to −1e4).
  Synthetic 1L probe: train 4-think burst, eval 32 thinks → ON 0.88 recall, OFF
  0.41 (`test_state_readonly_thinking.py`). Only the full-sequence forward is
  wired; `forward_step` leaves β unmasked.
- **PKM v7.1 live-diagnostic legend:** `pkm(αL=+0.32, αeff=+0.34, row=1.79,
  slots/H=33k/65k, top=0.003, ε=0.00, φ=0.00)`. αL learned scalar; αeff effective;
  row mean value-row norm / expected-init (>1 learning, ≈1 frozen); slots/H unique
  slots hit per microbatch; top hottest-slot mass share; ε/φ the ε-greedy and
  floor curricula. **PKM α-decay** (v7.0 before the fixes): αL grew 0→+0.085 by
  step 280 then decayed to +0.04 by step 400 — optimizer correctly saw random-init
  values as noise; fix = α-floor curriculum + value-LR-mult.
- **Cross-doc isolation** (2026-05-14): `MixedSourceStream(emit_doc_ids=True)`,
  `insert_think_bursts(aligned=)`, `model._build_cu_seqlens` (int32),
  `WorkingMemory` same-document read mask. `test_cu_seqlens.py` (packed==unpacked
  DeltaNet equality). `CROSS_DOC_ISOLATION_PLAN.md`. Open: RL replay-packing still
  passes doc_ids=None; FiLM `_shift_right` not doc-aware.
- **code_grader dense ladder:** syntax_error 0.0 < exec_error 0.05 < runtime_error
  0.2 < partial 0.2+0.7·(n_passed/n_tests) < pass 1.0. `_exec_target` AST-splits
  `check()` and runs statement-by-statement. `error_text` for repair.
  `test_code_grader.py`.
- **Trunk-diagnostic origin stories:** residual-stream collapse found by
  `diag_ckpt` (‖h‖@L0 shrank 8.1→3.5 between 500M/1B on v2 with WD=0.1; v3a WD=0.01
  un-collapsed it, ‖h‖@L0 23.5). LayerDrop v3b (`--layer_drop_max 0.2`) clean
  negative (L25–L29 vestigial, +0.02–0.10 CE every source). WD-equilibrium probe:
  |grad_α| ≈ 1.83 × WD·|α| → `--alpha_wd 0.0`.

---

## Lessons learnt (don't relive these)

- **PKM α-decay** means the bootstrap is incomplete — keep αL high long enough
  (α-floor) AND give values a separate higher LR (value-LR-mult). Without both,
  PKM stays inert. `v_std` is a misleading diagnostic; use `row`.
- **MQAR at T=512/K=128 doesn't discriminate feedback choices at our scale** — use
  T=256/K=32 for architectural ablations.
- **`compile_model` silently falling back to eager is a footgun** — always-on
  strict mode prevents it.
- **bf16-on-sparse-loss collapses logits to uniform** — run sparse-loss validation
  tasks (MQAR, recall) in fp32.
- **Cold latent co-train destabilizes from-scratch pretrain** (v12/v13): engaging
  latent_cotrain + gate_calib at step 3500 spiked VAL +10%, gnorm 20×, and
  collateral-damaged PKM routing. Cause: cold latent → huge CE_after → full-trunk
  R=4 grad, no ramp. Fix = aux OFF in pretrain, latent adapter-only post-hoc (or a
  ramped depth-matched `latent_reasoning` if it must run from day 1).
- **min_content_len killed magicoder+textbooks** across v10–v17 (filter only
  checked content/text fields → ~19% of the mix was 0 tokens). Fixed with a
  longest-string fallback.
- **Stateful chunking (rolling DeltaNet state across the 2048 boundary) not worth
  it** — cold-start penalty is real but front-loaded (+0.80 @pos0, ~0 by 1024);
  aggregate ceiling ~0.04 CE. Raise `--T` first if long-context matters.
- **Embedding-specific optimizers are a dead lever** — tuned embed LR −1.76%,
  rownorm dualizer −12.84%; keep shared-LR AdamW. The only real optimizer lever is
  the per-head-NS MATRIX optimizer (+2.46%).
- **Weight-teleportation / basin-hopping is a dead end** — loss-invariant
  transforms; minima are same-basin-mod-permutation. Working family = SWA / soups /
  SAM (generalization, not lower loss). SWA on WSD-plateau ckpts = free ~3% CE
  (0.952→0.919); adopt EMA/LAWA.
- **SOAP/Shampoo-family does NOT converge faster on wall-clock** at 287M — Muon
  wins. Don't re-run at this scale.
