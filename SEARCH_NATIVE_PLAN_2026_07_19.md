# Search-native decoding — pre-registered plan (2026-07-19)

## Claim under test (the high-risk/high-novelty bet)

Bounded-state models make program tree search cheap where transformers
cannot: a search-tree node costs a **flat 8.2 MiB state snapshot** (vs GBs of
KV per branch), so wide/deep search over code continuations is
memory-affordable; and the **trained neural interpreter** (Stage-A/B
executor) can serve as a *value function on non-executable prefixes* —
scoring partial programs where real execution is impossible. The
ideation-fleet literature check (2026-07) found this intersection
(state-checkpointed tree search + interpreter-supervised value function)
unpublished.

Two separable novel claims, tested in order:
1. **Interpreter-as-value-function**: the executor's simulated-execution
   signal ranks partial programs better than log-prob baselines.
2. **State-checkpointed search**: search at matched model-forward budget
   beats best-of-N sampling on pass@budget (the pass@k envelope work showed
   best-of-N+verifier ≈ the reachable ceiling — search must beat it on
   BUDGET, not just quality).

## Phase 1 (prereq): exec-trace re-attach on the long-context base

Stage-A text-scratchpad training on `production_lean_longctx.pt` (the
agent-preferred base) — a DATA fine-tune (the regime that works on this
lineage), recipe verbatim from the original Stage-A
(`configs/pretrain_mix_stageA_executor.yaml`, T=2048, isolation on, muon
6e-4/2e-3 wsd), 2,300 steps ≈ 300M tokens, 2-GPU manual allreduce (~1.5h).
→ `checkpoints/executor_longctx.pt`

**Phase-1 bars (pre-registered):**
- Exec heldout answer-with-trace ≥ 0.90 (original Stage-A: 0.94–0.97).
- HE-CE ≤ 0.6775 (longctx 0.6675 + 0.01; original Stage-A IMPROVED its
  base, so this is lenient).
- Secondary read (no veto): frozen-set sequential span lift (does T=2048
  isolated fine-tuning erode the longctx window gains? measure, don't
  guess).

## Phase 2: value-function validation (before any search infra)

On heldout exec-trace programs truncated at k of K steps (non-executable
prefixes with known final answers): rank candidate continuations by
(a) executor simulated-answer agreement, (b) mean log-prob, (c) random.
**Bar: (a) beats (b) by ≥10pp ranking accuracy** — if the interpreter can't
outrank log-prob on its home distribution, the search claim dies here
(cheap kill, no search code written).

## Phase 3: state-checkpointed search harness

Tree search over code continuations with per-node state snapshots
(`prefill` + snapshot/restore via the cartridges-validated state machinery);
grader for executable nodes, interpreter value for non-executable.
Benchmark: MBPP subset + HumanEval, **pass@matched-forward-budget vs
best-of-N** (the honest comparison the pass@k work demands), plus the
memory-per-node number vs a KV-cache baseline (the moat metric).
Bars finalized in this doc BEFORE phase 3 runs, after phase-2 effect sizes
are known.

## Kill discipline

Each phase gates the next; phase-2 failure kills the program at ~zero search
-infra cost. GPU0 flakiness: all runs use periodic ckpts + START_STEP resume.

## PHASE 2 RESULT (2026-07-19, run on stageA_executor while phase 1 trains): DECISIVE PASS

- Pooled (n=837, K∈{4,6,8}): executor ranking accuracy **0.956**
  [0.940–0.969] vs logprob 0.226 [0.198–0.256] vs random 0.257 —
  **executor − logprob = +73.0pp** (bar ≥10pp; ties-as-failures headline).
  Per-K: 0.98/0.96/0.92 — no depth cliff through K=8.
- Log-prob is AT RANDOM: the single-edit semantic mutations (constant/op
  changes) are exactly the surface-plausible traps a likelihood signal
  cannot see and a genuine interpreter can. The value-function claim holds
  on-distribution.
- Caveat carried forward: this is the executor's home distribution
  (synthetic exec traces). Phase 3's real risk is transfer to natural code
  (MBPP/HumanEval) — the CRUXEval mechanism-transfer failure is the prior.
  Phase-3 bars must be set with that in mind; re-run this harness on
  `executor_longctx.pt` when phase 1 lands (same bar, both results
  reported).

## PHASE 1 RESULT (2026-07-19): ALL BARS PASS — with one important negative finding

- Exec heldout answer-with-trace: **0.94–0.99 across K=2..8** (bar ≥0.90);
  trace state-acc 0.97–1.00; lifts +78–86pp over direct.
- HE-CE 0.6773 ≤ 0.6775 (photo-finish pass). `executor_longctx.pt` is the
  Phase-3 executor.
- Value-function re-check on the new executor: pooled 0.962 [0.949–0.974],
  **+70.0pp over logprob** — the value-function property survives the base
  swap essentially unchanged.
- **Negative finding (secondary read): sequential capability-stacking
  fails.** The T=2048+isolation Stage-A fine-tune ERASED the longctx window
  gains (frozen-set sequential lift −0.133 → −0.48). Same lesson as the
  attach program, now at the data level: capabilities must be CO-TRAINED
  (exec traces inside the long-context mix), not layered sequentially.
  Tolerable for Phase 3 (short search contexts); rules this ckpt out as an
  agent base; the eventual agent base needs a joint mix.

## PHASE 3 GATES (registered now that phase-2 effect sizes are known)

- **3a. Natural-code transfer probe (BEFORE any search infra; the CRUXEval
  prior says this is where it dies).** Ready-made natural mutations exist:
  the repair corpus (`gen_repair_triples`, 2,389 heldout buggy/fixed/error
  triples on executable MBPP-style tasks). Two-way ranking: does an
  executor-derived score rank FIXED above BUGGY better than mean log-prob
  does? **Bar: executor − logprob ≥ +10pp ranking accuracy** (same bar,
  hostile distribution). If FAIL: the search program is killed for the cost
  of one eval, and the on-distribution +73pp result stands alone as a
  scoped finding.
- **3b. Search harness (only if 3a passes):** state-checkpointed tree
  search, pass@matched-forward-budget vs best-of-N on MBPP subset +
  HumanEval, memory-per-node vs KV baseline. Quantitative 3b bars set after
  3a's effect size is known.
