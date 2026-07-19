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
