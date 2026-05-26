# Make the Thinking Mechanism Work — Plan

Top priority after Phase D pretrain completes (2026-05-26). The thinking
machinery (gate + think token + WM read + retrieval-as-input + WM gist)
is currently a CAPABILITY without a TRAINING SIGNAL that rewards using
it productively. Six concrete failure modes have been observed; the
plan addresses each with an ordered set of interventions.

## What we've concretely measured (the failure modes)

| failure | evidence |
|---|---|
| **Temperature fragility** | τ=0 → think_rate ≈ 0.30; τ=0.7 → collapse to 0 or 1 depending on gate_floor vs emit_threshold |
| **Always-think collapse in RL** | v6/v7/v7b depth_mean stuck at 120 (max budget) entire run |
| **Always-emit collapse in SFT** | sft_repair_v3 mean_gate=0.997, think_rate=0 across HumanEval |
| **Multi-think homogenization** | 8 consecutive thinks → median pairwise cosine +0.146 vs +0.060 at emit; effective rank ~210 vs ~560 |
| **Recurrence corruption** | Plain thinking dropped 512-distance recall 100% → 20%; every think token perturbs DeltaNet state |
| **Repair learning doesn't transfer** | v3c can fix `a-b → a+b` but not `round(x,2) → int(x)`; toy syntactic fix ≠ algorithmic insight |
| **WM gist supervision flat** | v5 lexical vs v6 multi-horizon-gist target produced identical `wm_off −5` HumanEval delta |

## Root cause

LM loss doesn't care if you think — only the final next-token prediction
matters. RL grader signal is for the END output; even with counterfactual
+ ponder cost, "this think token was useful" gets ~zero gradient. The
architecture HAS the mechanism (gate, WM, retrieval-as-input) but the
training distribution never explicitly rewards productive thinking.

## Phases

### Phase 1 — Diagnose what the model actually does with thinks

Three probes that run on any pretrain ckpt:

1. **Per-position CE delta**: for each position the gate elects to
   think, compare predicted-next-token CE with and without that think.
   If CE drops post-think on average → mechanism works mechanically.
   If flat or negative → the gate fires at wrong positions OR thinking
   adds noise.
2. **Counterfactual sampling on held-out HumanEval**: generate with
   `think_budget=120` vs `think_budget=0` on the same problems. Compare
   grader pass-rate. If thinking adds 0 problems → mechanism is
   decorative.
3. **Think position vs answer correctness correlation**: in RL
   rollouts, do "more think tokens at certain positions" correlate
   with passing rollouts? Or noise?

**Tells us whether we're fixing a USE problem (mechanism works,
training doesn't reward right) or an ARCHITECTURE problem (mechanism
doesn't even reduce CE).** Output: `THINKING_PROBE_RESULTS.md`.

### Phase 2 — Architectural: state-read-only thinking

CLAUDE.md flags this as the next architectural fix: think tokens
currently step the DeltaNet recurrence and corrupt long-range state.
Set `β=0` on the delta-rule write at think positions. Think positions
still READ from the state (so the model can think about current
context) but don't WRITE to it.

Implementation: ~30 lines in `model.py` `_FlaWrapper` + a constructor
flag `state_readonly_at_think`. Default OFF for backwards-compat;
opt-in via a new CLI flag.

Validation: `eval_longctx_recall.py` gives immediate signal — does
thinking stop hurting recall?

### Phase 3 — Diverse think input

The homogenization problem: 8 consecutive thinks all have the SAME
input embedding ([THINKING]), so the multi-think hidden states
collapse to a low-rank manifold.

Two interventions:

1. **Retrieval-as-input baked into pretrain** (not just SFT). The
   model trains end-to-end with retrieval-as-input from step 0 of
   future pretrain phases.
2. **Per-position think index embedding**: each think in a burst gets
   a small positional embedding (`think_emb + pos_in_burst_emb`).
   Cheap (~3k new params for an 8-position table at d_model=896).
   Breaks the all-thinks-look-alike at the input level.

### Phase 4 — Distill thinking from Qwen (CoT → thinks)

Currently the model has zero demonstrations of "good thinking that
improves the answer". Qwen-3.6 has reasoning enabled. Distill into
our format:

- Prompt Qwen for (problem, chain-of-thought, answer)
- Convert each CoT step into a sequence of OUR think tokens (with a
  small head that maps a CoT step → a fixed-length sequence of
  retrieval-as-input embeddings, OR via a vocabulary-projection)
- SFT on (problem → [thinks materialized from CoT] → answer)

This is the most data-intensive intervention but the highest expected
lift. Phi-1 / Phi-2 / Qwen all show that demonstration-style
distillation works at our scale.

### Phase 5 — Process-aware reward in RL

After Phase 4 the model has seen good thinking; RL can now reward
USING the thinking productively:

1. **Counterfactual gate target during pretrain/SFT**: at every
   position, compute next-K-token CE with and without a single think
   token. Gate target = 1 if think helps, 0 if not. Self-supervised
   process-reward signal — no external PRM needed.
2. **Per-think-token advantage in GRPO**: attribute reward fractionally
   to each think token based on whether the downstream emit was better
   than baseline.

### Phase 6 — Curriculum for hard reasoning

Synthetic tasks where thinking is REQUIRED to succeed:
- Multi-step arithmetic chains (compute X = a+b, Y = X*c, Z = Y-d, output Z)
- Conditional rule application (apply rule R1 if X else R2)
- Extend task #52 (synthetic memory tasks, completed) with multi-step
  reasoning families.

These give a clean signal: thinking is the difference between solving
and not.

## What we will NOT do

- **Pure thinking-token-count optimization** (depth_mean targeting):
  cycled through ponder_cost for months, never a clean win.
- **More gate-supervision aux losses without process signal**:
  entropy-aux is a proxy that didn't move the needle.

## Execution order

1. Phase 1 first (always) — measure before fixing.
2. Phase 2 (state-read-only) — cheap to build, clean validation probe
   (long-context recall), addresses the most-measured failure mode.
3. Phase 4 (distill from Qwen with CoT → thinks) — highest-EV
   training-signal intervention. Run in parallel with Phase 6 data prep.
4. Phase 3 (diverse think input) — light architectural touch-up,
   prerequisite for any new pretrain or SFT.
5. Phase 5 (process-aware RL) only AFTER Phase 4 has installed a useful
   thinking distribution.

## Decisions log

Important judgment calls made during execution are logged to
`THINKING_DECISIONS.md` for user review.
