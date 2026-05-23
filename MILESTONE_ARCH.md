# Milestone: Each Architectural Component Pulls Its Weight

**Goal:** Every load-bearing architectural choice in the model — PKM,
WorkingMemory, the thinking gate (including long thinking chains), and
memory writes — must be **measurably contributing** to the headline
metric (HumanEval pass@1, plus a future long-context recall benchmark).

If we can disable any one of them and the model is unchanged, that
component is dead weight. The thesis of this codebase ("small model
punches above its weight via clever architecture") only holds if every
piece earns its keep.

---

## Success criteria

A component **counts as load-bearing** when ablating it (e.g., zeroing
the relevant projection so the contribution is structurally 0) drops
the headline metric by at least the threshold below.

| component | ablation | metric | required drop | current status |
|---|---|---|---|---|
| PKM (static memory) | zero `pkm_layer.out_alpha` | HumanEval pass@1 | **≥30% relative** | ✅ **−50% relative on Phase C SFT** (10 → ~5 in ablation, 2026-05-22). PKM clears the bar on every post-Phase-C ckpt we've measured. |
| WorkingMemory (dynamic memory) | zero `memory.W_proj.weight` | HumanEval OR long-context recall | **≥10% relative** on at least one | ✅ **load-bearing on long-context recall** (the right probe for dynamic memory): WM-on **98.2 %**, WM-off ≈ chance on the distance-bucketed eval (`experiments/eval_longctx_recall.py`, 2026-05-22). HumanEval ablation is still near-noise — WM earns its keep on its proper probe. |
| Thinking gate | force `use_thinking=False` (always emit) | HumanEval pass@1 | **≥10% relative** | ✅ **task-adaptive at τ=0**: ~0 % think on long-context recall, ~33 % think on HumanEval; forced-emit drops HumanEval on the RL v2 ckpt. (Gate is still bimodal-fragile under sampling — see GEMINI.md "Open architectural finding: thinking gate is temperature-fragile".) |
| Long thinking chains | mean `burst_depth >= 4` on hard problems vs `<= 2` on easy ones | depth-by-difficulty correlation > 0.3 | qualitative | ❌ burst dist is currently bimodal (90% depth 0, 5% depth 8) — no graceful interpolation |
| Selective memory writes | per-position write gate entropy < uniform baseline; writes at "high info" positions (var defs, fact tokens) | qualitative | qualitative | ❌ write gate is uniform across position types; buffer composition matches baseline % think |

---

## Where we are (2026-05-19)

The most recent ablation experiment on the current best ckpt
(`sft_v7_pkm_film_combined.pt`, HumanEval pass@1 = 11/164):

```
baseline        11/164   (6.7%)
wm_off          12/164   (7.3%)   ← WM HURTS pass@1 by 1
pkm_off          (in-flight, ~5/80 = 6.2% so far, vs 13.8% baseline)
both_off        (pending)
```

PKM is doing real work — disabling it cuts pass rate roughly in half so
far. WM is decorative or slightly harmful in its current implementation.

Thinking gate fires at think_rate ≈ 0.30 on greedy eval, but earlier
ablation showed thinking-on == thinking-off on full HumanEval, meaning
the thinking content itself doesn't contribute. Under temperature
sampling, the gate collapses to bimodal (~0 or ~1) regardless of
config — fragile to deployment scenarios.

---

## Workstreams to hit the milestone

These are the levers known to plausibly move each component into the
load-bearing zone. Each has a concrete experiment behind it. Execute
top-down; the early items are cheaper and inform later choices.

### A. Make PKM contribute MORE (amplification — high-confidence lift)

PKM already passes the bar. Scaling it is the highest-confidence
single move on the headline metric.

  - **A1. PKM size scaling**: `n_heads 4 → 8`, `n_keys 256 → 384`
    (8 × 384² = 1.2M effective slots, up from 262k). Requires pretrain
    re-run. Reference: Lample 2024 "Memory Layers at Scale" scales
    PKM well beyond billions of slots without instability. Expected:
    +3–6 pass@1.
  - **A2. Two PKM layers** (instead of one mid-trunk): another PKM
    block at a different depth would let the model query memory at
    multiple representation levels. Expected: +1–3 pass@1, cheap.
  - **A3. Larger value tables (v_dim)**: each slot stores a richer
    value. Expected: marginal but free if A1 is happening anyway.

### B. Make WM contribute (currently dead; needs all of this)

The diagnostic-derived reasons WM is dead:

  1. No direct supervision — only indirect via next-token CE
  2. Random think-burst placement gives noisy gradient on WM
  3. Buffer holds arbitrary high-magnitude positions, not "info to
     remember"
  4. Think-position hidden states are too correlated (FIX A finding)
  5. Headline eval doesn't need WM (HumanEval is too short for any
     bounded recurrent state to fail)

Fixes:

  - **B1. Long-context synthetic recall data**: variable defined at
    token 50, asked for at token 1500+ (DeltaNet's recurrent state
    will lose it). Multiple difficulty levels (distance 500/1000/2000/
    4000). Use as both training data AND as a new eval benchmark
    distinct from HumanEval. **Without this, WM can never demonstrate
    value** because nothing in the eval requires it.
  - **B2. Direct supervision through WM read (Option A)**: at think
    positions, `wm_future_head(memory._last_injection_grad)` predicts
    `embed(input_ids[t+T])` via cosine loss. Gradient flows directly
    to `W_v`/`W_q`/`W_proj`/`W_write`. The `_last_injection_grad`
    stash already exists (2026-05-19); needs the head + loss wiring
    in `sft_code.py`.
  - **B3. Trajectory-style multi-offset prediction**: predict embedding
    at offsets `[1, 4, 16]` simultaneously through WM, with separate
    heads. Forces richer encoding of "what's coming". B2 is the basic
    version; B3 is the extended.
  - **B4. Retrieval-as-input thinking tokens** (v3, training now):
    forces the model to ACTUALLY USE the WM read (replaces input
    embedding at think positions). Solves homogeneity at root. Eval
    pending tonight.
  - **B5. Write gate conditioned on local salience**: instead of
    `g = σ(W_write(h))`, condition on a more informative signal —
    e.g., a small head that takes (h_t, h_{t-1}) and asks "is this
    token a binding / fact / important?". Forces selective writes.

### C. Make the thinking gate load-bearing

The thinking machinery (gate + memory injection at think positions)
is currently decorative on HumanEval. To make it work:

  - **C1. Gate-collapse fix**: gate produces low values under
    temperature sampling (RL rollouts) because it was only trained at
    greedy. Robustness fixes: train with token-level temperature
    augmentation (sample some training-time positions at τ>0 and
    measure gate output); or replace the sigmoid hard-threshold with
    a soft attention-style "thinking strength" head. Diagnostic
    `gate_collapse_diag` planned.
  - **C2. Long thinking chains**: currently `max_think_per_step=8`
    caps a burst. For genuinely hard problems, the model should be
    able to think for 30–50 tokens. Two paths: (a) increase the cap
    and add training data with long bursts of thought (Qwen
    completions with long CoT); (b) replace per-step cap with a
    "stop-thinking" gate that fires when the model has converged on
    a plan.
  - **C3. Eval-time prompting that exercises thinking**: prompt
    construction should give the model permission and reason to use
    long chains. Current sft_comment prompt is terse; richer
    multi-step problems may trigger more thinking.

### D. Make memory writes selective

The write gate currently fires uniformly. To get selective writes:

  - **D1. B5 above** is the architectural fix.
  - **D2. Direct write-gate supervision**: an auxiliary loss that
    rewards writing at positions where the value gets retrieved
    later (e.g., token-level mutual information between value content
    and downstream read attention). Indirect but ties writes to
    actual utility.
  - **D3. Sparser, longer-lived buffer**: currently `mem_size=1024`,
    every position competes equally. Try a tiered buffer (small
    "scratchpad" + larger "long-term") where the long-term has a
    higher write threshold.

---

## How we'll know we're done

A pass-counts table that looks like this:

```
                     baseline  -PKM   -WM    -think  -all
v? (next round)        20+      6      14    16      4
                                ↑      ↑     ↑       ↑
                              load-   load-  load-   trunk
                              bearing bearing bearing only
```

Each ablation column showing a meaningful drop = that component is
load-bearing. Today we have:

```
                     baseline  -PKM      -WM    -think  -all
v7-combined-v1         11      ~5 (-55%) 12 (+9%) 6 (-45%, prior run)  pending
                                ↑              ↑    ↑
                              PKM ✓        WM ✗  thinking neutral on full eval
```

Targeted result for the **next sprint** (after PKM amplification +
WM direct supervision + retrieval-as-input + long-context eval):

```
                     pass@1   long_ctx_recall  -PKM   -WM    -think
target                 18+    50%+ (vs <5% now)  ↓40%   ↓20%   ↓20%
```

---

## Execution path (build → train → re-test)

The current architecture is dead in places. More measurement on the
current state won't fix it. The pattern is: **make the architectural
change, then train under it, then re-run the ablation**. Three
iterations planned:

### Iteration 1: WM activation (next ~1.5 days)
  - Build B1 (long-context synthetic recall data) → ~2 hr
  - Build B2 (direct WM future-emb supervision) → ~1 hr code + tests
  - Re-train v5 SFT from pretrain with: existing recipe + retrieval-as-
    input (v3's mechanism) + Option A supervision + long-context data
    mixed in (target 30% of corpus). ~1 hr train.
  - Eval v5 on HumanEval AND on the new long-context benchmark.
  - Re-ablate (`zero W_proj`) on v5: does removing WM now hurt by
    ≥10%?
  - Decision gate: if WM ablation shows ≥10% drop, WM is load-bearing
    → proceed to iteration 2. If not, the architectural pieces missed
    the mark; revisit hypothesis.

### Iteration 2: Thinking robustification (after iteration 1)
  - Build C1 (gate temperature-robustness fix): augment training with
    token-level sampling so the gate sees varied hidden-state
    distributions. Re-train v6 from v5.
  - Re-run RL grader on v6 (which should now have a thinking gate
    that survives τ=0.9 sampling).
  - Verify: thinking ablation on v6 drops pass@1 ≥10%; RL adds
    pass@1 lift.

### Iteration 3: PKM amplification (after iterations 1+2, requires pretrain)
  - A1: PKM `n_heads 4 → 8`, `n_keys 256² → 384²`. Requires pretrain
    re-run (~24 hr).
  - This is iteration 3 because it's the most expensive and PKM is
    already meeting its load-bearing bar — the amplification is
    additive lift, not a "make it work" fix.

## Dependencies between workstreams

- **C1 (gate robustness) is a prerequisite for working RL** — currently
  RL rollouts collapse the gate, defeating any thinking-driven training
  signal. Without this, the RL stage will keep wasting compute.
- **B1 (long-context data) is a prerequisite for B2/B3 supervision** —
  the direct supervision through WM needs sequences where the future
  IS predictable from a stored past, which short HumanEval-style data
  doesn't provide.
- **A1 (PKM amplification) requires a pretrain re-run** — every other
  workstream can ride on the existing pretrain ckpt and just retrain
  SFT, which is 30 min cheap.

---

## Anti-goals (don't do these)

  - **Don't drop WM yet.** Tempting given the ablation, but it's
    untested on the right benchmark. Removing it before measuring
    against a task it should help would be premature.
  - **Don't add architectural complexity without a measurement that
    requires it.** The pattern is: hypothesis → diagnostic → fix →
    re-diagnostic. FIX A skipped the "diagnostic confirms the
    hypothesis" step and was discarded after burning 30 min of train.
  - **Don't optimize for HumanEval alone.** The milestone explicitly
    includes long-context recall because that's where WM should pay
    off. Headline metric is a number, the architecture has to do its
    job across the distribution we deploy on.

---

_Last updated: 2026-05-19. Owner: this milestone is tracked in
`MILESTONE_ARCH.md`; the live agent state and todo board are in
`GEMINI.md`._
