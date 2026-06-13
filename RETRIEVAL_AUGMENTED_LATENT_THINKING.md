# Retrieval-Augmented Latent Thinking (RALT) — design

## Motivation (measured, not assumed)

`probe_gate_placement.py` on `latent_code_adapteronly.pt` (the adapter-only
co-trained base) established two facts about latent thinking on code:

1. **Pure-trunk iteration has a low ceiling that depth does not raise.** Only
   ~10% of code positions benefit from a latent burst, by ≈+0.3 logp, FLAT
   across R=1,2,4,8. Unlike arithmetic (where exact-R unlocked +0.65–0.80),
   more latent steps add nothing on code. Mechanistically: code-token
   prediction is mostly *recall/pattern*, and a Coconut-style latent step only
   re-iterates the trunk's own computation — it cannot fetch a fact the trunk
   doesn't already have in-state.

2. **The gate is anti-aimed** (corr(P_think, Δlogp) ≈ −0.10) AND **cannot be
   linearly re-aimed**: a gate-head-only calibration on the dense `1{Δlogp>0}`
   teacher did NOT flip the correlation (−0.10 → −0.17). "Will thinking help
   here" is not linearly decodable from `out_norm(h_t)` — consistent with the
   value being *recall* (a property of what's retrievable, not of the current
   hidden).

**Thesis.** The missing ingredient for code is RETRIEVAL, not more iteration.
Give the latent step a channel to fetch a fact — from WorkingMemory (context
recall: a variable bound earlier, a signature) or PKM (parametric recall:
API/library knowledge) — and the content ceiling should rise above the
pure-iteration wall. This also makes PKM/WM load-bearing for *thinking*
specifically (the stated goal), not just for the emit path.

## Mechanism

Current adapter-only latent step (next think-slot input embedding):

    z = adapter(out_norm(h_t))                       # Coconut feedback, clean

RALT latent step (the sanctioned `mem_alpha` hybrid, already wired in
`latent_sft.py:57-69` trainer and `eval_humaneval.py:511-531` generator):

    z = adapter(out_norm(h_t)) + mem_alpha · inj_wm

where `inj_wm = model.memory._last_injection_grad[:, -1:, :]` (training) /
`._last_injection[:, -1:, :]` (inference) — the WM content-addressed retrieval
at the think slot. The WM read query is `W_q(out_norm(h_think))`, scored against
the buffer of past write-gated hidden states, so the retrieval is *conditioned
on the evolving thread state* and changes as the burst iterates.

PKM needs NO separate term: it already fires inside the trunk forward (after
layer 5) at the think slot, so its parametric read is already inside
`out_norm(h_think)`. The latent loop already lets the trunk re-query PKM with a
refined hidden over R steps.

### Why this is safe (the contamination footgun)

The 2026-06-05 bug: feeding back `out_norm(h) + α·W_proj(read)` raw is OOD for
the adapter (which learned to map *clean* `out_norm(h)` → input manifold) →
corrupted feedback → run-on output. RALT avoids it three ways, all already in
the codebase:

- **α-gated from ~0**: `mem_alpha` init 0.1, no weight decay (FiLM-α curriculum).
  A useless retrieval contributes ≈0; the adapter baseline always survives.
- **Added AFTER the adapter** (not inside the clean hidden the adapter consumes).
- **`_latent_feedback_premem`**: the thread source hidden is the *pre-memory*
  `h_premem`, so the carried thread stays clean while WM shapes the emit logits.

### No-think is preserved byte-identically

WM injection fires only where `input_ids == thinking_token_id`
(`model.py:1192`). No-think code generation contains no think tokens → WM never
injects → training the WM read path (`W_q`, `W_proj`) and `mem_alpha` cannot
change no-think behaviour. With the trunk + lm_head frozen, the no-think path is
exactly the base (the invariant the user requires: thinking ≥ no-think).

## What is trained (freeze-trunk, adapter-only-style)

Trainable (~tiny): `latent_feedback_adapter.*` (~805k), `mem_alpha` (1),
`memory.W_q`, `memory.W_proj` (the READ path — let WM specialize what to fetch
for code). Frozen: trunk, lm_head, PKM, `memory.W_write`/`memory.W_v` (keep the
buffer's *content* representation = the pretrain-learned one; only re-learn how
to *query/project* it). gate_head: frozen for the co-train (the gate is a
separate, later problem — and we proved it can't be linearly calibrated;
gate-driven RL on the retrieval-trained base is the follow-up).

Loss: `latent_cotrain_loss` (grad CE on the post-R-latent-think prediction of
the true next code token) — identical to the adapter-only co-train, but now the
latent step has the retrieval channel. Validation signal: the Δlogp ceiling
(frac helpful, mean Δlogp | helps) should RISE vs the adapter-only base.

## Staged validation plan (cheap → expensive; stop if a stage fails)

**Stage 0 — pre-train oracle probe (no training).** Attach `mem_alpha` to the
co-trained base, turn WM ON in the latent thread (premem), sweep
mem_alpha ∈ {0, 0.1, 0.3, 1.0}, re-run `probe_gate_placement`. Question: does
the (untrained) WM retrieval channel move the Δlogp ceiling AT ALL — does frac
helpful or mean Δlogp|helps rise above the adapter-only 10%/+0.3 wall at any
mem_alpha? If even untrained generic WM reads add nothing, the read path must be
trained (Stage 1); if it actively contaminates at all α>0, escalate to DKV
(addressing is the bottleneck). This costs minutes and tells us whether there's
headroom before any training.

**Stage 1 — RALT co-train.** Freeze-trunk co-train {adapter, mem_alpha, W_q,
W_proj} on code with WM-on premem hybrid latent. ~400–800 steps. Watch:
`mem_alpha` should GROW if WM is useful (stays ~0 → WM useless on code, negative
result). Re-run `probe_gate_placement` on the result: did the ceiling rise?

**Stage 2 — flip eval.** `mbpp_flip_eval.py` no-think vs thinking on the
RALT ckpt, ≥250 problems, confirm: (a) no-think exactly preserved, (b) thinking
net ≥ the adapter-only +1/+3, (c) zero hurt cases. Compare think-helped problem
ids to the adapter-only set — are they recall-shaped?

**Stage 3 (only if Stage 1–2 positive) — DKV escalation.** If legacy
address-by-value WM is the bottleneck (Stage 0/1 weak), train DKV cosine
addressing + fix the `forward_step` decode-parity gap (`model.py:2980`) so RL
rollouts use the same addressing. This is the "reliable semantic recall" path
from `WM_ADDRESSING_PROPOSAL_*`; defer until v1 proves retrieval helps at all.

## Parity requirements (must hold or we ship eval-only behaviour)

- The latent step formula `adapter(h) + mem_alpha·inj` must be identical in the
  grad trainer (`latent_cotrain_loss` / `_latent_think_logits_grad`) and the
  inference generator (`generate_latent_think`). Today both implement the
  mem_alpha hybrid — verify the trainer path (`thinking._latent_think_logits_grad`)
  ALSO adds the mem_alpha term (it may currently NOT, since the adapter-only
  co-train used WM-off). THIS IS THE KEY IMPLEMENTATION GAP TO CLOSE.
- `_last_injection_grad` (grad-keeping) in training; `_last_injection`
  (detached) in inference. Both pre-read-mask, sliced `[:, -1:, :]`.
- WM must be ON (not `clean_latent_thread` / `wm_off`) for RALT — opt out of the
  blunt WM-off toggle, rely on the α-gate + premem safeguards instead.

## UPDATE 2026-06-13 — channel decomposition result (REDIRECTS the plan)

`probe_retrieval_channels.py` on `latent_code_adapteronly.pt` (100 problems,
959 positions, no training), latent-think Δlogp ceiling by channel vs the
deployed no-think baseline:

| config | frac help | mean Δlogp\|helps | mean(all) | median |
|---|---|---|---|---|
| T (trunk only)        | 7.6%  | +0.327 | −1.226 | −0.348 |
| T+P (trunk+PKM)       | 12.8% | +0.293 | −0.848 | −0.121 |
| T+P+WM (native legacy)| 12.5% | +0.324 | −1.126 | −0.222 |

- **PKM is the working thinking-retrieval channel**: +0.052 frac-helpful
  (+68% rel), median −0.35 → −0.12 (near-harmless). It is already trained,
  load-bearing, fires at the think slot, and the co-trained base already
  exploits it (so the current +1/+3 win includes it).
- **Legacy WM HURTS thinking**: −0.003 frac, mean(all) −0.85 → −1.13. The
  address-by-value variant's diffuse-softmax addressing (flagged in
  model.py:1032-1042) is net-negative as a latent-step retrieval channel.

**Consequence: RALT-via-legacy-WM is the wrong build** (it would train the
channel that hurts). Two valid directions remain:

1. **DKV-WM for thinking (the principled "make WM effective" path).** Legacy
   addressing is broken by design; DKV cosine addressing (decoupled W_k +
   learned temperature) is the fix, but NO ckpt has trained it. Build =
   continuation-train DKV addressing (+ the contrastive address pretext from
   WM_ADDRESSING_PROPOSAL_*) on `sft_baked_pure.pt` / `pretrain_phase_c.pt`,
   fix the `forward_step` decode-parity gap (model.py:2980), then re-run this
   decomposition with DKV-WM — does it clear the wall legacy couldn't?
2. **PKM-exploitation for thinking (cheaper).** PKM already works passively;
   the lever is to train the latent step to query/iterate PKM better (adapter
   specialized for PKM retrieval), or gate thinking toward PKM-hit positions.
   Smaller headroom but cheap and targets the proven channel.

## UPDATE 2026-06-13 (2) — DKV-WM-for-code RULED OUT by cheap validation

Per the "validate fast/efficient before GPU-hours" mandate, two no-train probes
settled DKV-WM-for-code:

1. **No-train cosine addressing** (`probe_retrieval_channels.py` T+P+WMc, using
   existing W_q/W_v): +0.025 mag / −0.001 frac — does NOT fix WM on code.
2. **Oracle retrieval vs random control** (`probe_oracle_retrieval.py`, 181
   positions): oracle (best-of-16 buffer slots, peeking) frac 0.249, but
   **matched-norm RANDOM injection scored 0.298** — higher. Oracle − random =
   −0.05 frac ≈ 0. The apparent "oracle lift" was entirely max-of-K selection
   noise. **The WM buffer carries no addressable code-thinking signal**; no
   addressing scheme (trained DKV included) can manufacture it.

**Decision: do NOT build DKV-WM for code.** WM is effective for RECALL (MQAR
+11pp, long-context 98.2%) — short MBPP/HumanEval code generation simply doesn't
exercise long-range recall, so WM has nothing to fetch. PKM (+0.052 frac, the
one real retrieval channel) is modest and already exploited by the base.

**Pivot: execution-grounded objective** (option 4's second half) — the
higher-ceiling code lever that sidesteps per-token retrieval entirely:
supervise/gate thinking only at decision points that flip pass@1, not per-token
Δlogp. Validate cheaply first: does oracle-placed thinking (try a burst at each
position, keep what flips a no-think FAILURE to a pass) flip materially more
problems than the gate's current ~2? If yes, execution-grounded targeting has
headroom; if no, latent thinking's code ceiling is genuinely ~neutral and the
honest move is to stop polishing it.

## Open risks for the validation agents to scrutinize

1. Does turning WM on in the latent thread re-trigger contamination despite
   mem_alpha+premem, for the co-trained adapter specifically?
2. `latent_cotrain_loss` / `_latent_think_logits_grad` currently has NO WM term
   — adding it changes the measured Δlogp primitive. Is that the right place, or
   should RALT use a dedicated grad twin so the no-WM probe stays a valid
   control?
3. Is the WM buffer on a SHORT MBPP problem even populated with anything worth
   retrieving (only past write-gated hiddens of the same short sequence)? If
   the problems are too short, WM has nothing to recall and PKM (parametric) is
   the only real channel — in which case the lever is different.
4. Training W_q/W_proj on code: any path by which this leaks into no-think?
   (Claim: no, WM fires only at think positions. Verify.)
