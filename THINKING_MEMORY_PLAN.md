# Thinking ⊗ Memory Unification — Plan & Decision Log

Started 2026-05-29. Owner: this initiative makes the project's two validated-but-
separate bets — **latent thinking** (`THINKING_LATENT_2026_05_28.md`) and
**memory** (PKM static + WM dynamic) — into ONE co-trained mechanism, and wires it
through pretrain → SFT → RL.

## Super-coder thesis (crisp)

A small DeltaNet model punches above its weight on coding because **knowledge
lives in scalable memory (PKM facts + WM context) and is composed by latent
thinking**, instead of being crammed into dense weights. The recurrent state is
bounded (linear-RNN, saturates); memory is the scalable store; thinking is the
composition engine. Hardware budget: 2× RTX 5090.

## The unified mechanism — "think = retrieve + compute, write scratchpad"

At each THINK step (gate-triggered):
1. **Read**: query WM (dynamic buffer) + PKM (static table) with the current
   thought-latent.
2. **Feed-as-input**: fuse the retrievals and feed them as the next position's
   input embedding (retrieval-as-input; additive over a think baseline so a
   useless read can't overwrite). This is how the model pulls in NEW information
   as it thinks (user's design, 2026-05-29).
3. **State-readonly (β=0)**: the think READS the DeltaNet recurrent state but does
   NOT write it → the input/context bindings the recurrence carries are protected
   (fixes "thinking corrupts recall").
4. **Write scratchpad**: the think's intermediate latent IS written to the WM
   buffer (write-gated). So later think steps can retrieve earlier thoughts — WM
   becomes a working scratchpad, not just a context cache.
5. **FiLM** propagates the deep-layer computation from think-step r to r+1, so the
   running computation is carried across steps without needing explicit
   hidden-feedback (open question — validate; fall back to hidden-feedback hybrid
   if FiLM doesn't carry it).

The gate decides emit-vs-think and when to halt.

## Memory WRITES — the crux (three stores, three write policies)

| store | what it holds | write policy | read policy |
|---|---|---|---|
| **DeltaNet recurrent state** | input/context bindings | written at EMIT (normal β); **β=0 at THINK** (protected) | read everywhere |
| **WM buffer** (dynamic) | context + self-generated intermediate thoughts | write-gated at ALL positions INCLUDING think (think writes its scratchpad) | soft-attention read at THINK |
| **PKM** (static) | learned facts | parametric — "written" only by gradient during pretrain; read-only at inference | top-k read at THINK (thought-latent query) |

**Separation principle:** recurrence = *protected bindings*, WM = *mutable
scratchpad* (self-writes + context), PKM = *static knowledge*. Thinking can
corrupt neither the bindings (β=0) nor the facts (PKM read-only); it freely uses
the WM scratchpad.

**Why writes matter:** without WM writes during thinking, a think step can only
combine what's already retrievable — it can't record an intermediate conclusion
for a later step. The scratchpad write is what lets multi-step thinking
accumulate (e.g. hop r writes f^r(s); hop r+1 retrieves it).

## Training across phases

### Pretrain (`train_lm.py`)
- Wire the full mechanism; `--state_readonly_at_think`, `--output_gate`,
  `--gate_entropy_aux_weight` ON.
- Think-burst injection (existing, `data_mix.py`) gives think positions so
  reads+writes receive gradient from step 0.
- **Add memory-AND-depth-required synthetic data** (mixed in) with a curriculum,
  so the mechanism is *load-bearing*, not decorative. Natural-text LM loss alone
  won't force memory-thinking; the model must face tasks it can only solve by
  retrieving-while-thinking.

### SFT (`sft_code.py` / `latent_sft.py`)
- Continue the co-trained mechanism on code data (think-before/within solution).
- Co-train from day 1 — bolt-on was always inert.

### RL (`train_rl_grader.py`)
- Rollouts insert think; **gate-as-policy** (`--stochastic_gate`) + grader reward
  teach *productive* thinking and *when* to retrieve; KL anchor for stability.
- Memory reads/writes get terminal-reward gradient — the only signal that can
  teach "retrieve the thing that makes the code pass."

## Validation ladder (cheap → decisive, before scaling)

1. **Synthetic, memory-required + depth-required** (FiLM model): a fact table too
   big for the bounded recurrent state, so each reasoning hop MUST retrieve from
   WM/PKM. Three-way ablation — thinking-only (state saturates) fails,
   single-retrieval fails, **retrieve-while-thinking succeeds**. Answers "does
   FiLM carry the thread" and "do writes enable accumulation."
2. **Real 287M**, memory-required reasoning probe (arith/recall on the real
   tokenizer) with the unified mechanism co-trained.
3. **Scale** only the validated unified stack.

## Implementation steps

- [ ] S1. Synthetic harness: memory-required task + unified read/write think loop
      on a small FiLM+WM(+PKM) model; three-way ablation. (`experiments/latent_mem.py`)
- [ ] S2. Model: surface PKM per-position retrieval; fuse WM+PKM into the
      think-step input; β=0 at think; WM write-at-think. (`model.py`, `memory_layer.py`)
- [ ] S3. Pretrain wiring + memory-required curriculum data.
- [ ] S4. RL wiring (gate-as-policy rollouts read/write memory).
- [ ] S5. Scale.

## Decision log

**D1 (2026-05-29) — Scope the first build to the synthetic three-way ablation.**
Decided: before touching the 287M model or pretrain, validate the unified
mechanism on a small FiLM+WM model on a memory-required task. Why: it's cheap,
decisive, and answers the two open questions (does FiLM carry the running
computation across think steps; do WM writes enable multi-step accumulation)
that the no-FiLM/no-memory `latent_think` synthetic could not. Gate: with-memory-
thinking must beat both thinking-only and single-retrieval by a wide margin, or
the hidden-feedback hybrid is needed.

**D2 (2026-05-29) — WM is the thinking scratchpad; recurrence stays protected.**
Decided: think steps WRITE to the WM buffer (write-gated) but NOT to the DeltaNet
recurrence (β=0). Why: separates mutable working memory from protected input
bindings — resolves the long-standing "thinking corrupts recall" tension while
still letting multi-step thinking accumulate intermediate results. PKM stays
read-only at runtime (parametric; learned by pretrain gradient).

**D4 (2026-05-29) — Code audit in parallel with S1.** Launched a read-only audit
of the thinking+memory paths (model.py forward feature-composition, the
state-readonly b_proj hook under FiLM multi-pass, WM read/write + injection-grad
stashes, the iterative think-loop gradient/index alignment, the inference
generators). Why: the unified mechanism composes many features (inputs_embeds +
think_mask + FiLM K-self-feed + WM + β=0) — composition bugs / silent no-ops are
the top risk. Findings folded into S2 before wiring into the real model.

**D5 (2026-05-29) — Audit results: no blocking bug; two real issues + one S1
validity caveat.** The code audit found NO blocking correctness bug; the
state-readonly hook, WM read/write + causal mask, deep-supervision index
alignment, and inference generators are all sound, and FiLM *does* preserve
`inputs_embeds` across passes. Action items: (a) **`_film_bypass=True` during
`latent_sft` training FROZE FiLM** — contradicts "co-train FiLM to carry think
state"; if we revisit latent-SFT, train with FiLM live. (b) The gate is computed
on the POST-memory hidden — fine for now (we force R, no gate), revisit when the
gate halts. (c) **S1 validity caveat:** WM re-retrieves every think step, so the
ablation cannot cleanly separate "FiLM carried the running state" (D3) from "WM
re-retrieved it" — need a cleaner D3 probe (e.g. freeze the read, or test
FiLM-carries with mem OFF).

**D6 (2026-05-29) — S1 (FiLM+WM, N=40) does NOT learn (loss pinned at chance for
1800 steps).** Not slow bootstrap — zero learning even at K=1. Since
`latent_think` (no FiLM) and `latent_arith` (feedback=none) both learned fast,
and FiLM-ON was never tested with the think loop before, the suspect is the
FiLM(K=3-self-feed) × think-loop interaction (or memory co-learn). Running a
feature-toggle isolation sweep (FiLM/mem on/off) to pinpoint. [diagnosis pending]

**D7 (2026-05-29) — D6 RESOLVED: no bug; composition works. Now find the
crossover.** N=12: baseline (no FiLM/no mem) learns to 1.000 → harness/composition
fine. Full (FiLM+WM+thinking+β0) ALSO learns to 1.000 with a clean ablation
(no_think .09 / single_read .10 / think_only .23 / **both 1.00**) → the unified
mechanism composes and retrieve-while-thinking is load-bearing. BUT N=12 doesn't
*require* memory (recurrent holds 12 bindings), and N=24 failed for BOTH paths.
The money-shot experiment: sweep N to find the crossover where recurrent-only
(baseline) fails but memory rescues. If memory also walls at the same N as
recurrent, switch to a distractor-based memory-required task (recurrent forgets,
WM retains). Earlier "non-learning" was under-training at too-large N, not a bug.

**D8 (2026-05-29) — Two findings: (a) retrieval-as-input < hidden-feedback;
(b) the α-noise bug.** Crossover sweep: recurrent hidden-feedback holds N≤20
(think_only=1.0); retrieval-as-input "both" fails at N=20 (0.07). → **D3 refuted:
the HIDDEN feedback carries the running computation; FiLM does not.** Promote the
**hybrid** (hidden + α·retrieval). First hybrid attempt also failed at N=20
(0.035!) — because I used a FIXED α=0.5 on untrained-WM noise, swamping the
hidden thread (violates the FiLM-α/PKM-α/retrieval_input_alpha "init-small,
grow" lesson). Fix: **learned α init 0.1, no WD** so hybrid starts == pure
hidden-feedback and grows retrieval only if useful. Also: table-size is the
WRONG knob for "memory-required" — recurrent capacity (~N20-32) and WM
learnability overlap, so no clean band exists at this size. **Switch the
memory-required probe to distractor-based long-context recall** (the regime where
the repo already validated DN+memory breaking through), at small WM-storable N.

**D9 (2026-05-29) — CLEAN ISOLATION (N=20, equal 4000 steps): FiLM×WM is the
breaker.** A baseline 1.000 ✓ · B FiLM-only 0.999 ✓ · C **memory-only hybrid
1.000 ✓** · D FiLM+memory ALL chance ✗. So FiLM alone trains, thinking+memory
(hybrid) alone trains, but the **combination breaks learning in the iterative
think loop**. (Production pretrain uses FiLM+WM fine → it's specific to the
think-loop × FiLM-multipass × WM-read-as-input interaction; likely the K=3
self-feed warmup passes recomputing WM and feeding the lagged source states.)
Two consequences: (1) the CORE BET is validated — hybrid thinking+memory learns
and is load-bearing; (2) a real FiLM×WM think-loop bug to fix (try
feedback_self_k=1, or skip the FiLM multipass during think steps). **Plan:
proceed on the working memory-only config to get the decisive memory-required
result (distractor long-context), debug FiLM×WM in parallel.**

**D10 (2026-05-29) — Tiny synthetic CANNOT cheaply force "memory-required"; the
mechanism works, validate memory at real scale.** Distractor probe (mem-only,
N=8, K=3): distract=128 no-curriculum broke BOTH paths (think_only 0.262, hybrid
0.266 — WM didn't bootstrap). With a distractor CURRICULUM, **think_only ALSO
recovered to 0.999** — i.e. the bounded recurrent state, trained gradually,
holds 8 bindings through 128 distractors by itself. So at tiny scale the
recurrent state is proportionally too capable: the regime where memory is
strictly required (table > recurrent capacity) overlaps the WM-bootstrap limit →
no clean synthetic window. CONCLUSION: (1) the unified thinking+memory mechanism
**composes and learns** (hybrid=1.000) with the validated design (hidden-feedback
thread + learned-α retrieval augmentation); (2) memory's load-bearing value is
already established at PRODUCTION scale (this session: PKM value rows 100%
learned + broad slots; repo: WM 98.2% long-context recall, PKM −5 HumanEval
ablation). So the memory-required *validation* belongs at real scale/long real
contexts, not a tiny synthetic. **Revised next steps: (S2) FIX the FiLM×WM
think-loop bug (try feedback_self_k=1 / skip multipass during think); (S3) wire
the hybrid think-reads-memory mechanism into the real model where memory is
already load-bearing, with a real long-context curriculum.**

**D11 (2026-05-29) — FiLM×WM RESOLVED: bypass the FiLM multipass during think
steps. ALL FOUR BETS NOW CO-TRAIN.** FiLM(K=3)+WM+thinking+β0 with
`--film_bypass_think` → **hybrid=1.000** (was chance). Root cause: the FiLM K=3
self-feed MULTIPASS, run on every think-step forward and chained across the
iterative retrieval-feedback loop, breaks the gradient/optimization; bypassing it
(single-pass FiLM during think) fixes it. Principled + deploy-faithful: full FiLM
in the base/prompt forward (per-token depth), single-pass FiLM during think
(think-depth comes from hidden-feedback iteration; FiLM deploys single-forward
anyway). **Net: the unified mechanism = hidden-feedback thread + learned-α WM/PKM
retrieval + β0 (protect bindings) + full-FiLM-base / single-pass-FiLM-think — all
features compose and learn (hybrid=1.000).** Real-model wiring (S3) should adopt
exactly this: `_film_bypass` during the think burst.

**D3 (2026-05-29) — Validate the user's "retrieval-as-input + FiLM-carries-state"
design first, hybrid as fallback.** Decided: feed the *retrieval* as the think
input (not the raw hidden) and rely on FiLM to carry the running computation
across think steps; only add explicit hidden-feedback (hybrid) if S1 shows FiLM
doesn't carry it. Why: it's the cleaner bet, reuses existing FiLM machinery, and
is the design the user specified.
