# What would it take to build an actual intelligent AGENT on OUR architecture?

> Strategy brainstorm, 2026-06-21. Critical, falsifiable, prioritized. Cites our
> mechanisms by name. Where I speculate I say so. The goal is a roadmap that
> exploits what makes us *structurally different* — not "scale up + add tools."

## TL;DR

Our defensible edge is **not** "a smarter small brain." At ~287M–1B trained on
~5B tokens we are, and will remain, **knowledge-poor** vs any frontier or even
mid-size open model (SmolLM2-135M with *half* our params beats us on code because
it saw ~400–800× more tokens — `project_undertrained_not_undercapacity`). No
architecture trick and no distillation (`project_kd_distillation_pipeline`: KD
from a 7B teacher *hurt*, +13% ppl, capacity gap) buys back that gap. Concede
broad knowledge.

The edge is **structural agent economics**: a bounded-state linear-RNN
(DeltaNet `chunk_delta_rule`) gives **constant per-token decode cost and memory
regardless of horizon**, and it can be **rolled past any fixed context window**.
Pair that cheap-but-lossy horizon with an **addressable external memory** (PKM /
`WorkingMemory` / an append-only kNN session store) to recover the recall the
bounded state loses, with **adaptive latent compute** (the gate + state-readonly
latent ponder) spent only on the depth-bound substeps where it provably helps,
and **online RL on verifiable rewards** because *we own the weights*. That stack
targets the one quadrant where a big transformer LLM is structurally expensive
and an API user is structurally locked out: **long-horizon, stateful, verifiable
agentic loops on a narrow domain we can train against.**

---

## A. Architecture advantages — substantiated or debunked

### A1. O(1)/token constant-memory decode (bounded state) vs O(T) KV cache — **REAL, the flagship, but not yet wired**

- **Structural fact.** DeltaNet's recurrent state is fixed-size; PKM is a fixed
  side-table; `WorkingMemory` is a bounded buffer (K≤1024). So per-token decode
  compute and memory are **constant in sequence length**. A transformer's KV
  cache grows linearly with T and per-token attention is O(T). At a 100k-token
  agentic session (tool outputs, file reads, scratchpad), a 7B transformer's KV
  cache is multiple GB and the dominant cost; ours stays flat at ~MB of state.
- **How big.** Asymptotically unbounded advantage; the crossover is wherever
  sessions get long. The *sharpest* form isn't cost, it's **capability past the
  trained window**: a transformer trained at context W is structurally blind
  beyond W (or pays for fragile extrapolation); a DeltaNet can roll state to
  arbitrary T. That is a capability gap, not a cost gap.
- **Catch 1 (quality).** Bounded state = **lossy recall** — it saturates and
  forgets with distance (documented 100%→20% recall at distance 512 once a think
  corrupts it; `project_working_memory_win`: WM only helps *because* the state
  saturates). So the cheap-horizon advantage is **worthless on its own** — it is
  only real *paired with an external memory that restores long-range recall*.
- **Catch 2 (engineering debt).** True state-passing incremental decode through
  `TinyLM.forward_step` is **deferred/unwired** (FiLM K=3 self-feed + WM make it
  non-trivial; generation currently re-processes the prefix and leans on
  `_film_bypass`). So O(1) decode is today a *latent architectural property, not
  a realized capability*. We must build `forward_step` to claim it.
- **Verdict.** The single biggest structural differentiator a frontier
  transformer cannot cheaply copy (they'd have to switch backbones). Contingent
  on (a) wiring `forward_step`, (b) pairing with external memory.

### A2. Persistent external memory surviving beyond the context window — **mostly a build target, NOT structurally unique**

- **As-built we do not have this.** PKM is *parametric* (frozen at inference, not
  runtime-writable); `WorkingMemory` is a *within-sequence* buffer that resets
  each forward. Neither is a cross-session, runtime-writable long-term store.
- **The right primitive exists in our findings:** an **append-only kNN datastore**
  (key = pre-`lm_head` hidden, value = next token / binding) — validated to
  *escape the catastrophic-forgetting wall* because it adds knowledge with no
  gradient (`project_code_thinking_ceiling`). Its known limit — "only near-exact
  recall generalizes, not semantic similarity" — was a *negative for HumanEval
  generalization* but is **exactly right for agent memory**: an agent wants to
  recall *what it literally saw this session* (a file's contents, a prior tool
  result), not generalize across foreign problems.
- **Catch.** Big LLMs already do RAG / external vector stores. So external memory
  per se is **not a moat**. The only unique part is the *integration*: a
  bounded-state backbone that natively reads from addressable memory in-trunk
  (WM read-α, PKM, `mem_ctx_namekey`) at **constant decode cost**.
- **Verdict.** Real value, mostly unbuilt, and only differentiated when fused
  with A1's cheap horizon. Don't sell "RAG" as the edge.

### A3. Adaptive per-token latent compute via the gate — **REAL on depth-bound tasks, with proven safety; NOT a general lever**

- **Mechanism.** The output/thinking gate decides emit-vs-think per token; latent
  thinking (`latent_think_logp`, Coconut-style feedback under
  `state_readonly_at_think` β=0) extends *effective depth* beyond a single
  forward's ~2-hop budget; autonomous halting was validated (avg_steps tracks
  true depth on pointer-chase; fixed-point training makes over-thinking a no-op).
  This is genuine *continuous-bandwidth* test-time compute, finer and cheaper than
  discrete CoT tokens (which cost a full forward + KV growth each on a
  transformer).
- **Where it provably helps (matched bottleneck):** pointer-chase, execution
  traces, arithmetic chains, fixed-point iteration — **+0.47 to +0.80** vs a
  *fully-trained* no-think control (`project_latent_thinking_real_model`).
- **The hard catch.** On open-ended **code generation it is net-negative**
  (OOD; gate anti-aimed, corr(P_think,Δlogp)≈−0.10; ~10% of positions help by
  ~+0.3 logp, **flat across R=1–8** — `project_code_thinking_ceiling`). Code
  next-token is *recall, not iteration*, so iterating doesn't help.
- **Safety is solved (`project_pareto_safe_thinking`):** the `LatentFeedbackAdapter`
  fires *only* at think slots → no-think forward is byte-identical to base
  (proven max|Δlogits|=0); `state_readonly_at_think` β=0 means a think can't
  corrupt the recurrent state/recall. So thinking is **safe to leave on**; a
  verifier best-of (`best_of_think.py` + `code_grader`) makes it strictly ≥
  no-think.
- **Verdict.** Real, demonstrated, *safe* — but only on **depth-bound /
  homogeneous-iterated substeps**. The agentic bet is that real agent loops
  contain such substeps (state simulation, constraint propagation, multi-hop
  lookups). Speculative until shown on an agentic probe.

### A4. Cheap enough for massive parallel self-play / online RL / continual learning on 2 GPUs — **partly real, with real friction**

- **Verifiable-reward RL works on the rig.** `train_rl_grader.py` (GRPO + clipped
  PPO + KL-to-ref, `code_grader` reward) is validated (8→16/164 arc); rollouts
  are batched; cheap decode (once A1 is wired) makes long rollouts affordable.
- **Catch 1.** Latent-thinking + DDP are incompatible (`project_ddp_latent_incompatibility`):
  static_graph+no_sync is an unfixed PyTorch regression and our curriculum
  changes the graph → latent configs run **single-GPU** (manual-allreduce DDP is
  <2× on PCIe-no-NVLink). So "2 GPUs" is ~1.x for the interesting configs. (RL
  *rollouts* can still parallelize as independent inference processes across both
  cards.)
- **Catch 2.** Continual **weight** updates hit **catastrophic forgetting**
  (`project_code_thinking_ceiling`: cram 40 facts → lose ~73% of previously-solved
  problems; PKM does *not* escape it because it's read every token via shared
  addressing). Gentle update (replay 1.0 + KL 0.5) *halves* forgetting while still
  learning; LoRA/EWC untried. So "learn during the task" is real but lossy.
- **Verdict.** Verifiable-reward RL: feasible *now*. True online weight-learning:
  real differentiator, currently unsafe.

### A5. Fully trainable end-to-end by us — **REAL, the foundational moat (but it's an *experimentation* moat)**

- We control weights, reward shaping, architecture, memory mechanisms. API users
  can only prompt (+ maybe frozen fine-tune); open-weight users can fine-tune but
  can't get O(1) decode without our backbone. We can specialize *hard* to a narrow
  environment and do online RL against it.
- **Catch.** This is "we can run the experiment," not "the deployed agent is
  better." It converts to capability only when an experiment finds a real win.
- **Verdict.** The enabling moat behind every other bet; necessary, not
  sufficient.

---

## B. What "an intelligent agent" requires — where token-limit hurts vs where architecture compensates

| Capability | Token-limit impact | Architectural compensation | Net |
|---|---|---|---|
| **Broad knowledge** | **Fatal.** No fix. Confidently wrong (rates own wrong answer ~4.6×). | None (KD refuted). External retrieval supplies *facts seen*, not *facts known*. | **Concede.** Go narrow-domain. |
| **Reasoning / planning** | Hurts open-ended planning (needs knowledge). | Latent thinking + adaptive compute help *depth-bound, homogeneous* substeps (A3). | **Partial** — only structured substeps. |
| **Tool use / env interaction** | Mostly a *training-signal/format* problem, not capacity. Knowing *which* API exists is token-limited. | Cheap rollouts + verifiable reward → RL competent narrow tool-use. | **Winnable** in-domain. |
| **Long-horizon memory & state tracking** | This is our **sweet spot**. | Bounded-state cheap horizon (A1) + WM forced-recall + kNN session store (A2) + state-readonly thinking (can't corrupt bindings). | **Advantage.** Strongest compensate-for-token story. |
| **Self-correction / learning from feedback** | — | We own weights → online RL from execution/verifier feedback. Forgetting is the catch (A4). | **Unique-if-solved.** |
| **Calibration / knowing-what-it-doesn't-know** | Token-limited base is **miscalibrated**. | Don't fix the base — **bolt on external calibration**: cheap verifier-in-the-loop (best-of-N + `code_grader` / type-check / retrieval-hit-confidence). Cheap because decode is cheap. | **Winnable** in *verifiable* domains only. |

The pattern: token-limit dominates anything that needs *broad knowledge or
open-ended generation*; our architecture compensates anything that needs *cheap
unbounded horizon, long-range recall, depth-bound substeps, or verifiable
self-correction*. Build the agent so its hard parts fall in the second column.

---

## C. The unique-to-us bets (ranked by leverage × feasibility on 2×RTX5090)

**Bet 1 — Unbounded-horizon stateful agent at constant cost + external session memory. (LEVERAGE: high · FEAS: medium · FLAGSHIP)**
Wire true state-passing decode (`forward_step`) → O(1) decode → run agents over
100k+ token sessions where a transformer's KV cache explodes *and* past any fixed
window where a transformer is structurally blind. Compensate the bounded state's
lossy recall with (a) an **append-only kNN session store** (escapes forgetting,
near-exact recall = exactly "what did I see earlier"), (b) `WorkingMemory` for
in-context binding recall, (c) `state_readonly_at_think` so latent steps never
corrupt bindings. This fuses A1+A2+A3-safety into the one capability big LLMs pay
linearly/quadratically for and API users can't get at all.

**Bet 2 — Online RL self-improvement on a verifiable environment, owning the weights. (LEVERAGE: high · FEAS: medium-high)**
Extend validated `train_rl_grader.py` (GRPO + execution reward) to *multi-turn*
agentic RL (tool calls, repair loops). Because we own weights + decode is cheap,
the agent keeps improving on the *deployment* task — API agents cannot. The
ownership moat (A5) made concrete. Risk = forgetting for *truly* online updates
(use gentle update / LoRA / periodic offline consolidation).

**Bet 3 — Adaptive latent compute as cheap test-time depth for state-tracking / simulation substeps. (LEVERAGE: medium · FEAS: high)**
Reserve latent thinking for agentic substeps shaped like its *proven* wins:
simulate execution, trace state, multi-hop lookups, constraint/fixed-point
propagation. Structural-floor safety (A3) ⇒ never worse than no-think. Honest
scope: won't help open-ended generation; only depth-bound substeps. Cheapest to
try (mechanism + safety already validated).

**Bet 4 — Verifier-in-the-loop as external calibration. (LEVERAGE: medium · FEAS: high)**
Since decode is cheap, run best-of-N + a cheap verifier (`best_of_think.py`,
`code_grader`, type-check, retrieval-hit-confidence-gated λ) as the agent's
"knowing what it doesn't know," compensating token-limit miscalibration. Only
works in verifiable domains — which is fine, that's where we're playing.

**Bet 5 — Continual/online weight-learning *during* the task. (LEVERAGE: high-if-solved · FEAS: low — PARK)**
The dream differentiator (agent literally learns your codebase into its weights).
Blocked by catastrophic forgetting (unsolved) + DDP friction. Needs an
interference-free store or LoRA/EWC research. Ambitious-phase only.

**Ranked:** 1 > 2 > 3 ≈ 4 ≫ 5. Bets 3 and 4 are *free riders* you can ship while
building 1 and 2.

---

## D. Honest risks — where no architectural cleverness saves us

1. **Broad knowledge is unfixable at our token budget.** Any task needing to know
   an API/library/algorithm/fact → we lose. Concede general-purpose breadth.
2. **KD can't shortcut it** (capacity gap; refuted). More tokens is the only base
   lever, and that's the constraint the hardware imposes.
3. **Latent thinking does not help open-ended generation** — only depth-bound
   substeps. Don't oversell adaptive compute as general reasoning.
4. **Bounded state = lossy recall, and the external memory only does near-exact
   recall.** Our long-horizon agent recalls what it *saw*; it won't *synthesize
   across* distant context as well as a full-attention model. The cheap-horizon
   win is capped by the memory's recall fidelity.
5. **O(1) decode is unbuilt** (`forward_step` debt); the headline advantage is
   theoretical until we ship it.
6. **DDP+latent friction** roughly halves our 2-GPU throughput for the
   interesting configs.
7. **Calibration crutch only exists in verifiable domains** (code/math/typed
   tools). For non-verifiable agent tasks we have neither knowledge nor
   calibration → concede.
8. **The whole strategy is narrow.** We are betting on *long-horizon, stateful,
   verifiable, narrow-domain* agents. General open-domain agents go to big LLMs.

---

## E. Concrete 3-phase roadmap (validates the flagship bet first, cheaply)

The single most promising unique-to-us bet is **#1**. Its load-bearing claim —
the one that, if false, kills the whole flagship — is:

> *External memory recovers the long-range recall the bounded state loses, well
> enough that our model retains and acts on task-relevant state PAST a fixed
> transformer's context window, at flat cost.*

Test that **before** building `forward_step` infra.

### Phase 1 — First cheap experiment (days, single GPU): the "long agentic session" recall-and-act probe

- **Data.** Synthetic multi-turn session: a stream of tool-call/observation turns
  over T ∈ {2k, 8k, 32k} where early turns establish bindings (`file X defines f
  returning C`, var assignments, config values) with distractors, and a late turn
  must *recall and use* one binding (compute with C / call f). This is the
  agent-shaped analogue of MQAR + `var_binding`, where WM is proven load-bearing
  *only when the state saturates* — so push T past saturation deliberately.
- **Arms.** (a) DeltaNet base alone (bounded state → predict degradation with
  distance); (b) + `WorkingMemory` (read-α gate, `mem_ctx_namekey`); (c) +
  append-only **kNN session store** (key=hidden, value=binding); (d) **FAIR
  CONTROL** = a matched-class open transformer with full attention (e.g.
  SmolLM2-360M) — the perfect-recall *and* the *fixed-window* ceiling.
- **Metrics.** (1) recall-and-act accuracy vs binding distance; (2) peak
  GPU memory and tok/s vs T (ours flat, transformer growing); (3) **the killer
  comparison:** accuracy at T beyond the transformer's trained window (train ours
  at T=2048, roll state + kNN to T=32k) where the transformer is structurally
  blind.
- **Falsifiable success criteria.** (i) arm (c) ≥ transformer accuracy within a
  few % *inside* the window; (ii) arms (b)/(c) stay flat in cost while the
  transformer grows; (iii) **past the window, (c) holds recall while the fixed
  transformer drops to chance.** If (i)/(iii) fail — i.e. memory does *not*
  recover the recall — the flagship is downgraded and we pivot to Bet 2/3.
- **Fair-baseline guardrails (mandatory here):** de-leak any kNN store; run the
  100%-trained no-think / mean-vector WM-ablation controls; report cost A/B at
  matched batch. (These are exactly the traps prior loops hit.)

### Phase 2 — Medium (weeks): realize the decode advantage + a real agent harness

- Wire **true state-passing `forward_step`** (FiLM K=3 self-feed + WM buffer +
  per-row think-counter) → measured O(1) decode; regression-test bit-equivalence
  vs the one-shot forward.
- Build a multi-turn **agentic coding harness** (tool use, file read/edit, run
  tests) with the kNN session store + WM as the agent scratchpad, and route
  latent thinking (Bet 3) only to flagged depth-bound substeps (with best-of
  verifier, Bet 4).
- Measure cost + task success on a *long-horizon* agentic benchmark (synthetic
  multi-file repo task, or a small SWE-bench-style subset) vs a transformer
  baseline at matched cost. Headline target: **equal/better task success at a
  fraction of the long-session cost**, not raw HumanEval (the wrong probe —
  `project_humaneval_config_artifact`).

### Phase 3 — Ambitious (months): the ownership moat — online self-improvement

- Extend grader-RL to **multi-turn agentic RL** on the deployment environment;
  the agent improves its *weights* on your codebase/task via verifiable reward —
  what no API agent can do.
- Tackle Bet 5's forgetting safely: gentle update (replay+KL) / LoRA adapters /
  periodic offline consolidation / an interference-free growable store.
- Stretch: continual learning *during* a session (write to kNN now, consolidate
  to weights offline) → an agent that demonstrably gets better at your specific
  environment over time at constant inference cost. *This is the thesis-confirming
  agentic result: per-dollar, per-watt, per-horizon SOTA in a narrow verifiable
  domain — not by being smarter, by being structurally cheaper and trainable.*

---

## Speculation flags

- That real agent loops contain enough *depth-bound homogeneous* substeps for Bet
  3 to pay off is **unproven** — Phase 2 is the test.
- That near-exact kNN recall is "good enough" for agent state (vs needing
  cross-context synthesis) is **plausible but untested** — Phase 1 metric (i)/(iii)
  decides it.
- The "past-the-window capability gap" assumes our state-rolling holds targeted
  recall even though `project_stateful_chunking_notworth` found rolling is a NO-GO
  for *general* next-token CE. The bet is that *targeted recall via kNN* (not the
  rolled state) carries the long range — Phase 1 tests exactly this separation.
