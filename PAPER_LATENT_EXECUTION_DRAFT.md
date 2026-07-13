# Latent Execution: Internalizing Interpreter Traces into Continuous Thoughts

*Draft — 2026-07-13. Numbers are quoted verbatim from the run JSONs cited in
each table; nothing here is recomputed. `[PENDING: …]` marks a result a
currently-running experiment will fill in.*

**Second headline:** *externalize before you internalize.*

---

## Abstract

Prior work supervises latent chain-of-thought steps against *model-generated
text* rationales (SIM-CoT, 2509.20317), and separately trains language models
on interpreter execution traces rendered as *text* (Code World Model, Meta
FAIR, 2025). We combine the two into **latent execution**: a program
interpreter provides free, dense, *verifiable* per-step ground truth for a
sequence of continuous thoughts, so that each latent step decodes — through the
model's own shared output head — to the true machine state at that point in the
computation. We demonstrate this in a 402M bounded-state DeltaNet-class linear
RNN (linearized from SmolLM2-360M), where latent deliberation costs zero
context in an O(1)-decode model.

Our central experiment is a controlled *staging cell*. Trained latent-first,
with dense per-step interpreter supervision from the start, the model fails: all
latent-depth arms sit at ≈9–13% exact-answer and per-hop decode pins at the
value prior (hop cross-entropy ≈ ln 10). Trained staged — first teach the
executor function as a *text* scratchpad (Stage A), then gradually compress the
text steps into latent steps (Stage B, Coconut-style) — the same data and the
same per-hop machinery instead succeed: latent thoughts carry real program
state (per-hop decode 0.844 at K=4 vs 0.11 latent-first), and R=K latent steps
beat the same checkpoint's no-scratchpad baseline by +50.7/+51.7/+45.7pp at
K=4/5/6. The computation is *depth-true*: R=1 (under-thinking) and R=K+4
(over-thinking) both collapse; only R=K works. This is, to our knowledge, the
first working latent compression of a real (executed, not templated) program
trajectory, and the sharpest empirical statement to date of the
theoretically-predicted necessity of token-space competence *before* latent
internalization (2602.01148).

We report the limits plainly: the mechanism does not transfer zero-shot to
out-of-distribution real programs (CRUXEval latent arms ≈1.5–4%), though a
direct-answer *internalization* signal on CRUXEval rises 0.0688→0.1050
(z=2.58, n=800) — a promising but token-count-confounded observation whose
control is in flight. Latent state saturates at a ~6-hop horizon. Results are a
single model, single scale, single seed.

---

## 1. Introduction

Continuous ("latent") chain-of-thought replaces discrete reasoning tokens with
the model's own hidden state fed back as the next input embedding (Coconut,
2412.06769). It promises higher-bandwidth deliberation than text tokens and, in
a bounded-state recurrent model, deliberation that costs *no context* at all.
But two problems have kept latent reasoning narrow. First, **the supervision
targets for latent steps are unknowable** in general: outcome-only training over
latent steps provably collapses (2606.20075), and the leading fix supervises
latents against *teacher-generated text* rationales (SIM-CoT), which are
themselves noisy and unverifiable. Second, **latent-first training is unstable**:
theory predicts that a curriculum externalizing computation into tokens before
internalizing it is necessary (2602.01148), but the controlled empirical cell
isolating *why* has been missing.

Code is the one domain where both problems dissolve at once. Execute a program
and every intermediate interpreter state is a *free, infinite, and verifiable*
supervised target — one per latent step. This paper asks whether that free
process supervision lets a small bounded-state LM internalize program execution
into continuous thoughts, and what training regime is required to do so.

Our contributions:

1. **Latent execution (the C∩B combination).** We supervise each continuous
   thought against the *ground-truth machine state* from a real `sys.settrace`
   execution, decoded through the model's shared LM head. Both parent lines
   leave this hole open: execution-trace work keeps state in text (CWM, Neural
   Debugger 2603.09951, Code-Exec-as-Grounded-Supervision 2506.10343); latent-
   step-supervision work uses model/teacher text as the target (SIM-CoT). The
   interpreter supplies dense, verifiable process supervision — the answer to
   the "latent supervision targets are unknowable" critique.

2. **The staging cell.** In a controlled comparison holding data and the
   per-hop loss fixed, dense per-step supervision is *not sufficient*
   latent-first (§4.2): compression into latent space succeeds only once the
   computation already exists in token space (§4.4). This is the sharpest
   empirical evidence to date for the theoretically-predicted necessity of
   externalization-before-internalization (2602.01148); it also disconfirms the
   *natural reading* of SIM-CoT ("just supervise the steps"), since SIM-CoT
   never trains from a token-incompetent base.

3. **A depth-true signature** (§4.5). R=K latent steps work; R=1 *and* R=K+4
   both collapse; per-hop decode is 0.844 at K=4. This rebuts the line that
   latent tokens are pseudo-reasoning placeholders (Do Latent Tokens Think?
   2512.21711; 2411.15862): the latent steps are doing genuine sequential
   computation.

4. **Bounded-state framing** (§3.1). This is, to our knowledge, the first
   continuous-thought reasoning in a DeltaNet-class linear RNN; because latent
   steps consume no context and the model's decode state is O(1), internalized
   execution provides zero-context-cost deliberation for long-horizon coding
   agents.

We explicitly do **not** claim that latent reasoning works in general (Coconut),
that per-step supervision stabilizes latent reasoning in general (SIM-CoT's
headline / 2606.20075's theorem), that LMs can simulate Python execution (CWM at
32B in text — our Stage A alone is a small-scale replication of that with zero
novelty), or that curriculum is necessary as a bare claim (2602.01148). Our
delta is the *combination* and the *controlled staging cell*.

---

## 2. Related Work

**Latent / continuous chain-of-thought.** Coconut (2412.06769, COLM 2025)
established that feeding a model's last hidden state back as the next input
embedding lets it reason in a continuous space; Token Assorted (2502.03275) and
CODI (2502.21074) explore mixed and self-distilled variants; surveys 2505.16782
and 2509.02350 map the area. Stepwise Internalization (2405.14838) gradually
removes explicit CoT tokens during training — the curriculum ancestor of our
Stage A→B replacement. Huginn (2502.05171) recurs a latent block at test time
for extra depth; Tiny Recursive Mamba-2 (2602.12078) recurs inside a linear-RNN
block. Our Stage B is Coconut's gradual text→latent replacement, but the
replaced tokens are *interpreter-state* scratchpad lines and each latent step is
supervised against the true state rather than only the final answer.

**Supervising the latent steps.** SIM-CoT (2509.20317, ICLR 2026) adds an
auxiliary decoder that aligns each latent step to a token-level teacher
rationale, stabilizing latent training and preventing collapse at higher latent
counts. Information-theoretic analyses (2606.20075) prove that outcome-only
latent supervision collapses and that per-step signal is needed; "Thinking
States" (2602.08332) studies what latent states can encode. Our per-step target
is not teacher *text* but the *ground-truth interpreter state*, which is dense,
free, and verifiable — sidestepping the noisy-target problem these methods work
around.

**Capabilities and limits of latent CoT.** "Capabilities & Fundamental Limits
of Latent CoT" (2602.01148) proves a curriculum (externalize, then internalize)
is necessary for latent CoT; the Depth Ceiling paper (2604.06427) argues latent
planning capacity barely scales with more latent steps; "Do Latent Tokens
Think?" (2512.21711) and 2411.15862 question whether latent tokens perform real
computation or act as placeholders; "Soft Tokens, Hard Truths" (2509.19170)
reports a practical ~6-continuous-token limit for Coconut-style training. Our
staging cell is the controlled empirical instance of the 2602.01148 prediction;
our depth-true signature answers the placeholder critique; our ~6-hop horizon is
suspiciously close to the Soft-Tokens limit (§5).

**Execution traces and neural interpreters.** The Neural Programmer-Interpreter
(Reed & de Freitas, 2016) learned to imitate execution traces of algorithms;
the Scratchpad (Nye et al., 2021) taught transformers to emit intermediate
computation steps as text; LaSynth (2107.00101) and Jin & Rinard (2305.11169)
study learning-to-execute. Recent code models train on interpreter traces
rendered as text: Code World Model (CWM, Meta FAIR, Sep 2025, 32B), the Neural
Debugger (2603.09951), Code-Exec-as-Grounded-Supervision (2506.10343), and
SemCoder (2406.01006). All keep the execution state in *text*. Our Stage A is a
small-scale replication of this text-executor idea (and we claim no novelty for
it); the novelty is compressing that verified trace into *latent* steps
(Stage B). CRUXEval (2401.03065) is our out-of-distribution transfer probe for
input/output-of-execution prediction.

---

## 3. Method

### 3.1 Bounded-state substrate and continuous thoughts

The model is a 402M-parameter TinyLM whose sequence mixer is **DeltaNet**
(`--arch deltanet`, plain `chunk_delta_rule`) — a linear-attention RNN with a
*bounded* recurrent state and O(1)-per-token decode, linearized from
SmolLM2-360M by weight inheritance (32 layers, d_model 960, 15 heads × 64,
d_ff 2560, vocab 49152 tied). All experiments start from the same linearized
checkpoint, `feature_pilot_A.pt` (HumanEval-solution cross-entropy 0.7429).

A **latent thought** is one recurrence step in which the trunk's own last
hidden state, passed through a small `LatentFeedbackAdapter`, becomes the next
input embedding — Coconut's continuous feedback, realized inside a linear RNN.
Latent steps are run **state-readonly** where applicable (DeltaNet write-gate
β→0 at think positions) so the deliberation cannot corrupt the recurrent state
that carries long-range bindings. Because latent steps append to the growing
thread but are never emitted, K latent steps of computation cost *zero* output
tokens and leave decode complexity O(1) — the property that makes internalized
execution attractive for long-horizon agents (the bounded-state / agent-
economics framing).

The think-path is attached as an *adapter only*: with the adapter contributing
nothing (R=0), the forward is byte-identical to the base model — a structural
floor verified in the N1′ eval (`max_abs_delta = 0.0` over 3 prompts;
`runs/n3_killtest_N1prime.json`, `structural_floor_check`). Thinking can thus be
left on safely: it can never make the no-think path worse.

### 3.2 Execution traces as free, verifiable per-step supervision

Data is generated by `experiments/gen_exec_traces.py`: 30,100 validated
examples over rungs K=2–8 (plus length-generalization K=9/10/12), each a real
Python program whose ordered *state-change events* for one tracked variable are
read off a real `sys.settrace` execution — never hand-computed — inside a forked,
resource-limited subprocess, and independently re-verified with `--validate`.
Two sources: (a) synthetic-but-real Python (assignments, augmented ops,
value-dependent branches, small accumulator loops, table lookups, with
distractor scalar/list/dict mutations interleaved); (b) MBPP gold solutions run
on their real test call, keeping the most-assigned local variable's trajectory.
Values are mod-10 by construction so every intermediate is a single BPE token
(the single-token constraint keeps the per-hop loss drop-in; multi-token values
are future work). Each record's `intermediates` field is the ordered list of K
post-states; `answer` is its last entry.

The key property: the interpreter gives one *ground-truth, verifiable* target
per computation step, for free and without bound — precisely the supervision
signal that latent-step methods otherwise lack.

### 3.3 Two-stage training: externalize, then internalize

**Latent-first (the failed baseline, N1′).** Full fine-tune from
`feature_pilot_A`, training the latent thread with dense per-hop supervision:
latent slot j (1-indexed, at absolute position P+j−1) is decoded through the
shared `out_norm → lm_head` — with no causal shift — against `intermediates[j−1]`,
and the final slot predicts the answer. Total loss = answer CE + per-hop CE
(`experiments/latent_reasoning_cotrain.py::_answer_span_latent_loss`,
`perhop_weight` term). This asks the model to invent the executor function *and*
its compressed latent encoding simultaneously.

**Stage A — text-scratchpad executor.** The same programs rendered as *text*:
`prompt` + `# trace:` + one `# step j: x = v` line per state change + `# final:
N`. Trained as a 0.15-weight mix stream over the pilot mix ×0.85, full fine-tune
from `feature_pilot_A`. The executor function is learned in native token
machinery, with the interpreter states as ordinary next-token targets.

**Stage B — Coconut text→latent replacement.** Full fine-tune from the Stage-A
checkpoint with a fresh `LatentFeedbackAdapter`. A curriculum stage `s` replaces
the **first `s`** text-trace steps with `s` latent slots:

```
prompt + "# trace:\n" + [s latent slots] + remaining text step lines
         (original numbering kept) + "# final: N"
```

The loss is CE on the remaining text + final line (teacher-forced) *plus* per-hop
CE decoding latent slot j → `intermediates[j−1]` — the identical N1′ machinery
(`_answer_span_latent_loss_batched` with R=s, "solution" = remaining trace text +
final line, per-hop targets = `intermediates[:s]`). At s=K the trace is fully
latent. `s_max` ramps 0→8 over the first ~55% of steps (sampling s near the
frontier), then a consolidation phase samples s ~ uniform{0..min(K,8)} (the
validated ramp-then-consolidate recipe). The Stage-A mix stays on as the
background anchor keeping the text executor alive. Single GPU (a known
DDP+latent incompatibility). ~2,600 steps / ~340M tokens.

At inference the latent arm injects R=K latent slots after `# trace:`, greedily
decodes the continuation, and parses `# final: N`
(`experiments/eval_exec_trace_latent_trace.py`).

---

## 4. Experiments

### 4.1 Setup

All checkpoints derive from `feature_pilot_A`. Evals use n=300 held-out
examples per rung K unless noted; the held-out programs are drawn from the same
generator but never seen in training. We report exact-answer match, per-hop
state-decode accuracy, and (for code-competence guarding) HumanEval-solution
cross-entropy (HE-CE, lower is better). CRUXEval transfer uses n=200 (mechanism
arms) and n=800 (direct-answer sample). Single seed (0).

### 4.2 The staging cell — latent-first fails (N1′)

Trained latent-first with dense per-hop supervision, the model does not learn to
execute. **Source: `runs/n3_killtest_N1prime.json`.**

| K | no_think (R=0) | wrong_low (R<K) | latent R=K | wrong_high (R=K+4) | per-hop acc | lift vs no-think (pp) |
|---|---|---|---|---|---|---|
| 2 | 0.0867 | 0.110 | 0.110 | 0.0833 | 0.1417 | +2.3 |
| 3 | 0.0967 | 0.0733 | 0.0933 | 0.1267 | 0.1056 | −0.3 |
| 4 | 0.090 | 0.130 | 0.110 | 0.1167 | 0.1142 | +2.0 |
| 5 | 0.100 | 0.1133 | 0.080 | 0.090 | 0.110 | −2.0 |
| 6 | 0.1167 | 0.1233 | 0.1133 | 0.1233 | 0.0944 | −0.3 |
| 7 | 0.090 | 0.1067 | 0.110 | 0.110 | 0.1138 | +2.0 |
| 8 | 0.130 | 0.1133 | 0.1067 | 0.1133 | 0.1029 | −2.3 |

(greedy_exact shown; teacher-forced is within ≤1pp.)

All four latent-depth arms are statistically indistinguishable at every K
(no_think 8.7–13.0%, latent R=K 8.0–11.3%), and the pre-registered kill-line
(≥5pp lift at K≥4) is not met at any K (lift −2.3..+2.3pp). Per-hop decode
accuracy is 9.4–14.2%, matching the **marginal distribution over
single-digit values**: the training hop cross-entropy plateaued at ≈2.2 ≈ ln 10.
The latent slots learned the *prior over values*, never the simulation. The
structural floor held exactly (no-think byte-identical to the base).

We attribute the failure to three stacked issues: (1) an **inverted capability
gradient** — the base cannot do the task in *text* either (direct answer
≈9–14% ≈ digit prior), so latent training must invent the executor and its
compression at once, through R fragile bottleneck vectors; (2) **skipped
staging** — the Scratchpad→Coconut path teaches the function in tokens first;
(3) a **marginal-stats attractor** — under CE, matching the value prior is a
deep local optimum reachable without simulation.

### 4.3 Stage A — text-scratchpad executor (decisive pass)

Teaching the executor as text works cleanly. **Source:
`runs/stageA_executor_eval.json`.**

| K | trace step-acc | answer w/ trace | answer direct | lift (pp) |
|---|---|---|---|---|
| 2 | 0.980 | 0.9567 | 0.160 | +79.7 |
| 3 | 0.9867 | 0.970 | 0.1367 | +83.3 |
| 4 | 0.9875 | 0.970 | 0.150 | +82.0 |
| 5 | 0.9713 | 0.9633 | 0.1233 | +84.0 |
| 6 | 0.9806 | 0.950 | 0.1467 | +80.3 |
| 7 | 0.9633 | 0.9367 | 0.1367 | +80.0 |
| 8 | 0.9667 | 0.9467 | 0.1167 | +83.0 |

Trace step-accuracy is 0.96–0.99, answer-with-trace 0.94–0.97 vs direct
0.12–0.16 — a **+80–84pp causal lift** from writing the trace. The base model
control (`feature_pilot_A`, n=30/rung) has 0% format compliance and 0%
trace/answer accuracy — a clean null. Critically, executor training *improved*
general code competence: **HE-CE 0.7343, better than the base's 0.7429** — the
best code checkpoint in the project to date. Length-generalization is partial:
past the trained max depth the model stops at ~8 steps and reads off the wrong
line (K=9/10/12 step-acc 0.79/0.73/0.60, answers collapse to 0.09/0.02/0.00) — a
curriculum-exposure artifact, not a wall. Stage A on its own is a small-scale
replication of the text-executor idea (CWM at 32B); we claim no novelty for it.

### 4.4 Stage B — latent execution (not killed; mechanism validated)

Compressing the text trace into latent steps, staged from the Stage-A
checkpoint, works. **Source: `runs/stageB_latent_trace_eval.json` (per-hop
column re-measured in `runs/stageB_perhop_remeasure.json`; see below).**

| K | text step-acc | text-trace answer | direct answer | latent R=K answer | latent R=1 | latent R=K+4 | per-hop (R=K) | kill lift (pp) |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.9383 | 0.9033 | 0.2567 | 0.140 | 0.1867 | 0.120 | 0.925 | — |
| 3 | 0.940 | 0.8933 | 0.1933 | 0.590 | 0.2267 | 0.1833 | 0.9089 | — |
| 4 | 0.9158 | 0.860 | 0.1233 | 0.630 | 0.1533 | 0.140 | 0.8442 | +50.7 |
| 5 | 0.8987 | 0.8333 | 0.1167 | 0.6333 | 0.1367 | 0.220 | 0.8227 | +51.7 |
| 6 | 0.890 | 0.820 | 0.130 | 0.5867 | 0.1967 | 0.2167 | 0.7817 | +45.7 |
| 7 | 0.8876 | 0.760 | 0.1333 | 0.4967 | 0.1267 | 0.2167 | 0.7252 | +36.3 |
| 8 | 0.8467 | 0.7267 | 0.1367 | 0.3333 | 0.0467 | 0.260 | 0.6321 | +19.7 |

Against the pre-registered lines:

- **KILL line — NOT tripped.** Full-latent R=K answer beats the *same
  checkpoint's* direct (no-trace) baseline by +50.7/+51.7/+45.7/+36.3/+19.7pp at
  K=4..8 (bar: ≥5pp at K≥4).
- **Mechanism gate — PASS.** Per-hop latent-slot decode is **0.844 at K=4** (bar
  0.50; N1′ was 0.11). During training the per-hop CE broke the N1′ digit-prior
  plateau (~2.30 flat for N1′'s entire run) around step ~750 as the ramp
  deepened, falling to ~0.43–0.89 in consolidation.
- **Success bar — PARTIAL.** Retaining ≥50% of the Stage-A lift means R=K answer
  ≥ ~0.55: K=4/5/6 pass (0.63/0.63/0.59); K=7 (0.497) and K=8 (0.333) fail. So
  latent thoughts carry the computation to ~6 hops.
- **Code guard — PASS.** HE-CE 0.7491 (bar 0.755; Stage A 0.7343).

The text-trace skill degrades but survives on the Stage-B checkpoint (heldout
text answer 0.73–0.90 vs Stage A's 0.94–0.97) — a real but bounded cost of the
latent compression.

**Per-hop measurement note (do not omit).** As first run, the per-hop column read
0.000 structurally — a units bug in the eval harness that compared argmax *token
ids* to *raw int* values. It was fixed (`encode_inter_token_ids` in
`eval_exec_trace_latent_trace.py`) with a regression test and re-measured on the
identical checkpoint and records; the values above are the corrected ones
(`perhop_note` in `runs/stageB_latent_trace_eval.json`). The generative arms
were unaffected by this bug.

**The staged path is what made the difference.** N1′ and Stage B use the *same
data* and the *same per-hop machinery* and reach *opposite* outcomes. The only
difference is that Stage B starts from a checkpoint that already executes the
programs in text. This is the controlled cell for 2602.01148's prediction:
dense per-step supervision is not sufficient latent-first; prior token-space
competence is necessary. (On the confound that "the latent-first base is just
weaker," see §5 — the three checkpoints are matched on general code competence
to within ~0.015 HE-CE.)

### 4.5 Depth-true signature

The latent computation is genuinely sequential, not a bag of placeholder
tokens. From the Stage-B table (§4.4): only **R=K** works. **R=1**
(under-thinking) collapses to 0.05–0.23, near the direct/prior floor, and
**R=K+4** (over-thinking) collapses to 0.12–0.26. A placeholder mechanism would
be insensitive to R; a depth-true one needs exactly K steps for a K-hop program
— which is what we observe, replicating the synthetic pointer-chase depth
signature on real execution traces. The per-hop-by-step profile confirms it:
hops 1–5 decode at 0.78–0.96 across every K, with the drop always at the *last*
required hop (e.g. K=8: [0.947, 0.917, 0.893, 0.857, 0.773, 0.577, 0.093,
0.000]).

### 4.6 Transfer to CRUXEval — the negative, and an internalization signal

We probe whether the executor mechanism transfers to real, out-of-distribution
programs using CRUXEval-style output prediction.

**Mechanism transfer — NEGATIVE (report plainly).** On CRUXEval (n=200, arms
`direct` / `text_trace` / `latent_R4` / `latent_R8`), the trace and latent arms
do *not* transfer. **Sources: `results/cruxeval_transfer_{feature_pilot_A,
stageA_executor,stageB_latent_trace}.json`.**

| checkpoint | direct | text_trace | latent R4 | latent R8 |
|---|---|---|---|---|
| feature_pilot_A (base) | 0.075 | 0.000 | 0.000 | 0.005 |
| stageA_executor | 0.095 | 0.095 | 0.015 | 0.000 |
| stageB_latent_trace | 0.130 | 0.020 | 0.015 | 0.040 |

The Stage-B latent arms sit at ≈1.5–4% — the executor learned on our synthetic
generator does not fire on CRUXEval's distribution (different program shapes,
multi-token values, etc.). This is the honest ceiling on the current result:
**synthetic programs only.** (Note the base and Stage-A checkpoints carry no
trained latent adapter, so their latent arms are effectively random baselines;
only Stage B is a meaningful latent measurement.)

**Direct-answer internalization — a promising, confounded signal.** On a larger
direct-answer sample (n=800, no trace, no latent), CRUXEval accuracy rises
monotonically along the training pipeline. **Sources:
`results/cruxeval_direct800_{feature_pilot_A,stageA_executor,
stageB_latent_trace}.json`.**

| checkpoint | direct acc (n=800) |
|---|---|
| feature_pilot_A (base) | 0.06875 |
| stageA_executor | 0.08125 |
| stageB_latent_trace | 0.10500 |

Base → Stage B is 0.0688 → 0.1050 (z=2.58; we confirm z≈2.57 with a pooled
two-proportion test). The suggestive reading is that *internalizing execution*
improves zero-scaffold output prediction even where the explicit mechanism does
not transfer. **The confound:** Stage A + Stage B expose the model to more code
training tokens than the base, so the direct-answer gain could reflect extra
code exposure rather than internalized execution. A matched-token no-trace
fine-tune control is in flight. **[PENDING: attribution control — matched-token,
trace-free fine-tune; report whether the 0.0688→0.1050 direct-answer gain
survives.]**

---

## 5. Limitations

We state every weak point the novelty assessment flagged.

- **Synthetic programs only; mechanism does not transfer.** The executor is
  trained on our `gen_exec_traces` generator and does not fire on CRUXEval
  (§4.6, latent arms ≈1.5–4%). The direct-answer internalization signal
  (0.0688→0.1050, z=2.58) is promising but token-count-confounded; its control
  is running. Until the mechanism transfers to real, unseen program
  distributions, the positive result is a controlled-setting demonstration, not
  a general capability.

- **A ~6-hop latent horizon.** Latent state decodes cleanly for hops 1–5
  (0.78–0.96 at every K, including length-gen K=9–12) but hop 6 sits at ~0.60
  and hop 7+ falls off a cliff (≤0.28 → 0.00), so the answer — which needs the
  *last* hop — passes only to K≈6 (§4.4) and length-gen lift collapses past the
  horizon (latent−direct +4.7/+3.7/+3.0pp at K=9/10/12). We diagnose this as a
  training-exposure artifact rather than a hard wall: deep slots (7–8) receive
  gradient only from the rare (K≥7 rung) × (s≥7 stage) draws, while slots 1–5
  train in every s≥1 draw; the hop CE was still falling at run end. This is
  suspiciously close to the ~6-continuous-token practical limit reported for
  Coconut (Soft Tokens, Hard Truths, 2509.19170), and it is exactly the regime
  the Depth Ceiling paper (2604.06427) predicts latent planning capacity barely
  scales through. A depth-weighted consolidation-sampling arm (and possibly
  c=2 slots/step at higher aux weight) is running to distinguish exposure from
  structure. **[PENDING: depth-fix results — depth-weighted stage sampling;
  report whether the hop-7+ cliff lifts and K=7/8 answer clears ~0.55. Relate
  the outcome to Soft-Tokens' ~6-token limit and the Depth Ceiling either way.]**

- **Single model, single scale, single seed.** All results are one 402M
  DeltaNet checkpoint, seed 0. SIM-CoT reports that GPT-2-scale latent results
  can invert at 8B, and bounded state imposes its own scale limits; none of our
  conclusions should be assumed to hold across scale without replication.

- **The latent-first (N1′) base-competence confound.** One could argue N1′
  fails merely because "that base is a weaker model." We pre-empt this two ways.
  (i) N1′ and Stage A fine-tune from the *identical* checkpoint
  (`feature_pilot_A`), and the three checkpoints in the arc are matched on
  *general* code competence to within ~0.015 HE-CE (base 0.7429, Stage A 0.7343,
  Stage B 0.7491) — the base is not weaker in the broad sense. (ii) The relevant
  difference is *task-specific* token competence: `feature_pilot_A` can execute
  these programs in text at only ~12–16% direct-answer, and Stage A lifts that
  same base to 0.94–0.97 with a text trace. So the staging result should be read
  as: prior *token-space competence at the task* is what unlocks latent
  compression — which is the mechanism we are claiming, not a nuisance
  confound. We note honestly that Stage B's base is therefore *strictly more
  capable at the task* than N1′'s, so the comparison establishes the
  *sufficiency* of the staged path directly and the *necessity* of token-space
  competence by the N1′ failure from the token-incompetent base, rather than by
  a single fully-matched A/B.

- **Open anomaly at K=2.** Latent R=K answer at K=2 is 0.140 — *below* both the
  direct baseline (0.2567) and its own 0.925 per-hop decode — whereas K=3–6 sit
  at 0.59–0.63. The per-hop state is recovered but answer *emission* fails at the
  shallowest rung; we have not resolved the cause. **[PENDING: K=2 anomaly —
  inspect transcripts; determine whether this is an emission/format failure at
  R=2 or a genuine shallow-depth effect.]**

- **Single-token intermediate values.** The per-hop loss currently consumes
  single-BPE-token (mod-10) values. Multi-token per-hop decode is future work
  and is a prerequisite for the CRUXEval-distribution transfer above.

---

## 6. Conclusion

Code makes the two standing problems of latent reasoning tractable at once: the
interpreter supplies free, dense, verifiable per-step supervision, and a
Scratchpad→Coconut curriculum supplies the token-space competence that latent
compression turns out to require. On a 402M bounded-state DeltaNet, staged
training internalizes real, executed program trajectories into continuous
thoughts — per-hop decode 0.844 at K=4, R=K answer +45–52pp over the no-trace
baseline to ~6 hops, with a depth-true R=1/R=K/R=K+4 signature — where
latent-first training with the identical data and loss pins at the value prior.
The durable methodology finding is the staging cell: *externalize before you
internalize* is not just theoretically predicted but empirically load-bearing,
and dense per-step supervision alone does not substitute for it. The honest
frontier is transfer: the mechanism does not yet fire on out-of-distribution
programs, the latent horizon saturates near six hops, and the encouraging
direct-answer internalization signal awaits its token-matched control. In a
model whose decode state is O(1) and whose latent steps cost no context,
internalized execution points toward zero-context-cost deliberation for
long-horizon coding agents — once it generalizes beyond the synthetic setting.

---

## References

- Coconut — *Training Large Language Models to Reason in a Continuous Latent
  Space.* arXiv:2412.06769 (COLM 2025).
- Stepwise Internalization — *From Explicit CoT to Implicit CoT.*
  arXiv:2405.14838.
- CODI. arXiv:2502.21074.
- SIM-CoT — *Supervised Implicit Chain-of-Thought.* arXiv:2509.20317 (ICLR
  2026).
- *Information-theoretic analysis of latent-step supervision.*
  arXiv:2606.20075.
- *Capabilities and Fundamental Limits of Latent Chain-of-Thought.*
  arXiv:2602.01148.
- *Thinking States* / probing latent thought content. arXiv:2602.08332.
- Do Latent Tokens Think? arXiv:2512.21711; and arXiv:2411.15862.
- Soft Tokens, Hard Truths. arXiv:2509.19170.
- Token Assorted. arXiv:2502.03275.
- Huginn — latent recurrent-depth reasoning. arXiv:2502.05171.
- Tiny Recursive Mamba-2. arXiv:2602.12078.
- Depth Ceiling — latent planning capacity scaling. arXiv:2604.06427.
- Code World Model (CWM), Meta FAIR, Sep 2025
  (github.com/facebookresearch/cwm).
- Neural Debugger. arXiv:2603.09951.
- Code Execution as Grounded Supervision. arXiv:2506.10343.
- SemCoder. arXiv:2406.01006.
- LaSynth — learning to synthesize/execute. arXiv:2107.00101.
- Neural Programmer-Interpreter (NPI). Reed & de Freitas, 2016.
- Scratchpad — *Show Your Work.* Nye et al., 2021.
- Jin & Rinard — learning to execute. arXiv:2305.11169.
- CRUXEval. arXiv:2401.03065.
- Latent-reasoning surveys. arXiv:2505.16782, arXiv:2509.02350.

---

## Appendix A — Provenance of every reported number

| Result | Value | Source file |
|---|---|---|
| Base HE-CE | 0.7429 | `SESSION_FINDINGS.md` (feature_pilot_A) |
| Stage A HE-CE | 0.7343 | `SESSION_FINDINGS.md` / `EXEC_TRACE_LATENT_PLAN.md` |
| Stage B HE-CE | 0.7491 | `EXEC_TRACE_LATENT_PLAN.md` Stage-B RESULT |
| N1′ arms / per-hop / structural floor | table §4.2 | `runs/n3_killtest_N1prime.json` |
| Stage A trace/answer/lift | table §4.3 | `runs/stageA_executor_eval.json` |
| Stage B answer / per-hop / kill lift | table §4.4 | `runs/stageB_latent_trace_eval.json` |
| Stage B per-hop (re-measured) | 0.925…0.632 | `runs/stageB_perhop_remeasure.json` |
| CRUXEval mechanism transfer | table §4.6 | `results/cruxeval_transfer_*.json` |
| CRUXEval direct internalization | 0.0688 / 0.0813 / 0.1050 | `results/cruxeval_direct800_*.json` |
