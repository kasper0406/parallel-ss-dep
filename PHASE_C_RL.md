# Phase C RL: prediction-as-RL-signal for code generation

> **Ponder-cost shaping (Phase-C-prep, landed pre-pretrain-finish):**
> `experiments/thinking.py::compute_grpo_advantages` now supports
> `ponder_shape ∈ {linear, quadratic}`, `counterfactual=True`
> (clamps task component at the depth-0 baseline so thinking can never
> make the task reward worse than not thinking, then always charges
> the depth cost — encourages exploration of thinking while still
> minimising depth), and `separate_ponder_norm=True` (z-score task
> reward within the GRPO group, then subtract absolute ponder cost
> — prevents the group z-score from squashing small ponder values
> into noise). `experiments/train_rl.py` exposes
> `--grpo_ponder_shape`, `--grpo_ponder_counterfactual`,
> `--grpo_separate_ponder_norm`, and `--grpo_ponder_warmup_steps`
> (curriculum). Defaults are backward-compatible. **Recommended config
> for the next RL run:** `--grpo_ponder_shape quadratic
> --grpo_ponder_counterfactual --grpo_ponder_warmup_steps 300`. See
> the function docstring for the exact reward formula per mode.


The proposal: train the model to write code **and** predict the
behaviour of that code before execution, then use the gap between
predicted and actual behaviour as part of the RL reward. Code that
passes tests but was predicted wrong gets less reward than code that
passes tests *and* was predicted correctly. Code that fails but was
predicted to fail (and predicted *how* it would fail) gets partial
credit. The aim is to push the model toward an internal *world model*
of execution rather than toward surface pattern-matching that passes
unit tests.

This sits in **Phase C** of the project roadmap — *after* the
pretrain (Phase A, current) and SFT (Phase B) produce a base that
solves a non-trivial fraction of problems. It is not a replacement
for code-execution RL; it is a richer reward signal layered on top.

## Why this is a real idea, not just folklore

There are three concurrent research lines this combines:

1. **Process Reward Models (PRMs)** — "Let's Verify Step by Step"
   (OpenAI 2023), DeepSeek-Math, *PRMBench*. The lift over pure
   outcome-reward RL is well-documented for math reasoning at
   frontier scale. The same mechanism — grading intermediate
   reasoning, not just the final answer — applies cleanly to code.

2. **World-model RL** — Dreamer family, MuZero, planning-via-prediction.
   The agent learns a forward model of the environment as a side
   product; that forward model becomes the bottleneck for harder
   reasoning. For an LLM "agent" writing code, the environment is
   the Python interpreter, and the forward model is *the model's own
   prediction of what the code will do*.

3. **Execution-aware code generation** — TRACED, ExeDec, and several
   2024 papers that ask the model to predict execution traces as an
   auxiliary task and measure pass-rate lift at modest scale. Existing
   results: lift in the range +3–8 pp HumanEval on sub-7 B models with
   trace prediction as an SFT auxiliary. Nobody (publicly) has put this
   inside an RL loop with execution-grounded verification, which is
   where it should pay back more.

The combination — *predict, execute, compare, reward both correctness
and the prediction's match to reality* — is the obvious next step that
hasn't been combined cleanly in open work. At our small-model scale
this is exactly the kind of methodology-stack the thesis (`THESIS.md`)
predicts will compound visibly.

## Concrete training recipe

### Output protocol

Prompt the model with the problem + a short instruction:

```
Write a Python function for the task below. Before the function,
emit a one-line prediction of the function's return value (or raised
exception) on the example inputs, in this exact format:

# Expected on examples: <python repr of expected output, or
                          'raises <ExceptionType>'>

Then the function body.
```

Generation produces:

```
# Expected on examples: [1, 3, 5, 7]
def filter_odd(xs):
    return [x for x in xs if x % 2 == 1]
```

### Execution + reward

For each rollout:

1. Parse out the prediction comment(s); strip them before execution.
2. Execute the function against the hidden test inputs (the same set
   the unit tests use). Capture: actual return value, any exception,
   any stdout.
3. Compute three signals:

   | Signal | Range | How |
   |---|---|---|
   | `pass`  | {0, 1}      | All hidden unit tests pass |
   | `match` | {0, 1}      | Predicted output matches actual on `k` sampled example inputs (literal equality on Python repr, exception-type match) |
   | `format` | {0, 1}    | Prediction comment parses cleanly |

4. **Combined reward** (multiplicative, as the user specified — either
   signal being 0 kills credit):

   ```
   reward = format * (α + (1 - α) * match) * (β + (1 - β) * pass)
   ```

   with e.g. α = 0.1, β = 0.0. Multiplicative means: bad prediction
   *and* failing tests is 0; failing tests with a *correct* prediction
   of the failure is `α` (partial credit for "the model knew it was
   wrong"); passing tests with a wrong prediction is `(β)(1) = 0`
   under β = 0 — strong penalty for "lucky right code with wrong
   understanding". Tune α and β based on early curves.

### Hyperparams to sweep early

- α (credit for predict-correctly-but-fail): 0.0 vs 0.1 vs 0.25.
  Higher α encourages calibrated failure prediction; too high and
  the model learns to predict failure on everything.
- β (credit for pass-but-predict-wrong): 0.0 vs 0.1. Zero is the
  pure-thesis form; small positive avoids pathological "refuse to
  predict" behaviour at cold start.
- `k` (number of example inputs the prediction is checked on): 1 vs
  3. Lower is faster; higher is harder to game.
- Prediction format: Python literal vs free-form English vs both
  (parse the literal if present, fall back to English with an
  LLM-judge similarity score). Start with literal-only; it's the
  least gameable.

## Risks worth flagging

1. **Reward hacking via prediction shaping.** The model could learn
   to predict whatever it *expects to produce*, not whatever the code
   *actually* produces. The defence: the prediction is checked
   against execution on hidden inputs the model didn't see in the
   prediction step. Use sampled test inputs at reward time, not the
   ones surfaced in the prompt.

2. **Cold start.** A base that can't predict execution at all gets
   `match = 0` on everything, the multiplicative reward kills
   gradient, and the model can't escape. Defences:
   - Pretrain auxiliary task: predict-trace SFT before RL starts
     (this is the TRACED / ExeDec recipe). Cheap; gives a non-zero
     starting `match`.
   - Curriculum: start RL on problems where prediction is mechanical
     (basic arithmetic, list ops). Don't put `is_prime` in the first
     batch.
   - Soft floor: replace the multiplicative reward with
     `reward = max(α_floor, format * match * pass)` for the first
     N steps. Removes the cold-start trap; revert to pure
     multiplicative once predictions are non-trivially calibrated.

3. **Distribution mismatch between Python literals and complex
   outputs.** A function that returns a 50-element nested dict has a
   prediction that's hard to match exactly. Defence: limit the
   prediction-check to "essential structure" — type match, length
   match, first/last element match — for outputs > N bytes. Define
   this as a separate `partial_match ∈ [0, 1]` score and use that
   instead of binary `match` for large outputs.

4. **Compute cost per rollout.** We already pay execution cost for
   the existing test-pass reward. The prediction comparison adds
   string-compare or AST-diff cost — negligible compared to the
   subprocess sandbox spin-up.

5. **Interaction with the thinking gate.** The model's natural place
   to "think before writing" is the gate-controlled thinking burst.
   Prediction is a *structured form of thinking*. We should *not*
   double-train the gate during this RL phase — the
   prediction-emission step is the de facto thinking. Either
   freeze the gate, or count emitted think tokens as part of the
   "format" check.

## How this fits our existing pipeline

The current RL infrastructure (`experiments/train_rl.py`,
`experiments/thinking.py`, `experiments/code_grader.py`) already
implements:

- GRPO with group-relative advantages
- Code-execution reward (HumanEval / MBPP-style grader with a
  subprocess sandbox)
- Hard-example position sampling
- Working memory + gate-head trained from the SFT base
- Full-trajectory credit assignment

What needs to be added:

- **Prompt template** that asks for the `# Expected on examples:`
  comment first, then the code. Lives wherever the SFT prompts live.
- **Prediction parser**: a 30-line function that pulls the literal
  out of the generated text, attempts `ast.literal_eval`, returns
  `(prediction, parse_ok)`.
- **Trace-execute helper**: extend `code_grader.py` to additionally
  run the function on the prompt's example inputs and return the
  actual output (not just pass/fail on hidden tests). Most of the
  subprocess machinery is already there.
- **Reward shaping**: replace the existing scalar test-pass reward
  with the multiplicative formula above. The GRPO trainer's
  advantage computation is unchanged.

Implementation effort: ~2–3 days end-to-end, assuming Phase A
(pretrain) and Phase B (SFT) have produced a base scoring > 10 % on
HumanEval. Below that, the model can't usefully participate in this
RL signal and time would be better spent on the base.

## Success criteria

This experiment counts as a positive result if:

- **HumanEval pass@1** lifts ≥ 3 pp over the same base trained with
  plain code-execution RL (no prediction signal), at matched compute.
- **HumanEval pass@10 − pass@1** *narrows* — i.e., the model's
  greedy output is closer to its best-sampled output, evidence that
  the prediction step is reducing variance in the right direction.
- Inspecting failure cases: the model's predictions on its own wrong
  outputs are *correctly wrong* — i.e., when the model says
  "I think this returns [1,2,3]" and it actually returns [3,2,1],
  that's a sign the prediction head is grounded; when the model says
  "I think this returns [1,2,3]" and the code returns the right
  answer but the prediction was wrong, that's pure pattern-matching
  and not what we want to reinforce.

A null result (no HumanEval lift) is still informative — it would
say the prediction signal is either too easily hacked, or that our
small-model base is below the threshold where the auxiliary task
helps, or that the additive reward shaping killed gradient. In any
of those cases the *direction* of the result is more useful than
"keep tuning until it works".

## What this is NOT a substitute for

- A strong base. If Phase A + B don't get HumanEval above ~10 %,
  RL of any flavour will struggle. Don't run this experiment on
  a 0/50-HumanEval base; it will look like a failure of the idea
  when it's a failure of cold-start.
- Pure correctness signal. If we ever start training a model where
  the eval is "does this code pass tests", that signal stays in
  the loss. The prediction reward *layers on top*, it doesn't
  replace.

## When to do this

After Phase A (current pretrain, ~30 hr) and Phase B (SFT, ~1
day) produce a base scoring ≥ 10 % HumanEval. Open this doc, set
up the prompt template + trace executor + reward shaping, run for
24–48 hr, and report against the success criteria above.

---

## Dense execution-grounded reward (landed 2026-05-14)

Binary pass/fail kills GRPO when a weak model solves nothing — every
rollout scores 0, the group has zero advantage variance, zero
gradient. `experiments/code_grader.py` now returns a **dense**
`GradingResult`:

- Tier ladder, `score ∈ [0,1]`: `syntax_error` 0.0 < `exec_error`
  0.05 < `runtime_error` 0.2 < `partial` 0.2 + 0.7·(n_passed/n_tests)
  < `pass` 1.0. The gap between partial-at-max (0.9) and pass (1.0)
  is the fully-correct boost.
- `_exec_target` AST-splits `check()` and runs it
  statement-by-statement, so one failing assert no longer masks the
  rest — that's where the fractional signal comes from.
- `GradingResult.error_text` is the formatted diagnosis (SyntaxError
  + line, exec traceback, the failed-assert source lines).
- Tests: `experiments/test_code_grader.py` (15).

**Explicitly rejected: embedding-similarity-to-reference reward.**
It's reward-hackable (the model learns to produce code that *looks*
like the answer), there is no single target (test suites admit many
correct solutions), and code embeddings barely track functional
correctness (a one-token `<`→`<=` edit flips correctness but is ~99 %
cosine-similar). Code has an execution verifier — lean into it; make
*it* dense rather than substituting a noisy proxy.

**Remaining wiring:** `train_rl.py`'s GRPO reward path still consumes
binary pass/fail — swap it for `result.score`. The ponder-cost
shaping above layers on top unchanged.

## Iterative self-repair loop (designed 2026-05-14, Phase-C-next)

Turn single-shot generation into an iterative debugging loop: when a
rollout fails, **re-add it to the rollout pool as a new task**,
`prompt' = original_prompt + failed_code + error_text`, same target.
The model learns "code + error → fix" as a first-class skill because
those `(prompt', target)` pairs are now in the training distribution.

The framing that keeps it tractable: this is **curriculum
generation, not multi-turn trajectory credit assignment**. Each turn
is graded independently with the dense `score` above — no
long-horizon credit assignment. A failed attempt just manufactures a
harder, information-rich training example.

- **Justification**: the project target is *agent* tasks, where the
  model has execution feedback at inference. So an error-augmented
  training distribution *matches* deployment — not distribution shift.
- **Mechanics**: cap retry depth ~2–3, TTL the queue. The
  `experiments/thinking.py` queue infra
  (`ThinkContinuation`/`ThinkReplay`, `safety_max_depth`,
  `think_queue_ttl`) is structurally the same enqueue-with-capped-
  depth pattern — reuse it.
- **Reward-hacking check**: independent per-turn grading means the
  model always prefers a one-shot 1.0 over fail-then-fix; no
  incentive to fail on purpose.
- **Composes with the thinking gate**: the diagnosis turn is a
  natural "think" trigger — read error → think → emit fix.
- **Prerequisite**: `code_grader.error_text` — done. The loop logic
  itself is gated on a base that produces *fixable* failures (not
  garbage), i.e. post-v3-long/v4.
