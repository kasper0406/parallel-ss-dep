# Code Execution in Pre-training — design + honest go/no-go

> Status: DESIGN/RESEARCH proposal. No code changed, no training launched.
> The running v17 pretrain on GPU 0 (16 GB used) is untouched; GPU 1 is free.
> Date: 2026-06-18. Author context: the "small super-coder" (287M, 2×RTX5090,
> ~5B-token Chinchilla pretrains).

## 0. The idea (user, verbatim intent)

> "Could code execution be made part of pre-training? Something like including
> examples of full compiler/interpreter outputs and then the fixes applied as a
> git diff or similar."

i.e. pretrain on **execution-grounded sequences** — `(code → real
compiler/test output/traceback → the fix as a diff → fixed code)` — so the model
learns the *consequence* of code and repair behaviour at pretrain scale, not
only as an RL reward (which `experiments/train_rl_grader.py` already does
post-SFT).

## 1. TL;DR verdict

**Mostly SKIP it as a HumanEval-headline lever. Do a cheap, narrowly-scoped
PILOT only if the target is a repair / execution-awareness / agentic capability,
and measure it on a repair benchmark — NOT on greedy HumanEval pass@1.** Spend
the first marginal effort on **data hygiene + rare-algorithm tail up-sampling
(task #84)**, which is the better-evidenced lever for the headline.

Three load-bearing reasons, each grounded:

1. **Prior art says execution-grounded *pretraining* lifts code COMPREHENSION /
   execution-understanding, not from-scratch generation.** TRACED and
   CodeExecutor (below) improve runtime-value / trace prediction and
   comprehension tasks; none demonstrate a HumanEval-generation lift. The one
   clear generation lift from execution-adjacent data (OctoPack/CommitPack)
   came at **16B via instruction tuning**, and its cleanest, largest benefit was
   on the **repair** benchmark (HumanEvalFix), which our greedy HumanEval
   pass@1 does not test.

2. **The exact "inject a bug → train the fix" recipe has a fresh, direct
   NEGATIVE result.** "Synthetic Error Injection Fails to Elicit Self-Correction
   in Language Models" (arXiv 2512.02389, Dec 2025) finds synthetic-error
   supervision fails to improve self-correction *even on simple synthetic
   tasks*, because (a) synthetic-error distribution ≠ the model's *on-policy*
   error distribution, and (b) "parroting" — the model detects the bug and then
   reproduces it. Their conclusion: **on-policy RL is what works** — which is
   exactly what this repo already has in `train_rl_grader.py`.

3. **Capacity opportunity-cost at 287M / 5B tokens, measured in THIS repo.**
   The synthesis memo (`project_why_mechanisms_synthesis.md`) records the killer
   datapoint: v12 (recall/WM-heavy pretrain mix) scored **8/164** HumanEval vs
   Phase-C (code-focused mix) **14/164** under an identical SFT+eval recipe —
   pouring fixed pretrain capacity into a non-code objective *cost* 6/164. An
   execution-grounded stream at, say, 12% weight displaces ~600M code tokens at
   our budget; that is the single strongest argument for caution.

The user's framing is *better* than the failed recipe in one important way: they
want the model to see **execution consequence** (real output), which is closer
to TRACED/CWM (comprehension, which *did* improve) than to the failed
self-correction recipe. But comprehension gains have not been shown to move
short-context greedy generation — so the honest read is "real capability, wrong
probe for our headline."

## 2. Prior art (what worked, at what scale, repair-vs-generation)

| Work | What it did | Scale / stage | Where the benefit landed |
|---|---|---|---|
| **TRACED** (Ding et al., ICSE 2024, [2306.07487](https://arxiv.org/abs/2306.07487)) | Execution-aware *pretraining*: code + concrete inputs + execution traces (runtime variable values + branch coverage) | encoder-scale, pretrain | +12.4% execution-path prediction, +25.2% runtime-value prediction; downstream wins on clone/vuln/code-to-code (COMPREHENSION). No generation/HumanEval claim. |
| **CodeExecutor** (Liu et al., ACL Findings 2023, [aclanthology 2023.findings-acl.308](https://aclanthology.org/2023.findings-acl.308/)) | Pretrain to predict the execution trace (output) of code + curriculum | seq2seq, pretrain | Better execution simulation / semantic comprehension. Not a generation lever. |
| **Code World Models / "Debugging code world models"** (Rahmani et al., 2026, [2602.07672](https://arxiv.org/abs/2602.07672); Meta CWM) | Mid-train on Python interpreter traces (variable state after every line); model simulates execution as an alternative to NL CoT | up to ~5T-token mid-training | State propagation works; **fails on token-budget exhaustion (dense traces are long), string-valued state (subword tokenization), and action hallucination**. Direct warning for our token budget. |
| **OctoPack / CommitPack / CommitPackFT** (Muennighoff et al., ICLR 2024, [2308.07124](https://arxiv.org/abs/2308.07124)) | Git commits as (commit-msg instruction, code diff) pairs; **instruction tuning** (CommitPackFT, filtered 4TB→2GB) | **16B StarCoder**, post-pretrain SFT | 46.2% HumanEval Python (best non-OpenAI-data); introduced HumanEvalFix; **largest, cleanest gains on the REPAIR task**. |
| **Self-Debugging** (Chen et al., ICLR 2024, [2304.05128](https://arxiv.org/abs/2304.05128)) | Iterative debug loop using execution results, few-shot | **inference-time, no training** | Lifts pass@1, but it's a decoding-time scaffold, not a pretraining lever. |
| **Synthetic Error Injection Fails…** ([2512.02389](https://arxiv.org/abs/2512.02389), 2025) | Inject artificial errors → supervise recognition/correction | multiple LMs, SFT | **NEGATIVE.** Distribution shift (synthetic≠on-policy) + parroting. On-policy RL succeeds where this fails. |
| Synthetic bug / mutation gen (LEAM, contextual mutants, [2310.02407](https://arxiv.org/abs/2310.02407), [2107.06657](https://arxiv.org/pdf/2107.06657)) | Learn realistic mutants to train bug detectors/repair | SE-task scale | Useful for bug *detection/repair* training; quality of mutants (realistic vs trivial) is the crux. |

**Synthesis of prior art for us:** execution-in-pretraining is well-validated for
*comprehension/execution-awareness*; commit/repair data lifts generation only at
much larger scale + instruction tuning, with the cleanest benefit on *repair*
benchmarks; the precise synthetic-bug→fix recipe has a fresh negative result and
points back to on-policy RL (which we already do).

## 3. What this repo already gives us (the cheap parts are built)

- **`experiments/code_grader.py`** — `grade(problem, completion) -> GradingResult`
  runs code in a sandboxed subprocess (SIGALRM + bounded `q.get` + SIGKILL
  reap), AST-splits `check()` per-assert, and returns a dense tier
  (`syntax_error/exec_error/runtime_error/timeout/partial/pass`) **plus
  `error_text`** — the formatted SyntaxError+line / traceback / failed-assert
  sources, capped at `_ERROR_TEXT_CAP = 1000` chars. **This is a ready-made
  generator of "real compiler/interpreter output" for arbitrary code, already
  length-bounded.**
- **`experiments/distill_solutions.py`** — emits JSONL rows
  `{task_id, problem_prompt, qwen_completion, extracted_code, has_tests, tier,
  score}`. Run *without* `--keep_only_passing` it already keeps **failing
  rows** — i.e. we already have `(problem, on-policy-ish wrong code, tier,
  score)` pairs. `extract_code_block()` pulls the fenced code.
- **`experiments/gen_rejection_data.py`** — rolls out N samples from *our own*
  model and grades them; `--keep_all` keeps failures too. **This is the
  on-policy bug source the negative-result paper says actually matters.**
- **`experiments/iterative_repair.py::build_repair_prompt(original_prompt,
  failed_code, error_text, ...)`** — already formats a repair prompt from a
  failure + its grader error text. Reusable as the data-format template.
- **`experiments/data_mix.py`** — the pretrain pipeline. Key facts for the
  design:
  - A source can be a local JSONL (`jsonl_path`) and compose fields with
    `text_field: [a, b]` (joined by `\n\n`) **or** a named `text_builder`
    (registry `TEXT_BUILDER_REGISTRY`; e.g. `_builder_bigvul` already renders a
    **before/output-description/after fix block** — a working precedent for our
    format).
  - **Special tokens are plain-text strings, no vocab change** — FIM uses
    `<|fim_prefix|>` / `<|fim_suffix|>` / `<|fim_middle|>` as literal strings the
    tokenizer encodes. We follow the same precedent: **no vocab surgery.**
  - Loss masking in pretrain is coarse: `targets=-100` only at think-token
    positions and (optionally) EOS. There is **no per-segment loss mask** today
    beyond the recall `emit_read_mask` (an aligned 0/1 channel). So a variant
    that masks the EXEC-output span out of the loss needs a small code change;
    the **full-LM-loss variant needs ZERO new masking** (just a `text_builder`).
- **`bigvul` + `cybernative_dpo` sources** are *already* a real bug→fix /
  vulnerable→safe pretrain signal in every recent mix (v15: bigvul 0.08,
  cybernative 0.04). **We already do "before-bad / after-good code" in pretrain.**
  The user's idea is largely: *add the executed OUTPUT in between, and use
  general code (not just CVEs).*

## 4. Data format design (ranked)

Design constraints from prior art + repo: keep value sections SHORT (CWM
token-budget failure; `error_text` already capped at 1000 chars); avoid
line-numbered unified diffs (CWM string/format tokenization failure +
small-model brittleness); plain-text sentinels (no vocab change); prefer
full-LM loss (zero new masking, matches commit-pretraining practice).

### Variant A (RECOMMENDED) — before → real output → after, full code, full-LM loss

```
<|task|>{problem_prompt}
<|code|>
{buggy_code}
<|exec|>
{grader tier + error_text, ≤ ~40 lines}
<|fix|>
{corrected_code}
```

- **No special masking** → a single `text_builder` in `data_mix.py`, zero
  trainer change. Full next-token loss over the whole sequence trains BOTH
  "predict the execution consequence of `buggy_code`" (comprehension, the
  TRACED/CWM-validated part) AND "predict the corrected code given code+output"
  (repair, the OctoPack part), jointly.
- `<|exec|>` body = e.g. `tier=runtime_error\n` + the capped `error_text`.
  Keep it tight — it is the most token-expensive section.
- `{corrected_code}` is the FULL function/program, not a diff. BigVul's
  `_builder_bigvul` proves the before/after-full-code format is learnable here;
  a small model copies-with-edits a full function more reliably than it emits a
  valid `@@ -a,b +c,d @@` hunk.

### Variant B — search/replace minimal-edit "diff"

Replace `<|fix|>` body with one or more minimal edit blocks:

```
<|fix|>
<<<<<<< SEARCH
{exact buggy lines}
=======
{replacement lines}
>>>>>>> REPLACE
```

- Closest to the user's "git diff" intent while avoiding line-number fragility
  (this is the Aider/Coeditor-style edit format, more learnable than unified
  diff at 287M). Teaches *localized* repair — arguably the more useful skill for
  agentic/iterative-repair downstream (`iterative_repair.py`).
- Still full-LM loss; still a `text_builder`. Slightly higher format-error risk
  than A (the SEARCH block must match verbatim). **Use A first; B as a follow-up
  if the agentic/repair direction is greenlit.**

### Variant C (NOT recommended at this scale) — raw git unified diff

`<|diff|>` body = real `git diff` / `difflib.unified_diff`. Rejected: CWM showed
small models fail on exactly this kind of string-structured output; line numbers
are brittle; no upside over B.

### Variant D (separate objective) — execution-output prediction only (TRACED/CWM-lite)

```
<|code|>
{code}
<|input|>{args}
<|output|>{captured stdout / return repr / traceback}
```

- This is the *pure comprehension* lever (no repair). It is the part prior art
  most cleanly validated — but on comprehension tasks, not generation. Include a
  SMALL amount of this only if we explicitly want execution-awareness (e.g. for
  a future "predict-then-verify" decode); do not expect HumanEval movement.
  Token-expensive if traces are dense → keep to final stdout/return, not
  per-line state.

**Loss-masking note (if we ever want it):** to train "predict fix given output"
*without* spending loss on predicting the output, add an aligned loss-mask
channel mirroring the existing `emit_read_mask` plumbing (`_build_read_mask` →
`buffer_readmask` → 4th yield element). This is a real but bounded code change.
**Recommendation: don't — full-LM loss is simpler and is what commit-pretraining
uses; predicting the output is the comprehension signal we may actually want.**

## 5. Generation pipeline (cheap, repo-grounded)

The whole corpus can be built with existing tools + one new generator script
(`experiments/gen_exec_repair_tasks.py`, to be written later if greenlit):

1. **Seed correct code.** Use graded-passing rows from the existing distill
   corpus (`distill_solutions.py` output) and `gold_solution`s in
   `code_grader.LOADERS` (mbpp_combined / leetcode / humaneval-excluded). These
   are verified-correct anchors and the `<|fix|>` targets.

2. **Get buggy code — prefer ON-POLICY, per 2512.02389.**
   - **Primary (on-policy):** failing rollouts from our *own* model via
     `gen_rejection_data.py --keep_all` (or `distill_solutions.py` without
     `--keep_only_passing`). These match the model's real error distribution —
     the property the negative-result paper says is decisive.
   - **Secondary (cheap synthetic, lower value):** AST mutation operators (swap
     `+`/`-`, `<`/`<=`, `==`/`!=`, `and`/`or`, off-by-one in `range`, drop
     `return`, wrong variable). Easy via the `ast` module, but per prior art
     these are *trivial* bugs with a distribution mismatch — use sparingly for
     coverage, not as the bulk.
   - **Real bug→fix already present:** keep leaning on `bigvul` /
     `cybernative_dpo` (already in the mix).

3. **Execute → capture real output.** Run each buggy candidate through
   `code_grader.grade(problem, code)` (use `grade_in_parallel`, 8 workers). Take
   `result.tier` + `result.error_text` as the `<|exec|>` body. **The sandbox,
   timeout, and SIGKILL reaping are already implemented** — no new sandboxing
   needed for OUR generated code. (Caveat: never feed *arbitrary downloaded*
   code through exec; we only run code we generated/mutated.)

4. **Pair + verify.** Emit a row only if (a) the buggy code fails the grader and
   (b) the chosen `<|fix|>` *passes* it (re-grade the fix). This is the quality
   gate that avoids the "teach a fix that's also wrong" failure.

5. **Emit JSONL** in a schema `data_mix` can read directly with a `text_builder`
   (fields: `problem_prompt`, `buggy_code`, `exec_tier`, `error_text`,
   `fixed_code`). Register `_builder_exec_repair` in `TEXT_BUILDER_REGISTRY`.

**Volume + cost.** Target ~50–100k rows = a few hundred MB of text, comfortably
a *minority* stream (see §6). The model-rollout / vLLM cost is the distill cost
we already pay (run on **GPU 1**, never GPU 0/v17). The execution pass is
CPU-only subprocess work: at ~8 parallel workers and a few hundred ms typical
grade, ~50–100k grades is single-digit CPU-hours. **Cheap.**

**Caveats:** `code_grader` / `process_reward` assert `pad_id != thinking_token_id`
(pad-as-think silently triggers state-readonly on pads); the caller uses
`pad_id=0`. Keep `data_mix`'s think-burst insertion behaviour in mind — for the
exec stream, prefer `think_burst_prob` low or 0 on this source so we don't
scatter `[THINK]` tokens through structured exec/diff text (the current
`MixedSourceStream` applies bursts globally; isolating per-source burst prob is a
small follow-up if needed).

## 6. Minimal first experiment (cheap, high-signal)

**Design = a SHORT continuation with a matched control, NOT a full pretrain.**

- **Base:** the latest stable v16/v17 ckpt (after v17 finishes / on GPU 1; do
  **not** touch GPU 0).
- **Arm 1 (treatment):** continue +300–500M tokens with an `exec_repair`
  (Variant A) stream at **~12% weight**, anti-forgetting bulk kept exactly as
  v15 (code/instr/wiki ~84%), recall streams optionally dropped to free weight.
- **Arm 2 (control, the fair baseline):** identical continuation, identical
  token budget, the 12% replaced by **more of the existing code bulk** (e.g.
  codeparrot/magicoder). This isolates "exec data helped" from "we just trained
  longer / on more code."
- **Evaluations (fix ONE harness — `project_humaneval_config_artifact` warns the
  164-greedy signal is noisy):**
  1. **Repair benchmark (the matched probe, where the effect SHOULD appear):**
     build a held-out repair set by taking HumanEval/MBPP, injecting a bug,
     presenting `[problem][buggy code][grader error_text]`, and grading the
     model's fix with `code_grader`. (HumanEvalFix-style; we own all the
     tooling.) Metric: repair pass@1.
  2. **HumanEval generation (the headline):** greedy + **temperature pass@k**
     (k≥10) for a less-noisy estimate, same flags both arms
     (`--prompt_style sft_comment --extract_code_block`).

- **Success / decision rule:**
  - Treatment **lifts repair pass@1 by a clear margin** AND **does not regress**
    HumanEval pass@k vs control → it's a real *repair/agentic* lever; greenlight
    Variant B + scale, scoped to the agentic target (not the HumanEval headline).
  - Repair lifts but HumanEval **regresses** → confirms capacity opportunity-cost;
    keep exec data out of the general pretrain, use it only in a task-specific
    SFT/RL stage.
  - No repair lift → the synthetic/format transfer failed (as 2512.02389
    predicts for non-on-policy bugs); stop, lean on on-policy `train_rl_grader`.

This costs ~half a day of GPU-1 time + a few CPU-hours of data gen — small enough
to run *after* the cheaper data-hygiene work, as a clean A/B.

## 7. Risks / what NOT to do

- **Distribution mismatch (the headline risk).** Synthetic/mutated bugs ≠
  on-policy bugs → no transfer to real self-correction (2512.02389). **Mitigation:
  source bugs from our own failing rollouts (`gen_rejection_data --keep_all`),
  not mutation operators.**
- **Capacity opportunity-cost at 287M / 5B tokens.** Every exec token displaces a
  code token; the repo's own v12-vs-Phase-C (8 vs 14/164) shows a non-code mix
  *costs* HumanEval. **Mitigation: keep it a small minority stream; always run
  the matched control; consider it for a *continuation*, never a from-scratch
  pretrain budget.**
- **Repair-format → generation transfer is weak.** Prior art's generation lift
  from commit/repair data was at 16B + instruction tuning, cleanest on repair
  benchmarks. **Don't promise HumanEval movement; measure repair on a repair
  probe.**
- **Token-budget / format brittleness.** Dense traces and unified diffs are where
  small models fail (CWM). **Keep `<|exec|>` short (use the capped `error_text`),
  use full-code or search/replace fixes, never raw line-numbered diffs.**
- **Quality / hackability.** A "fix" that doesn't actually pass is worse than no
  data. **Re-grade every fix; emit only verified pass; dedup.**
- **Don't re-tread dead ends.** `process_reward.py` was REMOVED (2026-05-30);
  `train_rl_grader.py` already does execution-as-reward *on-policy* — that is the
  validated, prior-art-endorsed path. The novel/speculative claim here is only
  "execution belongs in PRETRAIN," and that claim is what the pilot tests.
- **Sandbox hygiene.** Only exec code we generated/mutated through the existing
  grader subprocess; never exec arbitrary scraped code. Respect `pad_id !=
  thinking_token_id`.

## 8. Recommendation

1. **First, spend effort on data hygiene + rare-algorithm tail up-sampling
   (task #84).** Best-evidenced lever for the HumanEval headline; the synthesis
   memo says the bottleneck is base knowledge, not mechanisms or inference-time
   tricks.
2. **Treat execution-in-pretraining as a separate, cheap PILOT for a
   repair/agentic capability**, not the headline. Run §6's matched A/B with
   Variant A, on-policy bugs, full-LM loss, on GPU 1. Decide by the repair-probe
   lift + the no-regression-on-HumanEval control.
3. **If (and only if) the pilot shows repair lift without HumanEval regression**,
   proceed to Variant B (search/replace edits) and fold it into the
   agentic/iterative-repair line (`iterative_repair.py`), keeping it a minority
   stream. Otherwise, keep execution where it already pays off: as the on-policy
   RL reward in `train_rl_grader.py`.

Net: **cheap pilot OK if the goal is repair/agentic; SKIP for the HumanEval
greedy pass@1 headline in favour of data hygiene.**

### Sources
- TRACED: Execution-aware Pre-training for Source Code — https://arxiv.org/abs/2306.07487
- CodeExecutor (Code Execution with Pre-trained LMs) — https://aclanthology.org/2023.findings-acl.308/
- Debugging Code World Models — https://arxiv.org/abs/2602.07672
- OctoPack / CommitPack — https://arxiv.org/abs/2308.07124
- Teaching LLMs to Self-Debug — https://arxiv.org/abs/2304.05128
- Synthetic Error Injection Fails to Elicit Self-Correction — https://arxiv.org/abs/2512.02389
- Challenging Bug Prediction and Repair Models with Synthetic Bugs — https://arxiv.org/abs/2310.02407
- DeepMutants (contextual mutation) — https://arxiv.org/pdf/2107.06657
