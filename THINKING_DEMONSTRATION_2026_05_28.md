# Thinking demonstration: does the "thinking" mechanism ever help? (2026-05-28)

## The question
The single most important question for the thinking primitive: **is there
ANY task where the model's "thinking" mechanism measurably improves task
performance?** The working hypothesis was that thinking is *emergent* — it
cannot help where the base model fails completely (floor) or succeeds
trivially (ceiling), but should help in the **competence band**: tasks the
model can *partially* do, where extra sequential computation tips
near-misses into hits. Prior work (`THINKING_AUDIT_2026_05_28.md`) found
thinking neutral on HumanEval (13=13) and the only synthetic reasoning probe
(`synth_reasoning_heldout`, 5–8-step arithmetic) was at the floor (3/300) —
nothing for thinking to amplify. So we built **easier rungs** to scan the
band.

## Method
- **Difficulty ladder**: arithmetic-chain reasoning tasks at
  `n_steps ∈ {1..6}` (chain length 1–6), 80 problems per rung, JSONL in the
  `load_synth_reasoning` schema. Generator: `_gen_multi_step_arith` is now
  parametrized on `n_steps_min/max`; `gen_synthetic_reasoning_tasks.py
  --arith_n_steps N` fixes a single rung. Data:
  `data/synth_arith_ladder_n{1..6}.jsonl`. Every task's gold solution is
  validated to pass its own `check()` (80/80 emitted per rung).
- **Ckpt**: `checkpoints/rl_grader_phase_c_v2_step300.pt` (the project-best,
  16/164 HumanEval).
- **Ablation**: harness `experiments/eval_thinking_ladder.py` runs each rung
  twice with **identical** generation config (greedy, temp 0, max_gen 128,
  deterministic grader, `code_fence` prompt style), the ONLY difference
  being the think flag:
  - **without-think**: plain `generate(use_thinking=False)`.
  - **with-think**: `generate_with_retrieval_as_input` (the gate decides
    when to think; additive injection per the ckpt's
    `retrieval_input_additive`).

## The headline table — pass@1 with vs without thinking

| n_steps | n  | no_think    | with_think | delta | think_tok/prob |
|--------:|---:|:------------|:-----------|------:|---------------:|
| 1       | 80 | 0/80  0.000 | 0/80 0.000 |   +0  |  76.4 |
| 2       | 80 | **17/80 0.212** | 0/80 0.000 | **−17** | 134.7 |
| 3       | 80 | 13/80 0.163 | 0/80 0.000 | **−13** | 111.9 |
| 4       | 80 | 1/80  0.013 | 0/80 0.000 |   −1  |  86.8 |
| 5       | 80 | 4/80  0.050 | 0/80 0.000 |   −4  |  26.2 |
| 6       | 80 | 0/80  0.000 | 0/80 0.000 |   +0  |  29.9 |

(Raw JSON: `THINKING_LADDER_RESULTS.json`.)

## Verdict: NO. Thinking never helps — and in the competence band it
## actively destroys performance.

There is **no rung where with-think > without-think**. with-think scores
**0/80 on every single rung**. The competence band is clearly visible in the
no-think column (n=2: 21%, n=3: 16% — the model can partially do these), and
that is *exactly* where thinking does the most damage (−17, −13). At the
floor (n=1, n=4, n=6) thinking can't subtract what isn't there, so delta is
0 or −1. The hypothesis that thinking would help in the band is **refuted**:
the band is where it hurts most.

## This is a genuine corruption effect, not a measurement artifact
The uniform 0/80 with-think demanded verification that the
retrieval-as-input generator wasn't structurally broken (e.g. never emitting
gradeable code). It is not. Side-by-side on rung-2 problems the no-think
model solves:

```
NO_THINK  (pass):  def solve():
                       v0 = 7
                       v1 = v0 - 4
                       v2 = v1 - 4
                       return v2          # correct: inlines seed, walks chain

WITH_THINK (partial, 194 think tokens):
                   def solve(v0, v1, v2):
                       return v0 - 4      # lost the seed value (v0 now a param),
                                          # collapsed the 2-op chain to one op,
                                          # then degenerates into repetition
```

with-think *does* emit extractable code — it just emits **worse** code. The
thinking process makes the model drop the seed binding and collapse the
arithmetic chain. This is the documented "thinking corrupts recall" failure
mode (CLAUDE.md / `eval_longctx_recall.py`): every think token steps the
DeltaNet linear-RNN recurrence and perturbs the precise binding the state
was carrying. Here the "binding" is the running arithmetic value, and 80–190
think tokens reliably destroy it. The with-think tier was `partial` (right
shape, wrong value), never `pass`.

## Confound: more compute
with-think grants the model extra forward passes (the think steps;
`think_tok/prob` column) that without-think lacks. If anything this confound
works *against* the verdict's direction — thinking gets MORE compute and
still loses. There is no "more compute" story that rescues thinking here; the
extra compute is spent corrupting the answer.

## Recommendation
**Do NOT scale a thinking micro-ladder.** Thinking does not help even in the
competence band that the emergence hypothesis predicted; it is net-negative
wherever the model has any competence to lose. The architectural primitive,
as currently trained (gate + retrieval-as-input + think-token-steps-the-
recurrence), is a liability on tasks that require carrying state — which is
most reasoning. Concretely:

1. The headline thinking claim should remain **"neutral-to-harmful"**, with
   this ladder as the clean negative demonstration.
2. The only documented positive for thinking is code *validity* (syntax),
   not task success — keep that framing narrow.
3. If thinking is to be revived, the fix must be **architectural**: stop
   think tokens from writing to the recurrent state
   (`--state_readonly_at_think`, β=0 on the delta-rule write gate at think
   positions) so they can read/compute without corrupting the binding. Only
   after a state-read-only variant shows a positive rung here is a scaling
   ladder justified. Until then, thinking should be off by default for
   reasoning/coding eval.

## Artifacts
- Generator: `experiments/gen_synthetic_reasoning_tasks.py` (parametrized
  `--arith_n_steps`).
- Data: `data/synth_arith_ladder_n{1..6}.jsonl` (80/rung, gold-validated).
- Harness: `experiments/eval_thinking_ladder.py`.
- Results: `THINKING_LADDER_RESULTS.json`.
- Tests: `experiments/test_synthetic_reasoning.py` (ladder rungs:
  chain-length control, gold-passes, default-range preservation, committed
  ladder files well-formed).
