# Thesis: small-model performance is under-served by training-methodology research

This doc states the bet behind the project. Empirical numbers go in
`README.md`; mechanism rationale in `THINKING_RAG_DIRECTION.md` and
`WORKING_MEMORY_FINDINGS.md`; this doc is the *framing*.

## The claim, stated carefully

Most training-methodology innovations of the past two years — curated
synthetic data (Phi), code-execution / unit-test RL rewards
(DeepSeek-Coder, R1), process-reward models on chain-of-thought,
architectural lifts like sparse attention / MoE / linear-RNN backbones,
distillation from teacher logprobs, working-memory + gated computation
— were *invented and benchmarked at frontier scale*. At those scales,
the relative effect of any single technique is squeezed between
massive base capacity above and noisy task evals below; a method that
adds 1–2 pp at 70 B looks unimpressive even when it's strong.

At small scale (≤ 1 B params), the same techniques *compound visibly*
because the base is weak enough that any genuine signal moves the
benchmark. **The structural under-investment in small-scale methodology
research means there is unclaimed performance there.**

This is *not* the claim that scaling laws are wrong, or that lottery
tickets predict dense small nets can match large ones on every task.
At a fixed (data, algorithm) the size–performance curve is monotonic
and real. The claim is narrower:

> Holding compute roughly fixed, **methodology** (data quality + curation,
> training stages, RL signal design, architecture refinements) is a much
> larger lever at small scale than the field's current research portfolio
> reflects. Therefore concrete small-model results obtained by combining
> small-scale-validated methods can land above the published per-parameter
> frontier — *not because the field doesn't know how*, but because the
> field hasn't combined the methods at this scale.

## What the evidence supports

- **Phi-1 (1.3 B, ~50 % HumanEval, ~7 B synthetic textbook tokens)** —
  beat 10× larger code models. Data-quality + curriculum, at small scale,
  not a new architecture.
- **Qwen2.5-Coder-0.5 B (~31 % HumanEval)** — beats many 7 B general
  models on code. Methodology (mostly RL + data) over scale.
- **DeepSeek-R1-Distill-Qwen-1.5 B** — distillation from R1 lifts a
  1.5 B base into competitive territory with 70 B-class reasoners on
  math benchmarks. The model size hasn't changed; the *training signal*
  has.
- **SmolLM2-360 M** — closing meaningful ground on text by curating
  pretraining data, not by capacity.
- **Our own internal evidence** — working-memory + saturated MQAR:
  +11 pp recall (mem-on vs mem-off) at the regime where DeltaNet state
  saturates. The same module's measurable contribution is much smaller
  at the un-saturated regime and would be near-invisible at 7 B+ where
  attention models can just memorise inside the KV cache. The
  architectural lift only *shows up* at small scale.

## What the evidence does NOT support (be honest)

- Frontier-class models (4o, o1, R1-full, Sonnet 3.7, Opus 4.7) are
  *not* 4 B-active MoEs. The very top of capability still tracks
  scale; the small-model bet does not eliminate this gap, it narrows
  the gap *per parameter*.
- "Lottery ticket" results are about finding sparse subnets *within
  a large overparameterized model*. They do not directly imply that
  small dense models with no overparameterization can match large
  models on every task. Citing them as a general justification for
  "small is enough" is overreach.
- On hard multi-step reasoning, long-context multi-file code, and
  multilingual tasks, the per-parameter gap *grows* with task
  difficulty. Methodology pays back most on tasks where the bottleneck
  is *training-signal quality*, not raw representational capacity.

## Where this project sits

We are building a **217 M code-focused linear-RNN + sparse FiLM
feedback + bounded working memory + thinking gate** model. Validated
internal lifts on canonical small-model tasks (MQAR, induction-heads,
dyck) say the architecture is real. The current bet — pretrain on a
mixed 9-source open corpus, 2 B tokens, mid-training HumanEval
auto-stop, then SFT, then RL with code-execution reward — is *exactly*
the kind of methodology-stack a frontier lab would only deploy at
70 B because that's where the eval moves enough to publish.

If this project achieves **HumanEval ≥ 25 % at 217 M after SFT**, or
≥ 30 % after RL, that's the thesis-confirming result: it would beat
StarCoderBase-1 B per-parameter by a clear margin and approach
Qwen2.5-Coder-0.5 B at less than half the parameter count, *using
only openly-published methodology assembled in the right order*. No
new ideas required — just the discipline to combine known small-
scale wins instead of chasing frontier-scale novelty.

## What this thesis does *not* commit us to

- A specific architecture. If a different small-model backbone
  produces stronger numbers at our compute, we should switch.
- A specific parameter count. 217 M is what we chose to derisk the
  pipeline; 360 M is the natural next size if 217 M plateaus low.
- A specific corpus. The mixed-corpus YAML is a swap-in artifact.
- Avoiding distillation. If a strong open-license code teacher
  (Qwen2.5-Coder-32 B, Qwen3-Coder, DeepSeek-Coder-V3) becomes cheap
  enough to distil 100 M+ tokens from, the thesis says *go for it*.

## What it does commit us to

- **Eval at small scale or don't claim the result.** A method's
  effect at 70 B is not evidence of its effect at 217 M and vice
  versa.
- **Compound methods, don't single-shot them.** Most published
  small-model papers vary one knob (e.g. "FiLM helps"); the project
  bets that the *combined effect* of FiLM + working memory + thinking
  gate + mixed corpus + RL with execution reward is materially larger
  than any one piece.
- **Be honest about the gap to frontier.** "Per-parameter SOTA on
  benchmark X" is a defensible claim. "Small models are as capable
  as large models" is not, and we should never make it.

## How to update this document

When real numbers from the current 2 B-token pretrain land — and
later from SFT and RL stages — append a "Status" section with the
results and a short note on which clauses of the thesis they
supported, contradicted, or sharpened. The thesis is a hypothesis to
be tested, not a position to defend.
