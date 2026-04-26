# 04 — Hierarchical / Multi-Timescale State

Coding LLMs face a structural mismatch with flat fixed-state RNNs: code is recursively scoped (token < expression < statement < block < function < module). A 256-dim state must simultaneously encode a half-finished slice expression *and* "we are 4 nested defs deep inside class `Foo` of module `bar.py`". Hierarchical state buys exponentially-different time constants without paying full attention's quadratic bill. Below: the five concretizations that look most plausible for our 135M / 5k-step Tinystories+Python regime, ordered by my prior on payoff.

## Top 5 concrete ideas

### Idea 1: Multi-timescale DeltaNet — banded decay heads
- **What it is:** keep DeltaNet, but split the H heads into K bands with fixed log-spaced decay priors λ_k ∈ {0.5, 0.9, 0.99, 0.999, 0.9999}. Each band still has data-dependent gating, but its *prior* is a different timescale. Effectively a parallel-scan analog of Clockwork-RNN, but every head still updates every step (clock = decay magnitude, not gating mask).
- **Hierarchy mechanism:** scale separation by spectral radius. Slow heads accumulate function/module-level facts; fast heads do tokenization-level edits. At read time, output is concat across bands.
- **Parallel-scan compatibility:** trivial. Same kernel, just different λ inits per head. No new ops; one matmul.
- **Why for code:** Python's hierarchy spans 1 → 1000 tokens; a single decay constant under-fits both edits-within-line and current-function context.
- **Key papers:** Clockwork-RNN (Koutník 2014); S5 (Smith 2023) — diagonal log-spaced timescales; MS-SSM (Kim 2025, arXiv 2512.23824).
- **Implementation cost:** S — one config knob in our existing DeltaNet kernel.
- **Code-LLM relevance:** high. Smallest diff that addresses the root mismatch.
- **Risk:** if data-dependent gating already learns these scales, the prior is wasted; mitigated by *fixing* fastest/slowest bands and letting the gate only modulate magnitude, not λ.

### Idea 2: Two-scale fast/slow scan with state-promotion gate
- **What it is:** FS-RNN reborn. A fast DeltaNet runs at every token. A slow DeltaNet receives an "elevator-input" — a learned content-dependent summary — only when a chunk-boundary gate fires (Hwang et al. H-Net dynamic chunking). The slow state is broadcast back into every fast step as a constant additive bias.
- **Hierarchy mechanism:** two recurrences, two timescales, asymmetric coupling: slow ← chunk-summary, fast += slow_state.
- **Parallel-scan compatibility:** fast scan is unchanged. Slow scan runs over T_slow ≪ T entries — also a scan. Coupling is a broadcast-add, not a join. Wall-clock cost: ~+5 percent.
- **Why for code:** chunk gate naturally fires at newlines, dedents, `def`/`class` — exactly the AST-coarsening boundaries.
- **Key papers:** Mujika & Steger Fast-Slow RNN (2017); Hwang H-Net (2025, arXiv 2507.07955); RMT (Bulatov 2022) for the segment-token interface.
- **Implementation cost:** M — need a chunk-boundary scorer and a broadcast hook in the kernel.
- **Code-LLM relevance:** high. The chunk gate is what lets a fixed-state model behave like an unbounded one.
- **Risk:** straight-through gate is fragile to train; H-Net used a routing-module stabilizer. Without it, the slow path goes silent.

### Idea 3: Scope-stack subspace — explicit AST-bias state slot
- **What it is:** reserve d_scope ≈ 32 of the recurrent state as a *push/pop stack register*. Whenever the tokenizer emits `:` after `def`/`class`/`if`/`for`/`while`/`with`, a learned "push" head writes; on dedent it "pops" by attenuating. This is a lightweight inductive bias on top of any flat scan.
- **Hierarchy mechanism:** a counter-like subspace explicitly representing scope depth and last-N enclosing-block summaries. The rest of the state is unconstrained.
- **Parallel-scan compatibility:** still a linear recurrence; the push/pop is just a structured contribution to A_t and B_t. Indent-level can be derived deterministically from input bytes (Python whitespace) and fed as an aux feature, so the scan stays causal-linear.
- **Why for code:** Python is parseable from indentation alone — we can hand the model the scope-depth feature for free, and it has known mechanistic value (variable-binding heads, Wu et al. 2025).
- **Key papers:** Wu/Milliere "How transformers learn variable binding" (2025); AST-T5 (Wang 2024); Tree-LSTM (Tai 2015).
- **Implementation cost:** M — preprocessing for indent feature is trivial; reserved-subspace constraint is a projection mask in DeltaNet.
- **Code-LLM relevance:** high. Aligns architecture with the one cleanly-extractable structural signal in the data.
- **Risk:** TinyStories has no scope, so eval may show no signal there; the gain may only appear on Python. Need an ablation that conditions only on the code subset.

### Idea 4: Autoregressive U-Net / H-Net hybrid scan
- **What it is:** byte-level encoder DeltaNet → learned chunk-pool → coarse DeltaNet → unpool → byte-level decoder DeltaNet, all causal. Each scale is its own scan. Skip connections from encoder to decoder.
- **Hierarchy mechanism:** explicit pooling/unpooling, like H-Net, but with linear-recurrent backbones at every level instead of attention.
- **Parallel-scan compatibility:** good per level, but the pool-unpool cross-level causality is fiddly — needs autoregressive masking similar to Hwang et al.
- **Why for code:** byte level removes BPE artifacts (Hwang shows ~4× data efficiency on code); coarse level is naturally function-scope.
- **Key papers:** Hwang H-Net (2025); MBLM Multiscale Byte LM (2025, arXiv 2502.14553); Funnel-Transformer (Dai 2020); Autoregressive U-Net (2025).
- **Implementation cost:** XL — three scan kernels + chunk router + autoregressive pool-unpool plumbing.
- **Code-LLM relevance:** high in principle, but at our 5k-step budget, the engineering risk dominates.
- **Risk:** straight-through chunking + 3 stacked scans makes the loss surface ugly; H-Net needed careful warmup. Probably a bad fit before we have a more stable training pipeline.

### Idea 5: SambaY-style GMU memory sharing — same state, more frequencies
- **What it is:** Phi-4-mini-flash's Gated Memory Unit. Lower DeltaNet layers compute a "memory readout" once; upper layers consume it through a cheap GMU instead of recomputing recurrence. Different layers therefore observe different effective timescales.
- **Hierarchy mechanism:** vertical-axis hierarchy (layer depth = timescale), not horizontal. The memory bank is shared and gated.
- **Parallel-scan compatibility:** native — only the bottom of the stack scans; everything above is gated readout. Big inference-latency win.
- **Key papers:** SambaY / Phi-4-mini-flash (Microsoft 2025); MEGA / Megalodon (Ma 2024) for gated EMA precedent.
- **Implementation cost:** M — adds a small gate per upper layer, removes redundant scan ops.
- **Code-LLM relevance:** medium. Helps inference economics more than perplexity per FLOP at 5k steps.
- **Risk:** at small scale the saved scans aren't bottlenecking us; benefit appears at long context which we are not yet evaluating.

## Recommendation

**Run #1 (Multi-timescale DeltaNet) first.** Cost is one afternoon. It's the smallest possible change that addresses the obvious flat-state pathology, it composes with our existing parity gate, and a clean negative result still tells us the data-dependent gate is already capturing scale separation. **Run #3 (scope-stack subspace) second**, conditioned on having a code-only eval split — this is the highest-leverage code-specific architectural prior we can cheaply test. Idea 2 (fast/slow with chunk gate) is the clear next milestone if #1 wins; #4 is too expensive until our pipeline stabilises.

## Open questions

- Do log-spaced fixed decays beat fully-learned ones when only 5k steps are available? (Cite: S5 ablations say yes for short training; selective Mamba says no for 100k+.)
- Can we get the indent-depth feature from the tokenizer cheaply, or do we need a pre-pass?
- Does a coarse slow state hurt the latency advantage of pure DeltaNet inference? Probably yes if its update is on every token; chunk-gated update should keep it cheap.
- At what scale does GMU memory sharing start to matter for our 32 GB / no-NVLink setup?
- How would we evaluate "scope tracking" specifically? Need a probe — e.g., does the model predict `self.` vs bare names correctly across block boundaries?
