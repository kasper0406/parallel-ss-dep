# The Latent-Thinking Challenge

> A bounded-state recurrent model learns to "think" its way through depth on a
> task where every step applies the **same** operation — but the **same
> mechanism, same recipe, same model** falls flat when every step applies a
> **different** operation. Make it work on the second task too.

This is a tiny, self-contained challenge (pure PyTorch, no `fla`/`triton`/
`transformers`/`flash-attn`). The whole thing trains in a few minutes on one
GPU and is meant to be read top to bottom in one sitting.

---

## The problem in one paragraph

We have a **linear-RNN** ("DeltaNet"): a fixed-size recurrent state, **no KV
cache, no attention over the input** — so inference is O(1) memory in sequence
length. We give it **latent thinking** (Coconut-style): after the prompt, the
model can append `R` "think" slots where it feeds its **own hidden state** back
in as the next input (full-bandwidth, not a discrete token), doing `R` extra
sequential compute steps before answering. We also give it a small
content-addressable **working memory**. On a **homogeneous** depth task — apply
the *same* function `f` n times, `f^n(s)` — latent thinking with `R=n` steps
nails it: **1.00 accuracy at every depth**, while a single forward sits near
chance. The lift is huge and grows with depth. But on a **heterogeneous** task
— a short program where each step is a *different* kind of operation — the same
latent thinking gives a meaningful lift only at shallow depth and **decays to
~zero as depth grows**, with the working memory contributing nothing useful.
**The challenge: make latent thinking + WM solve the heterogeneous task the way
they solve the homogeneous one.**

---

## The architecture (and why)

`model.py` (~250 readable lines):

- **DeltaNet layer** — a linear RNN. Per head, the state `S` is a fixed
  `d_head x d_head` matrix updated by the delta rule
  `S <- S + beta * (v - S k) k^T` and read by `S q`. No attention, no KV cache:
  the *only* thing that crosses time is the bounded state. A tiny causal
  depthwise conv (kernel 3) on q/k/v lets the model bind a value token to the
  key that preceded it (standard in linear-attention models; still bounded
  state). A `state_readonly` flag forces `beta -> 0` at think steps, so thinking
  reads the prompt's bindings but can never corrupt them.
- **Latent thinking** — `DeltaNetLM.think_forward(ids, R, mode=...)`. `mode="none"`
  answers from a single forward; `mode="latent"` feeds the model's own hidden
  state (through a small adapter, plus a WM read) back as the next input for `R`
  steps; `mode="token"` is a control that feeds a constant `[THINK]` embedding
  each step (no bandwidth between steps). The **same code path is used in
  training and eval.**
- **Working memory** — a write-gated buffer of past hidden states, read by
  soft-attention at think positions and injected into the think-slot input.
  Bounded size. On by default.

**Why bounded state?** Efficiency. A model that can think its way through depth
*without* a growing KV cache is cheap to deploy. The constraint is the point:
you may **not** add full attention over the input to "cheat" the recall.

---

## The two tasks

`tasks.py`. No tokenizer — the vocabulary is just small integers; the answer is
always a single value token in `0..V-1`. Each example also exposes the per-step
intermediates (the "chain"), and train/eval use **disjoint** RNG streams.

**Homogeneous — pointer-chase `f^n(s)`** (thinking *should* help). A random
table `f : {0..V-1} -> {0..V-1}` is presented as shuffled `(i, f(i))` pairs,
then `[QUERY, s]`. Target = `f^n(s)`: apply the **same** `f`, n times. One
forward can trace ~1 hop; n hops need n sequential steps — exactly what latent
thinking supplies. There is one reusable latent operator (`apply f`) to iterate.

**Heterogeneous — exec-trace** (the open problem). A start value `s` and a
program of `n` operations, **each a different kind**:

| op | effect |
|----|--------|
| `TABLE` | `x <- f(x)` (one shared random table `f`) |
| `ADD c` | `x <- (x + c) % V` |
| `MUL c` | `x <- (x * c) % V` |

The op sequence is encoded in the input (op-kind token + arg per step). Target =
the value after running all `n` ops in order. Because each step is a *different*
operation, there is **no single reusable latent operator** to iterate — which is
exactly why naive latent thinking collapses at depth.

---

## Baseline results (measured, final-answer-only supervision)

`V=10`, `n_max=6`, 3-layer DeltaNet `d=192` (~1.05M params), 3000 steps with a
depth curriculum, **final-answer-only** supervision (no per-hop labels — the
realistic regime; the per-hop reasoning thread must *emerge*). Held-out eval,
2048 examples per depth. Chance ≈ 0.10. `seed=0`.

**Homogeneous `f^n(s)` — latent thinking WORKS:**

| depth n | no-think | latent R=n | lift | token R=n (control) |
|--:|:--|:--|--:|:--|
| 1 | 0.098 | **1.000** | +0.902 | 0.886 |
| 2 | 0.206 | **1.000** | +0.794 | 0.119 |
| 3 | 0.185 | **1.000** | +0.815 | 0.209 |
| 4 | 0.308 | **1.000** | +0.692 | 0.206 |
| 5 | 0.203 | **1.000** | +0.797 | 0.287 |
| 6 | 0.406 | **1.000** | +0.594 | 0.228 |

Latent thinking hits **1.00 at every depth**; the **token control** (same think
slot but constant embedding, no bandwidth) collapses to chance for n>=2 — proving
it is the *fed-back content* (bandwidth), not just "extra steps", that carries
the computation.

**Heterogeneous exec-trace — latent thinking FAILS at depth:**

| depth n | no-think | latent R=n | lift | token R=n (control) |
|--:|:--|:--|--:|:--|
| 1 | 0.328 | 0.684 | +0.356 | 0.366 |
| 2 | 0.166 | 0.420 | +0.255 | 0.302 |
| 3 | 0.120 | 0.235 | +0.116 | 0.186 |
| 4 | 0.129 | 0.204 | +0.075 | 0.145 |
| 5 | 0.153 | 0.190 | +0.038 | 0.142 |
| 6 | 0.163 | **0.167** | **+0.004** | 0.141 |

The lift is real at shallow depth but **decays toward zero** as depth grows; by
n=6 latent thinking is at chance and adds nothing. The working memory is not
usefully utilized. (Reproduced across seeds — `seed=1` gives the same shape:
+0.37 at n=1 down to +0.08 at n=6.)

**The dichotomy at a glance (lift = latent − no-think):**

```
 lift  +0.9 |  H———H———H———H———H———H     homogeneous (H): flat at the top
            |
       +0.4 |  x
            |       x
            |            x
       +0.0 |                 x    x    x   heterogeneous (x): decays to 0
            +----------------------------------
             n=1   n=2   n=3   n=4   n=5  n=6
```

---

## The success criterion

Train on the **heterogeneous** task (final-answer-only supervision is fine —
that is the honest regime; if you use per-hop supervision, say so) and produce a
model whose **`latent R=n` lift over no-think becomes positive and *grows* (or
at least holds) with depth**, approaching the homogeneous curve — instead of
decaying to zero. Concretely, beat this bar:

> **mean lift at deep n (n>=3) > +0.40**, with **latent R=6 accuracy > 0.60**,
> while keeping the bounded-state constraint (no full attention over the input)
> and not regressing the homogeneous task.

A clean win shows the working memory or the latent channel actually carrying the
per-step *program* (which op to apply next, and the running value), so that
depth helps the way it does in the homogeneous case.

---

## How to run

```bash
pip install -r requirements.txt           # just torch
CUDA_VISIBLE_DEVICES=0 ./run.sh           # both tasks, ~10-14 min total

# or individually:
python train.py --task homogeneous   --V 10 --n_max 6 --steps 3000
python train.py --task heterogeneous --V 10 --n_max 6 --steps 3000
```

Useful flags: `--deep_supervision` (per-hop labels — note this is a *crutch*:
it lets even the heterogeneous task hit ~1.00, which is why the honest baseline
above is final-answer-only), `--no_memory`, `--state_write` (let think steps
write the recurrent state), `--d_model/--n_layers`, `--save model.pt`.

---

## What's been tried and failed (don't just redo these)

- **Final-answer-only + depth curriculum** (the recipe that *makes the
  homogeneous task work*, the per-hop thread emerging unsupervised): on
  heterogeneous it gives the decaying-lift table above.
- **Per-hop supervision** as a fix: it does make heterogeneous hit ~1.00 — but
  it works by *handing the model every intermediate*, so it doesn't tell you how
  to get there from final answers. It masks the problem rather than solving it.
  (Try it: `--deep_supervision`. The interesting question is doing it
  *without* per-step labels.)
- **Bigger working memory** / wider model: more capacity does not change the
  shape — the lift still decays with depth.
- **Constant-embedding ("token") thinking**: a control, not a fix — confirms
  bandwidth matters (it fails on homogeneous too).
- In the larger research program this challenge is distilled from, a literal
  **program-counter** input and **verbatim content lookup** in WM were also
  tried and did not transfer.

The open question is *representational*: in the homogeneous task there is a
single latent operator to iterate; in the heterogeneous task the model must, at
each latent step, (a) recall *which* operation comes next and (b) apply that
specific operation to the running value — and route that through a bounded state
+ a small memory. Making the latent channel and WM carry the **program** is the
crux.

---

## Rules

1. Keep the **bounded-state** constraint: no full attention / no KV cache over
   the input. The short causal conv (fixed tiny receptive field) is allowed;
   re-reading the whole prompt at answer time is not.
2. The think mechanism must use the **same code path in train and eval**.
3. Report **held-out** numbers on the disjoint eval split, per depth, as the
   table above (no-think vs latent R=n vs the token control).
4. State your supervision (final-answer-only vs per-hop) plainly.
5. Don't regress the homogeneous task.

## Files

```
model.py          tiny DeltaNet + latent thinking + working memory (~250 lines)
tasks.py          the two integer-token tasks (homogeneous / heterogeneous)
train.py          curriculum training + the per-depth eval table
run.sh            runs both tasks and prints the dichotomy
requirements.txt  torch
```
