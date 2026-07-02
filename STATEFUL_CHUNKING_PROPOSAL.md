# Stateful chunking — rolling the DeltaNet recurrent state across the T=2048 boundary

**Scope:** what it would take to carry the linear-RNN recurrent state from the
end of one T=2048 chunk into the start of the next chunk of the *same document*
during pretraining (truncated-BPTT / "stateful chunking"), instead of treating
every chunk as an independent sequence. Read-only investigation + value
estimate; no code changed.

**Bottom line up front:** ~**27.5% of training tokens** in `pretrain_mix_v18_arxiv.yaml`
sit beyond position 2048 inside a >2048-token document and would actually gain
new left-context from cross-chunk carry. But — contrary to the "this is an arXiv
lever" intuition — **~52% of that benefit is `codeparrot` (raw source files) and
~78% is code / code-adjacent** (codeparrot + bigvul + the programming textbook);
arXiv full-text is only ~19% of it. So it IS relevant to this code model. The
catch: DeltaNet's state is a *bounded, lossy* summary (unlike an attention KV
cache), so the realized gain is capped and concentrated near boundaries, and the
loader restructure carries real correctness risk. **Recommendation:** gate the
work behind a cheap inference-only "warm-vs-cold state" probe on the existing
ckpt (described in §8); if that shows a real CE gap, do it; otherwise raise `T`
first (a near-one-line partial substitute). Details below.

Files inspected (exact):
- `experiments/data_mix.py` — `MixedSourceStream.__iter__` / `fill_buffer` (L515–707), `_extract_text` (L283), `SourceConfig` (L144), `load_sources_from_yaml` (L713).
- `experiments/model.py` — `TinyLM.forward` (L3288–3690), `Block.forward` (L163–194), `_run_block`/`_ckpt_run_block` (L220–237), `_build_cu_seqlens` (L240–273), `_block_fwd`/block loop (L2987–3055).
- `experiments/layers.py` — `_FlaWrapper.forward` (L247–274), `_FlaWrapper.forward_step` (L276–335), `DeltaNetAttention` (L338–363).
- `~/ml/flash-linear-attention/fla/layers/delta_net.py` — `DeltaNet.forward` (L170–292).
- `~/ml/flash-linear-attention/fla/ops/delta_rule/chunk.py` — `chunk_delta_rule` (L214–321).
- `experiments/train_lm.py` — step body + grad-accum loop (L1072–1090, L1505–1660), DataLoader setup (L484–544), DDP wrap (L631–660), val loop (L2085–2096).
- `configs/pretrain_mix_v18_arxiv.yaml`, `launch_pretrain_v18_arxiv_resume.sh` (arch: `--arch deltanet --n_layers 10 --d_model 896 --n_heads 14 --d_head 64 --T 2048 --batch 4 --grad_accum 32`).
- Value estimate: `/tmp/estimate_longdoc_tokens.py` (CPU-only, run command in §6).

---

## 1. Current behavior (no carry)

The pretrain loader (`MixedSourceStream`) keeps **one running token buffer per
source**. Each iteration it:
1. picks ONE source by weight (`rng.choices(..., weights=weights)`, L597);
2. fills that source's buffer to `block_size+1` tokens, appending an EOS after
   every document (`fill_buffer`, L548–593);
3. slices `chunk = buffer[:block_size+1]; buffer = buffer[block_size:]` (L605–606),
   optionally injects think-bursts, and yields `(inputs, targets[, doc_ids[, read_mask]])`.

Consequences for state:
- Consecutive chunks of the **same document** are emitted *in order* from that
  source's buffer, but they are **interleaved with chunks from every other
  source** and are **not pinned to any batch row**. Chunk *k+1* of an arXiv
  paper may land in a different batch row, a different microbatch, and a
  different optimizer step than chunk *k* — with arbitrary other content in
  between.
- `TinyLM.forward` builds `cu_seqlens = _build_cu_seqlens(doc_ids)` (model.py
  L3352). `cu_seqlens` makes the FLA kernel **reset** the recurrent state at
  every row boundary and at every doc-id change *within* a chunk (cross-document
  isolation). Crucially, it resets at the **start of every chunk** too — so the
  recurrent state always begins at zero for each chunk. There is no path today
  that feeds a non-zero `initial_state`.
- The training loop (`train_lm.py`) carries **no** state between steps or
  microbatches. Each `_nonthink_forward_loss(...)` call is a fully independent
  forward.

So today every 2048-window is an island: a token at position 0 of chunk *k>1*
of a long document sees *nothing* of the preceding ~`(k-1)·2048` tokens.

---

## 2. The exact state object + FLA API for carrying it

`DeltaNetAttention` wraps `fla.layers.DeltaNet(mode="chunk", ...)`. Its
`forward` (delta_net.py L170–292) already speaks the full stateful protocol — we
just never invoke it in training:

```python
recurrent_state = last_state['recurrent_state'] if last_state is not None else None
o, recurrent_state = chunk_delta_rule(
    q, k, v, beta,
    initial_state=recurrent_state,      # <-- carried state IN
    output_final_state=use_cache,       # <-- ask for final state OUT
    cu_seqlens=cu_seqlens,
    use_qk_l2norm_in_kernel=(self.qk_norm == 'l2'),
)
```

`chunk_delta_rule` (chunk.py L214–321) signature: `initial_state` shape
**`[N, H, K, V]`** and, when `output_final_state=True`, it returns
`final_state` of the same shape `[N, H, K, V]`. For our config
(`expand_k=expand_v=1.0`): `K = V = d_head = 64`, `H = n_heads = 14`. So **per
layer, per sequence** the state is `[14, 64, 64] = 57,344` floats.

Two ways to thread it:

**(A) Explicit `initial_state` / `final_state` (recommended).** Add
`initial_state=` and `return_final_state=` kwargs down
`TinyLM.forward → Block.forward → _FlaWrapper.forward → DeltaNet`. The wrapper
calls `self.layer(x_flat, cu_seqlens=cu_seqlens, past_key_values=<Cache holding the state>, use_cache=True)`
OR threads `initial_state` directly into a thin call. Gives explicit,
per-step control over detach (TBPTT) and storage dtype. This is the cleanest for
training.

**(B) Reuse the FLA `Cache` machinery** that `forward_step` already uses
(layers.py L276–335: `past_key_values=Cache()`, `use_cache=True`). The Cache
carries `recurrent_state` **and** the short-conv state (see below) per
`layer_idx`. Lower-effort to wire (the chunk path honors `past_key_values`
identically to the decode path), but the Cache stores grad-carrying tensors and
its in-place `update_layer_cache` mutation is awkward to detach cleanly between
steps and interacts poorly with `torch.compile` + activation checkpointing. Use
(A); borrow only the conv-state insight from (B).

**Second piece of state — the short convolutions.** `use_short_conv=True`, so
each of q/k/v passes through a `ShortConvolution(kernel_size=conv_size=4)`. The
conv has its own cache (`conv_state_q/k/v`, delta_net.py L197–218) holding the
last `conv_size-1 = 3` tokens of left context. The `forward_step`/Cache path
carries it; a pure `initial_state` carry of only the recurrent matrix does
**not**. **Recommendation:** for v1, carry only `recurrent_state` and accept the
3-token conv discontinuity at each chunk boundary (3 of 2048 tokens lose their
4-gram conv context — a negligible approximation). Carry the conv state only if
the §8 probe shows it matters.

**Interaction with `cu_seqlens` — this is the crux.** When `cu_seqlens` is set,
the kernel requires `q.shape[0] == 1` (we already flatten `(B,T,d) → (1,B*T,d)`,
layers.py L261–262) **and** `initial_state.shape[0] == len(cu_seqlens) - 1 == N`
(chunk.py L303–307), where `N` is the number of *segments* (one per
row-boundary + one per intra-chunk doc change), so **`N ≥ B`**. The kernel
applies `initial_state[seg]` at the start of each segment. Therefore the carried
state is **only valid for the FIRST segment of each row** (the continuation of
that row's previous chunk); every other segment is a fresh document and must get
a **zero** initial state. Concretely the carry must build an `[N, H, K, V]`
tensor:
- `initial_state[first_segment_of_row_r] = carried_state[r]`
- `initial_state[all_other_segments] = 0`

and on the way out extract `final_state[last_segment_of_row_r]` as the next
carried state for row `r`. Mapping "row → its first/last cu_seqlens segment" is a
small index computation from `doc_ids`/`cu_seqlens`, but it is the load-bearing
bookkeeping and the most bug-prone part (it is exactly the class of off-by-one /
row-alignment bug the repo already hit building cross-document isolation).

---

## 3. The hard problem: per-batch-row document continuity

To carry state you need **consecutive chunks of one document fed to the same
persistent slot in order**, with that slot's previous `final_state` as the next
`initial_state` (detached). Today's global-weighted-sampling + shared-buffer
loader gives **none** of this.

### Required loader restructure — the Megatron/GPT "persistent batch slot" pattern

Define `S = batch × grad_accum` **persistent streams** per DDP rank (for v18:
`4 × 32 = 128`). Note from the step body (train_lm.py L1074 first microbatch,
L1519 microbatches 1..n-1) that one optimizer step consumes `batch` rows per
microbatch × `grad_accum` microbatches = `S` independent sequences. The mapping
that makes carry work:

- Slot `s` is touched **exactly once per optimizer step**, always in the same
  `(microbatch_index, row_index)` position. Its document advances by one
  2048-window per step. `state[s]` carries (detached) from step `k` to step
  `k+1`. ⇒ **TBPTT window = T = 2048** (one chunk per step per slot).
- Each slot independently: consume one document at a time, emit consecutive
  windows; **when the document ends, zero that slot's state and pick a new
  document by weight.** Short docs (<2048) still pack several per chunk — for
  those the "carry" is moot (the slot resets state at each doc boundary and the
  next doc starts fresh), so the benefit is entirely on the >2048 docs.

### Can the weighted-mix semantics be preserved?

Yes, but the weighting moves from *per-chunk* to *per-document*: when a slot
finishes a document it draws the next source by the same weight vector, then
streams a document from that source. This keeps the long-run token mix ≈ the
configured weights (a slot spends proportionally more *windows* on a long doc,
but it also *picked* that source proportionally to weight, so token share ≈
weight — the same approximation the current loader makes). The think-burst
injection, `mask_eos_in_targets`, `emit_read_mask`, and FIM all still apply
per-window.

### How much of `MixedSourceStream` must change?

This is effectively a **new** `PersistentSlotStream` (or a major rewrite of
`__iter__`). What's reusable verbatim: `_open_stream`, `_extract_text`,
`_build_filter`, `maybe_apply_fim`, `_build_read_mask`, `insert_think_bursts`,
the EOS/think/doc-id/read-mask packing logic. What changes: the single shared
per-source buffer becomes **`S` per-slot buffers**, each tied to one source's
iterator *for the duration of one document*; the chunk-emission order becomes a
fixed slot order so `DataLoader(batch_size=batch)` re-assembles the same slots
into the same microbatch positions every step. The yielded tuple gains a
**`new_doc_at_pos0`** flag per row (or equivalently the loader emits enough for
the train loop to know whether slot-state should be zeroed because the chunk
starts a fresh document rather than a continuation).

Worker subtlety: state continuity requires deterministic slot order. Easiest
correct option is `num_workers=0` (single-process loader), or each worker owns a
**disjoint, fixed** subset of slots. The current `num_workers=2` global-shuffle
setup is incompatible with carry as-is.

---

## 4. Model-forward threading

Add to `TinyLM.forward(..., initial_state: list[Tensor] | None = None,
return_final_state: bool = False)`:
- thread `initial_state[L]` and `return_final_state` through `_block_fwd →
  _run_block / _ckpt_run_block → Block.forward → _FlaWrapper.forward` for the
  DeltaNet attention only (gate on `accepts_cu_seqlens`/a new
  `accepts_state_carry`); every non-FLA attention ignores it.
- `_FlaWrapper.forward` builds the `[N,H,K,V]` `initial_state` (carried state at
  first-segment-of-row slots, zeros elsewhere — §2), calls `chunk_delta_rule`
  (via `DeltaNet.forward`) with `output_final_state=return_final_state`,
  reshapes `o` back to `(B,T,d)`, and returns `(out, final_state)` when asked.
- `TinyLM.forward` collects per-layer `final_state`, extracts
  `final_state[last_segment_of_row]` per row per layer, and returns it through
  `_finalize` as an extra output.

**Per-layer / multi-pass care:**
- **FiLM K=3 self-feed** (the v18 path) runs the block stack up to K times. All
  passes use the *same* `initial_state` (correct — each pass is a fresh forward
  over the same chunk from the same start). The carried-out `final_state` must
  come from the **last (grad-carrying)** pass only. Same for the xattn 2-pass
  path (pass-1 is `no_grad`; take `final_state` from pass-2).
- **Activation checkpointing** (`_ckpt_run_block`): `initial_state` is an input
  tensor (checkpoint saves it); `final_state` becomes an additional checkpoint
  *output* — `_run_block`/`_ckpt_run_block` signatures must return the tuple and
  `torch.utils.checkpoint(use_reentrant=False)` supports multi-output. Minor but
  must be done in both wrappers.
- **`torch.compile`**: returning a new per-layer output tensor changes the graph
  signature; expect a recompile and watch for the AOTAutograd ViewMeta-replay
  class of bugs the repo has hit when adding graph outputs (the gist-loss
  backward tripped exactly this; workaround
  `torch._functorch.config.view_replay_for_aliased_outputs = False` already in
  `speed_knobs.py`). De-risk by validating `--no-compile` first.

### State memory cost

Per rank: `S × n_layers × H × d_head × d_head` = `128 × 10 × 14 × 64 × 64` =
**73.4 M floats** = **~147 MB bf16 / ~294 MB fp32**, persistent on the GPU
across steps. Modest, but the live runs already sit at ~25–30 GiB / 32 GiB, so
budget it (carry bf16 — it matches the in-kernel precision). Carrying conv state
too would add `S × n_layers × 3 × (key_dim·2 + value_dim)` ≈ another few-hundred
MB — another reason to skip conv carry in v1.

---

## 5. Training-loop changes + gradient choice

- Allocate `carry_state[s]` for `s in range(S)` (a list of per-layer bf16
  tensors, or one `[S, L, H, d, d]` tensor), init zero, on GPU, persistent
  across steps.
- Per microbatch, look up the `batch` slots it serves, gather their
  `carry_state` into the forward's `initial_state`, **zero the rows whose chunk
  starts a fresh document** (the `new_doc_at_pos0` flag from §3), run the
  forward with `return_final_state=True`, then write `final_state.detach()` back
  to `carry_state[s]`.
- **Gradient: truncated BPTT (detach at every chunk boundary). Recommended.**
  Store `final_state.detach()`; gradient flows only within a single 2048-chunk.
  Cheap, standard, numerically stable, no extra retained graph across steps.
  **Full BPTT across chunks** (keep the graph) would need the previous step's
  activations retained → multiplies memory by the chain length and forbids the
  per-step `optimizer.step()`; not worth it — TBPTT is the established choice for
  exactly this (Mistral/GPT stateful packing, RNN-LM TBPTT).
- **grad_accum:** benign — each `(microbatch_index, row)` is its own persistent
  slot; detach decouples microbatches. No `no_sync`/reducer interaction beyond
  what already exists.
- **DDP:** each rank owns its own `S` slots and its own `carry_state`; no
  cross-rank state sync. (Note the repo's standing constraint: latent-thinking
  pretrain already runs single-GPU; if stateful chunking ships alongside latent
  thinking it inherits that single-GPU restriction, but stateful chunking by
  itself is DDP-compatible.)
- **bf16:** carry/detach in bf16; the kernel computes state in bf16 under
  autocast anyway.

---

## 6. Value estimate — fraction of tokens that benefit

CPU-only, replicating the data_mix extraction + filter path (no FIM; FIM only
adds ~3 sentinel tokens and doesn't change the length distribution), tokenizing
with `HuggingFaceTB/SmolLM2-135M`, 400 docs (or 75 s) sampled per source:

```
CUDA_VISIBLE_DEVICES="" MAX_DOCS=400 TIME_BUDGET=75 .venv/bin/python /tmp/estimate_longdoc_tokens.py
```

`%tok>T` = fraction of a source's tokens that live in docs > 2048.
`%tokCarry` = fraction beyond position 2048 (the tokens that literally gain new
left-context) = `Σ max(0, L−2048) / Σ L`.

| source | weight | median | p90 | max | %docs>T | %tok>T | %tokCarry |
|---|--:|--:|--:|--:|--:|--:|--:|
| multibind_recall_pretrain | 0.050 | 871 | 1671 | 1746 | 0.0% | 0.0% | 0.0% |
| code_recall_train | 0.050 | 1310 | 1747 | 1851 | 0.0% | 0.0% | 0.0% |
| agentic_recall_train | 0.030 | 1148 | 1641 | 1830 | 0.0% | 0.0% | 0.0% |
| synthetic_memory | 0.030 | 134 | 152 | 164 | 0.0% | 0.0% | 0.0% |
| **codeparrot** | **0.220** | 1205 | 5126 | 380742 | 30.8% | **82.1%** | **64.8%** |
| python_codes_25k | 0.080 | 74 | 133 | 269 | 0.0% | 0.0% | 0.0% |
| magicoder_oss | 0.090 | 533 | 775 | 1737 | 0.0% | 0.0% | 0.0% |
| code_exercises_jinaai | 0.020 | 188 | 263 | 293 | 0.0% | 0.0% | 0.0% |
| python_instr_18k_alpaca | 0.060 | 242 | 571 | 5949 | 1.5% | 15.9% | 7.4% |
| codealpaca_20k | 0.050 | 72 | 168 | 938 | 0.0% | 0.0% | 0.0% |
| **bigvul** | **0.080** | 967 | 3738 | 26871 | 23.0% | **67.3%** | **44.0%** |
| cybernative_vuln_dpo | 0.040 | 256 | 380 | 2134 | 0.2% | 2.0% | 0.1% |
| textbooks_lite_sciphi | 0.040 | 658 | 968 | 1924 | 0.0% | 0.0% | 0.0% |
| **textbook_quality_programming** | **0.040** | 30190 | 44791 | 68646 | 99.2% | **100.0%** | **92.8%** |
| wikipedia_en | 0.040 | 460 | 1454 | 8015 | 3.2% | 17.7% | 8.1% |
| **arxiv_fulltext** | **0.060** | 13121 | 31469 | 89920 | 99.2% | **99.9%** | **87.4%** |
| arxiv_abstracts | 0.020 | 236 | 370 | 553 | 0.0% | 0.0% | 0.0% |

**Weighted overall (weight ≈ token share):**
- fraction of training tokens in docs > 2048: **35.2%**
- fraction of training tokens beyond pos 2048 (carry-benefit): **27.5%**

**Where the 27.5% comes from** (contribution = `weight × %tokCarry`):

| source | contribution (pts) | share of benefit |
|---|--:|--:|
| codeparrot | 14.3 | **52%** |
| arxiv_fulltext | 5.2 | 19% |
| textbook_quality_programming | 3.7 | 13% |
| bigvul | 3.5 | 13% |
| wikipedia_en + python_instr_18k | 0.8 | 3% |

So the benefit is **dominated by large raw source files (codeparrot)**, with
arXiv full-text a secondary contributor. **Code / code-adjacent sources
(codeparrot + bigvul + textbook_programming) = ~21.5 of 27.5 pts ≈ 78%** of the
benefit. The framing "this is mainly an arXiv lever" is **not** supported by the
data — it is mainly a *large-source-file* lever, which is squarely on-mission for
a code model.

Caveats on the number: 400-doc samples per source (time-capped), so the heavy
tail (codeparrot max = 380 k tokens) is sampled and the fractions have sampling
noise (±a few points on the heavy-tailed code sources); the qualitative ordering
is robust. The script printed the full table and then hit a benign parquet-stream
teardown crash *after* all results were collected — the numbers are valid.

---

## 7. What else carries / resets

- **WorkingMemory buffer** (`WorkingMemory.forward`, model.py L1796): built
  **per-forward** from the current sequence's hidden states; it does NOT persist
  across forward calls today. **Reset per chunk** (do not carry in v1). Carrying
  the WM buffer across chunks is a much larger, separate change (and the repo's
  WM/recall work suggests it's not the bottleneck for the >2048-doc case). Note:
  WM already takes `doc_ids` for same-document read masking, so a carried WM
  would face the same first-segment-only validity issue.
- **Short-conv state:** see §2 — skip in v1 (3-token approximation).
- **FiLM K=3 self-feed lagged sources:** entirely within-forward (computed from
  the chunk's own pass-1 outputs) — unaffected, no carry needed.
- **gist loss / latent-reasoning aux / gate-calibration:** all operate on the
  current forward's outputs; no state to carry. The carried `initial_state` does
  make the gist/gate targets slightly *better* (the hidden states now reflect
  prior-chunk context) but needs no special handling. Their **extra forwards**
  (process_reward / gate_calibration run a second `[prefix, K*THINK]` forward)
  must pass `initial_state=None` / `return_final_state=False` so they don't
  clobber the carry — same snapshot discipline the loop already uses for
  `_last_gate` (train_lm.py L1539–1576).
- **Positional embed** (`self.pos_embed`, if `max_T>0`): indexed `0..T-1` per
  chunk. With carry, chunk *k>1* still gets positions `0..T-1`, not its true
  document offset. For DeltaNet this is benign (the recurrence carries order),
  but it's a known mild inconsistency — leave as-is in v1.

---

## 8. Cheap gating probe before building anything (inference-only)

Before paying for the loader restructure, measure the **upside ceiling** with
zero training, on the existing v18 ckpt (needs a GPU — run when the live job
frees one; do **not** co-run):

> Take long docs (codeparrot/arxiv >4096 tokens). For each, run chunk *k* both
> **cold** (fresh state, today's behavior) and **warm** (`initial_state` =
> `final_state` from a forward over chunk *k−1*). Compare next-token CE on chunk
> *k*'s tokens. The warm−cold CE gap (especially on the first few hundred tokens
> of chunk *k*) is the maximum PPL win stateful *training* could realize.

This is ~50 lines reusing `_FlaWrapper` + a manual `chunk_delta_rule(...,
initial_state=..., output_final_state=True)` call, no loader changes, no
training. If the gap is small (DeltaNet's bounded state may simply not retain
enough across 2048 tokens to matter), **stop** — the loader rewrite won't pay
off. If it's meaningful, proceed to Phase 1.

---

## 9. Phased implementation plan (effort / risk)

| Phase | Work | Effort | Risk |
|---|---|---|---|
| **0. Gating probe** (§8) | Inference-only warm-vs-cold CE on existing ckpt. | **0.5 day** | Low. Decides go/no-go. |
| **1. Model forward threading** | `initial_state`/`return_final_state` through `TinyLM.forward → _block_fwd → Block → _FlaWrapper → DeltaNet`; the `[N,H,K,V]` first-segment build + last-segment extract; FiLM/xattn last-pass-only; checkpoint multi-output. Unit test vs the equivalence the repo already has (`test_cu_seqlens.py` packed==unpacked) extended with a 2-chunk warm-state == single 4096-window equality test. | **2–3 days** | **Medium.** The `[N,H,K,V]` segment↔row mapping is the bug-prone core; gate behind the equality test. |
| **2. Persistent-slot loader** | New `PersistentSlotStream` (S per-slot buffers, per-doc weighted source pick, fixed slot order, `new_doc_at_pos0` flag), `num_workers=0` or disjoint-slot workers; preserve think-burst/EOS-mask/read-mask/FIM. Smoke: per-source token mix ≈ weights; slot order stable across steps. | **3–4 days** | **Medium-high.** Largest surface; reproducing the weighted mix + multi-worker determinism is fiddly. |
| **3. Train-loop state mgmt** | `carry_state[S]`, gather/zero-fresh/scatter-detach per microbatch, GPU memory budget, aux-forward isolation. | **1–2 days** | Medium. detach correctness + the ~150 MB state alloc. |
| **4. compile/DDP hardening** | Validate `--no-compile` first, then re-enable; confirm activation-checkpointing path; DDP smoke. | **1 day** | Medium (compile graph-output bugs). |
| **5. Ablation run** | Short pretrain continuation with vs without carry; compare per-source held-out CE (esp. codeparrot/arxiv/bigvul) and the chunk-2+ token CE. | **GPU time** | Low. |

Total ~**8–12 engineering days** + GPU ablation. The repo standing rule applies:
have a background agent review the diff before any scaled run (especially the
segment↔row index math).

---

## 10. Honest bottom-line recommendation

- The benefit is **real and on-mission**: 27.5% of tokens are carry-eligible and
  ~78% of that is code / code-adjacent (codeparrot large files dominate). This is
  **not** "just an arXiv lever."
- But it is **bounded**: DeltaNet's state is a fixed `[H,64,64]` lossy summary,
  not a full KV cache, so cross-chunk carry gives a *degraded* long-context
  signal concentrated near boundaries. The realized PPL win is unknown and
  plausibly modest — which is exactly why Phase 0 (the inference-only warm-vs-cold
  probe) must run first; it's cheap and decisive.
- The **cost/risk** is medium-high, concentrated in (a) the persistent-slot
  loader rewrite and (b) the `[N,H,K,V]` segment↔row alignment — the same family
  of subtle bug the cross-doc-isolation work already had to get exactly right.
- **Cheaper partial substitutes worth weighing first:** raising `--T` to
  4096/8192 directly shrinks the carry-eligible fraction (at T=4096, codeparrot
  p90=5126 → most code files fit in ≤2 chunks; cost is ~2–4× activation memory,
  addressable with the existing `--activation_checkpointing`), and better
  doc-packing reduces fragmentation for the short-doc bulk (no help for >T docs).

**Recommendation:** run Phase 0. If the warm−cold CE gap on long code/arXiv docs
is meaningful (say ≥~0.05 CE on chunk-2+ tokens), build it — it's a legitimate,
code-relevant lever. If the gap is small, prefer raising `T` and improving
packing, and shelve stateful chunking. Do **not** build the loader rewrite on the
27.5% headline alone; that number is the *addressable* token fraction, not the
*realized* gain.
