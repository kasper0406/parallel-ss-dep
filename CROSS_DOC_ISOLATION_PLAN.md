# Implementation Plan: Document-Boundary Isolation for DeltaNet

**Status:** IMPLEMENTED 2026-05-14 (Steps 1–5 complete; 88 tests green).
Open follow-ups remain — see "Open risks" #8 (RL replay-packing path still
passes `doc_ids=None`) and #6 (FiLM `_shift_right` is not doc-aware,
deliberate). Produced by a planning pass over the actual code.

## The problem

`data_mix.py` packs multiple independent documents into each T=2048 training
sequence, separated only by an EOS *token*. DeltaNet is a linear-RNN — its
recurrent state (and `WorkingMemory` reads) flow straight across those EOS
boundaries. The model wastes capacity learning spurious cross-document
dependencies. A transformer would prevent this with a block-diagonal attention
mask; FLA's chunk kernels support `cu_seqlens` for exactly this.

## Key findings from reading the code

**FLA signature (the load-bearing fact):** the local FLA fork supports
`cu_seqlens` end-to-end:
- `fla/ops/delta_rule/chunk.py::chunk_delta_rule(...)` takes `cu_seqlens` and
  `cu_seqlens_cpu`, threaded into `ChunkDeltaRuleFunction.apply` — **fwd + bwd
  both** carry it (backward is correct, not just inference).
- `fla/layers/delta_net.py::DeltaNet.forward` reads `cu_seqlens` from kwargs
  and passes it to `chunk_delta_rule` *and* all three `ShortConvolution` calls
  — so the short-conv is also boundary-aware (else it would leak 3 tokens).
- **Hard constraint** (`chunk.py:297-302`): when `cu_seqlens` is passed
  directly, `q.shape[0]` must be `1`. The kernel operates on a single
  flattened `(1, total_tokens, H, D)` ragged tensor; `cu_seqlens` is a `[N+1]`
  index over that flat axis and can encode arbitrarily many documents.
- The alternative `attention_mask` route is **unusable** — a padding mask is
  one-document-per-row and cannot express multiple packed documents per row.
  We must use the direct `cu_seqlens` route and do the `(B,T)→(1,B*T)` flatten
  ourselves.

**Consequence:** because `cu_seqlens` is a flat index over `B*T`, one global
`cu_seqlens` covers the whole batch as long as every `Block` flattens
`(B,T,d)→(1,B*T,d)` consistently before the DeltaNet wrapper and reshapes back
after. Row boundaries become document boundaries automatically.

**Wrapper location:** `experiments/layers.py::_FlaWrapper.forward` currently
takes only `x`. `cu_seqlens` must be threaded `TinyLM.forward → Block.forward
→ _FlaWrapper.forward → DeltaNet.forward(**kwargs)`.

## Step 1 — Data pipeline: compute and emit document boundaries

**File:** `experiments/data_mix.py`

- **1a.** In `MixedSourceStream.__iter__`, alongside `buffers` maintain
  `buffer_docids: list[list[int]]`. In `fill_buffer`, before `extend(ids)`,
  append a running per-source document counter for every token in `ids`, then
  once more for the `eos`, then increment. Slice `buffer_docids[idx]` in
  lockstep with `buffers[idx]`.
- **1b.** The EOS token is labeled with the *closing* document's id; the
  `cu_seqlens` boundary goes *after* the EOS (the model still learns
  "document ends → predict EOS"; state resets for the *next* doc's first real
  token).
- **1c.** `insert_think_bursts` (`sft_code.py`) shifts tokens — the `doc_id`
  array must be transformed by the same permutation. Preferred: make
  `insert_think_bursts` accept an optional aligned third array and insert a
  copy of the neighbouring position's doc_id at each think position. Keep the
  arg optional so existing callers/tests are unaffected. Make the
  aligned-array transform a small reusable helper (`_reorder_aligned`) — FIM
  (#65) needs the identical mechanism.
- **1d.** Yield `(inputs, targets, doc_ids)`. To bound blast radius, gate the
  third element behind a constructor flag `emit_doc_ids: bool = False`
  (default keeps the 2-tuple, matching the `mask_eos_in_targets` pattern);
  `train_lm.py` sets it `True`.
- **1e.** `val_ds` uses the same class — emits `doc_ids` automatically.

## Step 2 — Trainer: thread doc_ids into the model call

**File:** `experiments/train_lm.py`

- Batch unpack `x, y = batch` → `x, y, doc_ids = batch` (default collate turns
  per-sample `(T,)` into `(B,T)`); `.to("cuda")`.
- Forward calls: add `doc_ids=doc_ids`. Same for the validation loop unpack +
  forward.
- The aux-brackets synthetic path and the RL/replay-packing path pass
  `doc_ids=None` (default). Audit `train_rl.py`, `train_distill.py`,
  `sft_code.py`, `eval_*.py` callsites — `doc_ids=None` keeps current
  behaviour everywhere.

## Step 3 — Model: accept doc_ids, derive cu_seqlens, thread to every block

**File:** `experiments/model.py`, `experiments/layers.py`

- **3a.** `TinyLM.forward` signature: add `doc_ids=None`.
- **3b.** Add `_build_cu_seqlens(doc_ids, B, T, device)` called once at the top
  of `forward`. `doc_ids is None` → rectangular `[0, T, 2T, …, B*T]` (each row
  its own document — correct leak-free default, lets us *always* use the
  `cu_seqlens` route). Otherwise detect segment changes
  (`doc_ids[b,t] != doc_ids[b,t-1]` or `t==0`), flatten row-major, build
  `cu_seqlens` (int32) + a `.cpu()` copy. Build on-device; do not store on
  `self` (re-entrancy with activation checkpointing / multi-pass FiLM).
- **3c.** Thread `cu_seqlens` through every block callsite: `_block_fwd`,
  `_run_block`/`_ckpt_run_block` (`layers.py`), `Block.forward`, the plain
  block loop, the `_film_bypass` path, the `feedback_xattn_pairs` path (both
  passes), `_sparse_pass_collect_sources` + the K-self-feed loss-bearing loop,
  the standard 2-pass sparse FiLM, and the dense feedback path.
- **3d.** FiLM needs **no** `cu_seqlens` logic — it operates only on the
  residual stream `h` in `(B,T,d)` and never touches the kernel. The same
  `cu_seqlens` is reused for every pass. (Low-risk residual subtlety:
  `_shift_right_by_k` lags source state across doc boundaries — the FiLM
  feedback channel, not the recurrence. Documented non-fix; revisit only if
  tests/PPL show it matters.)
- **3e.** `_FlaWrapper.forward`: add `cu_seqlens=None`. When set, reshape
  `(B,T,d)→(1,B*T,d)`, pass `cu_seqlens` (+ cpu copy) into `self.layer`,
  reshape back. Gate with a class attribute `accepts_cu_seqlens = True` on
  `_FlaWrapper` (mirrors the `needs_input_ids` pattern); `Block.forward` only
  passes `cu_seqlens` when the attention advertises it. Pure-PyTorch
  attentions (`SoftmaxAttention`, …) don't receive it — transformer-baseline
  doc isolation is separate scope. **Verify** `Mamba2.forward` accepts
  `cu_seqlens` before enabling the flag for `Mamba2Attention`; if not, set
  `accepts_cu_seqlens = False` there.
- **3f.** Use `.reshape()` not `.view()` (post-FiLM `h` may be
  non-contiguous); `cu_seqlens` must be int32/int64 on `x`'s device.

## Step 4 — WorkingMemory: mask read/write across document boundaries

**File:** `experiments/model.py::WorkingMemory.forward`

- Thread `doc_ids` via `TinyLM._apply_memory` / `_finalize` (add the param to
  both; every `_finalize(...)` call in `forward` passes it).
- `WorkingMemory.forward` gains `doc_ids=None`. When provided:
  - **Write side:** gather each buffer slot's source doc id —
    `buf_doc = torch.gather(doc_ids, 1, top_idx)` → `(B,K)`.
  - **Read side:** add a document-match mask alongside the causal mask —
    `doc_mask = buf_doc.unsqueeze(1) != doc_ids.unsqueeze(-1)` → `(B,T,K)`;
    `scores.masked_fill(doc_mask, -inf)`. Recompute `all_masked` over
    `causal_mask | doc_mask` so rows with no in-document predecessor get a
    zero read (the existing NaN-guard handles the all-`-inf` row).
  - `doc_ids is None` → byte-identical to today.
- `mem_read_mask` is orthogonal (controls *where* injection lands) — composes
  with doc masking, no change.

## Step 5 — Tests (`experiments/test_cu_seqlens.py`, new)

1. `test_doc_ids_emitted` (CPU) — `MixedSourceStream` yields a 3-tuple;
   `doc_ids` shape `(block_size,)`, monotonic non-decreasing, increments after
   each EOS.
2. `test_doc_ids_survive_think_bursts` (CPU) — with `think_burst_prob=1.0`,
   `doc_ids` stays aligned with `inputs`.
3. `test_cu_seqlens_construction` (CPU) — unit-test `_build_cu_seqlens`:
   hand-built `doc_ids` → expected flat `cu_seqlens`; `None` → rectangular.
4. `test_packed_equals_unpacked` (CUDA) — **the headline test.** Two short
   sequences run standalone vs packed-into-one-row-with-doc_ids; logits over
   each document's positions must match (`atol~1e-2`, bf16). Fails on the
   second document without the fix.
5. `test_film_bypass_threads_cu_seqlens` (CUDA) — same equality with
   `feedback_self_k=3`, toggling `_film_bypass`.
6. `test_memory_no_cross_doc_attention` (CPU) — `WorkingMemory` attention onto
   any buffer slot from another document is exactly 0; `doc_ids=None`
   regression guard.
7. `test_forward_signature_backcompat` (CPU) — `model(input_ids)` still works
   and equals `model(input_ids, doc_ids=None)`.

Run the full suite afterwards — protocol/signature changes touch
`test_data_mix.py`, `test_eos_mask.py`, `test_sft_code.py`.

## Step 6 — Protocol-change fallout

`MixedSourceStream` 2-tuple → 3-tuple. The `emit_doc_ids` constructor flag
(Step 1d, default off) keeps the blast radius minimal: only `train_lm.py` opts
in. Still audit `data_mix.py::_smoke`, the train/val unpacks, and grep
`MixedSourceStream` repo-wide.

## Step 7 — FIM (#65) reuses this representation

FIM splits a single document into prefix/middle/suffix and reorders *within
one document* — it adds **no** new `cu_seqlens` boundaries, just permutes
tokens inside an existing `doc_id` segment. After FIM permutes a document's
tokens (+ inserts 3 sentinels), every token still carries the same `doc_id`.
The Step-1c aligned-array mechanism is exactly what FIM needs. **Build the
`doc_id` representation now; no rework expected for FIM.** Had we emitted raw
`cu_seqlens` from the dataset instead, token insertions would desync the
offsets — `doc_id` per-position is robust to insertion and permutation.

## Open risks

1. **`B==1` flatten cost** — post-FiLM `h` non-contiguous → `.reshape()`
   copies. Minor; measure with `profile_train.py`.
2. **`torch.compile` interaction** — `cu_seqlens` is data-dependent in shape
   (doc count varies per batch), may trigger recompilation at the wrapper
   boundary. If it thrashes, build `cu_seqlens` in the trainer (outside the
   compiled region) and pass it in.
3. **bf16 numerics** — packed-vs-unpacked won't be *bit*-identical (different
   chunk tilings); use `atol/rtol ~1e-2`. fp32 path unavailable (kernel
   asserts non-fp32).
4. **Short documents < chunk_size (64)** — `q_len` is the *total* flattened
   `B*T` so it stays on the chunk kernel; per-doc segments <64 are handled by
   `prepare_chunk_indices`. Smoke-test with a deliberately tiny document.
5. **`initial_state` count** — `chunk.py:303` checks
   `initial_state.shape[0] == len(cu_seqlens)-1`. We pass `initial_state=None`,
   so moot — but a future state-carry change must match document count.
6. **FiLM `_shift_right` across boundaries** — deliberate documented non-fix
   (3d); second-order vs the recurrent-state leak.
7. **`SoftmaxAttention` / transformer baseline** gets no doc isolation from
   this change (would need a block-diagonal mask). Out of scope; active arch
   is `deltanet`.
8. **RL replay-packing path** packs rows by `torch.cat`; not covered until
   that path also produces `doc_ids`. Follow-up.

## Critical files

- `experiments/data_mix.py`
- `experiments/model.py`
- `experiments/layers.py`
- `experiments/train_lm.py`
- `experiments/sft_code.py` (the `insert_think_bursts` aligned-array change)
