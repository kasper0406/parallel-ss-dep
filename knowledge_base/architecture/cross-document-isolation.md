# Cross-document state isolation

## Summary
`data_mix.py` packs multiple documents into each T=2048 sequence separated only by an EOS *token*. Because DeltaNet is a linear RNN, its recurrent state (and [[working-memory]] reads) would flow straight across those boundaries unless stopped. The fix: emit a per-position `doc_id` array, turn it into a ragged `cu_seqlens` that FLA's `chunk_delta_rule` respects, and a same-document read mask in WM. Source: `CLAUDE.md`, `experiments/data_mix.py`, `model.py::_build_cu_seqlens`, `CROSS_DOC_ISOLATION_PLAN.md`.

## Mechanism
- `MixedSourceStream(emit_doc_ids=True)` emits a per-position `doc_id` (kept aligned through `insert_think_bursts`).
- `train_lm.py` threads `doc_ids` into `TinyLM.forward`; `_build_cu_seqlens` builds an int32 ragged `cu_seqlens`; `_FlaWrapper` feeds it to `chunk_delta_rule` after flattening `(B,T,d)→(1,B*T,d)`.
- `WorkingMemory.forward` gained a same-document read mask.
- `doc_ids=None` (any non-data_mix stream, eval, MQAR) is **byte-identical** to the old path — only `DeltaNetAttention` opts in (`accepts_cu_seqlens=True`).
- The same `doc_id` representation is reused for FIM augmentation.

## Open follow-ups
- The RL replay-packing path still passes `doc_ids=None`.
- The FiLM `_shift_right` lag is not doc-aware (documented second-order non-fix).

## Related
[[deltanet-backbone]] · [[working-memory]] · #architecture
