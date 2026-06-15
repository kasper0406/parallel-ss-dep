# DeltaNet backbone

## Summary
The backbone is **DeltaNet** (`--arch deltanet`, plain `chunk_delta_rule` from the local FLA fork) — an efficient linear-RNN with a **bounded recurrent state** (no O(T²) attention, no growing KV cache). The bounded state is the whole reason the side mechanisms exist: it can't hold everything, so [[product-key-memory]] adds parametric storage, [[working-memory]] adds a buffer for when the state saturates, and [[latent-thinking]] adds sequential depth. Source: `CLAUDE.md`, `experiments/model.py` (`DeltaNetAttention`, `_FlaWrapper`).

## Why this backbone
- **No KV-cache cost at inference** — RNN inference state is ~74× smaller than a matched Transformer's KV cache (README headline). Critical under the [[hardware-and-compute]] memory budget.
- The bounded state is also the project's central tension: a linear-RNN compresses the past into a fixed-size state, so **fine-grained recall saturates** once there are more items than capacity — this is exactly the regime where [[working-memory]] wins (+11 pp MQAR) and below it WM is redundant.

## Key facts / settings
- **`gated_deltanet` is broken on sm_120** (Blackwell) — Triton autotuner "misaligned address" in the backward. Use plain `deltanet`. See [[hardware-and-compute]].
- **`--state_readonly_at_think`** forces the per-token write-gate β→0 at think positions (forward-hook on the inner FLA `b_proj`, pre-sigmoid logits clamped to −1e4). Think tokens READ the recurrent state but never WRITE it → preserves long-range bindings across multi-think bursts. This is the architectural guarantee that makes [[latent-thinking]] and [[pareto-safe-thinking]] safe for recall. Only the full-sequence forward path is wired; the per-token decode path leaves β unmasked (open follow-up).
- Recurrence is **content-addressable associative memory by construction** (`read = S·q`); this is why it already solves much recall (see [[working-memory-recall-saga]]) and why [[key-separability]] matters for any side store.
- Cross-document state would leak across packed docs without [[cross-document-isolation]] (`cu_seqlens`).

## Related
[[film-feedback]] · [[shallow-wide-trunk]] · [[working-memory]] · [[key-separability]] · #architecture
