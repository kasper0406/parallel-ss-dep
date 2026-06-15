# Principle: key separability (cosine addressing)

## Summary
**Cosine/softmax content addressing only works when the keys are separable.** When the things to be addressed have near-identical representations (e.g. synthetic `v7 = 4530`, `v12 = 8821` binding lines), cosine-over-hidden cannot tell them apart and the read collapses to **recency**. This is the mechanistic reason [[working-memory]] failed on multibind recall while it won on MQAR (which uses distinct random keys). The fix is not a better similarity function — it's making the keys **orthogonal/exact** by addressing on a discriminable channel. Source: `project_recall_discrete_key_direction.md`, `project_wm_addressing_root_cause.md`.

## The diagnosis
- WM read = `softmax(cosine(W_q·h_query, W_k·h_buf))`. On multibind, the value-bearing hiddens are near-identical → the soft read can't separate them → it puts mass on the most-recent slots (recency), 0 % on the queried binding.
- **MQAR won because its keys were DISTINCT RANDOM** (cosine-separable). The multibind keys are near-identical binding lines.
- Every cosine-on-hidden fix failed: frozen-trunk addressing FT, joint-trunk co-train, even explicit attention-placement supervision (`p_bind` stuck at chance ~0.01). The trunk representation itself is dominated by recency/positional structure; a thin linear `W_q/W_k` cannot isolate variable-identity against it.

## Tokenization fact (why naive re-keying is fragile)
There is **no single "variable-name token"**: `v7 → ['v','7']`, `v12 → ['v','1','2']`; the `'v'` is shared, the **digits are the only discriminator**, and 4-digit values are 4 tokens. So a single-token re-key collides on `'v'`. You need the identifier's digit **span** (or a hash).

## The fix (validated)
- **Address on token-identity, not the deep hidden.** Keying the read on the variable-name **input-embedding window** makes addressing perfect: top1 = 1.00, mass 39–65× chance, up to N=96 — while trained cosine-on-hidden sits at chance. (`wm_namekey_probe.py`)
- Equivalently, a **discrete/orthogonal-key store** = a delta-rule layer fed one-hot/VQ keys: `S = Σ onehot(c)·vᵀ`, `read = S·onehot(c_q)` → zero cross-talk, reuses the FLA `chunk_delta_rule` kernel, O(T·d). Separable by construction.
- Then a **copy/pointer readout** for multi-token values (a single additive read decodes only ~1 token). See [[working-memory]].

## Reusable lesson
When designing any content-addressable retrieval: **ask whether the keys are separable in the representation you're addressing on.** If the discriminating signal is token-identity (names, symbols), address on the input embedding / a discrete code over the identifier span — not on the deep hidden, which is recency-dominated. Real code identifiers (`CACHE_SIZE`) are naturally separable; that's why the recurrence already recalls them — and why WM's real headroom is the *non-separable saturating* regime, which code rarely has ([[working-memory-recall-saga]]).

## Related
[[working-memory]] · [[working-memory-recall-saga]] · [[route-around-principle]] · #principle #recall
