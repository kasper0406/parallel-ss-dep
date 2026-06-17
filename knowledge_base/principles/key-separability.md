# Principle: key separability (cosine addressing)

## Summary
**Cosine/softmax content addressing only works when the keys are separable.** When the things to be addressed have near-identical representations (e.g. synthetic `v7 = 4530`, `v12 = 8821` binding lines), cosine-over-hidden cannot tell them apart and the read collapses to **recency**. This is the mechanistic reason [[working-memory]] failed on multibind recall while it won on MQAR (which uses distinct random keys). The fix is to address on a **separable channel** (the identifier's token-identity / name-span), **not** the deep hidden. Source: `project_recall_discrete_key_direction.md`, `project_wm_addressing_root_cause.md`.

## CRITICAL REFINEMENT (2026-06-16) — separable CHANNEL is the fix, discreteness is NOT (and costs robustness)
The earlier conclusion "the fix is to make keys orthogonal/exact (discrete hash/VQ)" was **half right and half wrong**, corrected by a clean 3-arm head-to-head (`wm_vqkey_probe.py`, commit 72ec39a, leak-free): HASH (discrete exact) vs VQ (discrete learned) vs SOFT (continuous learned), identical copy readout, frozen trunk.
- **What was right:** you must address on a **separable channel** — the name-SPAN token-identity (not the deep hidden, not a contaminated fixed window that pools name+arrow+value).
- **What was WRONG: discreteness was never the lever — and it's actively harmful.** Given a separable name-span key, a **continuous SOFT (cosine/softmax) read wins on BOTH axes**: separability (top1 1.00 @N=128, matching the hash) AND surface-variant robustness (case 0.65–0.81, camel 1.00). The discrete arms LOSE: the HASH is spelling-locked (variant recall ~0), and VQ loses on *both* — argmax collapses near-but-distinct entities (41% collision → separability lost) AND breaks at cell boundaries (variant recall ~0.01–0.37).
- **⟹ The "soft attention diffuses to recency" WM failure was NEVER softmax** — it was the non-separable KEY. Softmax over a *separable name-span key* holds perfectly. Discretizing that key (hash/VQ) throws away the near-miss tolerance that buys variant-robustness.
- **Honest scope:** probe-scale (3L×256d from scratch, synthetic lexically-distinct words, N≤128); the encoder is trained directly on recall. Real-model transfer (does a name-span soft key TRAIN on the 287M, where the contaminated-window soft key historically failed?) is the open test before replacing the deployed hash.

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
When designing any content-addressable retrieval: **ask whether the keys are separable in the representation you're addressing on.** If the discriminating signal is token-identity (names, symbols), address on the **name-span input embedding** — not on the deep hidden (recency-dominated) and not on a contaminated fixed window (washes out the discriminator). Once the channel is separable, a **continuous soft read is preferred over a discrete code** (it adds surface-variant tolerance for free; discreteness only re-introduces brittleness — see the 2026-06-16 refinement above). Discrete codes (hash) remain useful only as a training-free, zero-cross-talk *exact*-match fallback. Real code identifiers (`CACHE_SIZE`) are naturally separable; that's why the recurrence already recalls them — and why WM's real headroom is the *non-separable saturating* regime, which code rarely has ([[working-memory-recall-saga]]).

## FOLLOW-UP (2026-06-17) — the trunk DOES linearly encode name→binding (cosine under-exploits it); reference-resolution is a separate, deeper wall
A fair contextual-addressing probe (alias_addressing_probe.py, capacity-matched + attention-supervised + whitening/learned-linear probes, frozen v12) separates two questions the earlier "cosine fails" lumped together:
- **EXACT name→binding identity IS linearly recoverable from the trunk's NAME-SPAN contextual hidden** (learned-linear/PCA holder-id 0.87–0.94 heldout, vs ~0.16–0.26 recurrence). So "cosine-over-hidden failed" was partly that **cosine under-exploits a separable-but-not-cosine-aligned signal** (dominant shared "value-register" component buries it; whitening/a learned linear key recovers it) AND that the prior probes used the wrong anchor (value-start register carries value, not name). ⟹ a *trained* semantic name-key is feasible and more general than a surface hash.
- **REFERENCE RESOLUTION (alias `C→B→A→value`) is NOT recoverable at any key (learned-linear + PCA-whiten all at chance) on the FROZEN trunk** — a deeper, separate wall: the linear-RNN doesn't FOLLOW the multi-hop reference chain in one forward (~2-hop budget), so the resolved value never reaches the query position in ANY recoverable form. This is a TRUNK-depth limitation, not an addressing one. Open control: trunk-LoRA/co-train (can a trainable trunk learn reference-following?). Connects to [[latent-thinking-verdict]] (depth) × WM (retrieval) cooperation.

## Related
[[working-memory]] · [[working-memory-recall-saga]] · [[route-around-principle]] · [[latent-thinking-verdict]] · #principle #recall
