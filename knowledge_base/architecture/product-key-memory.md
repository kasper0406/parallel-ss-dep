# Product-Key Memory (PKM)

## Summary
A **multi-head learned KV side-table** dropped in after one block: per query, top-k retrieve from `n_heads × n_keys²` slots via a factorised sub-key product (default 4 heads × 256² = **262 k effective slots**). It is the project's **one side mechanism that works out-of-the-box** (load-bearing on HumanEval, −5 in ablation) — because facts reduce next-token CE everywhere in text, so the LM loss automatically creates PKM's bottleneck. See [[pkm-verdict]] for the full verdict, [[objective-function-alignment]] for *why* it works where [[working-memory]] doesn't. Source: `experiments/memory_layer.py::PKMLayer` (line 37), `CLAUDE.md`.

## Why it nearly died, and the fix
The v5-pkm probe (2026-05-17) found **97 % of value rows still at random init** after 2.13 B tokens — PKM was structurally inert; the "v5 wins" were trunk lifts. The **v7.1 bootstrap-fix package** turned it into "always-positive per-source contribution". The five interlocking fixes:
1. **`--pkm_use_output_gate`** — scalar α (init 0) wrapping output; gradient grows α only as PKM proves useful (FiLM-α pattern).
2. **`--pkm_alpha_floor_start 0.3` / warmup 2000** — sign-preserving additive α-floor; forces minimum contribution during the value-table bootstrap window. Without it, α grew to +0.085 then **collapsed back** (v7.0 trace) because random-init values look like noise.
3. **`--pkm_epsilon_start 0.5`** — ε-greedy random slot replacement so *every* slot gets gradient (v5: only ~4 % of slots ever fired; v7.1 hits 65 k/65 k).
4. **`--pkm_value_init_std 1.0`** — init value rows at residual-stream magnitude (not `1/√d ≈ 0.04`).
5. **`--pkm_score_norm layer`** + **`--pkm_diversity_weight 0.01`** — LayerNorm on scores (vs noisy BatchNorm), entropy penalty against peaky retrieval.
- **`--pkm_value_lr_mult 100.0`** — separate AdamW group at 100× LR; compensates the ~10⁴× multiplicative dampening of per-row gradient.

## Diagnostics
- Live log per step: `pkm(αL=+0.32, αeff=+0.34, row=1.79, slots/H=33k/65k, top=0.003, ε=0.00, φ=0.00)`. **`row`** (mean value-row norm vs init; >1 = learning) is the real health signal; **`v_std` is misleading** (invariant under distribution-preserving updates).
- Healthy bootstrap: αL commits +0.27 by floor-end, self-sustains to +0.346 / row 2.54 (v7.1). The **failure path** (v7.0/v11/v13): αL never commits during the floor window → values freeze → slots collapse to ~2 %. See [[pretrain-run-history]].
- Probes: `probe_v5_pkm_utilization.py` (row drift, slot concentration, residual contribution), `probe_pkm_capacity.py` (capacity vs exposure).

## What PKM is bounded by
Not capacity (a 76 k-param model memorizes ~10 k facts; 287 M ≈ tens of millions) but **under-exposure** (a fact needs ~100+ gradient exposures to lock in; rare algorithms get ~22) and **catastrophic forgetting** (PKM is read at every token via shared addressing, so it forgets like dense). See [[thinking-on-code-verdict]].

## Related
[[pkm-verdict]] · [[objective-function-alignment]] · [[working-memory]] · #architecture
