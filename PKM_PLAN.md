# PKM_PLAN.md — Persistent learned-RAG / Product-Key Memory layer

## Motivation (recap)

Our 218 M dense backbone is undertrained (~5–10 tok/param) and the v3/v4 mix
contains heavy fact-bearing streams (Wikipedia, bigvul, cybernative) that a
small dense model cannot memorize at the parameter density required.
Symptom: per-source CE on **bigvul + cybernative drifts up** across every
run we have measured. The structurally-correct fix is to add a sparse
key/value lookup table — a *memory layer*, in the Lample 2019 / "Memory
Layers at Scale" 2024 sense — so library/algorithm/CVE facts can live in
65 k–1 M slots of dense storage instead of being amortised into the
shared residual stream.

This is **not** RAG over text documents; it is a learned KV table whose
lookups are end-to-end differentiable.

## Architecture (PKMLayer, single-head spec written multi-head-aware)

A drop-in residual module: `h_out = h + PKM(norm(h))`.

Forward, single position (broadcast over B, T):

1. `q = W_q · norm(h)` ∈ ℝ^{n_heads · 2 · k_dim}; reshape to
   `(n_heads, 2, k_dim)`.
2. Per head, split `q = (q1, q2)`, each ∈ ℝ^{k_dim}.
3. **Sub-key matrices** `K1, K2 ∈ ℝ^{n_heads × n_keys × k_dim}` (learnable,
   fp32). Scores: `s1 = q1·K1ᵀ`, `s2 = q2·K2ᵀ`, both ∈ ℝ^{n_keys}.
4. **Score BatchNorm** on `s1, s2` (per-head, BN over the B·T axis →
   n_keys features). The PKM "warmup" trick — without it, only a few
   sub-keys win every lookup and the rest never receive gradient.
5. Top-`k₁` from `s1`, top-`k₂` from `s2` (typically `k₁ = k₂ = top_k`).
6. Outer-sum: `S[i,j] = s1[top1[i]] + s2[top2[j]]` ∈ ℝ^{k₁ × k₂}.
   Final top-`top_k` over the `k₁ · k₂` candidates.
7. Indices into the full table: `idx = top1[i] · n_keys + top2[j]`
   (so the value table behaves as a single `(n_keys², v_dim_per_head)`
   table per head).
8. Softmax the `top_k` scores → attention weights.
9. Value lookup: `V ∈ ℝ^{n_heads × n_keys² × v_dim_per_head}`
   (stored bf16, math fp32). Gather the `top_k` rows.
10. Weighted sum of values → `(n_heads, v_dim_per_head)`.
11. Concat heads → `v_dim`. Optional output projection `W_o → d_model`.

### v0 hyperparameters

| param          | value | notes                                      |
|----------------|-------|--------------------------------------------|
| `n_heads`      | 4     | Lample-style independent value tables/head |
| `n_keys`       | 256   | per sub-key set → 65 536 slots / head      |
| `k_dim`        | 128   | per sub-key half                           |
| `top_k`        | 32    | retrieved entries per query                |
| `v_dim_per_head` | 144 | = `d_model / n_heads` (576/4)              |
| value storage  | bf16  | fp32 math via `.to(float32)` at lookup     |

**Param accounting**:
- Values: `4 · 65 536 · 144` ≈ 37.7 M (bf16: ~75 MB).
- Sub-keys: `4 · 2 · 256 · 128` = 262 k (fp32: ~1 MB).
- Query proj: `576 · (4·2·128)` = 590 k.
- Output proj: `576 · 576` = 332 k.
- **Total ≈ 38.9 M** params, ~78 MB persistent at bf16 storage.

## Integration into `TinyLM`

PKM is a residual side-table added after one mid-depth block. Concretely:

- Helper `TinyLM._maybe_pkm(self, h, L)` → returns `h + self.pkm_layer(h)`
  iff `L == self.pkm_after_layer and self.pkm_layer is not None`,
  else `h`.
- Call `_maybe_pkm(h, L)` after every `blk(...)` / `_block_fwd(...)` in:
  - `_block_fwd` — loss-bearing block forward (the obvious one).
  - `_sparse_pass_collect_sources` — K-self-feed `no_grad` warmup passes
    (passes 1 and 2 of K=3). **Critical**: if PKM is only in the
    loss-bearing pass, the FiLM source-state collected in passes 1/2
    differs from what the deployed model sees, breaking K=3
    self-consistency.
  - The `xattn_pairs` pass-1 vanilla forward (defensive — we don't use
    xattn but the contract is the same).
- Insertion point: **after layer 14** (mid of 30-block model). FiLM
  source layer is 28; 14 is comfortably between target (2) and source
  (28), so the PKM contribution participates in the FiLM signal without
  competing with either endpoint.

### Activation-checkpointing & cu_seqlens

- PKM has no per-position recurrence (it's a pointwise lookup), so it
  composes trivially with `cu_seqlens` cross-document isolation.
- Wrapping the PKM call in the existing `_ckpt_run_block` path adds it
  to the checkpointed graph; no separate handling needed. The lookup is
  cheap (one Linear + one gather) so checkpointing it is essentially free.

### Optimizer routing (`optim_utils.py`)

- `pkm_layer.values.weight` (the big embedding-like value table) →
  **AdamW** (added to `_EMBED_OR_HEAD_NAMES`-equivalent check).
  Muon's Newton-Schulz over a 65 k × 144 matrix is wasted compute; this
  table is exactly the same shape pattern as `embed.weight`.
- Sub-keys (`pkm_layer.subkeys` — a 4-D parameter `(n_heads, 2, n_keys, k_dim)`)
  → **AdamW** (Muon is 2-D only; current rule already routes ≥3-D to AdamW).
- Query/output projections (2-D Linears) → **Muon** (default).

### bf16 optimizer state

- Values are large (37.7 M); their AdamW exp_avg / exp_avg_sq would be
  another 75 + 75 = 150 MB at fp32. With `--bf16_optim_state` (just
  shipped in commit `60c9d1f`), that drops to ~75 MB. Default v5-pkm
  config will enable this.

## CLI / wiring

New flags in `train_lm_args.py`:

```
--use_pkm                  # bool, default False
--pkm_after_layer N        # int, default 14
--pkm_n_keys N             # int, default 256
--pkm_n_heads N            # int, default 4
--pkm_k_dim N              # int, default 128
--pkm_top_k N              # int, default 32
--pkm_value_bf16           # bool, default True (storage; math stays fp32)
```

`model_builder.build_model_from_args` threads these into `TinyLM(...)`.

## Tests (`experiments/test_memory_layer.py`)

1. **Shape**: `PKMLayer(d_model=64)(x_BTd)` returns `(B, T, d_model)`.
2. **Top-k correctness**: PKM-lookup top-k matches naive full
   `(n_keys²)` argsort to within float tolerance, on a small config
   (n_keys=8, top_k=4, n_heads=2).
3. **Gradient flow**:
   - Top-k values and sub-keys receive gradient.
   - Non-top-k values **do not** receive gradient (sparse-update sanity).
4. **Cold-start spread (BatchNorm)**: with BN enabled, after 5 steps on
   random inputs the *number of distinct value rows that ever made
   top-k* is ≥ 80 % of the table. With BN disabled the same test
   collapses to ≤ 10 %. (This is the warmup-stability assertion.)
5. **Determinism**: two forwards on the same input give bit-identical
   output in eval mode (BN-in-eval uses running stats).

Run:
```
PYTHONPATH=. .venv/bin/python -m pytest experiments/test_memory_layer.py -v
```

## Smoke test (before launching the full v5-pkm run)

Goal: catch OOM, FLA-kernel incompatibility, or compile graph-break
explosions in <10 minutes.

- Builds the v5-pkm config (full mix, full T, batch 12, all v4 knobs).
- 100 training steps, no checkpoint save, no mid-eval.
- Asserts: forward+backward complete; final `loss.item() < initial`;
  GPU peak memory ≤ 28 GB (4 GB headroom on the 32 GB card).

## v5-pkm small-scale pretrain config

Mirrors v4 except:

- `--use_pkm --pkm_after_layer 14` (defaults for the rest of the PKM
  flags).
- `--bf16_optim_state` (banks the recently-shipped saving).
- `--batch 12` instead of 20 (PKM adds ~2 GB activations + ~78 MB params;
  drop batch to keep ≥ 4 GB headroom; smoke test will confirm or revise).
- `--steps 4500 --grad_accum 8` → ~4500 · 12 · 2048 · 8 ≈ **883 M tokens**
  (about a fifth of v4's 4.26 B; "small scale" as requested).
- `--save_ckpt checkpoints/pretrain_mix_v5_pkm.pt`.
- `--tb_dir runs/tb/pretrain_mix_v5_pkm`.
- Log to `runs/pretrain_mix_v5_pkm.log`.
- `CUDA_VISIBLE_DEVICES=1` (v4 owns GPU 0).

## Success / failure criteria

**Primary (the existing pathology):**
- Per-source CE on `bigvul + cybernative` *stops drifting up* between
  the 250 M → 500 M → 800 M token marks (read off `diag_ckpt.py`'s
  per-source breakdown). v4 and earlier: monotonic up. v5-pkm target:
  flat or down.

**Secondary (don't break the rest):**
- Overall CE within 2 % of an extrapolation of v4's curve at matched
  tokens. Ideal: better. Acceptable: equal. Red flag: significantly
  worse → PKM is crutching.
- Throughput: ≥ 80 % of v4's tok/s (PKM lookup should be cheap).

**Hard fail conditions:**
- Cold-start collapse: <10 % of value table ever active after 1k steps.
- OOM during forward+backward.
- NaN / Inf in PKM scores or values.

## Sequencing

- This run is on GPU 1; v4 keeps GPU 0. They do not contend.
- v5-pkm is a small-scale probe. If primary criterion passes, scale to
  1 M-slot table (` --pkm_n_keys 1024`) for the full v5 run.
- If primary criterion fails, write up the negative result and move on.

## Open questions (defer to v5+ if probe succeeds)

- Multiple PKM layers (e.g. one early, one late). Lample showed 1 was
  sufficient at small scale; "Memory Layers at Scale" used 2-3 at
  larger scales.
- Memory-only fine-tune phase (freeze backbone, train PKM only on a
  fact-heavy mini-corpus) — analogous to LoRA but for facts.
- Value-table sharing across layers vs separate.
