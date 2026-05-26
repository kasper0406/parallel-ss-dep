# Phase B — Thinking-specialized Adapter

Implementation of `THINKING_PLAN.md` Phase B: a `ThinkAdapter` module
giving the trunk dedicated parameters that only fire at think positions.

## What was built

### New module `ThinkAdapter` in `experiments/model.py`

A 2-layer MLP `d_model -> hidden_mult*d_model -> d_model` (GELU activation)
plus a learnable scalar `alpha` (init 0). Applied as:

    h_out = h_in + alpha * think_mask * ThinkAdapter(h_in)

- `fc1`, `fc2` use PyTorch default Linear init (non-zero) so the
  gradient on `alpha` is non-zero from step 1 (the FiLM-alpha lesson:
  zero alpha + zero W => zero gradient on alpha).
- `alpha` init = 0 -> cold-start contribution is 0 -> byte-identical to
  no-adapter baseline. Mirrors the FiLM-alpha / PKM-alpha pattern.

### Wiring into `Block` (model.py)

- `Block.__init__` accepts `use_think_adapter`, `think_adapter_hidden_mult`.
- When on, instantiates `self.think_adapter`.
- Applied AFTER the standard `attn + MLP` residual updates, gated by
  `think_mask` (silently skipped when caller passes `think_mask=None`,
  e.g. plain LM eval with no thinking token).

### Wiring into `TinyLM` (model.py)

- New `__init__` kwargs `use_think_adapter`, `think_adapter_hidden_mult`.
- Stored on `self` so the forward pass knows to build the per-position
  `think_mask` even when Phase 2 `state_readonly_at_think` is OFF.
- `think_mask` build site updated: built whenever EITHER Phase-2 OR
  Phase-B is enabled.

### Wiring into `model_builder.py`

- `build_model_from_args` forwards `use_think_adapter` and
  `think_adapter_hidden_mult` from the CLI args to `TinyLM(...)`.

### Wiring into `eval_bracket_structure.py::build_model_from_ckpt`

- Auto-detects adapter weights via `".think_adapter." in k for k in sd_keys`.
- Infers `think_adapter_hidden_mult` from any block's
  `fc1.weight.shape[0] // d_model`.
- Default OFF for back-compat (older ckpts have no adapter keys).

### CLI flags in `experiments/train_lm_args.py`

- `--use_think_adapter` (action="store_true", default False)
- `--think_adapter_hidden_mult INT` (default 2)

### Optimizer routing in `experiments/optim_utils.py`

- New predicate `_is_think_adapter(name)`: matches names containing
  `.think_adapter.` (covers fc1, fc2, alpha across all blocks).
- In the Muon-mode branch, `_is_think_adapter` short-circuits BEFORE the
  ndim-based Muon routing — so the 2D fc weights go to AdamW too, not
  Muon. The 1D alpha would already go to AdamW (ndim != 2), but the
  predicate keeps it routed consistently with the rest of the adapter.
- In the pure-adamw mode branch, no special routing is needed (all
  params route to AdamW anyway) — the test still verifies they appear.

### Tests `experiments/test_think_adapter.py` (10 tests, all pass)

1. `test_default_off_no_params` — `use_think_adapter=False` creates no
   `think_adapter` submodule and no `.think_adapter.` params.
2. `test_on_creates_expected_params` — 5 params per block
   (fc1.w, fc1.b, fc2.w, fc2.b, alpha), correct shapes, alpha init 0.
3. `test_default_off_forward_byte_identical` (CUDA) — same-seed,
   adapter-off models produce identical outputs.
4. `test_alpha_zero_byte_identical_to_no_adapter` (CUDA) — adapter ON,
   alpha=0, non-adapter weights synced -> output byte-identical to
   adapter-OFF. The load-bearing invariant for existing ckpts.
5. `test_nonzero_alpha_changes_output` (CUDA) — alpha=0.5 -> forward
   output DIFFERS from no-adapter.
6. `test_state_dict_round_trip_preserves_output` (CUDA) — save +
   load reproduces adapter weights AND forward output exactly.
7. `test_state_dict_contains_adapter_keys` — expected keys present.
8. `test_is_think_adapter_predicate` — predicate correctness.
9. `test_adapter_params_routed_to_adamw_with_muon` — under
   `build_optimizer(optimizer='muon', ...)` every adapter param lands in
   AdamW, none in Muon.
10. `test_adapter_params_routed_to_adamw_with_pure_adamw` — under
    `optimizer='adamw'` every adapter param lands in the AdamW optimizer.

Run: `PYTHONPATH=. .venv/bin/python -m pytest experiments/test_think_adapter.py -v`

Result: **10 passed, 0 failed.**

## Byte-identical-when-off verification

Two independent paths verify the invariant:

1. **Default-off path**: `test_default_off_forward_byte_identical` — same
   seed, two builds with `use_think_adapter=False` -> identical outputs.
   This is enforced by `not hasattr(blk, 'think_adapter')` and the early
   exit in `Block.forward` (no adapter call attempted).

2. **Adapter-on, alpha=0 path**: `test_alpha_zero_byte_identical_to_no_adapter`
   — build adapter-ON, copy all non-adapter weights from an adapter-OFF
   reference, run both on identical inputs containing think tokens at
   positions 5 and 6. Output diff = 0 exactly (`atol=0, rtol=0`). Because
   alpha=0, `alpha * mask * adapter_out = 0` at every position; the
   adapter's fc1/fc2 still run but contribute exactly zero to the
   residual stream.

The second path is the load-bearing one for production: an existing v4
/ v7.1 / Phase-C ckpt loaded into a Phase-B-enabled `TinyLM` with
`strict=False` will retain its EXACT pre-Phase-B behaviour until alpha
moves off zero during training.

## Optimizer routing test result

Both `test_adapter_params_routed_to_adamw_with_muon` and
`test_adapter_params_routed_to_adamw_with_pure_adamw` pass:

- Under `optimizer='muon'`: the Muon optimizer's param_groups contain
  **zero** adapter params; the AdamW optimizer's param_groups contain
  **every** adapter param (5 per block: fc1.w, fc1.b, fc2.w, fc2.b,
  alpha).
- Under `optimizer='adamw'`: every adapter param is in the single
  AdamW's param_groups.

The routing is implemented by `_is_think_adapter(name)` short-circuiting
before the ndim-based Muon routing in `build_optimizer`. Adapter weights
land in the `regular` AdamW group (with `weight_decay=wd`), not the
`alpha` group (which has `weight_decay=alpha_wd=0`). This is a
deliberate choice — Phase B's `alpha` is one scalar per block; the
`alpha`-group split exists for the FiLM α's where there can be 30+ of
them and weight-decay equilibrium matters. For the few Phase-B alphas
(one per block), the default `wd=0.01` is fine; if a future experiment
wants no-WD on the Phase-B alpha specifically, the predicate is the
hook point.

## Regression check on existing tests

- `experiments/test_think_index_emb.py` — **8/8 pass** (Phase 3, unchanged).
- `experiments/test_state_readonly_thinking.py` — **4/4 pass** (Phase 2,
  unchanged; my `think_mask`-build change preserves the original
  `state_readonly_at_think` branch.).
- `experiments/test_activation_checkpointing.py`,
  `experiments/test_memory_layer.py`, `experiments/test_cu_seqlens.py`
  — all pass.
- `experiments/test_pretrain_knobs.py` — 4 pre-existing failures
  (`_nonthink_forward_loss` signature drift in the
  `thinking-token-gate-curriculum` branch; unrelated to Phase B).

## Open questions / follow-ups

1. **Adapter at K=3 self-feed cost**: the adapter runs once per Block
   forward; under FiLM K=3 self-feeding the Block runs 3x, so the
   adapter compute multiplies. For the 10L x 896d trunk with
   `hidden_mult=2`, that's 10 * 2 * 896^2 ≈ 16M params * 3 ≈ 48 MFLOPs
   extra per token per K=3 step. Negligible relative to the 287 M trunk
   but worth noting.
2. **Per-block alpha vs global alpha**: current design has one alpha per
   block. A simpler ablation would share one global alpha across all
   blocks; that's a small CLI change if Phase B's diagnostics show
   alpha converges to similar values across depth.
3. **`forward_step` (per-token decode) path**: like state-readonly-at-think,
   the per-token incremental-decode path (`TinyLM.forward_step` and
   `_FlaWrapper.forward_step`) is not yet adapter-aware. The Block's
   forward IS called from forward_step, but `forward_step` does not
   currently thread `think_mask`. **Documented non-fix** — same as the
   Phase-2 note in CLAUDE.md. For the production loss-bearing forward
   pass and for `eval_humaneval.generate` (which uses full-sequence
   forward), the adapter is fully wired.
4. **`feedback_xattn` path**: the xattn path also calls `_block_fwd` which
   threads `think_mask` correctly, so the adapter works there too. The
   sparse-feedback `_sparse_pass_collect_sources` calls `blk(...)`
   directly (not via `_block_fwd`) and DOES pass `think_mask`, so it
   works there too.
5. **Saving `use_think_adapter` to ckpt config**: `train_lm.py` builds
   the ckpt `config` dict; if it doesn't include `use_think_adapter` /
   `think_adapter_hidden_mult` automatically, `build_model_from_ckpt`
   auto-detects from state-dict, so reload works either way. Worth a
   one-line addition to `train_lm.py`'s config dict when wiring this
   into a launcher.

## Files modified

- `experiments/model.py` — `ThinkAdapter` class, `Block` adapter integration,
  `TinyLM.__init__` kwargs + threading, `think_mask` build-site update.
- `experiments/optim_utils.py` — `_is_think_adapter` predicate, Muon-branch
  routing short-circuit.
- `experiments/model_builder.py` — forward `use_think_adapter` /
  `think_adapter_hidden_mult` to `TinyLM`.
- `experiments/train_lm_args.py` — `--use_think_adapter`,
  `--think_adapter_hidden_mult` CLI flags.
- `experiments/eval_bracket_structure.py::build_model_from_ckpt` —
  auto-detect adapter from ckpt state-dict.
- `experiments/test_think_adapter.py` — 10 new tests (all pass).
