# Long-context continued-pretrain — pre-registered plan (2026-07-19)

## Why (the wall this removes)

`STATE_CARTRIDGES_PLAN_2026_07_19.md` run 1: sequential 8–32k repo ingestion
makes tasks WORSE (−0.54 span-CE) on `production_lean_soup3` — every pretrain
in this lineage ran T=2048 WITH cross-document state isolation, so the model
has never accumulated more than ~one document of state. The O(1) moat is
mechanical; usable context ≈ 2k. This run buys the missing substrate for:
the cartridges claim (8–32k re-run), the repo-agent north star, and honest
long-context capability behind the cost-moat bench.

## Design

Continued-pretrain of `checkpoints/production_lean_soup3.pt`:
- **T=8192** (4x the trained window), `--no_doc_isolation` (new flag,
  default-off: emit_doc_ids off → no cu_seqlens resets → state flows across
  documents within each window — standard LM packing, the thing this lineage
  never did). Recall streams lose per-doc isolation too; acceptable for this
  run (their weight is 25%, and the point is state-carry).
- Mix: the standard pilot mix (comparability; codeparrot files up to 30k+
  tokens exist and now actually exercise long state).
- **First production use of `--manual_allreduce` (2 GPUs)**: batch 1 × T8192
  × grad_accum 8 per rank = 131,072 tok/step global (matches the production
  campaign). Muon 2e-4/6e-5 (anneal-preserving continuation LR, the attach
  program's validated trunk LR), wsd warmup 50, decay_frac 0.3.
- 5,500 steps ≈ 720M tokens, ~7–8h wall-clock (est. from 27.9k tok/s at
  T2048; T8192 per-token cost slightly higher, 2 GPUs ≈ 1.9x).
- Periodic ckpts every 100M tokens; drift check every 500 steps.

## Decision rule (pre-registered)

Success = the frozen 8–32k cartridge experiment's sanity gate flips:
1. **PRIMARY: lift(sequential) ≥ +0.15 span-CE on the FROZEN
   `data/repo_episodes` eval** (the exact gate that voided cartridges run 1;
   same harness command).
2. Guards: HE-CE ≤ soup3 + 0.01 (0.6714) AND depdist total ≤ soup3 + 0.01
   (0.8226) — long-context must not cost short-context competence.
3. If primary passes AND guards hold → this ckpt becomes
   `production_lean_longctx.pt`, and the ORIGINAL cartridges experiment
   re-runs on it (same frozen bars 0.60/0.75/0.35).
4. If primary fails but guards hold → one registered escalation: 2x steps
   from the same start (state-accumulation may need more exposure). If that
   fails too → the wall is not exposure-fixable at this budget; record and
   re-scope (cartridges-at-working-scale line-CE result stands on its own).
5. If guards trip → kill, keep soup3, record the trade.

## Secondary reads (no veto power)

Line-CE lift on the frozen set; the short-context set re-run (should not
regress); per-source CE at val; VAL trajectory at T8192.

## RUN 1 RESULT (2026-07-19): primary FAIL, guards PASS-with-improvement → registered 2x escalation launched

- Primary (frozen 8–32k set): lift(sequential) span **−0.1330** — big move
  from soup3's −0.5437 (75% of the harm removed by 720M tokens) but short
  of the +0.15 gate. Line-CE lift flipped positive (+0.0127, was −0.1094).
  Structure check improved at every K.
- Guards: HE-CE 0.6675 (≤0.6714 ✓, +0.006 vs soup3); depdist total
  **0.7948 — BETTER than soup3's 0.8126** (long-context packing improves
  natural-code CE outright). Long-ctx training is currently guard-free.
- Manual-allreduce production debut: 5,500 steps, 50.4k tok/s sustained
  (1.9x), zero drift warnings, clean finish. The 2-GPU mode is now
  production-validated.
- **Escalation (per registered rule #4, the ONE permitted): 11,000 steps =
  1.44B tokens from the same soup3 start → `production_lean_longctx2.pt`.**
  Same gate, same guards. If it fails: wall not exposure-fixable at this
  budget; re-scope per the plan.

## ESCALATION RESULT / PROGRAM CLOSED at this rung (2026-07-19)

- Primary: lift(sequential) span **−0.1624** (run 1: −0.1330) — the extra
  720M tokens moved NOTHING. The −0.54 → −0.13 gain saturated. Per rule #4:
  the wall is not exposure-fixable at this budget with T=8192 packing.
- Guards: depdist **0.7778 — best natural-code CE ever measured here**
  (soup3 0.8126, longctx1 0.7948); HE-CE 0.6728 = photo-finish MISS of the
  ≤0.6714 bar (+0.0014, inside seed σ≈0.007; formal miss recorded).
- **Mechanistic read of the saturation:** training window T=8192 can only
  teach state accumulation UP TO 8k — the frozen eval's episodes are 8–32k,
  so 75%-harm-removal-then-saturation is exactly "fixed within the window,
  can't extrapolate past it". Revisit-suspect: a T=32k rung (new
  registration required; memory feasibility unverified).
- Base status: `production_lean_soup3.pt` REMAINS the official base
  (registration-clean). `production_lean_longctx.pt` (0.6675/0.7948, both
  guards passed at its rung, 75% less long-context harm) is the
  long-context-preferred variant — promotion is a strategy call, not a
  registered outcome; flagged to the user.
- **Cartridges re-scope (what survives, all measured):** (a) working-scale
  line-CE retention 0.99/0.82 — merging is essentially FREE; (b) merged
  cartridges BEAT one sequential state on span at working scale; (c) at
  8–32k cartridges hurt 5–7x less than sequential; (d) structure-carrying
  at every scale tested; (e) the quantified trained-context wall itself
  (state-accumulation OOD saturates under packing). Honest paper framing:
  "delta-rule states compose in parallel; sequential state accumulation,
  not merging, is the bottleneck."
