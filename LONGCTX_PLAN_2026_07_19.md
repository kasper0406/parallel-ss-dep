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
