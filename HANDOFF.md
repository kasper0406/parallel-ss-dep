# HANDOFF

## 2026-06-16 (overnight autonomous session — "why don't PKM/WM/latent help?" + Track A)

### What was settled (the "why" question — answered for all three)
- **WM = drop it for the code model.** Recurrence absorbs trained recall; WM trains
  itself off (copy_g→0, read_alpha→0.028, code-recall off=1.000 measured). Realistic
  recall data tops at N≤24, below WM's validated N≥48 niche, so off=1.0 is expected.
  One untested niche: a genuine N≥48 long-horizon agentic eval (task #83). Full saga:
  `knowledge_base/verdicts/working-memory-recall-saga.md`.
- **PKM = bounded sample-efficiency lever, NOT a capacity win** (adversarial review
  corrected an over-claim). Under-exposure not capacity; at matched params ≈ dense;
  doesn't escape catastrophic forgetting. +0.09 CE is an αL-toggle, not iso-param.
- **Latent thinking = real but narrow** (deep-sequential subtasks, day-1 only); the
  v13 "+0.45–0.60" was vs a weak control; flat on per-token code.
- **Unifying answer** (`memory project_why_mechanisms_synthesis`): an add-on helps only
  where (a) the recurrence fails AND (b) the add-on has a learnable mechanism there;
  AND on real code the dominant bottleneck is **base knowledge/data**, orthogonal to
  all three. The headline moves via data + scale, not mechanisms.

### Track A (v12 → SFT → RL → HumanEval) — ran to a decisive, sobering result
v12 pretrain finished clean (5B tok, VAL ppl 4.94). Then a SAME-CONFIG HumanEval sweep
(`--prompt_style sft_comment --extract_code_block --min_emit_before_eos 30 --max_gen 512`):

| ckpt | pass@1 |
|---|---|
| v12 SFT | 8/164 |
| **Phase C SFT (dirty)** | **14/164** |
| Phase C SFT (drop-broken corpus) | 9/164 |
| RL v11c | 8/164 |
| RL v2_step300 (the documented "16") | 13 off / 14 thinking-on |

- **Robust best = 14/164** (Phase C SFT no-think == RL with-think). The documented
  **"16/164" does not reproduce** same-config; **SFT→RL gain ≈ 0**; thinking ±1 (noise).
  The 0→10→14→16 "trajectory" conflated eval-config changes with capability.
  → memory `project_humaneval_config_artifact`.
- **v12 is a worse CODE base** (8 vs 14): its recall-heavy WM-experiment pretrain mix
  cost code capability. v12 arm is dead for the headline.
- **Naive data hygiene backfires**: dropping the 4,279 "verified-broken" distill rows
  HURT (14→9) — exec_error rows are mostly-valid code + MBPP coverage.

### DECISION NEEDED FROM YOU
Every cheap lever is now exhausted/negative — including **data pruning, which I
tested two ways and is DEAD**: drop-broken (tier-based) = 9/164 (hurt, removed valid
exec_error code); AST-clean (drop only won't-parse/no-def garbage, keep valid) = 13/164
(flat). The 54k distill is the only corpus on disk, so more volume needs a teacher.
To beat the robust ~14/164, pick a real investment:
1. **Data-regeneration build (ADDITIVE, the live lever — task #84)**: the SFT corpus is
   91.8% unverified + ~2× exposure/problem (vs ~100+ needed). Lever = generate MORE
   verified (problem, correct-code, tests) data + concentrate exposure on failing
   families. Needs a teacher (vLLM/Qwen) — a deliberate multi-hour build; I held off
   launching it unattended. Say go (+ teacher/scope) and I'll smoke-then-scale it.
2. **Scale** (bigger model / more code-focused tokens) — the only thing that buys
   composition at this size class.
3. **Better eval harness first** — HumanEval-164 greedy is too noisy as the dev signal
   (the whole "16" mirage came from this); standardize one harness (temp pass@k /
   HumanEval+ / MBPP+) before chasing any delta. Cheap; I can do this autonomously if
   you want a trustworthy bar first.

### Housekeeping
- **v14** still training on GPU0 (redundant WM run; its verdict is long-since answered).
  Safe to kill to reclaim GPU0 — left running pending your nod.
- All findings in `memory/` + `knowledge_base/`. Open tasks: #83 (N≥48 WM probe),
  #84 (data-regen build spec), #52 (re-measure latent results with the fixed harness).
- Ready-to-run: `launch_sft_v12.sh`, `launch_sft_phasec_clean.sh` (+ the clean distill
  `data/distill_v7_phase1_clean.jsonl`).
