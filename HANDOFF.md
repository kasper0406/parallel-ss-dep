# HANDOFF

## 2026-06-20 (capstone — the WHY loop bottomed out: TOKEN-limited, not capacity/mechanism)

Two decisive results closed the "why don't PKM/WM/latent help?" investigation:

1. **Iso-token mechanism ablation (`experiments/ablation_mechanisms_seq.sh`): mechanisms are ~FREE.**
   Lean trunk vs full v18 stack, IDENTICAL v4 code mix + budget + seed, 2000 steps. Per-source
   CE Δ(MECH−LEAN): codeparrot +0.018, magicoder +0.004, wiki −0.005; VAL tied 6.98 vs 7.00.
   ⟹ "mechanisms displace code capacity" REFUTED. v18's HumanEval regression (6 vs Phase-C 13)
   was MIX-DILUTION + token-budget, NOT the mechanisms. (memory: project_why_mechanisms_synthesis.)

2. **Capacity-vs-data reference probe (`/tmp/probe_capacity_ref.py`): TOKEN-limited, not param-limited.**
   HumanEval-solution CE: SmolLM2-135M=0.798, SmolLM2-360M=0.614 vs our Phase-C-SFT=0.969, v18-SFT=1.056.
   A HALF-size well-trained base beats our 287M (despite ours being format-matched SFT) — because
   SmolLM2 saw ~2-4T tokens vs our ~5B (400-800×). The binding "capacity" is the TRAINING BUDGET.
   (memory: project_undertrained_not_undercapacity — the session's headline strategic finding.)

**Net answer to the loop:** the mechanisms each work on their matched probe and are ~free, but the
287M base is severely UNDER-TRAINED (5B vs multi-T tokens), so it lacks the base competence the
mechanisms would amplify, and at a tiny token budget every data category competes (arXiv: +0.83
arXiv-CE for −0.20 code-CE). The dominant lever is TOKENS, not architecture/mechanisms/mix.

**STRATEGIC FORK (awaiting user):** A) continue-pretrain/fine-tune a strong open base (SmolLM2-360M
/ Qwen) on code — practical on 2×5090, inherits multi-T-token competence, but sets aside the custom
DeltaNet arch; B) keep DeltaNet + PKM/WM/latent as a validated-on-matched-probes research testbed,
accept sub-SmolLM2 HumanEval from scratch; C) 100×+ more code tokens, same arch (beyond 2×5090).
**v19 code-upweighted pretrain is built + staged** (`configs/pretrain_mix_v19_codeup.yaml`,
`launch_pretrain_v19_codeup.sh`) but is still ~5B tokens → run only as a mix-lever MEASUREMENT, not
a capability jump. Both GPUs idle. NOTE: GPU0 falls off the bus under sustained load — run single-GPU
on GPU1 (the ablation crashed twice on GPU0; see the mechanism-ablation saga below).

## 2026-06-19 (autonomous /loop — optimizer levers + v18 integrated run + the "why worse" root-cause)

### Optimizer side-quests — all settled & committed (default-off where wired)
- **DeltaNet per-head Newton-Schulz matrix optimizer** (`--matrix_optimizer fused_deltanet_ns`):
  production A/B = free, modest ~3% convergence-speed edge at identical wall-clock, no real
  mem cost. ADOPT as default for the next from-scratch pretrain. (commits 5ea1d5b/5bcdd99/
  aee0845/165b27b; `DELTANET_PRECONDITIONER*.md`; memory `project_deltanet_per_head_ns_preconditioner`.)
- **Batch/B_crit** (`BATCH_LR_ANALYSIS.md`, commit 3aef559): 262k tok/step is 5-9× above B_crit
  (signal-dominated) — keep batch + current LRs (lr_muon 5e-3, lr_adamw 1.4e-3).
- **Embedding optimizer** = DEAD lever (commit 325c936, memory `project_embed_optimizer_negative`):
  tuned embed LR 2x WORSE (-1.76%), rownorm dualizer -12.84%; keep shared-LR AdamW.
- **data_mix dead-source bug** (committed a662e84 earlier): `_filter_min_content_len` only
  checked content/text → magicoder+textbooks = 0 tokens across v10-v17. Fixed (longest-string
  fallback); revived in v18. CAUSALLY confirmed (magicoder CE 2.52→1.52 in-run).

### v18 (the maximally-integrated run) → SFT → HumanEval = a clean NEGATIVE
v18 = data-hygiene fix (revived magicoder+textbooks) + arXiv + day-1 PKM/WM(mem_ctx_namekey)/
latent. Finished clean (4.75B tok, VAL ppl 5.21). SFT'd on the ast-clean corpus (same recipe
as Phase C) → **HumanEval 6/164 thinking-off vs Phase-C-SFT 13/164 (matched same-session
harness).** Monotonic with v12 (8): v18 < v12 < Phase C. NOT proceeding to grader-RL on a
6/164 base. `launch_sft_v18.sh` is the ready launcher.

### WHY v18 is worse — measured, with an honest correction
- **Latent did NOT cause it** (eval was thinking-OFF). Latent only hurts code *if turned on*
  (Δlogp −2, OOD: it's a sequential-depth lever, code is recall); its day-1 co-train competes
  for capacity but is secondary.
- **Root cause = finite-budget MIX DILUTION, not "mechanisms are dead weight"** (per-source CE,
  `/tmp/probe_gen_vs_code.py`, v18 vs Phase C base): magicoder **−0.25 (v18 BETTER — the fix
  worked!)**, but codeparrot **+0.20** and wikipedia **+0.23 WORSE**. v18 spread a SMALLER budget
  (4.75B vs 5.28B) across MORE sources (arXiv/recall/magicoder) + MORE objectives (mechanisms);
  the added source improved, the un-upweighted core Python (codeparrot, what HumanEval resembles)
  regressed → worse headline. HE-solution CE confirms: v18 1.056 vs Phase C 0.969.
- **Capacity accounting:** PKM kill-gate claws back +0.14 code CE on v18 yet net is +0.087 worse
  → the rest of the stack costs more than PKM returns. Each mechanism helps its MATCHED probe
  (PKM→code CE on recall tokens; latent→depth; WM→leak-free recall) but those are orthogonal to
  short-context HumanEval. Full detail: memory `project_why_mechanisms_synthesis` (v18 section).

### The actionable conclusion + next steps (awaiting user go on the multi-hour run)
The code-headline lever is **DATA-MIX + full budget** more than "drop mechanisms":
1. **Code-upweighted, full-Chinchilla pretrain** — keep magicoder, CUT the arXiv/recall dilution,
   RESTORE codeparrot weight, train to 5.3B+. Optionally ± mechanisms (iso-token) to finally
   disentangle mechanism-cost vs mix-dilution vs token-budget. Multi-hour; not launched unattended.
2. **Standardize ONE eval harness** before quoting deltas (HumanEval-164 greedy is noisy — the
   matched-control re-run gave Phase C 13 not the historical 14).
- Uncommitted/local: README.md + docs/architecture.svg edits, BATCH_LR_ANALYSIS status tweak,
  configs/pretrain_mix_v18_arxiv.yaml, launch_sft_v18.sh, launch_pretrain_v18_arxiv_resume.sh.

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
