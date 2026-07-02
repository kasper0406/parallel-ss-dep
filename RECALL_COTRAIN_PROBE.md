> ⚠️ **METROLOGY CORRECTIONS (audited 2026-07-01, `SESSION_FINDINGS.md`).** Read first; these
> override the body where they conflict.
> 1. **"OURS beats both transformer configs at recall" is NOT a fair comparison — retracted.** OURS
>    was continue-trained on a stream from the SAME `_build_body` the eval uses; SmolLM2-360M is
>    ZERO-SHOT and 24% smaller. `full=0%@8192` is a RoPE/training-window artifact (max_pos=8192);
>    `window=0%` is by construction. OURS also LOSES to the zero-shot transformer at L=512 (62 vs 70)
>    and 1024 (57 vs 62) and collapses by 16k. Accurate: a COST/REACH result (recall retained past a
>    transformer's window at flat O(1) cost), NOT a quality win — a fine-tuned/context-extended
>    transformer would likely match/beat it.
> 2. **"Strict cross-distribution held-out" is partly mislabeled.** Only the arrow/eq CUE variants and
>    the codeparrot CROSS-SOURCE distractor row are genuinely held-out (codeparrot decays to 48%@4096).
>    "Natural code" is FILTERED to remove conflicting literals/reassignment/shadowing → the hard
>    real-code recall cases are UNTESTED.
> 3. **HumanEval-CE is a next-token PROXY, not pass@k**, and retention was NON-MONOTONIC (+0.043@160M,
>    recovered to +0.021 only via the WSD decay — interference is real). L=8192 recall is
>    NOISY/ckpt-unstable (0–83%); only L≤4096 is solid length-generalization.
> 4. Param count is **449M**, not 402M (line 5).
>
> Core finding stands: the recurrent state can be cheaply taught in-window recall (0%→100% ≤4096)
> without a large code-CE regression — TEACHABILITY, not general recall or a transformer win.

# Recall-cotrain feasibility probe (2026-06-30)

**Question (make-or-break for the "inherit knowledge + bounded-state cost moat"
thesis).** The lean LINEARIZED base
(`checkpoints/linearize/linearized_stage3.pt`; 32L×960d DeltaNet, 402M, tied
49152 vocab, `feedback=none`, no WM/gate — inherited SmolLM2-360M's knowledge via
MOHAWK+KD) scores **0% gen_acc** on long-range recall at every distance on the
scoreboard (`experiments/scoreboard_longctx_cost.py`). It recalls at distance 0
but loses a single binding within a couple of tokens. Diagnosis: linearization
transferred KNOWLEDGE but not recurrent-state RECALL behavior (the donor was a
full-attention transformer that never needed bounded-state recall).

**Can a SHORT recall-heavy continue-train teach the linearized base to USE its
recurrent state for long-range recall (lift scoreboard gen_acc off 0% at
L≥2048), WITHOUT destroying the inherited coding knowledge (HumanEval-solution CE
must stay near its 0.759 starting point)?**

## Plumbing

- **Base / arch:** `checkpoints/linearize/linearized_stage3.pt`, continue-trained
  with `--keep_base_vocab 49152` (NO embedding resize → loads with
  `missing=0 unexpected=0` keys, byte-preserved at step 0; verified), LEAN config
  (`--feedback none`, no WM/PKM/gate — the cleanest test of whether the recurrent
  state ITSELF can carry the binding). Trainer template = `launch_phase1_ab_A.sh`.
- **Data mix:** `configs/pretrain_mix_recall_cotrain.yaml` = v18's real-code anchor
  (codeparrot, magicoder, python-codes, instruction code, CVE, textbooks, wiki,
  arXiv) MINUS the four prose recall streams + ONE new recall stream
  (`data/scoreboard_recall_train.jsonl`, 40k examples) at weight 0.50 (≈37% of
  tokens after renormalization). The new stream **exactly matches the scoreboard
  elicitation** (`experiments/gen_scoreboard_recall_train.py`, reuses the
  scoreboard's own `_build_body`): `vN = <4-digit>` bindings near the top,
  distractor fill, query `print(vQ)  # VALUE`, value NEVER restated before its
  query (leak-free / first-occurrence → genuine long-range recall gradient).
  Curriculum: ~50% single-binding (n_keys=1) + multi-binding minority (n_keys
  2/3/6, every key queried once), distances 128–1900 tokens (fit the T=2048
  window; L=4096/8192 recall tested at eval via length-generalization of the
  position-agnostic recurrent mechanism).
  - WHY a new stream and not the existing `multibind_recall_pretrain` /
    `code_recall_train`: those use a DIFFERENT prose format AND restate the answer
    ("Answer: NNNN") → they train recency-copy in the wrong format. Training on
    them would not move the scoreboard metric (the repo's documented
    format-artifact + "supervise where recurrence fails" traps).
- **Arm:** trunk-only (no WM module) — the clean test of whether the recurrent
  state carries the binding. (WM `mem_ctx_namekey` as a 2nd arm was held in
  reserve; see verdict.)
- **Budget / schedule:** 6000 steps × (batch2 × grad_accum8 × T2048 = 32768
  tok) ≈ **197M tokens**; WSD (warmup 150, decay last 15%); Muon lr 3e-4 /
  lr_muon 1e-3 (the proven Arm-A continuation LR — low, to not degrade the base);
  `--bf16 --tf32 --no-compile --activation_checkpointing --mask_eos_in_targets`.
  Mid-ckpt saves at ~80M / ~160M tokens for the trajectory.
- **Eval (`experiments/eval_recall_ours.py`):** RECALL = the scoreboard's OWN
  machinery (`decode_bench.load_ours` bf16 + the TRUE incremental decode
  prefill+forward_step + the shared `print(vQ)  # ` cue + the first-4-digit
  extractor), single-binding (per_bucket=30, 1 key) + 6-binding. KNOWLEDGE =
  HumanEval-solution CE, fp32, CE only on canonical-solution tokens (the exact
  protocol that produced the base's 0.759).

## Elicitation sanity (the format-artifact check)

The 0% is **not** an elicitation/format artifact. With the EXACT
train/eval format, the base recalls perfectly at zero distance and collapses
almost immediately:

| binding→query distance | base gen_acc |
|---|---|
| 0 distractor lines (`v0=N` directly before query) | **100%** (40/40) |
| 2 distractor lines  | 0% |
| 5 distractor lines  | 0% |
| 10 distractor lines | 0% |

So the base CAN emit a just-bound value in this format; it simply cannot carry
the binding through its recurrent state past a couple of tokens. That is exactly
the behavior this probe tries to train in.

## BEFORE (linearized base) — single-binding needle, per_bucket=30

| L | gen_acc | top1 |
|---:|---:|---:|
| 64    | 10% | 17% |
| 512   | 0%  | 7%  |
| 2048  | 0%  | 3%  |
| 4096  | 0%  | 0%  |
| 8192  | 0%  | 0%  |

**HumanEval-solution CE (BEFORE): 0.7585** (9513 sol tokens) — matches the
documented 0.759 starting point.

## AFTER (recall-cotrained) — single-binding needle, per_bucket=30

Trajectory across the run (mid-ckpts at 80M / 160M, final decayed ckpt at 197M).
`gen_acc` (strict first-4-digit exact match):

| L | BEFORE | 80M | 160M | **FINAL 197M** |
|---:|---:|---:|---:|---:|
| 64    | 10% | 100% | 100% | **100%** |
| 512   | 0%  | 100% | 100% | **100%** |
| 2048  | 0%  | 100% | 100% | **100%** |
| 4096  | 0%  | 100% | 100% | **100%** |
| 8192  | 0%  | 73%  | 0%   | **43%** (top1 67%) |

**Harder 6-binding recall** (per_bucket=30; FINAL ckpt).
> ⚠️ **CORRECTED 2026-07-01 (code audit).** The original row was mislabeled
> "query each of the 6 keys": `run_ours_quality` queries `keys[:keys_per_task]`
> and the eval ran with `keys_per_task=3`, so only the 3 EARLIEST-bound keys
> (v0–v2, the best-retained ones) were scored (n=90). Re-run querying ALL 6
> keys (n=180, `eval_6key_all6.json`): L64/512 = **100%**, L2048 = **99.4%**,
> L4096 = **88.9%** (was 96%), L8192 = **30.6%** gen / **43.9%** top1 (was
> 47%/63%). In-window conclusion unchanged; the out-of-window tail was
> inflated ~1.5× by the early-key subset. Raw:
> `checkpoints/recall_cotrain/eval_after_final_6key_all6.json`.

**HumanEval-solution CE:** 0.7585 (before) → 0.7824 (80M) → 0.8017 (160M) →
**0.7793 (final, +0.021)**. The WSD decay phase recovered the slight mid-run
drift; the deployable final ckpt is essentially at the starting point.

Raw JSON: `checkpoints/recall_cotrain/eval_after_*.json`,
`checkpoints/recall_cotrain/results_summary.json`. Final ckpt:
`checkpoints/recall_cotrain/recall_cotrain.pt`.

## Verdict — **GREEN** (thesis feasible)

A short (~197M-token, ~2.2h, single-5090) recall-heavy continue-train teaches the
inherited LINEARIZED base to USE its bounded recurrent state for long-range
recall, WITHOUT destroying inherited coding knowledge:

- **Recall lifts decisively off the 0% floor.** 0% → **100%** at L=512, 2048
  (single- and 6-binding) and 100% single / **89%** all-6-key at L=4096.
  L=4096 is already **2×** the T=2048
  training window; **L=8192 (4×) reaches 31–43%** — never trained at that length,
  so the position-agnostic recurrent mechanism **length-generalizes**. (For
  reference, the scoreboard's `control_full` SmolLM2 cliffs to 0% past its 8192
  window and `control_window2048` cliffs to 0% past 4096; ours holds 100% at 4096
  and is non-zero at 8192 at constant O(1) decode cost.)
- **Coding knowledge retained.** HumanEval-solution CE 0.7585 → 0.7793 (**+0.021**,
  far inside the ≤~+0.1 GREEN bar). No catastrophic interference.
- **It's cheap and fast.** In-window recall (L≤4096) was already **saturated at
  100% by 80M tokens** — the recurrent state learns this behavior almost
  immediately given matched supervision.

⟹ **Green-light the full competent-base run.** The "inherit knowledge +
bounded-state recall + cost moat" thesis is feasible: the linearized base does
not structurally resist recall; it simply never had the supervision, and a thin
recall-cotrain supplies it without forgetting.

## Honest caveats

- **Synthetic, train/eval-matched recall.** The recall stream and the scoreboard
  eval share the SAME `_build_body` distribution (identical `vN=MMMM` bindings +
  canned distractor pool). So this proves the recurrent state CAN be *taught* to
  carry a binding cheaply and that it length-generalizes (L=8192 at 4× the train
  distance, and the 6-binding variant, are genuine within-family generalization),
  but it does NOT yet show recall on NATURAL code (varied binding forms, real
  distractors). Cross-distribution recall is the next test.
- **L=8192 is the out-of-window extrapolation regime and is noisy** across
  mid-ckpts (73% → 0% → 43%); only the WSD-decayed final ckpt is the deployable
  number, and even it is partial. In-window (L≤4096) is the rock-solid result.
- **HumanEval-CE is a next-token CE proxy**, not pass@k; +0.021 CE is small but
  real and its (likely negligible) pass@k cost was not separately measured.
- **Trunk-only.** No WM module was needed — the bare DeltaNet recurrent state
  carries single- and 6-binding recall to 4096 on its own. The WM `mem_ctx_namekey`
  channel (held in reserve) is therefore not required for THIS regime; its value
  is for higher-saturation / capacity-exceeding recall, untested here.

## Recommended next step

1. **Fold a recall-cotrain phase into the full competent-base run** (the format-
   matched, leak-free `print(vQ)  # VALUE` stream at ~15–35% weight; recall
   saturates fast so a minority weight over the larger token budget is plenty).
2. **De-risk generalization to NATURAL code recall** before scaling: build a
   recall eval/stream with real-code distractors + varied binding forms (config
   constants, returned values, imports) and confirm the lift transfers off the
   synthetic family.
3. **Re-run the scoreboard** (`experiments/scoreboard_longctx_cost.py`) on the
   recall-cotrained base — the cost columns are fixed by the architecture, so the
   quality curve should now lift off the floor while OURS decode stays flat
   (8.2 MiB state, ~14 ms/tok). That is the headline the board was built to show.


---

# Generalization de-risk (2026-06-30) — is the recall skill general or format-bound?

Before green-lighting the multi-day full base run, test whether the recall skill
learned by `checkpoints/recall_cotrain/recall_cotrain.pt` GENERALIZES beyond the
trained synthetic format. Leak-free / first-occurrence throughout; GPU1 only.
Harness: `experiments/gen_recall_variants.py` (held-out task gen, 3 axes) +
`experiments/eval_recall_variants.py` (same bf16 incremental-decode path as the
scoreboard, per-variant extractors). Three orthogonal axes, each isolated against
the trained-format control (cue=`print(vQ)  # `, value=int4, distractor=synthetic):
(a) CUE/format, (b) VALUE/binding form, (c) natural-code DISTRACTORS.

## Step 1 — ZERO extra training (cotrained ckpt on held-out variants)

`gen_acc` @ L=512/2048/4096 (per_bucket=25):

| axis · variant | 512 | 2048 | 4096 | transfer? |
|---|---:|---:|---:|---|
| cue · pyout (control) | 100 | 100 | 100 | — |
| cue · assert | 100 | 100 | 88 | **yes** |
| cue · prose | 100 | 100 | 56 | **yes** (weaker far) |
| cue · indirect | 100 | 100 | 100 | **yes** |
| value · int4 (control) | 100 | 100 | 100 | — |
| value · func-return | 100 | 100 | 100 | **yes** |
| value · dict-element | 100 | 100 | 72 | **yes** |
| value · bigint (7-digit) | 100 | 92 | 24 | partial (top1 100 — drops trailing digits far) |
| value · **string** | 28 | 0 | 0 | **NO** |
| distractor · synthetic (control) | 100 | 100 | 100 | — |
| distractor · **natural (Magicoder code)** | 4 | 0 | 0 | **NO** |

**Zero-training verdict: PARTIALLY format-bound.** Recall transfers across HOW you
ask (cue) and across NUMERIC binding forms (dict/func; bigint at short distance),
but FAILS on the two realistic axes: **string values** and **natural-code
distractors** (real Python between bind and query). It learned "carry a 4-digit
int through canned `_z=...` filler", not a general binding-carry.

## Step 2 — short DIVERSE cotrain (~85M tokens), then re-eval ALL axes

Continued from `recall_cotrain.pt` on `configs/pretrain_mix_recall_diverse.yaml`
(`experiments/gen_recall_diverse_train.py`: varied cues × value forms incl
strings × 50% synthetic + 50% Magicoder-natural distractors), 2600 steps × 32768
tok ≈ **85M tokens**, same LEAN config / low LR. `checkpoints/recall_cotrain/recall_diverse.pt`.

**Held-out variants (Magicoder-natural), `gen_acc` @ 512/2048/4096:**

| axis · variant | before | after diverse |
|---|---|---|
| cue · assert / prose / indirect / pyout | 100/100/{88,56,100,100} | **100/100/100** (all) |
| value · int4 / dict / func | 100/100/{100,72,100} | **100/100/100** (all) |
| value · bigint | 100/92/24 | **100/100/100** |
| value · **string** | 28/0/0 | **96/96/92** |
| distractor · synthetic | 100/100/100 | 100/100/100 |
| distractor · **natural** | 4/0/0 | **100/100/100** |

**STRICT cross-distribution held-out** (NONE of these in the diverse training:
NOVEL cues `# vQ -> ` / `# vQ == `, and natural distractors sampled from
**codeparrot** — a DIFFERENT source than the Magicoder used for training), `gen_acc`/`top1`:

| axis · variant | 512 | 2048 | 4096 |
|---|---:|---:|---:|
| cue · arrow (novel) | 100/100 | 100/100 | 100/100 |
| cue · eq (novel) | 100/100 | 100/100 | 100/100 |
| value · string | 96/68 | 96/76 | 80/60 |
| value · int4 | 100/100 | 100/100 | 100/100 |
| distractor · natural **(codeparrot, cross-source)** | 100/100 | 92/96 | 48/92 |

**Single-binding scoreboard recall (diverse ckpt):** 100% at 512/2048/4096,
**8192 improved 43% → 83%** (the diverse data also strengthened length-extrapolation).
**HumanEval-solution CE: 0.7585 (base) → 0.7793 (1st cotrain) → 0.7717 (diverse)** —
retention excellent (well within "near 0.78"; the diverse run was actually slightly
better-retained than the first).

## Generalization verdict — **GENERALIZES (with diverse supervision)**

Bounded-state recall is a **real, generalizable capability, not a fixed format
trick** — BUT only if it is supervised diversely. Trained on a single synthetic
format it is partially format-bound (string + natural-distractor axes fail). A
thin (~85M-token) cotrain over varied cues + value forms + natural-code
distractors makes recall robust across ALL three axes, INCLUDING truly held-out
distributions: novel cues never seen (`arrow`/`eq`) hit 100%, and a DIFFERENT
natural-code source (codeparrot, trained only on Magicoder) holds 100%@512 /
92%@2048 / 48%@4096 (top1 ≥92% throughout) — that is cross-distribution
generalization, not memorization. Coding knowledge retained (CE 0.7717).

⟹ **Green-light the full base run, WITH diverse recall supervision** (not the
single-format stream). Remaining honest soft spots, to watch: (1) the longest
out-of-window distances (4096+ for cross-source natural distractors, 8192 needle)
are where the recurrent state still partially decays — the full run's larger token
budget + the planned WM channel are the levers; (2) string-value recall is strong
(80–96%) but a notch below numeric — broaden value forms further in the full mix.

## Recommended recall-supervision recipe for the full base run

- **Use a DIVERSE recall stream, never a single format.** Mix: cues
  {`print(x) #`, `assert x ==`, prose comment, indirection, and a couple held-out-
  style cues}; value forms {int4, big int, **string**, dict/list element,
  function return}; distractors **50% real code lines** (sampled from the actual
  pretrain code corpus, filtered to not introduce a conflicting literal / clobber
  the binding) + 50% synthetic. Leak-free / first-occurrence (value supervised at
  the query, never restated before it). Generators ready:
  `gen_recall_diverse_train.py`.
- **Weight ~15–35% of tokens** — recall saturates fast (in-window hit 100% by 80M
  tokens), so a minority weight over the full token budget is plenty; keep the
  code/instruction anchor dominant for knowledge retention.
- **Keep it LEAN / low-LR as a continuation** (no base degradation observed:
  CE 0.7585 → 0.7717). WM module not required for this single-binding regime; add
  it only for higher-saturation (many-competing-binding) recall.
- **Re-run the scoreboard + the variant battery after the full run** to confirm
  the quality curve holds at flat O(1) decode cost and recall stays general.

## Step 4 — full scoreboard on the diverse ckpt (quality × cost vs distance)

`experiments/scoreboard_longctx_cost.py` on `recall_diverse.pt`, HARDER 6-binding
task (per_bucket=20, 3 keys). `gen_acc/top1` %, cost single-stream bf16 eager.
(OURS "before" column = the original lean base from `SCOREBOARD.md`, all 0% gen.)

| L | OURS before | **OURS diverse** | oMs/oPeak/oState | FULL gen/top1 | WIN-2048 | mem× |
|---:|---:|---:|---|---:|---:|---:|
| 512   | 0%/15% | **62%/67%** | 14.5 / 897 / 8.2 | 70%/75% | 70%/75% | 0.83× |
| 1024  | 0%/12% | **57%/62%** | 14.6 / 897 / 8.2 | 62%/63% | 62%/63% | 0.85× |
| 2048  | 0%/3%  | **52%/60%** | 14.6 / 897 / 8.2 | 47%/53% | 47%/53% | 0.90× |
| 4096  | 0%/5%  | **45%/53%** | 14.6 / 897 / 8.2 | 38%/42% | **0%/7%** ←blind | 0.99× |
| 8192  | 0%/13% | **35%/45%** | 14.7 / 897 / 8.2 | **0%/13%** ←past window | 0%/20% | 1.17× |
| 16384 | 0%/7%  | 2%/15%      | 14.6 / 897 / 8.2 | 0%/5%  | 0%/10% | 1.53× |
| 32768 | 0%/7%  | 0%/12%      | 14.6 / 897 / 8.2 | 0%/10% | 0%/10% | 2.26× |
| 65536  | — | (cost) | 14.5 / 898 / 8.2 | (cost) — KV 2561 MiB | — | 3.70× |
| 131072 | — | (cost) | 14.7 / 898 / 8.2 | (cost) — KV 5121 MiB | — | 6.60× |

**Headline (the artifact the board was built for):** the QUALITY curve lifted off
the 0% floor at FLAT O(1) decode cost (14.5 ms/tok, 897 MiB peak, **8.2 MiB**
bounded state from 512 → 131072, while the transformer's KV grows 20 → 5121 MiB,
mem crossover ~8192, 6.6× at 131072). And the quality now **beats BOTH transformer
configs exactly where bounded state is supposed to win**: at **L=4096** ours 45% vs
full 38% while the windowed transformer is **blind (0%)**, and at **L=8192** ours
35% vs full **0%** (past its 8192 training window) and window 0% — ours is the only
arm with non-trivial long-range recall, at constant cost. (Single-binding needle is
even stronger: 100% to 4096, 83% at 8192 — see `eval_after_diverse_1key.json`.)
Past L≈16384 ours also decays (state saturates under 6 competing bindings); that is
the next frontier, not a regression.
