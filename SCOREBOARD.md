> ⚠️ **METROLOGY NOTE (audited 2026-07-01, `SESSION_FINDINGS.md`).** This board (lean base) is honest:
> OURS reads 0% at every distance (the quality floor, not dressed up), and the cost columns are the
> solid, architecture-determined result. Two scoping corrections: (1) the later "OURS 62/57/52/45/35%,
> beats both transformer configs" number is from the recall-COTRAINED ckpt (in `RECALL_COTRAIN_PROBE.md`)
> and is a TRAIN-ON-EVAL-DISTRIBUTION vs ZERO-SHOT comparison — a COST/REACH result, not a fair quality
> win (see that doc's banner); ours also loses to the zero-shot transformer at L≤1024. (2) "Dramatic
> cost moat" applies to the MEMORY/reach axis only — single-stream latency favors the transformer
> (~2.3× faster/token until ~131k), and a windowed/paged transformer is also bounded-memory. The
> `window=0%` and `full=0%@8192` foils are by-construction / OOD-training-window artifacts, not general
> transformer inabilities.

# SCOREBOARD — long-context coding-recall QUALITY × decode COST (2026-06-30)

The project's committed north-star is **a cheap, bounded-state, long-context
coding agent whose moat is COST (O(1) / constant-memory decode) at long
horizon.** HumanEval-164 greedy is the wrong headline for that thesis (noisy,
and short-context so the moat is invisible). This is the ONE board that makes
the moat the metric: it jointly measures, **vs context distance**, both (a) the
TASK quality that *requires* using long-range context, and (b) the COST
(ms/tok, peak memory) to deliver it — for our bounded-state model vs a
transformer — into the regime where the transformer is expensive.

- Harness: `experiments/scoreboard_longctx_cost.py` (reusable). Cost machinery
  reused from `experiments/decode_bench.py`; leak-free task + 4-digit extractor +
  windowed-transformer control reused from `experiments/flagship_recall_probe.py`
  / `flagship_recall_probe_gen.py`.
- Raw: `checkpoints/scoreboard/scoreboard_results.json` (6-binding) and
  `…_1key.json` (single-binding needle). Tasks: `…/scoreboard_recall*.jsonl`.
- Run: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python
  experiments/scoreboard_longctx_cost.py --buckets 512,1024,2048,4096,8192,16384,32768
  --cost_buckets 65536,131072 --per_bucket 20 --keys_per_task 3`

## Task — leak-free, first-occurrence recall

Bind K facts near the START of a Python program (`vN = <4-digit int>`),
distractor-fill (no 4-digit literals, never clobbers a `vN`) to a controlled
token length L so the binding→query distance ≈ L, then query ONE key at the END
with the raw-Python output-comment cue `print(vQ)  # `. The bound value is
**NEVER restated before the query**, so a recency copy cannot answer — the model
must RECALL across the full distance. This is the repo's mandated **leak-free /
first-occurrence** protocol (NOT the leaky `eval_code_recall --mode
teacher_forced` that scores a restated answer). Values are 4-digit and
distractors carry no 4-digit literal ⇒ the first isolated 4-digit number in the
continuation is the recalled value (identical extractor for every arm).

**Elicitation (equal-opportunity mandate).** Both arms are *base* LMs, so both
get the SAME best-shot format — the raw-Python `print(vQ)  # ` continuation.
Validated 2026-06-30: the prose / "The value of X is" instruction cue is OOD for
the lean linearized base (reads ~0) while the comment-continuation recalls
cleanly at zero distance; it is also natural for the SmolLM2 code base.

Arms: **ours** = lean linearized DeltaNet (`checkpoints/linearize/linearized_stage3.pt`,
32L×960d, 449M, bounded recurrent state, scored with the TRUE incremental decode
`prefill`+`forward_step` — the same O(1) path `decode_bench` costs);
**control_full** = `HuggingFaceTB/SmolLM2-360M` (361M, full O(L)-KV attention,
8192 training window); **control_window2048** = the SAME transformer truncated to
the last 2048 tokens (a fixed-COST agent — past L=2048 the early binding leaves
the window). Quality = `gen_acc` (strict first-4-digit exact match) / `top1`
(gold's first token is the post-prompt argmax — a sensitive floor probe).

## Results — joint QUALITY × COST vs distance L

Single-stream, bf16, eager (the deployed `forward_step` path). `gen_acc/top1` in
%. Cost: steady-state median ms/tok over 32 decode steps after 8 warmup; peak =
decode-phase peak MiB (reset after prefill so the O(L) prefill transient doesn't
mask the decode footprint); state|KV = bounded recurrent state (ours) / KV cache
(xf). `mem x` = xf-peak / ours-peak.

### Canonical single-binding "needle" (per_bucket=30, 1 key)

| L | OURS gen/top1 | FULL gen/top1 | WINDOW-2048 gen/top1 |
|---:|---:|---:|---:|
| 512   | **0% / 13%** | 100% / 100% | 100% / 100% |
| 1024  | **0% / 7%**  | 100% / 100% | 100% / 100% |
| 2048  | **0% / 3%**  | 100% / 100% | 100% / 100% |
| 4096  | **0% / 0%**  | 100% / 100% | **0% / 7%**  ← window blind |
| 8192  | **0% / 0%**  | **0% / 17%** ← past 8192 train window | 0% / 13% |
| 16384 | **0% / 0%**  | 0% / 13% | 0% / 13% |
| 32768 | **0% / 20%** | 0% / 13% | 0% / 20% |

### Harder 6-binding recall (per_bucket=20, 3 keys) + COST columns

| L | OURS gen/top1 | oMs | oPeak | oState | FULL gen/top1 | fMs | fPeak | fKV | WIN gen/top1 | mem x |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512   | 0% / 15% | 14.4 | 919 | 8.2 | 70% / 75% | 6.3 | 756 | 20 | 70% / 75% | 0.82× |
| 1024  | 0% / 12% | 14.4 | 919 | 8.2 | 62% / 63% | 6.3 | 776 | 40 | 62% / 63% | 0.84× |
| 2048  | 0% / 3%  | 14.4 | 919 | 8.2 | 47% / 53% | 6.3 | 819 | 81 | 47% / 53% | 0.89× |
| 4096  | 0% / 5%  | 14.4 | 919 | 8.2 | 38% / 42% | 6.3 | 901 | 163 | **0% / 7%** ← window blind | 0.98× |
| 8192  | 0% / 13% | 14.4 | 919 | 8.2 | **0% / 13%** ← past window | 6.2 | 1063 | 322 | 0% / 20% | 1.16× |
| 16384 | 0% / 7%  | 14.3 | 919 | 8.2 | 0% / 5%  | 6.3 | 1386 | 640 | 0% / 10% | 1.51× |
| 32768 | 0% / 7%  | 14.4 | 920 | 8.2 | 0% / 10% | 6.4 | 2036 | 1280 | 0% / 10% | 2.21× |
| 65536  | (cost) | 14.4 | 920 | 8.2 | (cost) | 8.7  | 3336 | 2561 | — | 3.63× |
| 131072 | (cost) | 14.4 | 920 | 8.2 | (cost) | 13.9 | 5937 | 5121 | — | 6.45× |

## Headline read

**1. The COST moat is real and dramatic.** OURS decode is genuinely O(1): **flat
14.4 ms/tok and 919–920 MiB peak with an 8.2 MiB bounded recurrent state from
L=512 all the way to L=131072.** The transformer's KV cache grows linearly
(20 → 5121 MiB) and its decode-phase peak climbs 756 → 5937 MiB; per-token
latency rises 6.3 → 13.9 ms. **Memory crossover** is ~L=8192 (xf peak passes
ours) and the gap widens to **6.45× at 131072 and keeps growing**. Honest:
single-stream the transformer is ~2.3× *faster per token* at short context
(optimized softmax-decode, fewer params) — the latency only crosses near 128k;
the decisive throughput + OOM win lives in the **batched** regime (see
`DECODE_COST_BENCH.md`: ~1.8× throughput at ~5.8× less memory, and whole
(batch, context) regions that OOM the transformer while ours sails through).

**2. The windowed transformer hard-collapses at its window — the foil works.**
`control_window2048` is perfect (100%, single-binding) up to L=2048, then
**cliffs to 0% at L=4096** the moment the binding leaves the 2048-token window
(`n_blind` confirms the binding was truncated out). This is the exact
hard-collapse a bounded-state model is *supposed* to beat at long range.

**3. Even "full context" has a wall at the training length.** `control_full`
(SmolLM2-360M, 8192 training window) is perfect to L=4096 then **cliffs to 0% at
L=8192+** — RoPE extrapolation past the training length fails. So neither
transformer config serves recall past ~its window without retraining/extension.

**4. OURS quality is the gap to close — currently below the floor.** The lean
linearized base reads **0% gen_acc at every distance** (single- AND
multi-binding); inspection shows it emits tiny default print values ("7","10",
"0") and pattern-matches code structure instead of recalling the 4-digit binding
from ~490 tokens back. It DOES recall at zero distance, so the architecture has
the capability — but it has **not learned to USE its recurrent state for
long-range recall.** This is consistent with the repo's "undertrained, not
undercapacity" finding: the lean base saw only ~800M KD tokens (vs SmolLM2's
2–4T). So bounded state gives us **constant cost and unbounded context REACH**,
but the current base cannot yet convert reach into recall.

**Does bounded-state degrade gracefully where the windowed transformer
collapses? Not yet on this base.** The COST half of the moat is fully
demonstrated; the QUALITY half is the open target this board now measures.

## Honest caveats

- **Base weakness is expected and is the point.** Absolute OURS accuracy is at
  floor; the board is the TARGET, not a victory lap. We did not dress this up.
- **Leak-free, first-occurrence only.** The bound value appears once (at the
  top binding) and is never restated before the query; we score the model's
  generated value, not a teacher-forced restatement.
- **Synthetic task.** Bindings + distractors are generated (variable-binding
  "needle"); it isolates long-range recall but is not natural code.
- **Both arms eager, single-stream.** The throughput/OOM crossover (the larger
  cost win) is batched and lives in `DECODE_COST_BENCH.md`; this board is
  single-stream because quality is inherently per-sequence.
- **Full-transformer past 8192** is out-of-training-distribution (RoPE
  extrapolation), so its 0% past 8192 is a window limit, not a fixed verdict —
  a context-extended transformer would push that wall out (at growing KV cost,
  which is exactly the axis this board prices).
- We did not separately score the from-scratch 287M base: it needs the thinking
  generator and is expected at/below this floor on the same task.

## Recommendation — what "competent enough base" looks like on this board

The Step-2 base work should target a base that makes the QUALITY axis *readable*,
i.e. converts the architecture's unbounded reach into actual recall:

- **Minimum (makes the headline true):** single-binding `gen_acc ≥ ~50%` at
  **L = 4096 and 8192**, *held while `control_window2048` is 0%* (and ideally
  while `control_full` is 0% past 8192). That is the literal "graceful where the
  windowed transformer collapses" claim, now backed by a number.
- **Stretch (beats BOTH transformer configs):** match `control_full` (~100%) up
  to L=4096 AND stay non-zero past 8192 where the full transformer cliffs —
  proving bounded-state reach beats the window at constant cost.
- **How to get there (what the board says to train):** the recall behavior must
  be trained INTO the recurrent state — far more continue-pretrain / KD tokens
  (the lean base saw ~800M; SmolLM2 saw 2–4T), and/or explicit long-range recall
  supervision (the `multibind` / `longctx` recall streams), and/or making the
  WorkingMemory channel actually load-bearing for content recall. Re-run this
  board after each base iteration; the cost columns are fixed (architecture), so
  any base improvement shows up purely as the quality curve lifting off the floor
  while OURS cost stays flat.
