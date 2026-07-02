# Flagship capability probe — long-context multi-key recall

**Date:** 2026-06-21
**Question (capability only; cost/O(1)-decode deferred):** Can a bounded-state
**DeltaNet + WorkingMemory** recall facts bound EARLY in a long sequence and
queried LATE — at sequence lengths BEYOND a fixed transformer's window — where a
fixed-window transformer goes blind? And how does it compare to a transformer
control?

**TL;DR.** Yes for the foil that matters, with an honest ceiling. Our 287M
bounded-state model (`checkpoints/pretrain_v12.pt`) **retains 56.6% multi-key
recall at L=4096 and 17.0% at L=8192, where a fixed 2048-token-window transformer
collapses to exactly 0.0%**. It also matches/edges the *full-context* SmolLM2-360M
(the perfect-but-expensive O(L) ceiling) past 2048. BUT ours **also degrades past
its 2048 training length** (90% → 57% → 17%) — it is graceful, not solved — and the
recall is carried by the **DeltaNet recurrence, not the WorkingMemory** (the gate
thinks ~1% of the time, and *forcing* the WM channel makes recall much worse). So
the result validates **bounded recurrent state** as a length-extrapolating recall
substrate, and tells us the flagship needs **long-horizon training** (and a WM that
is actually load-bearing) to push past ~4k.

---

## Results

### Headline: generation-graded recall accuracy vs total length L (N = 900 queries/cell)

150 tasks/bucket × 6 keys each = 900 independent first-occurrence recall queries
per cell. Greedy decode. A query is correct iff the model generates the exact
4-digit value bound to the queried variable. Chance ≈ 1/9000 ≈ 0%.

| arm | L=1024 | L=2048 | L=4096 | L=8192 |
|---|---|---|---|---|
| **OURS** (v12, DeltaNet+WM, gate decides) | **99.4%** (895/900) | **90.1%** (811/900) | **56.6%** (509/900) | **17.0%** (153/900) |
| CONTROL-full (SmolLM2-360M, full ctx, O(L) KV) | 78.9% (710/900) | 67.0% (603/900) | 56.2% (506/900) | 0.0% (0/900) |
| CONTROL-window2048 (SmolLM2-360M, last 2048 tok) | 78.9% (710/900) | 67.0% (603/900) | **0.0%** (0/900) | **0.0%** (0/900) |

Tokens per bucket (all arms share the SmolLM2 tokenizer → identical counts):
L=1024 ≈ 1018 tok, L=2048 ≈ 2042, L=4096 ≈ 4090, L=8192 ≈ 8186.
CONTROL-window2048 truncates its input to the last 2048 tokens.
OURS `think_rate = 0.009` (the gate almost never thinks → recall comes from the
recurrence). `overflow>8192 = 0` for both controls (prompts capped ≤ 8192).

Sanity check: CONTROL-full == CONTROL-window2048 exactly at L≤2048 (the window
covers the whole prompt there, so it is literally the same computation), and they
diverge precisely once the early bindings leave the 2048-window at L=4096.

### Diagnostic: does FORCING the WorkingMemory channel help? (N = 150/cell subsample)

| arm | L=1024 | L=2048 | L=4096 | L=8192 |
|---|---|---|---|---|
| OURS, gate decides (think≈0.01) | 99.4% | 90.1% | 56.6% | 17.0% |
| OURS, **8 forced WM/think steps** (think=0.20) | 35.3% (53/150) | 8.7% (13/150) | 0.0% | 0.0% |

**Forcing the WM/think channel HURTS recall at every length.** This is the
documented "thinking corrupts recall" failure mode: every think token still steps
the DeltaNet recurrence and perturbs the precise binding the linear-RNN state was
carrying. ⟹ the recall capability here is the **bounded recurrent state**, not the
explicit WorkingMemory module.

### Fairness sensitivity (why each arm uses its own prompt wrapper)

Single-format probe at L=1024 (6 keys, one task), to justify per-arm elicitation:

| prompt format | SmolLM2-360M (base) | OURS (v12) |
|---|---|---|
| PROSE ("Run the following… `print(v)`") | 0/6 | 6/6 |
| COMPLETION ("…```\n\nThe value of v is ") | 6/6 | 0/6 |

The two model families have **disjoint best-shot formats**: OURS was pretrained
*only* on the prose multibind-recall format and is OOD on bare completion; the
*base* SmolLM2 does not follow the instruction but completes the value cue. A
single shared format would cripple one arm and inflate the other, so — per the
repo's equal-opportunity mandate — each arm uses its own best-shot wrapper around
**identical task content** (same body, bindings, values, queried key) and the
**identical answer extractor**.

---

## Verdict

1. **Does OURS retain recall past 2048 where CONTROL-window2048 collapses?**
   **YES, decisively.** The fixed-cost 2048-window agent drops to exactly 0.0% at
   L=4096 and L=8192 (the early bindings fall outside its window). OURS retains
   **56.6% @4096 and 17.0% @8192**. This is the foil the flagship was meant to beat,
   and it beats it cleanly.

2. **How does OURS compare to CONTROL-full (the perfect-but-expensive ceiling)?**
   OURS is **tied at L=4096** (56.6% vs 56.2%) and **wins at L=8192** (17.0% vs 0.0%
   — the transformer is blind there). It also reads higher at L=1024/2048, **but
   that gap is confounded**: OURS was pretrained on this exact `multibind_recall`
   task family, while SmolLM2-360M saw none of it, so the absolute in-window numbers
   partly measure task exposure, not architecture. The architecture-clean signals
   are (a) OURS vs the window foil and (b) the *shape* of degradation with distance.

3. **Honest negative — OURS is graceful, not solved.** OURS degrades steadily past
   its 2048 training length (90% → 57% → 17%). At L=8192 it recovers mostly the very
   first binding and loses the rest. Both architectures degrade with absolute
   distance; OURS just degrades more slowly and has no hard window cliff. A flagship
   long-recall system needs **long-horizon training** (train at >2048, multi-key
   saturation) — the bounded-state architecture is necessary but not sufficient.

4. **Mechanism caveat (important).** This validates the **bounded recurrent
   DeltaNet state**, not the WorkingMemory: the gate thinks ~1% of the time, and
   forcing the WM channel degrades recall badly. The WM module is currently not the
   thing doing the recall.

---

## Caveats (read before quoting any number)

- **CAPABILITY ONLY.** No O(1)/constant-cost claim is made. We ran the
  full-forward-equivalent decode path (verified bit-identical to the incremental
  path on this task); true incremental `forward_step` decode is an unwired perf
  detail and deliberately out of scope. Cost/throughput is not measured here.
- **In-distribution confound for OURS.** v12 was pretrained on `multibind_recall`
  (same prose format, same `vN = <4-digit>` scheme). Its absolute in-window recall
  benefits from that; the base SmolLM2 had zero exposure. Trust the *vs-window* and
  *degradation-shape* conclusions over the raw OURS-vs-full gap at L≤2048.
- **Per-arm prompt formats** (prose for OURS, completion cue for the base
  transformer). Task content/bindings/values/queried key are byte-identical across
  arms; only the surface wrapper differs (justified by the disjoint-format table).
- **CONTROL-full at L=8192:** the prompt is capped ≤8192 tokens, but greedy
  generation pushes a few positions just past SmolLM2's 8192 RoPE training length.
  Its 0.0% at 8192 reflects both genuine long-distance recall decay *and* mild
  position extrapolation. Either way it is blind at 8192 while OURS holds 17%.
- **Bounded WM buffer** kept at its trained `mem_size=1024` (the honest bound); it
  was not enlarged to cover L=8192.
- **Leak-free by construction:** the bound value is never restated before the query,
  so a recency-copy cannot answer — the model must recall across the full
  binding→query distance ≈ L.
- N=900/cell (main) gives tight estimates; queries within a task share a body, so
  effective independent samples are somewhat fewer than 900.

---

## Exact commands / scripts (to re-run)

New scripts (eval-only; no training code paths were modified):
- `experiments/flagship_recall_probe_gen.py` — generates the multi-key tasks,
  binary-search-trimmed to land at each target length L.
- `experiments/flagship_recall_probe.py` — runs all arms, per-arm best-shot
  wrappers, shared first-4-digit answer extractor.

```bash
cd /home/knielsen/ml/parallel-ss-dep
export PYTHONPATH=.

# 1) Generate the probe set (600 tasks: 150 per bucket × 4 buckets, K=6 keys).
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/flagship_recall_probe_gen.py \
    --out data/flagship_recall.jsonl \
    --buckets 1024,2048,4096,8192 --per_bucket 150 --n_keys 6 --seed 0

# 2) Main 3-arm run (N=900/cell). ~45 min on one RTX 5090.
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/flagship_recall_probe.py \
    --tasks data/flagship_recall.jsonl \
    --ckpt checkpoints/pretrain_v12.pt \
    --hf_model HuggingFaceTB/SmolLM2-360M \
    --keys_per_task 6 --max_gen 32 \
    --arms ours,control_full,control_window2048 \
    --out_json results/flagship_recall_main.json

# 3) Forced-think diagnostic (subsample, N=150/cell).
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/flagship_recall_probe.py \
    --tasks data/flagship_recall.jsonl --ckpt checkpoints/pretrain_v12.pt \
    --keys_per_task 6 --max_gen 32 --force_think 8 \
    --arms ours_forcethink --max_problems_per_bucket 25 \
    --out_json results/flagship_forcethink.json

# Optional: single-shared-format fairness sensitivity (--shared_format flips
# OURS to the completion wrapper and the controls to the prose wrapper).
```

Result JSON: `results/flagship_recall_main.json`, `results/flagship_forcethink.json`.
Notes: `--max_gen 32` is required so OURS's prose preamble ("The program assigns N
variables. `vX` is set to …") reaches the value before truncation; the completion
arms emit the value immediately. All arms share the SmolLM2 tokenizer.
