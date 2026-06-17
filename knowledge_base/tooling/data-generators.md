# Data generators

## Summary
The synthetic task and recall-data generators used to (a) validate mechanisms on matched-bottleneck tasks and (b) inject those bottlenecks into pretrain so a mechanism becomes load-bearing ([[objective-function-alignment]]). The key design rule: a recall/reasoning corpus must include **answer supervision** (so the bound value is actually a target) and, for [[working-memory]], be **capacity-exceeding AND non-memorizable** (fresh random content). All paths `experiments/gen_*.py`.

## Reasoning / depth (for [[latent-thinking]])
- `gen_pointer_chase.py` — `f^K(s)` over a fresh in-context permutation table (the clean depth separator; arithmetic is parallelizable and collapses). `--fixed_point` (halt-when-stable), `--dict_format` (multi-line Python dict = code-format).
- `ptr10dict` corpus (`data/ptr10dict_train_n{2..8}.jsonl`) — the depth-bound, single-token-answer corpus the v13 `LatentReasoningCotrain` loss trains on.
- `gen_exec_trace.py` / `data/trace_train` — real Python execution traces (thread x through n single-op lines). Code-distribution depth task; surfaced the heterogeneity wall.
- `gen_framings.py`, `gen_incontext_ops.py`, `gen_incontext_tables.py`, `gen_compose.py` — mapped the generalization boundary (presentation-invariance yes, per-step-op variety no).

## Recall (for [[working-memory]])
- `gen_multibind_recall.py` — assign N vars, print the queried one; N∈{8..128}. The **capacity-exceeding, non-memorizable** headroom-bearing probe (the realistic MQAR analog). `data/multibind_recall_{train,heldout,pretrain}.jsonl`.
- `gen_code_recall_tasks.py` — realistic code recall (config-constant / signature / import-alias / function-name), real magicoder Python as distractor interference. `data/code_recall_{train,heldout}.jsonl`. **Finding: the recurrence already solves this ~89 %** (real identifiers are separable).
- `gen_agentic_recall_tasks.py` — ReAct-style transcripts, recall an early tool-output value (semi-synthetic). `data/agentic_recall_{train,heldout}.jsonl`. **WM's clearest real opportunity** (baseline ~0.42, real capacity wall).
- `gen_longctx_recall_tasks.py` — single-binding recall by distance (NO headroom — recurrence is 100 %).
- `gen_synthetic_memory_tasks.py` — 5 families (var_binding, chain_arithmetic, list_index_recall, dict_lookup, multi_step_arithmetic).

## The two non-negotiable data lessons
1. **Answer supervision is required.** The pretrain recall stream that fed `problem_prompt` only gave WM no recall gradient at all (the bound value was never a target). Fix: `text_field: [problem_prompt, qwen_completion]` so "outputs ___ / Answer: ___" positions force recall. See [[working-memory-recall-saga]].
2. **`mem_read_mask` over the answer span** (`emit_read_mask: true` in the v14 config) routes answer-token CE into the WM read — the gradient that was structurally missing in v10–v13.

## Distillation / SFT data
- `distill_solutions.py` (vLLM, Qwen) → ~38 k (problem, CoT, code) pairs. Code-grader loaders: `mbpp_*`, `leetcode`, `magicoder_oss`, `codefeedback`, `distill_corpus`, `super_combined`.
- **Data hygiene matters — the headline lever (bottleneck is data, not architecture).** MEASURED on `data/distill_v7_phase1_with_tests.jsonl` (54,468 rows; already carries `tier`/`score`/`has_tests`, so existing rows need no re-grading): **91.8 % are `no_tests`/unverified** and default to `score=1.0` (so the "92 % passing" is an artifact); of the **4,468 verified** rows only **120 pass (2.7 %)** — 95.8 % broken (632 syntax_error + 3,647 exec_error); every problem appears only **~2× (median 2, max 2)** vs ~100+ exposures needed to lock a fact (≈25× under-exposed). The naive filter `has_tests AND score>=1.0` keeps **120 rows (0.2 %) — a trap**. Real levers: DROP the 4,279 verified-broken rows; verify/clean the `no_tests` bulk; CONCENTRATE exposure (upsample important families >100×). The model learned `degrees_to_radians` from malformed examples. See task #84 + [[thinking-on-code-verdict]]. (The older "~6 % broken / ~70 % unverified" estimate was for the combined corpus; the above is the precise distill-file measurement.) **De-leak** any retrieval datastore (magicoder/codefeedback contain MBPP).

## Related
[[evals-and-probes]] · [[objective-function-alignment]] · [[working-memory-recall-saga]] · [[latent-thinking]] · #tooling
