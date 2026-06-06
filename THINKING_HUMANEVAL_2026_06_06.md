# Making latent thinking actually help HumanEval (2026-06-06)

**TL;DR.** Latent thinking went from *catastrophically hurting* HumanEval
(thinking-ON 0/164 vs no-think 8/164) to *modestly helping* it
(thinking-ON **11/164** vs a matched no-think control **7/164**) — on the same
287M checkpoint, same decode path. The fix was **not** a better thinking
computation; it was **routing the gate so thinking fires rarely and
selectively** instead of compulsively. The collapse was a gating bug, exactly as
hypothesized.

Magnitude caveat: +3–4 on 164 greedy problems is modest and near the ±2–3
sample-noise band. The *collapse fix* is solid and reproduced; the *lift
magnitude* should be confirmed with temperature-sampled pass@1 before being
quoted as a headline.

---

## What was broken

On the baked-then-SFT'd 287M code model, turning thinking on **collapsed**
generation:

| condition (same ckpt) | think_rate | HumanEval |
|---|---|---|
| no-think | — | 8/164 |
| thinking-ON, autonomous gate | **0.46** | **0/164** |

Diagnostic generations showed the failure is a **degenerate attractor**, not
"slightly worse code": the model emits `"2222222…"` or halts after one token.

**Mechanism.** Latent thinking feeds the model's own hidden state back in as the
next input. On code positions the gate fires its think op ~46% of the time;
because that op is out-of-distribution for code generation, each mid-stream think
perturbs the state, and the perturbation **compounds token-after-token** into a
repeating/halting loop. A single upfront think (prefix-only) merely dented the
score (5/164); it was the *repeated mid-stream* firing that compounded to 0.

## What we changed (the 4 things)

1. **FLA Blackwell detection fix (infrastructure).** `fla/utils/_device.py`
   `IS_NVIDIA_BLACKWELL` only matched capability `== 10`; the RTX 5090 is
   sm_120 → `(12,0)`, so every Blackwell compiler workaround was silently off →
   intermittent `cudaErrorLaunchFailure` in the latent forwards. Fixed to
   `[0] in (10, 12)`. **This is uncommitted in the fork and reverts on any
   pull/refactor — re-apply and verify `IS_NVIDIA_BLACKWELL is True`.** (Bit us
   twice in one day; worth committing in the fork.)

2. **Train the latent op on CODE data, not arithmetic** (the user's call). The
   prior bake trained thinking on synthetic pointer-chase (`ptr10dict`), so the
   op was OOD for code. `launch_latent_bake_code.sh` re-bakes the op on real
   Python **execution traces** (`data/trace_train`, code-distribution,
   depth-requiring) while the LM loss stays on the real code pretrain mix.
   *Necessary but not sufficient* — the trace op still hurt generation until (4).

3. **Preserve competence AND the op together.** Neither base had both:
   `latent_bake_code` has the 804k-param latent adapter but isn't code-competent
   (0/164); `sft_phase_c_combined` is competent (10/164) but has no adapter. Fix:
   a **full code SFT with `--with_thinking` OFF** on `latent_bake_code`. The
   adapter is only used in the latent-thinking loop, never in the normal forward,
   so it gets **zero gradient → preserved exactly**, while the trunk learns code
   competence (→ `sft_baked_code.pt`, no-think 8/164).

4. **THE KEY — route the gate to EMIT on code, then think *selectively*.**
   `experiments/route_emit_finetune.py`: freeze everything except the 897-param
   gate head and train it to emit (BCE→1) on code positions (400 steps, ~30 s).
   This drove the gate's mean σ(emit) on code 0.49 → 1.00. Then at inference,
   lower `--emit_threshold` (≈0.1–0.3) so the gate fires think only at the
   *strongest-signal* positions (~2% of tokens). Result (`route_emit_code.pt`):

   | condition (same ckpt, same generator) | think_rate | HumanEval |
   |---|---|---|
   | no-think (standard) | — | 8/164 |
   | **control: latent_think, 0 thinks** (`--emit_threshold 0.0`) | 0.000 | **7/164** |
   | thinking-ON `--emit_threshold 0.5` | 0.028 | 6/164 |
   | thinking-ON `--emit_threshold 0.3` | 0.022 | **11/164** |
   | thinking-ON `--emit_threshold 0.1` | 0.016 | **11/164** |

   The matched **0-think control (7/164)** rules out a decode-path artifact, and
   **11/164 reproduces at two thresholds**, so the +4 (7→11) is attributable to
   thinking, not the generator.

## The one-sentence insight

**Thinking helps only when it is rare and selective; firing the same op
compulsively collapses generation.** The collapse (0/164) and the lift (11/164)
are the *same op* at think_rate 46% vs ~2%. Routing the gate to emit on code,
plus a low inference emit_threshold, is what makes thinking selective — and
therefore useful.

## Reproduce

```bash
# 1. (once) ensure the FLA Blackwell fix is live
python -c "import fla.utils as u; assert u.IS_NVIDIA_BLACKWELL"
# 2. bake the latent op on code traces (or reuse checkpoints/latent_bake_code.pt)
GPU=0 bash launch_latent_bake_code.sh         # then cash a mid-eval ckpt -> latent_bake_code.pt
# 3. full code SFT, thinking OFF, to restore competence + preserve the op
#    sft_code.py --load_ckpt latent_bake_code.pt (NO --with_thinking) -> sft_baked_code.pt
# 4. route the gate to emit on code (freeze all but gate head)
python experiments/route_emit_finetune.py --base checkpoints/sft_baked_code.pt \
    --save checkpoints/route_emit_code.pt
# 5. eval: selective thinking
python experiments/eval_humaneval.py --ckpt checkpoints/route_emit_code.pt \
    --prompt_style sft_comment --extract_code_block --min_emit_before_eos 10 \
    --use_thinking --generator latent_think --emit_threshold 0.3   # -> ~11/164
```

## Honest status / next

- **Solid:** the collapse was a gate-routing bug; routing + selectivity fixes it
  (0 → ~no-think) and gives a credible modest lift (→11/164), **reproduced via the
  canonical `eval_humaneval` harness** at emit_threshold 0.1 and 0.3, with a
  matched 0-think control (7/164) ruling out a decode-path artifact.
- **Magnitude NOT bulletproof.** The +3–4 is within the greedy ±2–3 noise band.
  Two attempts to confirm it independently both failed on **eval plumbing, not on
  the model**: (a) MBPP via `eval_broader` — no-think baseline itself scored 0/200
  (syntax_error 88%), so the harness/format is broken for this ckpt → inconclusive;
  (b) a per-problem flip script (`per_problem_flip.py`) — its ad-hoc grading
  reproduces 0/80 for no-think (vs 8/164 canonical), i.e. it doesn't replicate
  `eval_humaneval`'s exact decode→extract→grade pipeline → inconclusive. Lesson:
  trust the canonical harness; a clean magnitude confirmation needs the
  confirmation eval to **reuse `eval_humaneval.evaluate()` verbatim** (multi-sample
  or a properly-wired MBPP), which is a separate focused task.
- **To confirm (next):** multi-sample pass@1 *through* `evaluate()` (mind the gate's
  temperature fragility), or fix the MBPP harness to reuse the same grading.
- **Bigger lever for the headline** remains execution-grounded **RL** (grader-RL
  previously reached 16/164) and **scale** — latent thinking is a modest add-on
  here, and its primary, validated value is on **sequential-reasoning** tasks
  (pointer-chase / execution-trace eval: +0.47–0.78), not code generation.

See `memory/project_latent_thinking_real_model.md` for the full experimental
trail.
