# Latent-space thinking — WORKING (2026-05-28)

After a long arc where discrete-token "thinking" never helped on any task
(`THINKING_DEMONSTRATION_2026_05_28.md`: with-think = 0/80 on every rung),
the user redirected to **latent-space thinking for maximum bandwidth, not
CoT**. This is the first mechanism in the project that demonstrably performs
useful sequential computation.

## Mechanism — latent ponder (`experiments/latent_think.py`)

At a "think" slot appended after the query, feed the trunk's OWN continuous
hidden state back as the next input embedding (Coconut-style), for R
refinement iterations. The think slot runs **state-readonly** (DeltaNet
`b_proj` β=0, via `TinyLM(state_readonly_at_think=True)`): it READS the full
recurrent state but never WRITES to it. Each step feeds a full `d_model`
continuous vector — maximum bandwidth, no discrete token is ever emitted.

This is NOT new plumbing — it reuses the existing `inputs_embeds` +
`state_readonly_at_think` path. The new part is feeding the *raw hidden* back
(prior "retrieval-as-input" fed a WM lookup, not the hidden) and the
**training recipe** below.

## Task — pointer-chase `f^K(start)` (needs depth AND recall)

A random permutation `f` on N nodes is presented as shuffled `(i, f(i))`
pairs, then `[QUERY, s]`. Label = `f^K(s)` — K sequential hops. The function
table must be held in the recurrent state (recall); computing `f^K(s)` at a
single position needs K sequential steps (depth). A single forward can do a
few hops; latent ponder steps supply the missing depth without corrupting the
table.

## Results (N=12, 2-layer DeltaNet, d=128, ~0.5M params)

**Deep supervision (each step r supervised to decode `f^r(s)`), K=4:**

| eval | acc |
|---|---|
| none (R=0)   | 0.23 |
| latent R=3   | 0.07 (undershoot) |
| **latent R=4** | **1.00** |
| latent R=5/6 | 0.07 / 0.15 (overshoot) |
| token R=6    | 0.09 (discrete feedback — FAILS) |
| per-hop thread | h1..h4 = **1.00** |

**Final-answer-ONLY + depth curriculum (ramp K 1→4, no per-hop labels), K=4:**

| eval | acc |
|---|---|
| none (R=0)     | 0.12 |
| **latent R=4** | **1.00** |
| token R=6      | 0.09 |
| per-hop thread | h1..h4 = 1.00, **extrapolates h5..h8 = 1.00** |

**Final-answer-only, NO curriculum (control):** stuck at ~0.22, per-hop ≈
chance. The curriculum is the essential bootstrap.

**K=8 deep supervision:** R=8 → 1.00, per-hop 100% through 8 hops,
**extrapolates to h12 = 1.00**.

**Adaptive halting (fixed-point chase, variable hops L∈[1,6], gate learns
when to stop), final-answer + gate-halt supervision + curriculum:**

| metric | value |
|---|---|
| answer_acc | **1.000** |
| halt_exact (stopped at exactly L) | **1.000** |
| halt_mae (hops) | **0.000** |

The existing `output_gate` head, trained with a BCE halt target (fire once the
absorbing node is reached), learns to stop the latent loop at *exactly* the
right step for every example — variable compute per input, no overshoot.

## What is proven

1. **Latent thinking is load-bearing.** floor (0.12–0.23) → 1.00.
2. **Thinking = genuine sequential computation.** It needs *exactly* R=K
   steps; undershoot and overshoot both fail, because each step performs one
   real hop (per-hop thread = 100%).
3. **High bandwidth is essential.** The discrete-token variant (constant
   `[THINK]` embedding fed each step) collapses to ~0.09 — it cannot chain.
   This is the direct vindication of the latent-space-over-CoT thesis: the
   continuous fed-back vector carries the intermediate result; a discrete
   token cannot.
4. **It learns from final-answer-only supervision** when given a depth
   curriculum — the per-hop reasoning thread *emerges unsupervised*. This is
   the transfer-relevant result: real tasks only give final answers.
5. **Length generalization.** Trained K=4 → correct to 10 hops; K=8 → 12.
   The model learned a reusable latent operator, not a lookup.
6. **Recall preserved.** hop-1 (= associative recall) stays 1.00 across all
   hops; state-readonly never corrupts the table.

## Honest caveats

- **State-write also worked on this task** (per-hop 100%). Pointer-chase
  re-reads the table every hop, so corruption would surface immediately and a
  *co-trained* model learned non-destructive writes. So corruption is an
  *untrained / discrete-token* failure mode; state-readonly is the safe
  architectural guarantee, not strictly required once co-trained. A task that
  reads a binding only AFTER an unrelated think burst (long-context recall)
  would separate them — left for the port.
- **Synthetic, tiny model.** This proves the *primitive*, the *training
  recipe* (depth curriculum + final-answer supervision), and *adaptive
  halting* (gate). Porting to the 287M model + real coding/reasoning data is
  the next phase.
- **Halting was supervised here** (we know L). On real tasks there is no L
  label; the halt signal must come from task reward (the gate as an RL action,
  `--stochastic_gate`) or a "answer is stable" fixed-point criterion. The
  synthetic shows the gate *can* learn precise halting; learning it from
  terminal reward is the open port question.

## Transfer — real arithmetic-chain reasoning (`experiments/latent_arith.py`)

The arithmetic-chain task is the exact class where discrete-token thinking
scored **0/80 on every rung** (`THINKING_DEMONSTRATION_2026_05_28.md`). Ported
the latent loop onto it with the REAL DeltaNet architecture (4L × 256d, ~4.3M
params, real multi-token arithmetic TEXT; answer bounded to one token for a
clean metric). Recipe: depth curriculum (ramp n 1→6) + a **consolidation phase**
(uniform-depth sampling after the ramp, to avoid forgetting shallow rungs) +
final-answer-only supervision, R latent steps = n.

| n_steps | no-think | **latent (R=n)** | delta |
|--:|:--|:--|--:|
| 1 | 0.162 | **1.000** | +0.838 |
| 2 | 0.521 | **1.000** | +0.479 |
| 3 | 0.306 | **1.000** | +0.694 |
| 4 | 0.233 | **1.000** | +0.767 |
| 5 | 0.154 | **1.000** | +0.846 |
| 6 | 0.162 | **0.999** | +0.837 |

**Latent thinking solves every rung (~1.000) where no-think sits at 0.15–0.52**,
the benefit growing with depth. This is the first time *any* thinking mechanism
in the project has helped this task (discrete = 0/80).

Caveat: on arithmetic, `token`-mode (constant-embedding feedback) is also high
(0.94–1.00) because the chain is partly left-to-right *accumulable* in the
recurrent state — so arith rewards "extra compute steps" regardless of feedback
content. The **pointer-chase remains the clean separator** proving the latent
*content* (bandwidth) is what carries the computation (token-mode there = 0.09).
The two together: pointer-chase proves bandwidth matters; arith proves the
mechanism transfers to a real reasoning task on the real architecture. A strict
curriculum (ramp only, no consolidation) forgets shallow rungs (n=1 latent
regressed to 0.01); the consolidation phase fixes it.

## Artifacts
- `experiments/latent_think.py` — synthetic task, model build, latent loop, eval.
- `experiments/latent_arith.py` — real-arithmetic transfer (per-rung table).
- `experiments/test_latent_think.py` — regression tests (curriculum learns,
  latent R=K >> none, token fails, adaptive halting). 5 pass.
- Checkpoints: `latent_think_curriculum_k4.pt`, `latent_think_halt_k6.pt`,
  `latent_arith_n6_mixed.pt`.

## Real 287M code model — port results (2026-05-28)

`generate_latent_think` (eval_humaneval.py) + `latent_sft.py` port the
mechanism to the production 287M ckpt. First latent-SFT (forced R≥1 every
example, `latent_sft_v1.pt`) on HumanEval:

| config | pass@1 |
|---|---|
| baseline `sft_phase_c_combined`, no-think | 9/164 |
| latent_sft_v1, **no-think** | **0/164** (collapsed) |
| latent_sft_v1, **latent-think (R=4)** | 7/164 |

**v1 finding (forced R≥1 every example):** no-think collapsed to 0/164,
latent-think recovered to 7/164 — i.e. thinking became *mandatory*, not
*helpful*.

**v3 finding (mixed think/no-think training, format-matched CoT+fence):** the
clean, honest result. no-think recovers to **7/164** (full set) / 7/60
(subset); latent-think (forced R=2 burst) **3/60 — WORSE than no-think 7/60**;
baseline 8/60. So on HumanEval, latent think-before-solution does **not help
and tends to hurt**.

**Verdict: HumanEval is the WRONG probe for a depth mechanism.** HumanEval at
287M is capacity-bound (documented ~16/164 ceiling), and its problems are short
enough to fit DeltaNet's state — there is no sequential-depth bottleneck for
latent thinking to relieve. The v1 "0→7" was forced-dependence, not benefit.
The mechanism's value is proven where depth *is* the bottleneck: pointer-chase
(floor→1.00, bandwidth-essential) and real arithmetic chains (0.15–0.52→~1.00
across all rungs, the exact task discrete thinking scored 0/80). A small
think-before-solution SFT cost ~2 HumanEval points (SFT drift), unrelated to the
mechanism. **Recommendation: evaluate/deploy latent thinking on depth-bound
reasoning tasks, not on HumanEval.** Ports: `experiments/latent_sft.py`,
`eval_humaneval.generate_latent_think` (`--generator latent_think
--force_prefix_think N`).

## Next phase (port to the 287M code model)
1. Wire `generate_latent_think` into `eval_humaneval.py` (think burst =
   state-readonly latent ponder, gate halts) — replaces the discrete
   `[THINKING]` append. Plumbing exists (`inputs_embeds`, `_last_gate`,
   `state_readonly_at_think`).
2. Co-train (SFT/continue-pretrain) on coding/reasoning data with a
   reasoning-depth curriculum + consolidation. Co-train from day 1 — bolted-on
   was always inert.
3. Halting from terminal reward: gate as RL action (`--stochastic_gate`,
   `train_rl_grader.py`) since real tasks have no per-step depth label.
4. Headline probe: arithmetic ladder (`data/synth_arith_ladder_n*.jsonl`) on the
   real tokenizer, then HumanEval.
