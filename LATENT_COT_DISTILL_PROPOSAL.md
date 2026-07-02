# Latent-CoT Distillation — design proposal (2026-06-20)

**The idea (user, 2026-06-20):** the teacher (Qwen 3.6) reasons in *explicit CoT
tokens*; our student reasons in *continuous latent steps*. Don't make the student
emit the teacher's CoT verbatim (we proved discrete CoT barely helps a small model
— `project_latent_thinking_code_gen`). Instead, **distill the teacher's CoT
*computation* into the student's latent steps**: train the latent steps to reproduce
the *effect* the CoT has on the teacher's answer distribution.

## Core mechanism

Teacher produces, for problem P: chain-of-thought C, then code/answer A. The key
asymmetry we exploit:

  P_teacher(A | P, C)   >>   P_teacher(A | P)        (CoT sharpens the answer)

The student has no C — it has R latent steps. Objective:

  **KD: P_student(A | P, [R latent steps]) → P_teacher(A | P, C)**

i.e. after the prompt P, the student runs a think-burst of R **state-readonly latent
steps** (no discrete tokens emitted), then predicts the answer tokens A — and is
KD-supervised to match the teacher's *CoT-conditioned* per-token distribution on A.
The latent steps are thereby trained to do, in continuous space, whatever the CoT did
to improve the answer. **This is "compress CoT into latent."**

Why condition the teacher target on C (not just P): if we KD'd to P_teacher(A | P)
(no CoT), the latent steps would have nothing to learn beyond the no-think path. The
CoT-conditioned target is what gives the latent steps a reasoning signal to absorb.

## Why this is the right fix (not a new gamble)

`project_latent_thinking_code_gen` already established the fix for "latent hurts code"
is **on-code adapter co-training** (the reasoning-trained adapter is OOD for code;
train it ON code → flips net-negative → net-≥0). Latent-CoT distillation IS that
on-code co-train — **supercharged**: instead of CE on the lone true-next code token
(weak signal), the target is the teacher's full CoT-conditioned distribution over the
whole answer (rich, dense signal). So it directly attacks the documented OOD problem
with a far stronger objective, and it's the latent-thinking analog of the #101 token-KD
efficiency lever.

## Build (reuses existing infra)

- **Data:** `distill_solutions.py` already emits `(problem_prompt, qwen_completion)`
  where qwen_completion = CoT + code. Split into C (reasoning) and A (code) — the
  script already `extract_code_block`s A. Need the teacher's per-token logits on A
  *conditioned on [P, C]* — generate once (teacher forward over [P,C,A], keep top-k
  logits on the A span). (Storage: top-k ~ 64 per A token.)
- **Student loss:** generalize `thinking.latent_cotrain_loss` — currently CE on the
  post-R-latent prediction of true-next; add a `teacher_logits` path doing
  T²·KL(teacher_topk ‖ student) on the A span after the R latent steps (same KD math
  as #101, same vocab-slice + shared-tokenizer requirement).
- **R curriculum:** scale R with teacher CoT length (longer reasoning → more latent
  steps), or fixed R with a ramp. Start easy (short-CoT problems) to bootstrap.
- **State-readonly latent** (β=0) so the latent steps don't corrupt the recurrent
  state carrying P (the validated [[project_pareto_safe_thinking]] property).
- **Compose with #101:** token-KD over the general corpus (no-think path) + latent-CoT
  KD on problem→answer pairs. Both use the same teacher; both need shared vocab.

## Gating / order (do NOT jump ahead)

1. **#101 token-KD A/B must validate first** (does KD buy token-efficiency at all). RUNNING.
2. Resolve the **tokenizer fork** (Qwen ≠ SmolLM2): either adopt Qwen's tokenizer for
   the distilled model, or shared-tokenizer teacher (SmolLM2-1.7B) for the prototype.
   For latent-CoT we want a *strong reasoner* teacher (Qwen) — its CoT is the value —
   so this likely forces the Qwen-tokenizer decision. Prototype can still use a
   shared-tokenizer reasoner if one exists; otherwise this is the production gate.
3. **Prototype latent-CoT on a small scale** (a few k problems, frozen trunk, adapter-
   only) and measure with the FAIR control from [[project_latent_thinking_real_model]]:
   post-latent answer CE/pass@1 vs a fully-trained 100%-no-think control — NOT the
   within-model R=0 path (which inflated v13's numbers).

## Risks / honest caveats

- The CoT may carry more info than R latent steps can absorb → partial matching;
  still expected to help (absorbs *some* of the CoT effect). Curriculum mitigates.
- Needs the teacher's CoT-conditioned logits on A → teacher forward + top-k storage;
  non-trivial data-gen compute (but one-time, and seeded by our datasets — the
  "datasets guide distillation" point).
- Same tokenizer constraint as all logit-KD.
- Success metric is the proper fair control, not the inflated within-model baseline.

## Status
Design only. Gated on #101 (token-KD A/B, running) + tokenizer decision. Tracked as
task #102. Connects: [[project_latent_thinking_code_gen]] (the on-code fix this
supercharges), [[project_undertrained_not_undercapacity]] (distillation = the
token-efficiency lever), [[project_pareto_safe_thinking]] (state-readonly floor).
