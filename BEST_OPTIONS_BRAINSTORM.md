# Best options to make THIS model punch way above its weight

> Strategy synthesis, 2026-06-21. Ruthlessly honest, ranked, falsifiable. Cites
> our own results by name. Speculation is flagged. NOT a "scale up + add tools"
> doc — the point is the one or two moves that have a real chance of an
> impressive, *defensible* small-model result on 2×RTX5090.

## The one-paragraph thesis

Our binding constraint is **tokens, not parameters or mechanisms**
(`project_undertrained_not_undercapacity`: SmolLM2-135M, *half* our params, beats
our 287M on HumanEval-solution CE 0.798 vs 0.969 purely because it saw ~400–800×
more tokens). We cannot out-token SmolLM2/Qwen on this rig. KD from a
*far-bigger* teacher can't buy the gap back (`project_kd_distillation_pipeline`:
capacity-gap, KD *hurt* every source). So the only way to get a competent base
is to **inherit someone else's trillions of tokens** — and the only way to do
that *while keeping our architectural edge* (bounded-state O(1) decode) is to
**linearize a strong same-size open model into our DeltaNet backbone**. That single
move converts our biggest weakness (token-poverty) into a non-issue, and finally
makes the rest of the stack pay off — because mechanisms and the cost-moat are
*worthless on a weak base* and `project_undertrained_not_undercapacity`'s capstone
shows **mechanism benefit GROWS monotonically with base competence**.

---

## A. The achievable axes of "above its weight" — honest ratings

| Axis | Reachable for us? | Verdict |
|---|---|---|
| **(i) Broad capability vs big LLMs** | **No — from scratch.** Token-bound, conceded (`project_undertrained_not_undercapacity`, `INTELLIGENT_AGENT_BRAINSTORM`). **Maybe — by INHERITANCE** (linearize a strong base; see C). | Concede the from-scratch version; the inheritance version is the whole bet. |
| **(ii) COST / latency / efficiency at long horizon** | **Yes — our real structural edge.** Bounded state ⇒ O(1)/token decode & constant memory vs a transformer's O(T) KV cache; rolls past any fixed window (`FLAGSHIP_PROBE_RESULTS`: 17%@8k where a 2048-window transformer is 0%). | **Our defensible moat.** Engineering-gated (`forward_step` unwired) and only valuable on a competent base. |
| **(iii) Narrow-domain saturation** | **Partially.** Phi-1 playbook: saturate a bounded vertical within budget. Hard part is finding a vertical where big general models are *genuinely* weak AND we can source/generate enough quality tokens. | Real but pond-selection-dependent; best as a *complement* to inheritance, not standalone. |
| **(iv) On-device / private deployment** | Yes, but it's a *packaging/GTM* axis, not an *impressiveness* axis. Enabled by (ii). | Keep as a downstream benefit, not the headline. |
| **(v) Controllability / own-the-weights** | Yes — but it's an *experimentation* moat (we can run the experiment), not itself an impressive result (`INTELLIGENT_AGENT_BRAINSTORM` A5). | Enabling, not sufficient. |

**Best shot at a genuinely impressive + defensible result:** the **fusion of (i)-by-inheritance and (ii)**. The flagship claim that is both true and hard to copy:
*"a linear-RNN small coder carrying frontier-small-model knowledge, decoding at
constant cost/memory over 100k-token agentic sessions where the transformer
baseline is multiples more expensive or blind past its window — at equal task
quality."* A transformer lab would have to **switch backbones** to match it. That
is a structural moat; raw HumanEval pass@1 is not (`project_humaneval_config_artifact`:
the 8–16/164 band is eval noise).

---

## B. Full option survey, scored (leverage × feasibility, 1–5 each)

### KEEP / PURSUE

**B1. Linearize a strong same-size open model into our DeltaNet backbone.** **Lev 5 · Feas 3 = TOP.**
Init a DeltaNet+mechanisms model from SmolLM2-360M (or Qwen2.5-Coder-0.5B/1.5B):
transfer MLPs / embeddings / LayerNorms directly, replace self-attention sublayers
with DeltaNet layers, and run cheap **attention-transfer distillation** (MOHAWK /
Mamba-in-Llama / LoLCATs / SUPRA style — layerwise output-matching then short
end-to-end CE) on ~0.1–1B code tokens. This is the *explicit* answer to the
prompt's "continue-pretrain/init from an open small model; is a transformer
backbone better?" question: **the better vehicle is a *linearized* transformer =
our backbone that inherited a transformer's tokens.** Crucially it is **NOT** the
refuted whole-corpus logit-KD: teacher ≈ student in *size* (no capacity-gap KL
drag — the exact failure mode in `project_kd_distillation_pipeline`), and the
+3–5% iso-token result from the *small-gap* SmolLM2-1.7B teacher
(`project_undertrained_not_undercapacity`) is the in-house signal that same-size
distillation transfers. Output is still a DeltaNet ⇒ O(1) decode + all mechanisms
attach unchanged. *Speculation flag:* published mostly for plain/gated linear
attention & Mamba; **delta-rule as the conversion target is less established** —
this is the core technical risk and the thing the first experiment must de-risk.

**B2. Wire `forward_step` (true O(1) decode) + long-horizon agent harness + append-only kNN session store.** **Lev 4 · Feas 3.**
Realizes the cost moat (axis ii). `INTELLIGENT_AGENT_BRAINSTORM` Bet 1. Worthless
on a weak base, so **sequence it after B1**. The kNN store recovers the lossy
recall of bounded state for "what did I literally see this session" (near-exact
recall is exactly right here, even though it failed to *generalize* for HumanEval).

**B3. Data-/sequence-level distillation = use the teacher as a clean-data factory.** **Lev 3 · Feas 5.**
Generate teacher *completions* (not logits) on our domain and CE-train on them.
Sidesteps the capacity-gap KL drag (it's just curated targets). Cheap, complements
B1/narrow-domain. NB: the *latent-CoT* variant is dead (`project_kd_distillation_pipeline`:
coding corpus is recall-bound, CoT *hurts* the teacher's own answer) — but plain
code-completion data distillation is just good SFT data.

**B4. Aggressive data quality on a narrow code vertical (dedup + exec-filter + FIM + tail-upsample).** **Lev 3 · Feas 3.**
The Phi-1 lever. `project_code_thinking_ceiling`: rare algorithms get <100 gradient
exposures and never lock in; SFT corpus ~6% broken / ~70% unverified. Narrowing
the target makes ~5–50B tokens go further. Best as the *specialization* phase on
top of B1, after picking a pond where big models are actually weak.

**B5. Strategy-RL in a curricularized verifiable environment (RL on tool/memory/think USE).** **Lev 3 · Feas 4.**
`train_rl_grader.py` validated (8→16/164, noisy). RL is **elicitation, not
acquisition** — bounded by base pass@N. A *multiplier* on a good base (B1), not a
base-builder. Use multi-turn agentic reward; keep KL-to-ref (v1 collapsed without it).

**B6. Mechanisms (PKM/WM/latent) — keep, but demonstrate on matched probes, revisit as headline AFTER B1.** **Lev 2 now · Lev 4 post-B1.**
`project_why_mechanisms_synthesis`: ~free at iso-token (definitive ablation:
MECH vs LEAN tied), load-bearing *only* on matched bottlenecks (WM→long-range
recall, latent→sequential depth, PKM→rare-fact tail). Capstone: benefit grows
with base competence ⇒ they are most likely to finally show up as a *big* lever
on a B1 base, not a from-scratch 287M.

### KILL (do not re-propose as open)

- **Whole-corpus logit-KD from a far-bigger teacher** — refuted, capacity gap (`project_kd_distillation_pipeline`).
- **latent-CoT / reasoning distillation on the coding corpus** — recall-bound, 4× confirmed.
- **1B-from-scratch (via KD or not)** — *more* undertrained than 287M at our budget.
- **Mechanism micro-optimization as the headline** — free riders; orthogonal to short-context HumanEval.
- **WM as a HumanEval lever / "non-decaying unbounded recall"** — WM persistently inert; recurrence does the recall; flat-recall fix is generic RAG (`project_agent_economics_flagship`).
- **Stateful chunking, embedding-optimizer** — NO-GO (`stateful_chunking_notworth`, `embed_optimizer_negative`). (Per-head-NS preconditioner is a free ~7–13% iso-step speedup — keep, but it's not a headline.)

---

## C. The single highest-EV path

**LINEARIZE a strong small open code model into our DeltaNet+mechanisms backbone,
then attach the O(1) cost-moat (B2) + verifier/RL (B5).** (B1 → B2 → B5.)

**Why this and not the others.** It is the *only* move that breaks the proven
binding constraint (tokens) **without needing tokens we can't get** — it inherits
SmolLM2/Qwen's 2–4T-token competence with a ~0.1–1B-token *conversion* budget that
fits the rig. It is also the only move that makes everything else we've built pay
off: O(1) decode (axis ii) is worthless on a weak base; mechanism benefit *grows*
with base competence (`project_undertrained_not_undercapacity` capstone). And the
result is **structurally unique** — a competent linear-RNN coder that a transformer
lab can't cheaply copy. Every other option is either bounded by token-poverty
(B3/B4 from scratch), a multiplier needing a good base first (B5/B6), or pure
plumbing (B2 alone).

### First cheap experiment (days, single GPU) — the linearization feasibility probe

- **Setup.** Take **SmolLM2-360M** (transformer; *matched 49152 tokenizer = clean
  weight inheritance*, no embedding tax). Build a DeltaNet with matched
  `d_model`/`n_layers`; copy MLP + embedding + LayerNorm weights; replace each
  attention sublayer with a DeltaNet layer. Run **layerwise attention-transfer
  distillation** (match each block's output to the original transformer's block
  output, MOHAWK-style) on ~100–500M tokens of code, then a short end-to-end CE
  finetune. Single GPU, ~days.
- **The metric that already exposed the gap.** Teacher-forced **HumanEval-solution
  CE** (the exact probe from `project_undertrained_not_undercapacity`: SmolLM2-360M
  = **0.614**, ours from-scratch = **0.969**). Does the linearized DeltaNet land
  near **0.614** (inheritance worked) or near **0.969** (lost it all)?
- **FAIR CONTROL** (mandatory, `feedback_fair_baselines`): our existing Phase-C
  from-scratch base at an **equal post-conversion token budget** (same ~0.1–1B
  tokens of the same code), so the comparison isolates "inherited weights +
  attention-transfer" from "just more training."
- **Falsifiable success:** linearized CE within **~0.1 of 0.614** after **<1B
  tokens** ⇒ inheritance transfers to delta-rule ⇒ green-light B1 at full scale.
  **Kill:** CE collapses toward from-scratch 0.969 ⇒ delta-rule can't absorb
  attention cheaply ⇒ fall back to **narrow-domain from-scratch (B4) + cost-moat
  (B2)** and concede axis (i).
- **Free secondary read:** the output is a DeltaNet ⇒ confirm the O(1)-decode /
  bounded-state property is preserved by construction (it is), so a *win here*
  immediately also delivers the cost-moat substrate.

---

## D. Honest concessions & the biggest risk

**What we cannot win:**
1. **Broad open-domain capability from scratch** — token-bound, conceded.
2. **Beating frontier models on raw HumanEval/MBPP pass@1** — wrong probe; the
   8–16/164 band is eval noise (`project_humaneval_config_artifact`); coding
   benchmarks are recall-bound and recall ⇒ knowledge ⇒ tokens.
3. **A capability moat from memory** — WM is persistently inert; recall is the
   recurrence; the flat-recall fix is generic RAG. Our long-horizon edge is
   **cost**, not a recall others can't match.
4. **Online weight-learning during a task** — catastrophic forgetting unsolved;
   DDP+latent friction halves 2-GPU throughput (`project_ddp_latent_incompatibility`). Park it.

**The biggest risk to the recommended path (B1):** *delta-rule may not absorb
attention cheaply.* Linearization is published mainly for plain/gated linear
attention and Mamba; DeltaNet as the conversion target is **under-validated**
(speculation flag). Two failure modes: (a) the converted model loses too much of
the inherited knowledge (back to token-poverty), or (b) it keeps the knowledge but
a fixed-size linear-RNN structurally caps long-range associative recall vs full
attention — i.e. we buy the cost-moat with a *recall haircut* that may erase the
inherited edge on exactly the long-context tasks we wanted to win. The
feasibility probe above is engineered to surface both within days and cheaply,
before any full-scale commitment. Secondary risk: SmolLM2-360M is a *general*
model (weaker on code than Qwen-Coder); if its inherited code competence is
insufficient, the stronger inheritance target is **Qwen2.5-Coder-0.5B/1.5B** at
the cost of the 151k-vocab embedding tax (`project_kd_distillation_pipeline`) —
a tradeoff to run only if the SmolLM2 probe is promising-but-short.
