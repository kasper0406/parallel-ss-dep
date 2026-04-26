# 07 — Distillation Strategies for Coding LLMs

The 2024–2026 literature has converged on a clear pattern: linear-RNN students cannot
match softmax teachers from-scratch at small scale, but **cross-architecture distillation
closes 70–95% of the gap with <1% of pre-training tokens**. For our hybrid (rotation+delta)
on 2× RTX 5090, distillation is the highest-leverage move available — and arguably the
only one with a clear path to a useful coding model on this hardware.

## Top 4 concrete ideas

### Idea 1: HALO/HypeNet-style hybrid distillation from Qwen2.5-Coder-1.5B
- **What it is:** Replace a fraction of attention layers in a pretrained code-LLM with our
  rotation+delta block; train via blockwise hidden-state alignment (L2) → KL on logits →
  long-context fine-tune. Keep ~25% softmax layers (the ones HALO's attention-selection
  metric flags as load-bearing for long-range copy/induction).
- **Teacher / student / objective:** Qwen2.5-Coder-1.5B (or DeepSeek-Coder-1.3B) → hybrid
  rotation+delta + 25% kept attention; loss = L2(hidden) + KL(logits) + SFT.
- **Key papers:** HALO/HypeNet (Hybrid Linear Attention Done Right, 2026); MOHAWK
  (Goomba Lab, 2024); Mamba-in-Llama (NeurIPS 2024).
- **Implementation cost:** M.
- **Compute estimate:** HALO reports 2.3B tokens for Qwen3 conversion; for 1.5B class on
  2× 5090 with bf16 + grad checkpointing that's ~4–7 GPU-days. Phase 1 (hidden-state
  alignment) is per-layer and embarrassingly parallel.
- **Code-LLM relevance:** **High.** Teacher is already a SOTA coder; distillation
  preserves its FIM/repo training. Hybrid keeps the few attention layers where exact copy
  matters (function names, identifiers, brace matching).
- **Risk:** Generation quality can lag perplexity by 20pp ("When Perplexity Lies", 2026)
  — must validate on HumanEval/MBPP autoregressively, not via log-likelihood ranking.

### Idea 2: LoLCATs-style two-stage attention transfer + LoRA SFT
- **What it is:** (a) Train rotation+delta block in isolation to match per-layer softmax
  attention output (MSE on layer outputs, teacher frozen). (b) LoRA fine-tune QKVO + new
  block params end-to-end on code-instruct data.
- **Teacher / student / objective:** Qwen2.5-Coder-1.5B → fully linearized variant; MSE
  on attention output, then standard LoRA SFT.
- **Key papers:** LoLCATs (Hazy Research / arXiv 2410.10254, 2024); Hedgehog (ICLR
  2024); RADLADS (2025).
- **Implementation cost:** S.
- **Compute estimate:** LoLCATs linearized Llama-3-8B in 5 hours on a single A100 with
  40M tokens. For 1.5B on 2× 5090: <1 GPU-day for stage 1, ~2 GPU-days for SFT. **Cheapest
  meaningful experiment in this brief.**
- **Code-LLM relevance:** Medium. Pure linearization tends to drop a few HumanEval points
  vs. teacher; works best as a quick sanity check that our cell can absorb softmax
  behavior. Use this to validate the rotation+delta block is even capable before
  committing to bigger distillation runs.
- **Risk:** Pure linear (no kept attention) reliably loses on hard recall/copy — 1.5B
  coder distillates without a softmax fallback rarely hold up on >2k context tasks.

### Idea 3: On-policy distillation with sequence-level teacher correction
- **What it is:** Roll out from the (already-distilled, Idea 1 or 2) student on coding
  prompts; teacher provides per-token reverse-KL feedback on the student's actual
  trajectories. Only one teacher forward pass per token.
- **Teacher / student / objective:** Qwen2.5-Coder-1.5B (or 7B if memory permits via
  offload) → distilled hybrid; reverse-KL on student rollouts.
- **Key papers:** Thinking Machines Lab "On-Policy Distillation" (2025); MiniLLM (Gu
  et al. 2024); GKD (Agarwal et al. 2024).
- **Implementation cost:** M (need rollout infra + teacher serving on the second 5090).
- **Compute estimate:** Thinking Machines reports 9–30× compute efficiency over RL.
  Realistically 2–4 GPU-days on top of Idea 1 for meaningful coding gains. Pin teacher
  to GPU 1, student training on GPU 0.
- **Code-LLM relevance:** **High.** Code is exactly the regime ("specialized models for
  math, code, RAG, tools") that Thinking Machines explicitly recommends this for.
  Addresses exposure bias which kills naive KD on long autoregressive generations like
  function bodies.
- **Risk:** Requires Idea 1/2 first — student must already be a competent draft. Cold-start
  on-policy diverges.

### Idea 4: Phi-style synthetic textbook generation from teacher → from-scratch student
- **What it is:** Use Qwen2.5-Coder-7B (or larger via API briefly) to generate a synthetic
  "code textbook" — explained snippets, pedagogical exercises, FIM examples — then
  pretrain our hybrid from scratch on this distilled curriculum.
- **Teacher / student / objective:** Strong coder LM → synthetic data → from-scratch
  hybrid; standard CE loss.
- **Key papers:** Phi-1 / "Textbooks Are All You Need" (2023); OpenCodeInstruct (2025);
  OpenCodeReasoning (2025).
- **Implementation cost:** L (data pipeline).
- **Compute estimate:** Data gen is the bottleneck — ~1B-token textbook costs $300–800 of
  open-weights inference on rented H100s, or 2–3 GPU-weeks on our 5090s. Pretrain ~5–10
  GPU-days.
- **Code-LLM relevance:** Medium. Phi-1 hit 50.6% HumanEval at 1.3B from this recipe,
  but it required GPT-3.5 quality. Open coders can do it but the data takes care.
- **Risk:** Skips the teacher's representations entirely; we waste the strongest signal
  available. Only worth it if architectural distillation fails (e.g., rotation+delta
  fundamentally can't absorb softmax).

## Recommendation

**Run Idea 2 (LoLCATs-style, ~3 GPU-days) first as a smoke test**, then commit to
**Idea 1 + Idea 3 stacked** as the main bet. Concrete plan:

1. **Week 1:** LoLCATs single-layer attention-transfer on Qwen2.5-Coder-1.5B with our
   rotation+delta cell. Goal: confirm cell can fit softmax attention with MSE < some
   threshold per-layer. If MSE blows up, we have an architectural problem distillation
   won't fix.
2. **Week 2–3:** HALO-style staged distillation, keep ~6 of 28 attention layers (placement
   per HALO's selection metric), 2–3B tokens of the-stack-v2 Python + OpenCodeInstruct.
   Evaluate generation-side (HumanEval, MBPP) per "When Perplexity Lies".
3. **Week 4:** On-policy distillation with reverse-KL on student rollouts of programming
   prompts. Teacher pinned to GPU 1.

**Honest compute estimate to a useful coder on 2× 5090:** ~3 GPU-weeks total (~10–15
GPU-days of the 5090 pair). Comparable budget got LoLCATs/HALO real wins on much larger
teachers — at 1.5B class we should be able to land within 2–4pp of the teacher on
HumanEval/MBPP if the cell is sound.

**Skip Idea 4 for now.** Synthetic textbook is heroic-effort; we'd be reinventing what
distillation gives almost free.

**Teacher choice:** Qwen2.5-Coder-1.5B over DeepSeek-Coder-1.3B — better HumanEval
(43.8) at the size, recent-enough, open weights, well-supported in fla-friendly tooling.
Don't go to 7B teacher unless we add CPU-offload — teacher memory matters because we
need the second 5090 free for student rollouts in Idea 3.

## Open questions

- Does our rotation+delta cell admit a per-layer "attention-output match" objective at
  all? (The rotation parameterization may not span the softmax output manifold for
  arbitrary keys/values — LoLCATs phase 1 will tell us.)
- Should we keep the teacher's RoPE or use HALO's HyPE for hybrid position encoding?
  HyPE is designed exactly for hybrid length-generalization — likely worth porting.
- The kill-gate parity result (heisenberg vs linear, +32.6pp end-tok at T=64) — is the
  rotation+delta cell or a heisenberg-augmented variant the right student? Re-run the
  parity study but with a teacher signal to see if heisenberg's advantage compounds or
  collapses under distillation.
- Code-eval set: we should add SWE-bench-lite or LiveCodeBench-Lite alongside
  HumanEval/MBPP; the latter saturate fast and won't show distillation deltas at our
  size.
