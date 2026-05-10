# GEMINI.md - Project Instructions

## Project Context
`parallel-ss-dep` is a research project focused on scaling **state-dependent, parallelizable RNN cells** for language modeling, specifically targeting coding tasks.

## Core Research Direction: Continuous RAG & The Thinking Head
The project has pivoted toward the **Thinking Head** architecture:
1. **The Thinking Head:** A discrete gate that triggers recurrent in-place state updates to the DeltaNet matrix ($c_t$). This enables Adaptive Computation Time (ACT) without increasing context length or KV-cache size—a major advantage over Transformer-based pause tokens.
2. **Continuous RAG:** The thinking steps are designed to eventually trigger external vector-database lookups, projecting retrieved facts directly into the RNN's persistent state matrix.

## Key Constraints & Mandates
- **Backbone:** Always prefer **Linear RNNs** (DeltaNet / DeltaProduct) over standard Softmax Attention.
- **Sparse Feedback:** Maintain the **sparse (2, 28) FiLM feedback** connection as it is a proven architectural win for routing high-level context.
- **Training Loops:** Use the **asynchronous `ThinkContinuationQueue`** for training thinking passes to ensure GPU saturation.
- **Normalization:** Use **`aux_items`** (actual item count) rather than `fresh_tokens` (batch size × T) for normalizing auxiliary losses in the thinking head to stabilize curriculum learning.
- **Memory Management:** For "thinking" experiments with `safety_max_depth > 2`, always enable **`--think_checkpointing`**. This uses activation checkpointing for continuation passes, which is essential to prevent OOM errors during the deep-recurrence phases of the curriculum.

## Documentation Index
- `README.md`: High-level overview and headline findings.
- `PLAN.md`: Active and future research milestones.
- `THINKING_RAG_DIRECTION.md`: Detailed mechanistic rationale for the Thinking Head.
- `NEXT_DIRECTIONS.md`: Strategic roadmap and recent pivots.
- `SESSION_FINDINGS.md`: Chronological log of empirical results.
