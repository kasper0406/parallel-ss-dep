# Direction: Continuous RAG & The Thinking Head

This document outlines the mechanistic rationale and roadmap for the "Thinking Head" architecture, a discrete gated path that enables Linear RNNs (like DeltaNet) to perform Adaptive Computation Time (ACT) and sets the foundation for **Continuous RAG**.

## 1. The Core Problem: Sequence Expansion in Transformers
Existing research on "Pause Tokens" or "Thinking" in standard Transformers relies on extending the sequence length by appending dummy tokens (e.g., `<pause>`). This has two major drawbacks:
1. **KV-Cache Bloat:** Every "thought" step increases the KV-cache size, which scales quadratically (or linearly at best with GQA), leading to memory-bound inference bottlenecks.
2. **Context Fragmentation:** Long "thinking" trajectories can displace actual relevant prompt context from the model's finite window.

## 2. The Solution: In-Place Refinement with Linear RNNs
Because this project uses **Linear RNN backbones (DeltaNet)**, the hidden state is a fixed-size matrix ($c_t$). This allows for a fundamentally more efficient "thinking" mechanism:
- **Zero Sequence Expansion:** A "thinking" step is simply a recurrent update to the existing state matrix. The context length does *not* increase.
- **In-Place Refinement:** The model uses its weights to refine the hidden representation iteratively. It "chews" on a token for multiple cycles before emitting an output.
- **Mechanism:** A binary **Thinking Gate** (d_model → 1) outputs a probability $g_t$. If $g_t$ exceeds a threshold, the model executes a "thought pass" (re-running the same hidden state through the layers) instead of generating a token.

## 3. Systems-Level Training: The Thinking Queue
Training variable-length ACT is notoriously difficult to batch on GPUs. We solve this using an asynchronous **`ThinkContinuationQueue`**:
- **Decoupled Resolution:** When the gate triggers "think," the current context and hidden state are pushed to a CPU-backed queue.
- **High Saturation:** The training loop pulls batches from this queue to resolve thinking trajectories in parallel with fresh data, ensuring the GPU stays fully saturated despite the varying lengths of "thought" per token.

## 4. The Roadmap: Continuous RAG
The "Thinking Head" is the bridge to **Continuous RAG**. 
- **Current State:** A thinking step updates the $c_t$ state based purely on internal weights.
- **Target State:** A "thinking" step can trigger an **External Retrieval**. The retrieved facts (represented as key-value embeddings) are projected directly into the DeltaNet state matrix.
- **Benefit:** This completely bypasses the need to prepend large amounts of RAG context to the prompt. The model can literally "pause," load external function definitions or facts into its state, and then resume generation with an updated, fact-aware memory.

## 5. Temporal Sparse Feedback
This direction is the temporal extension of our finding that **Sparse Far-Distance Feedback** (e.g., Layer 28 -> Layer 2) provides significant PPL gains. While sparse feedback routes high-level context back through *space* (depth), the Thinking Head routes it back through *time* (recurrence).
