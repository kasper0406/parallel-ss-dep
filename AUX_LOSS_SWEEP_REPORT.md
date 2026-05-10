# Report: Auxiliary Loss Normalization Sweep (Phase 23)

## Objective
To stabilize the "Thinking Head" curriculum by comparing two auxiliary loss normalization strategies:
1.  **`fresh_tokens` (Control):** Queued aux loss is divided by the total token budget ($Batch \times T$).
2.  **`aux_items` (Experimental):** Queued aux loss is divided only by the actual number of continuation/replay items, with varying scales (0.25, 0.5, 1.0).

## Experimental Setup
- **Base:** 217M DeltaNet, sparse (2,28) FiLM feedback.
- **Data:** 16,000 segments (40% of baseline token budget).
- **Curriculum:** Depth 1→5, $\lambda$ 0.5→0.1, threshold 0.05→0.5.

## Results
| Run ID | Normalize | Scale | VAL PPL | Think Rate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Control** | `fresh_tokens` | 1.0 | **81.22** | 0.4% | Finished |
| **Scale 0.25** | `aux_items` | 0.25 | **156.73** | 5.9% | Finished |
| **Scale 0.5** | `aux_items` | 0.5 | **185.97** | 5.9% | Finished |
| **Scale 1.0** | `aux_items` | 1.0 | ~244 (est) | 5.9% | Terminated |

## Key Findings

### 1. The "Maladaptive Thinking" Trap
The sweep successfully "uncorked" the thinking head. By using `aux_items` normalization, we removed the damping effect of the total token budget, allowing the model to think **15x more frequently** (from 0.4% to 5.9%).

However, this increase in thinking was **negatively correlated** with model quality. As the model thought more, its perplexity nearly doubled. The model essentially learned "daydreaming"—it optimized the auxiliary loss (matching "thought" states to target states) in a way that corrupted the hidden state's ability to predict the next token.

### 2. Supervised BPTT Failure
The results prove that supervised auxiliary losses for recurrent thinking are extremely fragile:
- **Under-optimized (`fresh_tokens`):** The model ignores the thinking path because the gradient is too dilute.
- **Over-optimized (`aux_items`):** The model prioritizes the thinking proxy task over the language modeling task.

### 3. Data Gap
The baseline of ~51 PPL was trained on 2.5x more data. The Control's 81.22 PPL is consistent with its lower token budget, confirming that supervised thinking provided no "learning efficiency" shortcut.

## Strategy Pivot: Reinforcement Learning (GRPO)
The failure of supervised auxiliary losses validates the need for a **Reward-Based** approach. We will now pivot to **GRPO**, where the model is only rewarded if a thinking trajectory actually improves the final token prediction. This eliminates the "daydreaming" proxy and aligns the Thinking Head directly with intelligence.
