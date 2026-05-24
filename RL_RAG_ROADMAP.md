# Technical Roadmap: RL for Continuous RAG & Deep Thinking

This document outlines the technical path for transitioning the "Thinking Head" architecture from supervised BPTT to Reinforcement Learning (RL). This transition is necessary to scale "thought" chains beyond depth 20 and to effectively integrate the Continuous RAG database.

## 1. MDP Formulation
The thinking process is modeled as a Markov Decision Process (MDP):
- **State ($S$):** The persistent DeltaNet hidden state matrix ($c_t$).
- **Action ($A$):** Discrete choice $\{ \text{Think/Retrieve}, \text{Emit} \}$.
- **Transition ($P$):** 
  - `Think` triggers a recurrent pass through the layers and an external RAG query that projects new facts into $c_t$.
  - `Emit` executes the LM head and terminates the thinking episode for that token.
- **Reward ($R$):**
  - $R_{\text{task}}$: Sparse reward based on the correctness of the final generated token or sequence (NLL or code-verifier pass).
  - $R_{\text{ponder}}$: A small negative penalty proportional to the number of thinking steps to encourage efficiency.

## 2. Training Algorithm: GRPO
We use **Group Relative Policy Optimization (GRPO)** to maximize memory efficiency and training stability:
- **Sampling:** For each token, sample a group of $N$ trajectories with different thinking depths and retrieval results.
- **Relative Advantage:** Normalize trajectory rewards within the group. The model learns that "Thinking 10 steps to find Fact B" is better than "Thinking 3 steps and guessing" if it results in lower final NLL.
- **BPTT Decoupling:** By treating the recurrent passes as actions and sampling log-probs, we only backpropagate through the gate and the retrieval projection weights, bypassing the $O(\text{Depth})$ memory wall.

## 3. Technical Integration Phases

### Phase 1: Trajectory Sampling Infrastructure
- Extend `experiments/thinking.py` with `generate_thought_trajectories()`.
- Use `torch.no_grad()` to compute state updates and retrieval queries during the rollout.
- Store log-probabilities of each gate decision.

### Phase 2: RAG-Aware Reward Function
- Define a reward function that balances accuracy improvement against computational cost.
- Measure the NLL "delta" specifically attributed to RAG retrievals to reward the model for searching effectively.

### Phase 3: GRPO Driver
- Implement `experiments/train_rl.py`.
- Incorporate a frozen **Reference Model** (the best BPTT-trained checkpoint) to compute the KL-divergence penalty, preventing the policy from diverging too far from natural language.

## 4. Scaling Goals
- **Target Depth:** Move from the current supervised limit (depth 5) to autonomous chains of depth 50-100.
- **Goal:** Enable the model to autonomously pause, retrieve multiple pieces of evidence from the RAG DB, and "reason" through them in-place before responding.
