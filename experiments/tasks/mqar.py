"""
MQAR — Multi-Query Associative Recall.

Following the Zoology formulation (Arora et al., ICLR 2024,
https://arxiv.org/abs/2312.04927). The task probes the architecture's
ability to *retrieve* specific past tokens conditional on a query —
the canonical separator between architectures with KV-style memory
(linear-attention, DeltaNet, Mamba) and architectures with bounded
isometric state (LRU, AUSSM, plain SO(n) scan).

Structure of one example:
  - A learned "k-v dictionary": K random distinct keys with K random values.
  - The first 2K tokens are alternating (k_i, v_i) — defines the lookup.
  - The remaining T − 2K tokens are queries: a random key already seen.
  - Label at each query position is the corresponding value.
  - At non-query positions (the lookup phase), the label is ignored
    (we mask the loss).

Hard mode: T should be much greater than 2K so that the model must
*store* all K-V pairs. With K=20 and T=512, the model has 20 keys to
remember and 472 query positions — exactly the Zoology setup at MQAR-512.
"""
from __future__ import annotations

import torch


def make_batch(B: int, T: int, vocab_size: int = 64, n_pairs: int = 16,
               device: torch.device | str = "cuda",
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of (input_ids, labels, loss_mask) for MQAR.

    Args:
        B: batch size.
        T: sequence length. Must be >= 2 * n_pairs.
        vocab_size: total vocab; must be >= 2 * n_pairs (need distinct
            keys + distinct values).
        n_pairs: number of (key, value) pairs per example.

    Returns:
        input_ids:  (B, T) int64
        labels:     (B, T) int64 — value ID at query positions, else 0
        loss_mask:  (B, T) bool   — True at query positions where loss
                                    should be computed
    """
    assert T >= 2 * n_pairs, f"T={T} must be ≥ 2·n_pairs={2*n_pairs}"
    assert vocab_size >= 2 * n_pairs, \
        f"vocab_size={vocab_size} must be ≥ 2·n_pairs={2*n_pairs}"

    # Per-example keys and values: pick 2*n_pairs distinct tokens, split.
    # Implementation: randperm per example; the first n_pairs are keys,
    # next n_pairs are values, by index.
    perms = torch.stack([
        torch.randperm(vocab_size, device=device, generator=generator)
        for _ in range(B)
    ])                                                  # (B, vocab_size)
    keys = perms[:, :n_pairs]                           # (B, n_pairs)
    vals = perms[:, n_pairs:2 * n_pairs]                # (B, n_pairs)

    # First 2·n_pairs tokens: alternating k, v, k, v, ...
    lookup = torch.empty(B, 2 * n_pairs, dtype=torch.int64, device=device)
    lookup[:, 0::2] = keys
    lookup[:, 1::2] = vals

    # Remaining T - 2·n_pairs positions: random queries (index into keys).
    n_queries = T - 2 * n_pairs
    q_idx = torch.randint(0, n_pairs, (B, n_queries), device=device,
                          generator=generator)
    queries = torch.gather(keys, 1, q_idx)              # (B, n_queries)
    targets = torch.gather(vals, 1, q_idx)              # (B, n_queries)

    input_ids = torch.cat([lookup, queries], dim=1)     # (B, T)

    labels = torch.zeros(B, T, dtype=torch.int64, device=device)
    labels[:, 2 * n_pairs:] = targets

    loss_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    loss_mask[:, 2 * n_pairs:] = True

    return input_ids, labels, loss_mask
