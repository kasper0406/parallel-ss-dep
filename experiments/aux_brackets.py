"""
Auxiliary objective: per-token bracket depth prediction.

Direction E (verifier-coupled training) in concrete form: forces the
model's hidden state to encode bracket depth, a structural signal in
code that DeltaNet doesn't have built-in. Code with deeply nested
expressions, function definitions, and class hierarchies has bracket
depth as a real long-range dependency — at the time we close a paren,
the model needs to know which scope we're returning to.

The aux signal is computed cheaply: precompute a (vocab_size,) vector of
bracket-deltas per token (count `({[` minus count `)}]` in the token's
text). Then per-position depth = cumsum of deltas along the sequence.
"""
from __future__ import annotations

import torch


def compute_bracket_deltas(tokenizer) -> torch.Tensor:
    """For each token id, compute (#open_brackets - #close_brackets) in its text.

    Returns: (vocab_size,) int8 tensor.
    """
    V = tokenizer.vocab_size
    deltas = torch.zeros(V, dtype=torch.int8)
    for tid in range(V):
        try:
            text = tokenizer.decode([tid])
        except Exception:
            continue
        opens = sum(text.count(c) for c in "({[")
        closes = sum(text.count(c) for c in ")}]")
        deltas[tid] = opens - closes
    return deltas


def bracket_depth(input_ids: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Per-position bracket depth (after this token).

    Args:
        input_ids: (B, T) int64
        deltas:    (vocab_size,) int8 from compute_bracket_deltas

    Returns:
        depth: (B, T) int64 — cumulative bracket depth.
    """
    # Index deltas by id, cumsum along T.
    per_tok = deltas.to(input_ids.device)[input_ids].to(torch.int64)
    return per_tok.cumsum(dim=-1)
