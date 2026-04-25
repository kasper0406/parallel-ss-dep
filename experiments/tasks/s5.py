"""
S₅ word problem — the canonical NC¹-complete state-tracking benchmark.

Spec:
  - Vocabulary: 4 generators of S₅ (adjacent transpositions s_1=(1 2),
    s_2=(2 3), s_3=(3 4), s_4=(4 5)) plus a "no-op" identity → vocab=5.
  - Sequence: random tokens, each labels a generator (or identity).
  - Label at position t: whether the running composition (left-to-right
    multiplication) of all generators 0..t equals the identity in S₅.
    Binary classification.

Why this task:
  - Per Barrington's theorem (1989), every NC¹ Boolean function reduces
    to a width-5 branching program over a non-solvable group; S₅ is the
    canonical such group. A scan that recognises this language is
    expressively NC¹-complete.
  - SO(n) for n ≥ 3 contains A₅ (the icosahedral subgroup, non-solvable).
    In principle, an SO(n)-scan can realise S₅ word-problem acceptance.
    The empirical question is whether SGD finds the non-abelian rotations
    or stays in the abelian subgroup (per the expRNN literature, gradient
    descent strongly prefers abelian solutions).
  - DeltaProduct (Yang et al., NeurIPS 2025) achieves this via K
    Householder products per token (each contributes a reflection ∈ O(n)).
    With K = n, DeltaProduct's image spans O(n) and contains A₅.
    Single-Householder DeltaNet (any β) only realises Z₂; cannot recognise
    S₅.
  - Hybrid `[ortho, deltanet]` should solve via the ortho layers'
    rotation primitive; deltanet_negeig (Z₂ only) should fail.

Concrete encoding:
  - Tokens 0-3: the four adjacent transpositions s_1, s_2, s_3, s_4.
  - Token 4: identity (does nothing).
  - At each position t, label is 1 iff the composition s_{x_0} · s_{x_1}
    · … · s_{x_t} equals identity in S₅, else 0.
"""
from __future__ import annotations

import torch


def _gen_table() -> torch.Tensor:
    """Permutation table for the 4 adjacent transpositions in S₅ (and identity).

    Returns shape (5, 5) — entry [g, i] is the position that index i maps to
    after applying generator g.
    """
    n = 5
    table = []
    # Adjacent transpositions s_k = (k k+1) for k = 0, 1, 2, 3.
    for k in range(4):
        perm = list(range(n))
        perm[k], perm[k + 1] = perm[k + 1], perm[k]
        table.append(perm)
    # Identity.
    table.append(list(range(n)))
    return torch.tensor(table, dtype=torch.int64)


_GEN_TABLE = _gen_table()                     # (5, 5)
_IDENTITY = torch.arange(5, dtype=torch.int64)


def _compose(perm: torch.Tensor, gen_idx: torch.Tensor) -> torch.Tensor:
    """Compose a batch of permutations with a batch of generators.

    perm:    (B, 5) — current permutation per example.
    gen_idx: (B,)   — generator index in {0..4}.
    Returns: (B, 5) — perm composed with the generator.
    """
    # Composition is permutation-of-permutation: out[b, i] = perm[b, gen[gen_idx[b], i]].
    gen = _GEN_TABLE.to(perm.device)[gen_idx]                     # (B, 5)
    out = torch.gather(perm, 1, gen)                              # (B, 5)
    return out


def make_batch(B: int, T: int, vocab_size: int = 5,
               device: torch.device | str = "cuda",
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (input_ids, labels) for S₅ word-problem.

    Args:
        B: batch size.
        T: sequence length.
        vocab_size: must be 5 (4 transpositions + identity).
    Returns:
        input_ids : (B, T) int64, values in {0, 1, 2, 3, 4}.
        labels    : (B, T) int64, 1 iff running composition = identity, else 0.
    """
    assert vocab_size == 5, "S₅ task uses 5 tokens"
    inp = torch.randint(0, vocab_size, (B, T), device=device, generator=generator,
                        dtype=torch.int64)

    # Run the composition step-by-step and emit the identity-test label per step.
    state = _IDENTITY.to(device).unsqueeze(0).expand(B, -1).clone()  # (B, 5)
    identity = state.clone()
    labels = torch.empty(B, T, dtype=torch.int64, device=device)
    for t in range(T):
        state = _compose(state, inp[:, t])
        labels[:, t] = (state == identity).all(dim=-1).long()

    return inp, labels
