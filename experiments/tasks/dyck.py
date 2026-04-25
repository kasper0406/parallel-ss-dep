"""
Dyck-2 nested-brackets state-tracking task — coding-relevant probe.

Spec:
  - Vocabulary: 4 brackets {(, ), [, ]}, indices 0-3.
  - Sequence is a *random walk* — at each position we either open a
    random bracket type or close the topmost open bracket (50/50). If
    there's nothing to close, we always open. This guarantees every
    sequence is a valid Dyck-2 prefix.
  - Label at position t: the *current nesting depth* clipped to
    `max_depth=15` (16-way classification).

Why this task:
  - Tracking nesting depth is the canonical *structural* state-tracking
    in code (matching brackets, function/class scopes, indentation).
  - Failure mode is fail-late: easy to predict depth at small T, hard
    at long T because cumulative state-tracking errors grow.
  - This is the closest synthetic analog to "track scope depth" that
    a coding LLM needs to do.
  - SO(n)-rotation is well-suited to mod-p counting (each open
    increments depth; rotation by 2π/d encodes mod-d). DeltaNet's
    Householder eigenvalues ∈ {±1} only encode mod-2.
  - This is the *coding-relevant* generalisation of the modular
    addition result from Phase 9.

Following the convention from `tasks/parity.py` and `tasks/modular.py`,
this returns (input_ids, labels) where labels is per-position depth.
"""
from __future__ import annotations

import torch


def make_batch(B: int, T: int, n_types: int = 2, max_depth: int = 15,
               device: torch.device | str = "cuda",
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (input_ids, labels) for Dyck depth-tracking.

    Args:
        B: batch size.
        T: sequence length.
        n_types: number of bracket types (Dyck-1 = 1, Dyck-2 = 2).
        max_depth: clip current depth to [0, max_depth] for classification.
    Returns:
        input_ids : (B, T) int64, values in [0, 2 * n_types). Even indices
                    are open brackets, odd indices are matching closers
                    (so token 2k is "open type k", 2k+1 is "close type k").
        labels    : (B, T) int64, current depth clipped to max_depth.
    """
    n_tokens = 2 * n_types

    # Stack-tracking state per example. Each example maintains its own stack
    # of open bracket types. We avoid a Python loop over batch by using
    # tensor ops where possible.
    inp = torch.empty(B, T, dtype=torch.int64, device=device)
    labels = torch.empty(B, T, dtype=torch.int64, device=device)

    # Stack as a (B, max_depth+1) buffer holding bracket types; depth tracks
    # the size. Initialise empty stacks.
    stack = torch.zeros(B, max_depth + 1, dtype=torch.int64, device=device)
    depth = torch.zeros(B, dtype=torch.int64, device=device)

    # Per-step decision: open (50%) or close (50% if depth > 0, else open).
    # If depth = max_depth we also force a close (to keep within capacity).
    for t in range(T):
        rand_action = torch.rand(B, device=device, generator=generator)
        rand_type = torch.randint(0, n_types, (B,), device=device,
                                  generator=generator)
        # action = 1 if open, 0 if close (or open when depth==0).
        can_close = depth > 0
        forced_close = depth >= max_depth
        # 50/50 default; force open if depth==0; force close if depth==max.
        open_mask = (
            (~forced_close) & ((rand_action < 0.5) | (~can_close))
        )

        # Open: emit token 2*type, push type, increment depth.
        # Close: emit token 2*top + 1, pop, decrement depth.
        idx_b = torch.arange(B, device=device)
        # For closing, top of stack is at index depth-1 (clamped for safety).
        top_idx = (depth - 1).clamp_min(0)
        top_type = stack[idx_b, top_idx]

        token_open = 2 * rand_type
        token_close = 2 * top_type + 1
        token = torch.where(open_mask, token_open, token_close)
        inp[:, t] = token

        # Update stack/depth. Vectorised:
        #   if open: stack[depth] = type, depth += 1
        #   else:    depth -= 1
        # Use scatter for the conditional write.
        new_depth = torch.where(open_mask, depth + 1, depth - 1)
        # Write rand_type at index `depth` for opening examples.
        write_idx = depth.clamp_max(max_depth)
        stack_to_write = torch.where(open_mask, rand_type, top_type)
        # Scatter-write into stack only where we open.
        stack[idx_b, write_idx] = torch.where(
            open_mask, stack_to_write, stack[idx_b, write_idx]
        )
        depth = new_depth

        labels[:, t] = depth.clamp_max(max_depth)

    return inp, labels
