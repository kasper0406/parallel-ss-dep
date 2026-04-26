"""
Variable-binding pointer-chasing task — the smoke test for symbol-grounded
state.

Sequence template (toy Python-like syntax):

    a = 7 ; b = 3 ; c = 9 ;  ...filler...  a = 12 ; ... ?a 5

Where:
    - tokens are chosen from a small structured vocab,
    - "x = N" is a binding event,
    - "?x" is a query: the model should predict the *latest* binding for x,
    - the next token (after ?x) carries the label,
    - many distractor bindings + filler tokens between the relevant bindings
      and the query.

Why this task is the right separator:
    - Standard linear-RNN cells (DeltaNet, Mamba2) need to compress the
      *sequence* of bindings into their fixed state. Recovering the latest
      binding for a specific id is hard: the relevant key signal
      ("when did x get rebound last?") is buried under unrelated bindings.
    - Softmax attention solves it via attention to the prior binding token,
      but at O(T·d) cost.
    - A symbol-grounded cell can solve it with O(1) read + O(1) write per
      token via direct id-keyed lookup — *if* the inductive bias works.

This is *not* MQAR. MQAR's keys are random tokens; here the keys are the
SAME variable names appearing many times, with the model needing to track
the latest assignment.
"""
from __future__ import annotations

import torch


# Vocabulary layout (deterministic, used for both task and parsing):
#
#     id  0:    PAD / cue
#     id  1:    "="  (assignment marker)
#     id  2:    ";"  (statement separator)
#     id  3:    "?"  (query marker)
#     id  4..4+n_vars-1:                 variable identifiers v_0, v_1, ...
#     id  4+n_vars..4+n_vars+n_vals-1:   value tokens  N_0, N_1, ...
#
# Total vocab size = 4 + n_vars + n_vals.


def vocab_size(n_vars: int = 8, n_vals: int = 16) -> int:
    return 4 + n_vars + n_vals


def _var_ids(n_vars: int) -> tuple[int, int]:
    return 4, 4 + n_vars


def _val_ids(n_vars: int, n_vals: int) -> tuple[int, int]:
    lo = 4 + n_vars
    return lo, lo + n_vals


def make_batch(B: int, T: int, n_vars: int = 8, n_vals: int = 16,
               n_bindings_min: int = 6, n_bindings_max: int = 24,
               device: torch.device | str = "cuda",
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (input_ids, labels, loss_mask) for var_binding.

    Each example contains a random number of bindings of the form
    `var = val ;`, padded with filler tokens (random vars/vals not in any
    structural slot — but staying within vocab so embeddings are sensible)
    until length T-3. The final 3 tokens are `? var label`, where label is
    the latest binding for var.

    The loss is computed only at the position of `label` (one position per
    example). The model sees `... ? var` and must predict label.

    Args:
        B: batch size.
        T: sequence length. Should be >= n_bindings_max * 4 + 3 to fit.
        n_vars: number of distinct variables in the vocab.
        n_vals: number of distinct values in the vocab.
        n_bindings_min/max: number of bindings drawn uniformly per example.

    Returns:
        input_ids: (B, T) int64
        labels:    (B, T) int64 — label only at the prediction position
        loss_mask: (B, T) bool  — True only at the prediction position
    """
    V = vocab_size(n_vars, n_vals)
    var_lo, var_hi = _var_ids(n_vars)
    val_lo, val_hi = _val_ids(n_vars, n_vals)

    EQ_TOKEN = 1
    SEMI_TOKEN = 2
    QUERY_TOKEN = 3

    # Cap n_bindings to fit in T-3 (each binding is 4 tokens + 3 query tokens).
    cap = max(1, (T - 3) // 4)
    n_bindings_max = min(n_bindings_max, cap)
    n_bindings_min = min(n_bindings_min, n_bindings_max)
    assert n_bindings_min >= 1, f"T={T} too small even for one binding"

    # Build sequences as Python lists then stack — clearer than vectorising
    # the variable-length per-example structure.
    inp = torch.zeros(B, T, dtype=torch.int64, device=device)
    lab = torch.zeros(B, T, dtype=torch.int64, device=device)
    msk = torch.zeros(B, T, dtype=torch.bool, device=device)

    g = generator
    for b in range(B):
        # Latest-binding map for this example (Python dict; var_id -> val_id).
        latest: dict[int, int] = {}
        seq: list[int] = []

        n_bindings = int(torch.randint(
            n_bindings_min, n_bindings_max + 1, (1,), device=device, generator=g
        ).item())

        for _ in range(n_bindings):
            v = int(torch.randint(var_lo, var_hi, (1,), device=device, generator=g).item())
            n = int(torch.randint(val_lo, val_hi, (1,), device=device, generator=g).item())
            seq.extend([v, EQ_TOKEN, n, SEMI_TOKEN])
            latest[v] = n

        # Pick query var: must be one we've actually bound.
        bound_vars = list(latest.keys())
        q_var = bound_vars[int(torch.randint(
            0, len(bound_vars), (1,), device=device, generator=g
        ).item())]
        q_val = latest[q_var]

        # Truncate or pad with filler bindings if we don't fill T-3.
        # Simpler: cap n_bindings so the natural length fits, then pad with
        # extra "noise" bindings to (T-3).
        while len(seq) < T - 3:
            # Add a filler binding to a random non-query var so the query
            # answer isn't accidentally overwritten.
            while True:
                v = int(torch.randint(var_lo, var_hi, (1,), device=device, generator=g).item())
                if v != q_var:
                    break
            n = int(torch.randint(val_lo, val_hi, (1,), device=device, generator=g).item())
            need = T - 3 - len(seq)
            if need >= 4:
                seq.extend([v, EQ_TOKEN, n, SEMI_TOKEN])
            elif need == 3:
                # tight fit
                seq.extend([v, EQ_TOKEN, n])
                break
            else:
                seq.extend([SEMI_TOKEN] * need)
                break

        seq = seq[:T - 3]
        # Final 3 tokens: ? q_var q_val
        seq.extend([QUERY_TOKEN, q_var, q_val])
        assert len(seq) == T, f"len={len(seq)} != T={T}"

        inp[b] = torch.tensor(seq, dtype=torch.int64, device=device)
        lab[b, T - 1] = q_val
        msk[b, T - 1] = True

    return inp, lab, msk


def describe(inp_row: torch.Tensor, n_vars: int = 8, n_vals: int = 16) -> str:
    """Pretty-print a single sequence for sanity-checking."""
    out = []
    for tok in inp_row.tolist():
        if tok == 0:
            out.append("PAD")
        elif tok == 1:
            out.append("=")
        elif tok == 2:
            out.append(";")
        elif tok == 3:
            out.append("?")
        elif tok < 4 + n_vars:
            out.append(f"v{tok - 4}")
        else:
            out.append(f"N{tok - 4 - n_vars}")
    return " ".join(out)


if __name__ == "__main__":
    # Quick sanity check.
    inp, lab, msk = make_batch(4, T=32, n_vars=4, n_vals=8, device="cpu")
    for b in range(4):
        print(f"\nseq[{b}]: {describe(inp[b], n_vars=4, n_vals=8)}")
        print(f"label[T-1]: N{lab[b, -1].item() - 4 - 4}")
