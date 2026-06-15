"""
Two integer-token tasks for the latent-thinking challenge. No tokenizer:
the "vocabulary" is just small integers.

Both tasks share a layout so the SAME model/training code runs on either:
  - values live in 0 .. V-1   (single token each)
  - structural / special tokens occupy the TOP of the vocab:
        SEP, QUERY, THINK, PAD
  - the answer is always a single value token in 0 .. V-1
  - every example also exposes the per-step intermediates (the "chain"), so we
    can do optional per-hop supervision and per-hop diagnostics.

----------------------------------------------------------------------------
HOMOGENEOUS  (pointer-chase  f^n(s)) — thinking SHOULD help
----------------------------------------------------------------------------
A single random function table f : {0..V-1} -> {0..V-1} is encoded as shuffled
(i, f(i)) pairs. Then [QUERY, s]. The target is f^n(s): apply the SAME
operation f, n times. One forward can trace ~1 hop; tracing n hops needs n
sequential steps -> latent thinking supplies the depth. This is the regime
where the mechanism is already proven to work.

----------------------------------------------------------------------------
HETEROGENEOUS  (exec-trace) — thinking FAILS (the open problem)
----------------------------------------------------------------------------
A small program: a start value s and a sequence of n operations, each a
DIFFERENT *kind* of op drawn from:
    - TABLE  : x <- f(x)            (one shared random table f)
    - ADD c  : x <- (x + c) % V
    - MUL c  : x <- (x * c) % V
The op sequence is encoded in the input (op-kind token + arg token per step).
The target is the value after running all n ops in order. Each step is a
DIFFERENT operation, so there is no single reusable latent operator to iterate
-- which is exactly why naive latent thinking collapses at depth here.

The challenge: make latent thinking + WM solve THIS the way they solve the
homogeneous task.
"""
from __future__ import annotations

import torch


# Op kinds for the heterogeneous task
OP_TABLE, OP_ADD, OP_MUL = 0, 1, 2
N_OP_KINDS = 3


# ---------------------------------------------------------------------------
# Vocab layout helpers
# ---------------------------------------------------------------------------
class Vocab:
    """Lays out value tokens 0..V-1 then special tokens above them."""

    def __init__(self, n_values: int, n_op_kinds: int = 0):
        self.V = n_values
        self.n_op_kinds = n_op_kinds
        nxt = n_values
        # op-kind tokens (heterogeneous only): OPKIND_BASE + kind
        self.OPKIND_BASE = nxt
        nxt += n_op_kinds
        self.SEP = nxt; nxt += 1
        self.QUERY = nxt; nxt += 1
        self.THINK = nxt; nxt += 1
        self.PAD = nxt; nxt += 1
        self.size = nxt

    def opkind(self, kind: int) -> int:
        return self.OPKIND_BASE + kind


# ---------------------------------------------------------------------------
# Homogeneous: pointer-chase f^n(s)
# ---------------------------------------------------------------------------
def make_homogeneous_batch(B: int, V: int, n: int, device="cuda",
                           generator: torch.Generator | None = None):
    """Returns (ids, answers, chain, vocab).

    ids:     (B, 2V + 2)   [i0,f(i0), i1,f(i1), ..., QUERY, s]
    answers: (B,)          f^n(s)
    chain:   (B, n)        [f(s), f^2(s), ..., f^n(s)]  (per-hop targets)
    vocab:   Vocab
    """
    vocab = Vocab(V, n_op_kinds=0)
    g = generator
    perm = torch.rand(B, V, generator=g).argsort(dim=1)            # f(i) = perm[i]
    order = torch.rand(B, V, generator=g).argsort(dim=1)           # present shuffled
    tgt = perm.gather(1, order)
    pairs = torch.stack([order, tgt], dim=2).reshape(B, 2 * V)
    s = torch.randint(0, V, (B, 1), generator=g)
    x = s.clone()
    chain_cols = []
    for _ in range(n):
        x = perm.gather(1, x)
        chain_cols.append(x.clone())
    chain = torch.cat(chain_cols, dim=1)
    answers = chain[:, -1]
    query_col = torch.full((B, 1), vocab.QUERY, dtype=torch.long)
    ids = torch.cat([pairs, query_col, s], dim=1)
    return ids.to(device), answers.to(device), chain.to(device), vocab


# ---------------------------------------------------------------------------
# Heterogeneous: exec-trace (different op each step)
# ---------------------------------------------------------------------------
def make_heterogeneous_batch(B: int, V: int, n: int, device="cuda",
                             generator: torch.Generator | None = None):
    """Returns (ids, answers, chain, vocab).

    Layout:
      [i0,f(i0), ..., i_{V-1},f(i_{V-1}),        # the shared table f (2V toks)
       SEP, s,                                    # start value
       opkind0, arg0, opkind1, arg1, ..., opkind_{n-1}, arg_{n-1},
       QUERY]                                     # ask for the result
    answers: (B,)   value after running all n ops
    chain:   (B, n) intermediate value after each op (for per-hop supervision)
    """
    vocab = Vocab(V, n_op_kinds=N_OP_KINDS)
    g = generator
    perm = torch.rand(B, V, generator=g).argsort(dim=1)            # table f
    order = torch.rand(B, V, generator=g).argsort(dim=1)
    tgt = perm.gather(1, order)
    pairs = torch.stack([order, tgt], dim=2).reshape(B, 2 * V)     # (B, 2V)

    s = torch.randint(0, V, (B,), generator=g)                     # start value
    # op kinds and args per step
    kinds = torch.randint(0, N_OP_KINDS, (B, n), generator=g)      # (B, n)
    # args: TABLE ignores arg (use 0); ADD/MUL use a constant in 0..V-1.
    # avoid c=0 for MUL (collapses x->0) and keep ADD non-trivial.
    args = torch.randint(1, V, (B, n), generator=g)

    # run the program to get intermediates
    x = s.clone()
    chain_cols = []
    for j in range(n):
        kind_j = kinds[:, j]
        arg_j = args[:, j]
        x_tab = perm.gather(1, x.unsqueeze(1)).squeeze(1)
        x_add = (x + arg_j) % V
        x_mul = (x * arg_j) % V
        x = torch.where(kind_j == OP_TABLE, x_tab,
                        torch.where(kind_j == OP_ADD, x_add, x_mul))
        chain_cols.append(x.clone().unsqueeze(1))
    chain = torch.cat(chain_cols, dim=1)                           # (B, n)
    answers = chain[:, -1]

    # build the op token stream: [opkind, arg] per step
    opkind_tok = vocab.OPKIND_BASE + kinds                        # (B, n)
    ops = torch.stack([opkind_tok, args], dim=2).reshape(B, 2 * n)  # (B, 2n)

    sep = torch.full((B, 1), vocab.SEP, dtype=torch.long)
    s_col = s.unsqueeze(1)
    query = torch.full((B, 1), vocab.QUERY, dtype=torch.long)
    ids = torch.cat([pairs, sep, s_col, ops, query], dim=1)
    return ids.to(device), answers.to(device), chain.to(device), vocab


# ---------------------------------------------------------------------------
# Dispatch + sizing helpers
# ---------------------------------------------------------------------------
def make_batch(task: str, B, V, n, device="cuda", generator=None):
    if task == "homogeneous":
        return make_homogeneous_batch(B, V, n, device, generator)
    elif task == "heterogeneous":
        return make_heterogeneous_batch(B, V, n, device, generator)
    raise ValueError(task)


def max_seq_len(task: str, V: int, n: int) -> int:
    """Max prompt length (excluding the appended think slots)."""
    if task == "homogeneous":
        return 2 * V + 2
    else:  # heterogeneous: 2V table + SEP + s + 2n ops + QUERY
        return 2 * V + 3 + 2 * n


def vocab_for(task: str, V: int) -> Vocab:
    return Vocab(V, n_op_kinds=(0 if task == "homogeneous" else N_OP_KINDS))
