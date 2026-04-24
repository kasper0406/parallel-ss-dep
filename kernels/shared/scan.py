"""
Associative-scan reference helpers, in pure PyTorch.

We use these as ground-truth references — their correctness follows
from `StateDep.Scan.Tree.eval_eq_prod`: any binary-tree reduction of an
associative operator gives the same result as the sequential left-fold.

The three schedules implemented here are the ones a Triton kernel
typically picks between:

  - `sequential_scan`  — O(n) work, O(n) depth; trivially correct baseline.
  - `blelloch_scan`    — O(n) work, O(log n) depth, in-place.
  - `hillis_steele_scan` — O(n log n) work, O(log n) depth; easiest to
                           vectorise but does redundant work.

The `combine` callable is the monoid operation; states are opaque
(tuples, tensors, whatever the cell uses). Everything here is
pure-Python / pure-torch, shape-agnostic.
"""

from __future__ import annotations
from typing import Callable, List, TypeVar

S = TypeVar("S")


def sequential_scan(
    xs: List[S],
    combine: Callable[[S, S], S],
    identity: S,
) -> List[S]:
    """Inclusive left scan: `out[i] = x_0 * x_1 * ... * x_i`."""
    out: List[S] = []
    acc = identity
    for x in xs:
        acc = combine(acc, x)
        out.append(acc)
    return out


def blelloch_scan(
    xs: List[S],
    combine: Callable[[S, S], S],
    identity: S,
) -> List[S]:
    """
    Blelloch (work-efficient) parallel prefix scan, computed in-place.
    Returns the *inclusive* scan. Requires len(xs) to be a power of two;
    callers pad with `identity` if needed.

    This is the schedule a Triton kernel uses within a block / chunk.
    """
    n = len(xs)
    assert n & (n - 1) == 0, f"blelloch_scan needs power-of-two length, got {n}"
    a = list(xs)

    # Up-sweep (reduction).
    d = 1
    while d < n:
        for i in range(0, n, 2 * d):
            a[i + 2 * d - 1] = combine(a[i + d - 1], a[i + 2 * d - 1])
        d *= 2

    # Down-sweep (distribute partial sums). Converts to exclusive scan.
    #
    # For each internal node, the right child's exclusive prefix is
    # `combine(parent_excl_prefix, left_subtree_total)` — in that order,
    # because the parent prefix covers *earlier* positions than the left
    # subtree. For a commutative monoid the order is immaterial; for
    # non-commutative monoids (Heisenberg, unipotent, quaternions, …) it
    # is load-bearing. An earlier version of this routine had the
    # arguments swapped, silently passing for linear-attention (abelian)
    # but yielding wrong results for every non-commutative cell.
    a[n - 1] = identity
    d = n // 2
    while d >= 1:
        for i in range(0, n, 2 * d):
            t = a[i + d - 1]                           # left-subtree total (saved)
            a[i + d - 1] = a[i + 2 * d - 1]            # left child ← parent's excl prefix
            a[i + 2 * d - 1] = combine(a[i + 2 * d - 1], t)  # right child ← parent excl · left total
        d //= 2

    # Exclusive → inclusive: shift left by one and combine with the input.
    incl: List[S] = []
    for i in range(n):
        incl.append(combine(a[i], xs[i]))
    return incl


def hillis_steele_scan(
    xs: List[S],
    combine: Callable[[S, S], S],
) -> List[S]:
    """Naive parallel scan; included as a schedule sanity-check only."""
    n = len(xs)
    a = list(xs)
    step = 1
    while step < n:
        b = list(a)
        for i in range(step, n):
            b[i] = combine(a[i - step], a[i])
        a = b
        step *= 2
    return a


def pad_to_pow2(xs: List[S], identity: S) -> List[S]:
    n = len(xs)
    if n == 0:
        return xs
    p = 1
    while p < n:
        p *= 2
    return xs + [identity] * (p - n)
