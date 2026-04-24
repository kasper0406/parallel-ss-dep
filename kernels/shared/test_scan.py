"""
Unit tests for the three scan schedules in `shared/scan.py`.

Includes a **non-commutative** test (scalar Heisenberg H_3) — the agent
who wrote U_4 caught that an earlier Blelloch implementation had swapped
operands in the down-sweep. That bug was invisible to a commutative
sanity-check (plain int addition), so we now exercise both.

Run:
    python kernels/shared/test_scan.py
"""
from __future__ import annotations
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from kernels.shared.scan import (
    sequential_scan,
    blelloch_scan,
    hillis_steele_scan,
    pad_to_pow2,
)

# ---------------------------------------------------------------------------
# Monoid 1: plain integer addition (commutative). Smoke test.
# ---------------------------------------------------------------------------

def _test_int_add():
    add = lambda a, b: a + b
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = [1, 3, 6, 10, 15, 21, 28, 36]
    assert sequential_scan(xs, add, 0) == expected
    assert blelloch_scan(xs, add, 0) == expected
    assert hillis_steele_scan(xs, add) == expected


# ---------------------------------------------------------------------------
# Monoid 2: scalar Heisenberg H_3 (non-commutative). The acid test for
# any parallel-scan schedule claiming to support non-abelian monoids.
# (a, b, c) * (a', b', c') = (a + a', b + b', c + c' + a * b')
# ---------------------------------------------------------------------------

def _heisenberg_combine(x, y):
    a1, b1, c1 = x
    a2, b2, c2 = y
    return (a1 + a2, b1 + b2, c1 + c2 + a1 * b2)


H_ID = (0, 0, 0)


def _test_heisenberg():
    # Deliberately pick integer values with a, b ≠ 0 so the c-cross-term
    # makes non-commutativity visible.
    xs = [
        (1, 2, 0),
        (3, 5, 0),
        (7, 11, 0),
        (13, 17, 0),
    ]

    seq = sequential_scan(xs, _heisenberg_combine, H_ID)
    bl  = blelloch_scan(xs, _heisenberg_combine, H_ID)
    hs  = hillis_steele_scan(xs, _heisenberg_combine)

    assert seq == bl, f"Blelloch diverges: seq={seq} bl={bl}"
    assert seq == hs, f"Hillis-Steele diverges: seq={seq} hs={hs}"

    # Cross-check the final c against the explicit pair-sum Σ_{i<j} a_i * b_j.
    a_list = [x[0] for x in xs]
    b_list = [x[1] for x in xs]
    expected_c = sum(
        a_list[i] * b_list[j] for i in range(len(xs)) for j in range(i + 1, len(xs))
    )
    assert seq[-1][2] == expected_c, f"c={seq[-1][2]} expected {expected_c}"


def _test_heisenberg_nonpow2():
    # Non-power-of-two path (requires pad_to_pow2 into identity).
    xs = [(i, i * 2, 0) for i in range(1, 7)]  # length 6
    seq = sequential_scan(xs, _heisenberg_combine, H_ID)
    bl  = blelloch_scan(pad_to_pow2(xs, H_ID), _heisenberg_combine, H_ID)
    # After padding, the first 6 entries of the padded scan must match seq.
    assert bl[: len(xs)] == seq


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    _test_int_add()
    print("shared.scan  int-add                     OK")
    _test_heisenberg()
    print("shared.scan  Heisenberg H_3 (len 4)      OK")
    _test_heisenberg_nonpow2()
    print("shared.scan  Heisenberg H_3 (len 6 pad)  OK")
    print("OK")


if __name__ == "__main__":
    main()
