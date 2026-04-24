/-
  StateDep.Rotor

  Unit-quaternion rotors as a state-transition monoid.

  An element is a quaternion `q ∈ ℍ[R]`. The state update rule is
      h_{t+1} = q_t * h_t
  so the accumulated transformation after `n` steps is the ordered product
      q_{n-1} * q_{n-2} * ... * q_0
  of the per-step rotors. Since `ℍ[R]` is a `Ring` in mathlib, it is in
  particular a `Monoid` under `*`, and the abstract parallel-scan theorem
  `StateDep.Scan.Tree.eval_eq_prod` automatically gives parallel-scan
  correctness for any re-association schedule.

  This realization is the 4D analogue of the Heisenberg / Jet constructions:
  the nonabelian nature of quaternion multiplication is what makes the
  accumulated transformation expressive (it tracks compositions of
  rotations in 3-space, once restricted to unit quaternions), while
  associativity is free from the ring structure.

  We also expose a tiny wrapped form `Rotor R` = `ℍ[R] × R` which pairs a
  quaternion with an auxiliary scalar (e.g. a cumulative log-scale). This
  is just a product monoid and inherits the scan theorem trivially.
-/
import Mathlib.Algebra.Quaternion
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.Rotor

open Quaternion

/-! ## Quaternions form a monoid

Mathlib already gives `ℍ[R] = Quaternion R` a `Ring` structure whenever
`R` is a `CommRing`, so monoid-ness is immediate. -/

section Monoid

variable {R : Type*} [CommRing R]

example : Monoid (Quaternion R) := inferInstance

/-- Associativity of quaternion multiplication, delegated to the ring
instance. This is the only algebraic fact the scan theorem consumes. -/
example (q₁ q₂ q₃ : Quaternion R) : (q₁ * q₂) * q₃ = q₁ * (q₂ * q₃) :=
  mul_assoc q₁ q₂ q₃

end Monoid

/-! ## Non-commutativity

We witness nonabelianness with the classical `i * j = k` vs `j * i = -k`
identity, specialized to quaternions over `ℤ`. Concretely, using the
mathlib mul formula (which takes `c₁ = -1`, `c₂ = 0`, `c₃ = -1`):

* `⟨0, 1, 0, 0⟩ * ⟨0, 0, 1, 0⟩ = ⟨0, 0, 0, 1⟩`  (i·j = k)
* `⟨0, 0, 1, 0⟩ * ⟨0, 1, 0, 0⟩ = ⟨0, 0, 0, -1⟩`  (j·i = −k)
-/

/-- The `imK` component separates `i * j` from `j * i`: the former is
`+1`, the latter is `-1`. This witnesses non-commutativity of
quaternion multiplication over `ℤ`. -/
example : ((⟨0, 1, 0, 0⟩ : Quaternion ℤ) * ⟨0, 0, 1, 0⟩).imK = 1 := by decide
example : ((⟨0, 0, 1, 0⟩ : Quaternion ℤ) * ⟨0, 1, 0, 0⟩).imK = -1 := by decide

/-- Non-commutativity: multiplying `i` and `j` in different orders gives
different quaternions. We reduce the inequality to inequality of the
`imK` components (`1 ≠ -1` in `ℤ`), which `decide` handles. -/
example :
    (⟨0, 1, 0, 0⟩ : Quaternion ℤ) * ⟨0, 0, 1, 0⟩ ≠
      (⟨0, 0, 1, 0⟩ : Quaternion ℤ) * ⟨0, 1, 0, 0⟩ := by
  intro h
  have hK : ((⟨0, 1, 0, 0⟩ : Quaternion ℤ) * ⟨0, 0, 1, 0⟩).imK =
            ((⟨0, 0, 1, 0⟩ : Quaternion ℤ) * ⟨0, 1, 0, 0⟩).imK := by
    rw [h]
  revert hK
  decide

/-! ## Parallel-scan correctness

The abstract scan theorem applied at the monoid `Quaternion R`: any
binary-tree re-association of a sequence of rotors yields the same
product as the sequential left-to-right scan. -/

example {R : Type*} [CommRing R] (t : StateDep.Scan.Tree (Quaternion R)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

/-! ## Bonus: the wrapped `Rotor` structure

A `Rotor R` carries a quaternion together with an auxiliary scalar
(interpretable as e.g. a log-scale or a per-step gain). Multiplication
is componentwise — this is literally the product monoid `ℍ[R] × R` —
and the scan theorem instantly transfers. -/

/-- An RNN step carrying a quaternion `q` (the rotor) plus an auxiliary
scalar `scale`. Both components compose multiplicatively. -/
@[ext]
structure Rotor (R : Type*) [CommRing R] where
  q : Quaternion R
  scale : R

namespace Rotor

variable {R : Type*} [CommRing R]

instance : Mul (Rotor R) where
  mul x y := ⟨x.q * y.q, x.scale * y.scale⟩

instance : One (Rotor R) := ⟨⟨1, 1⟩⟩

@[simp] theorem mul_q (x y : Rotor R) : (x * y).q = x.q * y.q := rfl
@[simp] theorem mul_scale (x y : Rotor R) : (x * y).scale = x.scale * y.scale := rfl
@[simp] theorem one_q : (1 : Rotor R).q = 1 := rfl
@[simp] theorem one_scale : (1 : Rotor R).scale = 1 := rfl

/-- The wrapped `Rotor R` is a monoid: this is the product monoid
`Quaternion R × R`, and every axiom reduces to the corresponding fact
for each component. We close each goal with `Rotor.ext` directly (not
`ext`) so we don't accidentally recurse into the quaternion's own
componentwise extensionality. -/
instance : Monoid (Rotor R) where
  mul_assoc x y z :=
    Rotor.ext (mul_assoc x.q y.q z.q) (mul_assoc x.scale y.scale z.scale)
  one_mul x := Rotor.ext (one_mul x.q) (one_mul x.scale)
  mul_one x := Rotor.ext (mul_one x.q) (mul_one x.scale)

/-- Parallel-scan correctness for wrapped rotors, inherited from the
abstract scan theorem at monoid `Rotor R`. -/
example (t : StateDep.Scan.Tree (Rotor R)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end Rotor

end StateDep.Rotor
