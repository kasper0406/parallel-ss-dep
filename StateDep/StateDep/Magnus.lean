/-
  StateDep.Magnus

  The Magnus-truncation / BCH-at-order-2 monoid over `d × d` matrices.

  An element is a pair `(A, B)` where `A : Matrix (Fin d) (Fin d) R` is
  the *base* matrix (accumulates linearly) and `B : Matrix (Fin d) (Fin d) R`
  is the *commutator reservoir* that accumulates the Lie bracket between
  base matrices of different steps.

  The motivation is the Baker–Campbell–Hausdorff formula
      exp(X) · exp(Y) = exp(X + Y + ½·[X,Y] + O([·,·,·])),
  truncating after the degree-2 bracket term. The resulting monoid law on
  the pairs is
      (A₁, B₁) * (A₂, B₂) = (A₁ + A₂, B₁ + B₂ + ½·(A₁·A₂ − A₂·A₁)).

  Within this truncation the Jacobi identity is *not* needed: it lives
  one degree up. Associativity only requires bilinearity of the
  commutator, which follows from `Matrix.mul_add` / `Matrix.add_mul` and
  `abel`.

  The factor of ½ forces us off `CommRing`. We take `R` to be a `Field`
  for simplicity — this is entirely sufficient for the research prototype
  and keeps the algebra one-line.

  Once `Magnus R d` is a `Group` (in particular a `Monoid`), the abstract
  `StateDep.Scan.Tree.eval_eq_prod` theorem immediately licenses a
  parallel-scan kernel for this state family.
-/
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Abel
import Mathlib.Tactic.NormNum
import Mathlib.Algebra.Group.Defs
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.Magnus

open Matrix

/-- Element of the Magnus-truncation monoid over `d × d` matrices with
entries in `R`. The `A` slot accumulates the base matrix (linear); the
`B` slot accumulates the weighted sum of commutators `½·[Aᵢ, Aⱼ]` across
pairs `i < j` of steps. -/
@[ext]
structure Magnus (R : Type*) (d : ℕ) where
  A : Matrix (Fin d) (Fin d) R
  B : Matrix (Fin d) (Fin d) R

variable {R : Type*} [Field R] {d : ℕ}

instance : Mul (Magnus R d) where
  mul x y := ⟨x.A + y.A,
              x.B + y.B + (1/2 : R) • (x.A * y.A - y.A * x.A)⟩

instance : One (Magnus R d) := ⟨⟨0, 0⟩⟩

/-- Inverse at this truncation: the quadratic cross-term
`½·(A·(−A) − (−A)·A) = ½·(−A·A + A·A) = 0`, so flipping signs on both
slots really *is* a two-sided inverse. -/
instance : Inv (Magnus R d) where
  inv x := ⟨-x.A, -x.B⟩

@[simp] theorem mul_A (x y : Magnus R d) : (x * y).A = x.A + y.A := rfl
@[simp] theorem mul_B (x y : Magnus R d) :
    (x * y).B = x.B + y.B + (1/2 : R) • (x.A * y.A - y.A * x.A) := rfl
@[simp] theorem one_A : (1 : Magnus R d).A = 0 := rfl
@[simp] theorem one_B : (1 : Magnus R d).B = 0 := rfl
@[simp] theorem inv_A (x : Magnus R d) : (x⁻¹).A = -x.A := rfl
@[simp] theorem inv_B (x : Magnus R d) : (x⁻¹).B = -x.B := rfl

/-- **The Magnus-truncation monoid is a group.**

Associativity in the `A`-slot is just additive associativity (`abel`).
In the `B`-slot it is bilinearity of the commutator: expand both
`(x.A + y.A) · z.A` and `x.A · (y.A + z.A)` (and their reverses) via
`Matrix.add_mul` / `Matrix.mul_add`, pull the scalar out with
`smul_add`/`smul_sub`, and close with `abel`. No Jacobi identity is
invoked — that identity first becomes necessary one BCH degree higher.

The inverse identities collapse because the cross-term in
`A · (−A) − (−A) · A` is `−A·A + A·A = 0`. -/
instance : Group (Magnus R d) where
  mul_assoc x y z := by
    apply Magnus.ext
    · -- A-slot: additive associativity in the matrix group
      change x.A + y.A + z.A = x.A + (y.A + z.A)
      abel
    · -- B-slot: bilinearity of the commutator plus abelian bookkeeping
      change x.B + y.B + (1/2 : R) • (x.A * y.A - y.A * x.A) + z.B
              + (1/2 : R) • ((x.A + y.A) * z.A - z.A * (x.A + y.A))
            = x.B + (y.B + z.B + (1/2 : R) • (y.A * z.A - z.A * y.A))
              + (1/2 : R) • (x.A * (y.A + z.A) - (y.A + z.A) * x.A)
      rw [Matrix.add_mul, Matrix.mul_add, Matrix.mul_add, Matrix.add_mul]
      -- everything is now a sum of `(1/2)•(matrix monomial)` terms, so
      -- linear algebra in the matrix additive group closes it
      simp only [smul_add, sub_eq_add_neg, neg_add_rev]
      abel
  one_mul x := by
    apply Magnus.ext
    · change (0 : Matrix (Fin d) (Fin d) R) + x.A = x.A
      abel
    · change (0 : Matrix (Fin d) (Fin d) R) + x.B
              + (1/2 : R) • ((0 : Matrix (Fin d) (Fin d) R) * x.A
                             - x.A * 0) = x.B
      rw [Matrix.zero_mul, Matrix.mul_zero, sub_zero, smul_zero, add_zero,
          zero_add]
  mul_one x := by
    apply Magnus.ext
    · change x.A + (0 : Matrix (Fin d) (Fin d) R) = x.A
      abel
    · change x.B + (0 : Matrix (Fin d) (Fin d) R)
              + (1/2 : R) • (x.A * 0 - (0 : Matrix (Fin d) (Fin d) R) * x.A)
            = x.B
      rw [Matrix.zero_mul, Matrix.mul_zero, sub_zero, smul_zero, add_zero,
          add_zero]
  inv_mul_cancel x := by
    apply Magnus.ext
    · change -x.A + x.A = (0 : Matrix (Fin d) (Fin d) R)
      abel
    · change -x.B + x.B + (1/2 : R) • ((-x.A) * x.A - x.A * (-x.A)) = 0
      rw [Matrix.neg_mul, Matrix.mul_neg, sub_neg_eq_add]
      have h : -(x.A * x.A) + x.A * x.A
              = (0 : Matrix (Fin d) (Fin d) R) := by abel
      rw [h, smul_zero, add_zero, neg_add_cancel]

/-- Commutator of two `d × d` matrices, written as `⁅A, B⁆ := A·B − B·A`.
We use this only in the non-abelian sanity check below. -/
def commutator (X Y : Matrix (Fin d) (Fin d) R) :
    Matrix (Fin d) (Fin d) R :=
  X * Y - Y * X

/-- **Sanity check: the monoid is non-abelian.**

Take `A = [[0,1],[0,0]]` (2 × 2) and its transpose `A' = [[0,0],[1,0]]`.
Then `A · A' = [[1,0],[0,0]]` and `A' · A = [[0,0],[0,1]]`, so
`[A, A'] = diag(1, −1) ≠ 0`. Composing `(A, 0) * (A', 0)` produces a
`B`-slot equal to `½·[A, A']`, which is visibly nonzero — so the result
is not the identity element `(0, 0)`. -/
example :
    let A  : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 0, 0]
    let A' : Matrix (Fin 2) (Fin 2) ℚ := !![0, 0; 1, 0]
    ((⟨A, 0⟩ : Magnus ℚ 2) * ⟨A', 0⟩).B 0 0 = (1/2 : ℚ) := by
  simp [mul_B]

/-- The commutator-slot of `(A, 0) * (Aᵀ, 0)` for the specific `A` above
picks up the *diagonal* matrix `½·diag(1, −1)` — the characteristic
signature of a non-abelian composition at this truncation level. -/
example :
    let A  : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 0, 0]
    let A' : Matrix (Fin 2) (Fin 2) ℚ := !![0, 0; 1, 0]
    ((⟨A, 0⟩ : Magnus ℚ 2) * ⟨A', 0⟩).B 1 1 = (-(1/2) : ℚ) := by
  simp [mul_B]

/-- The product `(A, 0) * (Aᵀ, 0)` is *not* the identity element of
`Magnus ℚ 2`: its `B`-slot differs from `0` in the `(0,0)` entry,
because the `(1/2)`-weighted commutator is non-zero there. -/
example :
    ((⟨!![0, 1; 0, 0], 0⟩ : Magnus ℚ 2) * ⟨!![0, 0; 1, 0], 0⟩)
      ≠ (1 : Magnus ℚ 2) := by
  intro h
  have hB :
      ((⟨!![0, 1; 0, 0], 0⟩ : Magnus ℚ 2) * ⟨!![0, 0; 1, 0], 0⟩).B 0 0
      = (1 : Magnus ℚ 2).B 0 0 := by rw [h]
  simp [mul_B] at hB

/-- **Parallel-scan correctness for the Magnus-truncation monoid, for
free.** Any binary-tree re-association of a sequence of elements yields
the same group product as the sequential left-fold. This is the instance
of `StateDep.Scan.Tree.eval_eq_prod` at the monoid `Magnus R d` and is
all we need to license a parallel-scan kernel. -/
example (t : StateDep.Scan.Tree (Magnus R d)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.Magnus
