/-
  StateDep.HeisenbergD

  The d-dimensional Heisenberg group H_d(R) as a state-transition monoid.

  An element is a triple (a, b, c) where `a, b : Fin d → R` are vectors and
  `c : Matrix (Fin d) (Fin d) R` is a matrix. The composition law is
      (a₁, b₁, c₁) · (a₂, b₂, c₂)
        = (a₁+a₂, b₁+b₂, c₁+c₂ + vecMulVec a₁ b₂)
  where `vecMulVec a b` is the outer product with entry `(i,j) = a i * b j`.

  This is the matrix generalisation of `StateDep.Heisenberg`: the scalar
  cross-term `a₁·b₂` is promoted to the rank-one outer product `a₁ b₂ᵀ`.
  Once we show `H R d` is a `Group`, the abstract `Tree.eval_eq_prod`
  theorem in `StateDep.Scan` automatically gives parallel-scan correctness
  for any binary-tree re-association of a sequence of elements.
-/
import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Abel
import Mathlib.Algebra.Group.Defs
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.HeisenbergD

open Matrix

/-- Element of the d-dimensional Heisenberg group over a commutative
ring `R`. The `a` and `b` coordinates are vectors; `c` is a matrix that
accumulates the rank-one cross term `a₁ b₂ᵀ` at each composition. -/
@[ext]
structure H (R : Type*) (d : ℕ) where
  a : Fin d → R
  b : Fin d → R
  c : Matrix (Fin d) (Fin d) R

variable {R : Type*} [CommRing R] {d : ℕ}

instance : Mul (H R d) where
  mul x y := ⟨x.a + y.a, x.b + y.b, x.c + y.c + vecMulVec x.a y.b⟩

instance : One (H R d) := ⟨⟨0, 0, 0⟩⟩

instance : Inv (H R d) where
  inv x := ⟨-x.a, -x.b, -x.c + vecMulVec x.a x.b⟩

@[simp] theorem mul_a (x y : H R d) : (x * y).a = x.a + y.a := rfl
@[simp] theorem mul_b (x y : H R d) : (x * y).b = x.b + y.b := rfl
@[simp] theorem mul_c (x y : H R d) :
    (x * y).c = x.c + y.c + vecMulVec x.a y.b := rfl
@[simp] theorem one_a : (1 : H R d).a = 0 := rfl
@[simp] theorem one_b : (1 : H R d).b = 0 := rfl
@[simp] theorem one_c : (1 : H R d).c = 0 := rfl
@[simp] theorem inv_a (x : H R d) : (x⁻¹).a = -x.a := rfl
@[simp] theorem inv_b (x : H R d) : (x⁻¹).b = -x.b := rfl
@[simp] theorem inv_c (x : H R d) :
    (x⁻¹).c = -x.c + vecMulVec x.a x.b := rfl

/-- **The d-dimensional Heisenberg group is a group.** The vector
components reduce to `abel`/`ring` identities; the matrix component
uses the distributivity lemmas `add_vecMulVec` and `vecMulVec_add`
to split the cross term, after which `abel` closes the equation. -/
instance : Group (H R d) where
  mul_assoc x y z := by
    apply H.ext
    · change x.a + y.a + z.a = x.a + (y.a + z.a)
      abel
    · change x.b + y.b + z.b = x.b + (y.b + z.b)
      abel
    · change x.c + y.c + vecMulVec x.a y.b + z.c
            + vecMulVec (x.a + y.a) z.b =
           x.c + (y.c + z.c + vecMulVec y.a z.b)
            + vecMulVec x.a (y.b + z.b)
      rw [add_vecMulVec, vecMulVec_add]
      abel
  one_mul x := by
    apply H.ext
    · change (0 : Fin d → R) + x.a = x.a; abel
    · change (0 : Fin d → R) + x.b = x.b; abel
    · change (0 : Matrix (Fin d) (Fin d) R) + x.c
            + vecMulVec (0 : Fin d → R) x.b = x.c
      rw [show vecMulVec (0 : Fin d → R) x.b = 0 from by
            ext i j; simp [vecMulVec_apply]]
      abel
  mul_one x := by
    apply H.ext
    · change x.a + (0 : Fin d → R) = x.a; abel
    · change x.b + (0 : Fin d → R) = x.b; abel
    · change x.c + (0 : Matrix (Fin d) (Fin d) R)
            + vecMulVec x.a (0 : Fin d → R) = x.c
      rw [show vecMulVec x.a (0 : Fin d → R) = 0 from by
            ext i j; simp [vecMulVec_apply]]
      abel
  inv_mul_cancel x := by
    apply H.ext
    · change -x.a + x.a = (0 : Fin d → R); abel
    · change -x.b + x.b = (0 : Fin d → R); abel
    · change -x.c + vecMulVec x.a x.b + x.c
            + vecMulVec (-x.a) x.b = 0
      rw [neg_vecMulVec]
      abel

/-- Explicit closed-form for a three-fold product. The accumulated `c`
picks up three rank-one outer products: one for each ordered pair of
`a`-index before `b`-index. -/
example (x y z : H R d) :
    x * y * z =
      ⟨x.a + y.a + z.a,
       x.b + y.b + z.b,
       x.c + y.c + z.c
         + vecMulVec x.a y.b
         + vecMulVec x.a z.b
         + vecMulVec y.a z.b⟩ := by
  apply H.ext
  · rfl
  · rfl
  · change x.c + y.c + vecMulVec x.a y.b + z.c
          + vecMulVec (x.a + y.a) z.b =
         x.c + y.c + z.c
          + vecMulVec x.a y.b
          + vecMulVec x.a z.b
          + vecMulVec y.a z.b
    rw [add_vecMulVec]
    abel

/-- **Parallel-scan correctness for the d-dimensional Heisenberg group,
for free.** Any binary-tree re-association of a sequence of elements
yields the same group product as the sequential left-fold. -/
example (t : StateDep.Scan.Tree (H R d)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.HeisenbergD
