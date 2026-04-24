/-
  StateDep.Unipotent

  The unipotent group `U₄(R)` of 4×4 upper-triangular matrices with
  1s on the diagonal, presented as a state-transition monoid on six
  scalar parameters — one per above-diagonal entry.

  An element is
      ⎡1 x₁₂ x₁₃ x₁₄⎤
      ⎢0 1   x₂₃ x₂₄⎥
      ⎢0 0   1   x₃₄⎥
      ⎣0 0   0   1  ⎦
  and matrix multiplication gives the entry-wise rule
      (M₁ M₂)_{i,j} = M₁_{ij} + M₂_{ij} + Σ_{i<k<j} M₁_{ik}·M₂_{kj}.

  Concretely:
    * x₁₂, x₂₃, x₃₄ are additive (length-1 paths)
    * x₁₃, x₂₄ pick up a bilinear cross-term (length-2 paths)
    * x₁₄ picks up *two* independent bilinear cross-terms — a
      length-2 path through index 2 and a length-2 path through
      index 3. Iterating the product across a sequence therefore
      accumulates a **trilinear** correlation
           Σ_{i<j<k} (step i).x₁₂ · (step j).x₂₃ · (step k).x₃₄
      in the (1,4) slot: the "correlation-order knob" sits here.

  `U₄(R)` is the n=4 analogue of Heisenberg (`U₃`), which lives in
  `StateDep.Heisenberg` under the name `H`.

  Once we have `Group (U4 R)`, the abstract `Tree.eval_eq_prod` in
  `StateDep.Scan` immediately licenses parallel-scan kernels.
-/
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Group.Defs
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.Unipotent

/-- Element of the 4×4 unipotent upper-triangular group `U₄(R)`,
represented by its six above-diagonal entries. -/
@[ext]
structure U4 (R : Type*) where
  x12 : R
  x13 : R
  x14 : R
  x23 : R
  x24 : R
  x34 : R
  deriving Repr

variable {R : Type*} [CommRing R]

instance : Mul (U4 R) where
  mul M N :=
    { x12 := M.x12 + N.x12
      x23 := M.x23 + N.x23
      x34 := M.x34 + N.x34
      x13 := M.x13 + N.x13 + M.x12 * N.x23
      x24 := M.x24 + N.x24 + M.x23 * N.x34
      x14 := M.x14 + N.x14 + M.x12 * N.x24 + M.x13 * N.x34 }

/-- Identity: the zero 6-tuple represents the identity matrix. -/
instance : One (U4 R) := ⟨⟨0, 0, 0, 0, 0, 0⟩⟩

/-- Inverse via the truncated Neumann series `I - N + N² - N³` where
`N = M - I`. For U₄ we have `N⁴ = 0`, so three correction terms
suffice. The entries of `N²`, `N³` are the "missing" path sums:
`N²_{13} = x₁₂·x₂₃`, `N²_{24} = x₂₃·x₃₄`,
`N²_{14} = x₁₂·x₂₄ + x₁₃·x₃₄`, `N³_{14} = x₁₂·x₂₃·x₃₄`. -/
instance : Inv (U4 R) where
  inv M :=
    { x12 := -M.x12
      x23 := -M.x23
      x34 := -M.x34
      x13 := -M.x13 + M.x12 * M.x23
      x24 := -M.x24 + M.x23 * M.x34
      x14 := -M.x14 + M.x12 * M.x24 + M.x13 * M.x34 - M.x12 * M.x23 * M.x34 }

@[simp] theorem mul_x12 (M N : U4 R) : (M * N).x12 = M.x12 + N.x12 := rfl
@[simp] theorem mul_x23 (M N : U4 R) : (M * N).x23 = M.x23 + N.x23 := rfl
@[simp] theorem mul_x34 (M N : U4 R) : (M * N).x34 = M.x34 + N.x34 := rfl
@[simp] theorem mul_x13 (M N : U4 R) :
    (M * N).x13 = M.x13 + N.x13 + M.x12 * N.x23 := rfl
@[simp] theorem mul_x24 (M N : U4 R) :
    (M * N).x24 = M.x24 + N.x24 + M.x23 * N.x34 := rfl
@[simp] theorem mul_x14 (M N : U4 R) :
    (M * N).x14 = M.x14 + N.x14 + M.x12 * N.x24 + M.x13 * N.x34 := rfl

@[simp] theorem one_x12 : (1 : U4 R).x12 = 0 := rfl
@[simp] theorem one_x13 : (1 : U4 R).x13 = 0 := rfl
@[simp] theorem one_x14 : (1 : U4 R).x14 = 0 := rfl
@[simp] theorem one_x23 : (1 : U4 R).x23 = 0 := rfl
@[simp] theorem one_x24 : (1 : U4 R).x24 = 0 := rfl
@[simp] theorem one_x34 : (1 : U4 R).x34 = 0 := rfl

@[simp] theorem inv_x12 (M : U4 R) : (M⁻¹).x12 = -M.x12 := rfl
@[simp] theorem inv_x23 (M : U4 R) : (M⁻¹).x23 = -M.x23 := rfl
@[simp] theorem inv_x34 (M : U4 R) : (M⁻¹).x34 = -M.x34 := rfl
@[simp] theorem inv_x13 (M : U4 R) :
    (M⁻¹).x13 = -M.x13 + M.x12 * M.x23 := rfl
@[simp] theorem inv_x24 (M : U4 R) :
    (M⁻¹).x24 = -M.x24 + M.x23 * M.x34 := rfl
@[simp] theorem inv_x14 (M : U4 R) :
    (M⁻¹).x14 =
      -M.x14 + M.x12 * M.x24 + M.x13 * M.x34 - M.x12 * M.x23 * M.x34 := rfl

/-- **`U₄(R)` is a group.** Associativity splits into six component
goals via `ext`; each reduces to a ring identity. The (1,4)
component is the only nontrivial one — expanding it produces the
trilinear `x.x12 · y.x23 · z.x34` cross-term on both sides, which
`ring` cancels. -/
instance : Group (U4 R) where
  mul_assoc x y z := by
    ext
    · change x.x12 + y.x12 + z.x12 = x.x12 + (y.x12 + z.x12); ring
    · change x.x13 + y.x13 + x.x12 * y.x23 + z.x13
              + (x.x12 + y.x12) * z.x23
            = x.x13 + (y.x13 + z.x13 + y.x12 * z.x23)
              + x.x12 * (y.x23 + z.x23)
      ring
    · change x.x14 + y.x14 + x.x12 * y.x24 + x.x13 * y.x34 + z.x14
              + (x.x12 + y.x12) * z.x24
              + (x.x13 + y.x13 + x.x12 * y.x23) * z.x34
            = x.x14
              + (y.x14 + z.x14 + y.x12 * z.x24 + y.x13 * z.x34)
              + x.x12 * (y.x24 + z.x24 + y.x23 * z.x34)
              + x.x13 * (y.x34 + z.x34)
      ring
    · change x.x23 + y.x23 + z.x23 = x.x23 + (y.x23 + z.x23); ring
    · change x.x24 + y.x24 + x.x23 * y.x34 + z.x24
              + (x.x23 + y.x23) * z.x34
            = x.x24 + (y.x24 + z.x24 + y.x23 * z.x34)
              + x.x23 * (y.x34 + z.x34)
      ring
    · change x.x34 + y.x34 + z.x34 = x.x34 + (y.x34 + z.x34); ring
  one_mul x := by ext <;> simp
  mul_one x := by ext <;> simp
  inv_mul_cancel x := by
    ext
    · change -x.x12 + x.x12 = 0; ring
    · change -x.x13 + x.x12 * x.x23 + x.x13 + -x.x12 * x.x23 = 0; ring
    · change -x.x14 + x.x12 * x.x24 + x.x13 * x.x34 - x.x12 * x.x23 * x.x34
              + x.x14
              + -x.x12 * x.x24
              + (-x.x13 + x.x12 * x.x23) * x.x34 = 0
      ring
    · change -x.x23 + x.x23 = 0; ring
    · change -x.x24 + x.x23 * x.x34 + x.x24 + -x.x23 * x.x34 = 0; ring
    · change -x.x34 + x.x34 = 0; ring

/-- Explicit closed form for a three-fold product. The (1,4) entry
exhibits the trilinear cross-term `x.x12 · y.x23 · z.x34` that is
the signature of U₄ composition. -/
example (x y z : U4 R) :
    x * y * z =
      { x12 := x.x12 + y.x12 + z.x12
        x23 := x.x23 + y.x23 + z.x23
        x34 := x.x34 + y.x34 + z.x34
        x13 := x.x13 + y.x13 + z.x13
                + x.x12 * y.x23 + x.x12 * z.x23 + y.x12 * z.x23
        x24 := x.x24 + y.x24 + z.x24
                + x.x23 * y.x34 + x.x23 * z.x34 + y.x23 * z.x34
        x14 := x.x14 + y.x14 + z.x14
                + x.x12 * y.x24 + x.x12 * z.x24 + y.x12 * z.x24
                + x.x13 * y.x34 + x.x13 * z.x34 + y.x13 * z.x34
                + x.x12 * y.x23 * z.x34 } := by
  ext
  · change x.x12 + y.x12 + z.x12 = x.x12 + y.x12 + z.x12; ring
  · change x.x13 + y.x13 + x.x12 * y.x23 + z.x13 + (x.x12 + y.x12) * z.x23
          = x.x13 + y.x13 + z.x13
              + x.x12 * y.x23 + x.x12 * z.x23 + y.x12 * z.x23
    ring
  · change x.x14 + y.x14 + x.x12 * y.x24 + x.x13 * y.x34 + z.x14
            + (x.x12 + y.x12) * z.x24
            + (x.x13 + y.x13 + x.x12 * y.x23) * z.x34
          = x.x14 + y.x14 + z.x14
              + x.x12 * y.x24 + x.x12 * z.x24 + y.x12 * z.x24
              + x.x13 * y.x34 + x.x13 * z.x34 + y.x13 * z.x34
              + x.x12 * y.x23 * z.x34
    ring
  · change x.x23 + y.x23 + z.x23 = x.x23 + y.x23 + z.x23; ring
  · change x.x24 + y.x24 + x.x23 * y.x34 + z.x24 + (x.x23 + y.x23) * z.x34
          = x.x24 + y.x24 + z.x24
              + x.x23 * y.x34 + x.x23 * z.x34 + y.x23 * z.x34
    ring
  · change x.x34 + y.x34 + z.x34 = x.x34 + y.x34 + z.x34; ring

/-- **Parallel-scan correctness for U₄, for free.** Any binary-tree
re-association of a sequence of U₄ elements yields the same group
product as the sequential left-fold. This is the instance of
`Tree.eval_eq_prod` at the monoid `U4 R`. -/
example (t : StateDep.Scan.Tree (U4 R)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.Unipotent
