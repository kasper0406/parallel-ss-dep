/-
  StateDep.Affine

  The affine-map monoid. An element `(a, b)` represents the map
  `x ↦ a · x + b`. Composition (sequence-order: "first f, then g") is:
      (a₁, b₁) ∘ (a₂, b₂) = (a₂·a₁,  a₂·b₁ + b₂)

  This is the parametric workhorse underlying:
    • Mamba / S4 / S6 selective SSM  (h ← A_t h + B_t x_t)
    • Linear attention / RetNet / RWKV
    • DeltaNet at *chunk boundaries*  (affine summary of the chunk)

  Associativity is all we need for parallel-scan correctness, and it
  follows from the ambient ring laws on `R`.

  The matrix version — where `a ∈ R^{d×d}`, `b ∈ R^d` — is the same
  structure lifted through mathlib's `Matrix` API. Kept scalar here to
  keep the proof transparent; the matrix version is an exercise in
  applying `Matrix.mul_assoc` and `Matrix.mulVec_add` and we return to
  it when we need it for Sherman-Morrison closure.
-/
import Mathlib.Tactic.Ring
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.Affine

/-- Scalar affine map: `(a, b)` represents `x ↦ a·x + b`. -/
@[ext]
structure Affine (R : Type*) where
  a : R
  b : R
  deriving Repr

variable {R : Type*} [CommSemiring R]

/-- Composition in **reading order**: `f * g` represents "apply `f` first,
then `g`", i.e. `x ↦ g(f(x)) = g.a·(f.a·x + f.b) + g.b`. -/
instance : Mul (Affine R) where
  mul f g := ⟨g.a * f.a, g.a * f.b + g.b⟩

instance : One (Affine R) := ⟨⟨1, 0⟩⟩

@[simp] theorem mul_a (f g : Affine R) : (f * g).a = g.a * f.a := rfl
@[simp] theorem mul_b (f g : Affine R) : (f * g).b = g.a * f.b + g.b := rfl
@[simp] theorem one_a : (1 : Affine R).a = 1 := rfl
@[simp] theorem one_b : (1 : Affine R).b = 0 := rfl

instance : Monoid (Affine R) where
  mul_assoc f g h := by
    ext
    · change h.a * (g.a * f.a) = h.a * g.a * f.a; ring
    · change h.a * (g.a * f.b + g.b) + h.b
            = h.a * g.a * f.b + (h.a * g.b + h.b)
      ring
  one_mul f := by
    ext
    · change f.a * 1 = f.a; ring
    · change f.a * 0 + f.b = f.b; ring
  mul_one f := by
    ext
    · change 1 * f.a = f.a; ring
    · change 1 * f.b + 0 = f.b; ring

/-- The action on state: `Affine.apply (a,b) x = a·x + b`. -/
def Affine.apply (f : Affine R) (x : R) : R := f.a * x + f.b

/-- Composition of affine maps corresponds to function composition —
the main "correctness" theorem for an affine scan. -/
theorem Affine.apply_mul (f g : Affine R) (x : R) :
    (f * g).apply x = g.apply (f.apply x) := by
  simp [Affine.apply]; ring

/-- Parallel-scan correctness for affine, inherited from the abstract
theorem. -/
example (t : StateDep.Scan.Tree (Affine R)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.Affine
