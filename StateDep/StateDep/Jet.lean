/-
  StateDep.Jet

  First-order jets (dual numbers) as a state-transition monoid.

  Element: `(val, tan) ∈ R × R`. Multiplication is the Leibniz / chain rule:
      (a₁, ȧ₁) · (a₂, ȧ₂) = (a₁·a₂, a₁·ȧ₂ + ȧ₁·a₂)
  which is the product in R[ε]/(ε²): if we write `x = a + ȧ·ε` then
  `(a₁+ȧ₁ε)(a₂+ȧ₂ε) = a₁a₂ + (a₁ȧ₂+ȧ₁a₂)·ε + ȧ₁ȧ₂·ε²` and ε²=0 kills
  the last term.

  Sequence-modeling angle: a state that *simultaneously* carries a value
  and its derivative. Composing updates applies the chain rule
  automatically. Any parallel scan accumulates the derivative along the
  sequence for free.
-/
import Mathlib.Tactic.Ring
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.Jet

@[ext]
structure Jet (R : Type*) where
  val : R  -- the value
  tan : R  -- the tangent (coefficient of ε)
  deriving Repr

variable {R : Type*} [CommRing R]

instance : Mul (Jet R) where
  mul x y := ⟨x.val * y.val, x.val * y.tan + x.tan * y.val⟩

instance : One (Jet R) := ⟨⟨1, 0⟩⟩

@[simp] theorem mul_val (x y : Jet R) : (x * y).val = x.val * y.val := rfl
@[simp] theorem mul_tan (x y : Jet R) :
    (x * y).tan = x.val * y.tan + x.tan * y.val := rfl
@[simp] theorem one_val : (1 : Jet R).val = 1 := rfl
@[simp] theorem one_tan : (1 : Jet R).tan = 0 := rfl

/-- Jet multiplication is a monoid. Associativity is the chain rule
applied twice; `one` is the constant-value jet. -/
instance : Monoid (Jet R) where
  mul_assoc x y z := by
    ext
    · change x.val * y.val * z.val = x.val * (y.val * z.val); ring
    · change x.val * y.val * z.tan + (x.val * y.tan + x.tan * y.val) * z.val
            = x.val * (y.val * z.tan + y.tan * z.val) + x.tan * (y.val * z.val)
      ring
  one_mul x := by
    ext
    · change 1 * x.val = x.val; ring
    · change 1 * x.tan + 0 * x.val = x.tan; ring
  mul_one x := by
    ext
    · change x.val * 1 = x.val; ring
    · change x.val * 0 + x.tan * 1 = x.tan; ring

/-- Parallel-scan correctness for jet products, inherited from the
abstract scan theorem. Concretely: whatever binary-tree schedule we use
to multiply a sequence of jets, we recover the same accumulated
`(val, tan)`. -/
example (t : StateDep.Scan.Tree (Jet R)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.Jet
