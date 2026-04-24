/-
  StateDep.Tropical

  The tropical (min-plus / max-plus) semiring as a state-transition monoid.

  Key idea: the tropical "multiplication" is the underlying additive
  monoid's `+`, and the tropical "addition" is `min` (or `max`). State
  updates of the form `h ← min(A + h, b)` are *nonlinear* in the usual
  sense yet fall inside an associative semiring. That buys parallelism
  for free.

  The sequence-modeling angle: tropical matrix-vector products compute
  *best-path / Viterbi* dynamics. Replacing `softmax(Qᵀ K / √d) · V`
  with min-plus matrix products gives hard attention with a parallel-scan
  form. Whether this is *useful* is an empirical question; this module
  just certifies that it is *legal*.

  Everything is a one-line consequence of mathlib's `Tropical` type plus
  the abstract scan theorem in `StateDep.Scan`.
-/
import Mathlib.Algebra.Tropical.Basic
import StateDep.Scan

namespace StateDep.Tropical

open StateDep.Scan

/-- **Tropical numbers form a monoid.** The monoid operation is the
underlying additive monoid's `+`. If `R` has `0` and associative `+`,
so does `Tropical R` with `1_trop = trop 0` and `trop a * trop b = trop (a+b)`. -/
example {R : Type*} [AddMonoid R] : Monoid (Tropical R) := inferInstance

/-- **Tropical scan correctness, for free.** Any binary-tree schedule
for computing the "product" of a list of tropical numbers equals the
left-fold. -/
example {R : Type*} [AddMonoid R] (t : Tree (Tropical R)) :
    t.eval = t.toList.prod :=
  Tree.eval_eq_prod t

/-- For a state `h_t = min (A_t + h_{t-1}, b_t)` style Viterbi cell, the
combine operator over pairs `(A, b) : R × R` with
    `(A₁, b₁) ∘ (A₂, b₂) = (A₁ + A₂, min (A₁ + b₂, b₁))`
is associative. We use `Min.min` via `LinearOrder`, and the proof is
just `min_assoc` + `add_assoc`. This is a *bona-fide* nonlinear
state-dependent update that admits a parallel scan. -/
@[ext]
structure Viterbi (R : Type*) where
  A : R  -- additive gate (cumulative delay / cost)
  b : R  -- bias injected at this step
  deriving Repr

namespace Viterbi
variable {R : Type*} [LinearOrder R] [Add R]

/-- Combine rule derived from composing two Viterbi updates
    `h' = min(A₁ + h, b₁)` and `h'' = min(A₂ + h', b₂)`. Unfold:
    h'' = min(A₂ + min(A₁+h, b₁), b₂)
        = min(min(A₂+A₁+h, A₂+b₁), b₂)
        = min((A₁+A₂)+h, min(A₂+b₁, b₂))
    so the composite is `(A = A₁+A₂, b = min(A₂+b₁, b₂))`. -/
def comp (x y : Viterbi R) : Viterbi R :=
  ⟨x.A + y.A, min (y.A + x.b) y.b⟩

end Viterbi

/-- Viterbi-cell associativity, over a linearly-ordered commutative
additive monoid with `min` distributing over `+` from the left (which
holds for any ordered additive group like ℝ or ℤ).

This doesn't need mathlib's `Tropical` — it's the same algebraic content
spelled out directly, closer to what a kernel implements. The proof
uses `min` associativity and the distributivity `c + min a b = min (c+a) (c+b)`. -/
theorem Viterbi.comp_assoc {R : Type*} [LinearOrder R] [AddCommMonoid R]
    [CovariantClass R R (· + ·) (· ≤ ·)]
    (x y z : Viterbi R) :
    Viterbi.comp (Viterbi.comp x y) z = Viterbi.comp x (Viterbi.comp y z) := by
  ext
  · -- A component
    change x.A + y.A + z.A = x.A + (y.A + z.A)
    exact add_assoc _ _ _
  · -- b component: distribute `z.A +` over `min`, then re-associate `min`.
    change min (z.A + min (y.A + x.b) y.b) z.b
          = min (y.A + z.A + x.b) (min (z.A + y.b) z.b)
    rw [← min_add_add_left, min_assoc]
    congr 1
    rw [← add_assoc, add_comm z.A y.A]

end StateDep.Tropical
