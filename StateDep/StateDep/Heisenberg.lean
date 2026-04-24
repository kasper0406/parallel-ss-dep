/-
  StateDep.Heisenberg

  The Heisenberg group H(R) as a state-transition monoid.

  An element is a triple (a, b, c) ∈ R³. The composition law is
      (a₁, b₁, c₁) · (a₂, b₂, c₂) = (a₁+a₂, b₁+b₂, c₁+c₂+a₁·b₂)
  which is the 3×3 unipotent upper-triangular matrix product:
      ⎡1 a₁ c₁⎤ ⎡1 a₂ c₂⎤   ⎡1 a₁+a₂  c₁+c₂+a₁b₂⎤
      ⎢0 1  b₁⎥ ⎢0 1  b₂⎥ = ⎢0 1      b₁+b₂     ⎥
      ⎣0 0  1 ⎦ ⎣0 0  1 ⎦   ⎣0 0      1         ⎦

  Interpretation as an RNN state: `a` and `b` are two running sums that
  act like keys and values, and `c` accumulates a *bilinear cross-term*
  `Σᵢ<ⱼ aᵢ·bⱼ`. Composition is O(1), the operation is nonlinear (because
  of the a·b coupling), and the group is nonabelian.

  Once we show `H R` is a `Monoid` (actually a `Group`), the abstract
  `Tree.eval_eq_prod` theorem in `StateDep.Scan` automatically gives
  parallel-scan correctness.
-/
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Group.Defs
import StateDep.Scan

namespace StateDep.Heisenberg

/-- Element of the Heisenberg group over a commutative ring `R`. -/
@[ext]
structure H (R : Type*) where
  a : R
  b : R
  c : R
  deriving Repr

variable {R : Type*} [CommRing R]

instance : Mul (H R) where
  mul x y := ⟨x.a + y.a, x.b + y.b, x.c + y.c + x.a * y.b⟩

instance : One (H R) := ⟨⟨0, 0, 0⟩⟩

instance : Inv (H R) where
  inv x := ⟨-x.a, -x.b, -x.c + x.a * x.b⟩

@[simp] theorem mul_a (x y : H R) : (x * y).a = x.a + y.a := rfl
@[simp] theorem mul_b (x y : H R) : (x * y).b = x.b + y.b := rfl
@[simp] theorem mul_c (x y : H R) : (x * y).c = x.c + y.c + x.a * y.b := rfl
@[simp] theorem one_a : (1 : H R).a = 0 := rfl
@[simp] theorem one_b : (1 : H R).b = 0 := rfl
@[simp] theorem one_c : (1 : H R).c = 0 := rfl
@[simp] theorem inv_a (x : H R) : (x⁻¹).a = -x.a := rfl
@[simp] theorem inv_b (x : H R) : (x⁻¹).b = -x.b := rfl
@[simp] theorem inv_c (x : H R) : (x⁻¹).c = -x.c + x.a * x.b := rfl

/-- **The Heisenberg group is a group.** Every axiom reduces to
component-wise ring identities. We break the proof into the three
component goals so each can use exactly the tactic it needs. -/
instance : Group (H R) where
  mul_assoc x y z := by
    ext
    · change x.a + y.a + z.a = x.a + (y.a + z.a); ring
    · change x.b + y.b + z.b = x.b + (y.b + z.b); ring
    · change x.c + y.c + x.a * y.b + z.c + (x.a + y.a) * z.b =
             x.c + (y.c + z.c + y.a * z.b) + x.a * (y.b + z.b)
      ring
  one_mul x       := by ext <;> simp
  mul_one x       := by ext <;> simp
  inv_mul_cancel x := by
    ext
    · change -x.a + x.a = 0; ring
    · change -x.b + x.b = 0; ring
    · change -x.c + x.a * x.b + x.c + -x.a * x.b = 0; ring

/-- Sanity-check: H is nonabelian. The first-coordinate commutator
shows up in the c-entry. -/
example (x y : H R) :
    ((x * y).c - (y * x).c) = x.a * y.b - y.a * x.b := by
  simp; ring

/-- Explicit closed-form for a three-fold product. Useful for eyeballing
what the accumulated state looks like, and for kernel tests. -/
example (x y z : H R) :
    x * y * z =
      ⟨x.a + y.a + z.a,
       x.b + y.b + z.b,
       x.c + y.c + z.c + x.a * y.b + x.a * z.b + y.a * z.b⟩ := by
  ext
  · rfl
  · rfl
  · change x.c + y.c + x.a * y.b + z.c + (x.a + y.a) * z.b =
           x.c + y.c + z.c + x.a * y.b + x.a * z.b + y.a * z.b
    ring

/-- **Parallel-scan correctness for Heisenberg, for free.** Any binary-tree
re-association of a sequence of Heisenberg elements yields the same group
product as the sequential left-fold. This is the instance of
`Tree.eval_eq_prod` at monoid `H R` and is all we need to license a
parallel-scan kernel. -/
example (t : StateDep.Scan.Tree (H R)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.Heisenberg
