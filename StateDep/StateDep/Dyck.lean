/-
  StateDep.Dyck

  The Dyck / parenthesis-balance monoid.

  An element is a pair `(c, d) ∈ ℕ × ℕ` where:
  * `c` = number of *unmatched closers* at the start of the segment,
  * `d` = number of *unmatched openers* at the end of the segment.

  Any paren sequence reduces — via cancelling adjacent `()` — to a
  canonical word `)ᶜ (ᵈ`. Composition concatenates two such canonical
  words; the `d₁` openers at the end of segment 1 cancel against the
  `c₂` closers at the start of segment 2 until one runs out. If
  `m = min d₁ c₂` is the number of cancellations, then

      (c₁, d₁) * (c₂, d₂) = (c₁ + c₂ - m, d₁ + d₂ - m).

  Identity is `(0, 0)`. Token generators: `(` is `(0, 1)` and `)` is
  `(1, 0)`. A sequence is *balanced* iff its product is `(0, 0) = 1`.
-/
import Mathlib.Algebra.Group.Defs
import Mathlib.Tactic
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.Dyck

/-- Element of the Dyck-balance monoid: `c` unmatched closers at the
start, `d` unmatched openers at the end, of the canonical form `)ᶜ(ᵈ`. -/
@[ext]
structure Dyck where
  c : Nat  -- excess closers at the start
  d : Nat  -- excess openers at the end
  deriving Repr, DecidableEq

instance : Mul Dyck where
  mul x y :=
    let m := Nat.min x.d y.c
    ⟨x.c + y.c - m, x.d + y.d - m⟩

instance : One Dyck := ⟨⟨0, 0⟩⟩

@[simp] theorem mul_c (x y : Dyck) :
    (x * y).c = x.c + y.c - Nat.min x.d y.c := rfl
@[simp] theorem mul_d (x y : Dyck) :
    (x * y).d = x.d + y.d - Nat.min x.d y.c := rfl
@[simp] theorem one_c : (1 : Dyck).c = 0 := rfl
@[simp] theorem one_d : (1 : Dyck).d = 0 := rfl

/-- The Dyck-balance monoid. Identity `(0,0)` is the empty word.
Associativity is a cancellation bookkeeping argument: in any triple
product `(x*y)*z` or `x*(y*z)`, the total number of cancelled
`()`-pairs equals `min x.d y.c + min (x.d + y.d - min x.d y.c) z.c`
(resp. symmetric for the other association), and both expressions
reduce — by `omega` once the `min`s are case-split — to the same
closed form in `c₁, d₁, c₂, d₂, c₃, d₃`. -/
instance : Monoid Dyck where
  mul_assoc x y z := by
    ext <;> (simp only [mul_c, mul_d, Nat.min_def]; split_ifs <;> omega)
  one_mul x := by
    ext <;> (simp only [mul_c, mul_d, one_c, one_d, Nat.min_def]
             split_ifs <;> omega)
  mul_one x := by
    ext <;> (simp only [mul_c, mul_d, one_c, one_d, Nat.min_def]
             split_ifs <;> omega)

/-- Single-token generator: an open paren `(`. -/
def openTok : Dyck := ⟨0, 1⟩
/-- Single-token generator: a close paren `)`. -/
def closeTok : Dyck := ⟨1, 0⟩

@[simp] theorem openTok_c : openTok.c = 0 := rfl
@[simp] theorem openTok_d : openTok.d = 1 := rfl
@[simp] theorem closeTok_c : closeTok.c = 1 := rfl
@[simp] theorem closeTok_d : closeTok.d = 0 := rfl

/-- One open followed by one close cancels to the empty word. -/
@[simp] theorem open_close : openTok * closeTok = 1 := by
  ext <;> simp [openTok, closeTok]

/-- But one close followed by one open does *not* cancel: `)(` has one
unmatched closer at the start and one unmatched opener at the end. -/
example : closeTok * openTok = ⟨1, 1⟩ := by
  ext <;> simp [openTok, closeTok]

/-- **Dyck is not a group.** The element `openTok = (` has no two-sided
inverse: `y * openTok = 1` forces `y.d + 1 = 0` (contradiction) because
`Nat.min y.d 0 = 0`. -/
theorem not_group : ¬ ∀ x : Dyck, ∃ y : Dyck, x * y = 1 ∧ y * x = 1 := by
  intro h
  obtain ⟨y, _, h2⟩ := h openTok
  have hd := congrArg Dyck.d h2
  simp only [mul_d, openTok_c, openTok_d, one_d, Nat.min_zero] at hd
  -- hd : y.d + 1 - 0 = 0, hence y.d + 1 = 0
  omega

/-- **Semantic correctness of the scan.** A list of paren tokens is
balanced iff its accumulated Dyck product is the identity, iff both
components of the product are zero. The two sides of the `↔` are
definitionally equal, but stating it gives the usable rewriting
lemma and documents the meaning of the monoid. -/
theorem balanced_iff_scan_zero (xs : List Dyck) :
    xs.prod = 1 ↔ (xs.prod.c = 0 ∧ xs.prod.d = 0) := by
  refine ⟨fun h => ?_, fun ⟨hc, hd⟩ => ?_⟩
  · rw [h]; exact ⟨one_c, one_d⟩
  · ext
    · simpa using hc
    · simpa using hd

/-- Worked example: the word `(())` is balanced. -/
example : [openTok, openTok, closeTok, closeTok].prod = 1 := by
  simp only [List.prod_cons, List.prod_nil, mul_one]
  ext <;> simp [openTok, closeTok]

/-- Worked example: the word `())(` is *not* balanced; it reduces to
its canonical form `)(`. -/
example : [openTok, closeTok, closeTok, openTok].prod = ⟨1, 1⟩ := by
  simp only [List.prod_cons, List.prod_nil, mul_one]
  ext <;> simp [openTok, closeTok]

/-- **Parallel-scan correctness for Dyck, for free.** Any binary-tree
schedule for combining a sequence of paren tokens yields the same
accumulated balance-state as the sequential left-fold. -/
example (t : StateDep.Scan.Tree Dyck) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.Dyck
