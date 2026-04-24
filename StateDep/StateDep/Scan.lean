/-
  StateDep.Scan

  The abstract parallel-scan correctness theorem.

  Any associative combine operation (i.e. any monoid) admits a parallel scan:
  however we re-associate the fold into a binary tree, the result equals the
  left-fold (i.e. the sequential scan). This is the *only* algebraic property
  needed to license a parallel-scan implementation.

  Every specific state family we investigate (Heisenberg, Tropical matrices,
  DeltaNet rank-1, first-order jets, ...) inherits this correctness the
  instant we exhibit a `Monoid` instance on its parameter type.
-/
import Mathlib.Algebra.BigOperators.Group.List.Basic

namespace StateDep.Scan

/-- Binary tree with leaves of type `α`. A particular parallel-reduction
schedule on a list corresponds to a tree with leaves in list order. -/
inductive Tree (α : Type*) where
  | leaf : α → Tree α
  | node : Tree α → Tree α → Tree α
  deriving Repr

namespace Tree

variable {α : Type*}

/-- In-order flattening of a tree to a list. -/
def toList : Tree α → List α
  | leaf x   => [x]
  | node l r => l.toList ++ r.toList

/-- Evaluate a tree whose leaves are monoid elements by recursive
multiplication. Concretely: left subtree's product times right subtree's
product. This is *any* parallel reduction schedule. -/
def eval {M : Type*} [Mul M] : Tree M → M
  | leaf x   => x
  | node l r => l.eval * r.eval

/-- **Parallel-scan correctness.** For any monoid `M`, any tree of elements
evaluates to the product of its flattened list. In particular, *any* parallel
schedule gives the same result as the sequential left-to-right scan. -/
theorem eval_eq_prod {M : Type*} [Monoid M] (t : Tree M) :
    t.eval = t.toList.prod := by
  induction t with
  | leaf x        => simp [eval, toList]
  | node l r ihl ihr =>
      simp [eval, toList, List.prod_append, ihl, ihr]

/-- Two different trees over the same list flatten to the same product,
hence evaluate to the same element. This is the statement used when a
kernel replaces one tree shape (e.g. a pairwise scan) with another (e.g.
a block-parallel scan). -/
theorem eval_eq_of_toList_eq {M : Type*} [Monoid M] {t₁ t₂ : Tree M}
    (h : t₁.toList = t₂.toList) : t₁.eval = t₂.eval := by
  rw [eval_eq_prod, eval_eq_prod, h]

end Tree

end StateDep.Scan
