/-
  StateDep.Delta

  The algebra of rank-1 perturbations of identity — the backbone of
  DeltaNet's parallel-scan form.

  Each token contributes a transition `I - β·kkᵀ` (times `S` on the right,
  plus a `v·kᵀ` additive term; here we isolate the multiplicative part).
  The key fact is that the product of two rank-1 perturbations is a rank-2
  perturbation expressible via a single extra outer product and the dot
  product `k₁·k₂`:

      (I - β₁·k₁k₁ᵀ)(I - β₂·k₂k₂ᵀ)
        = I - β₁·k₁k₁ᵀ - β₂·k₂k₂ᵀ + β₁β₂·(k₁·k₂)·k₁k₂ᵀ

  So the space of rank-1 perturbations is **not** closed under composition;
  composition increases the rank by 1 per step. Within a chunk of length
  `n`, the accumulated matrix lives in the rank-≤-n family
  `{I - K D Kᵀ : K ∈ R^(d×n), D ∈ R^(n×n)}`, and Sherman-Morrison tells
  us exactly how to update `K` and `D` at each step.

  The practical implication is that DeltaNet's "chunkwise parallel" form
  is forced by the algebra: we run an exact scan within a chunk (state
  size grows linearly inside the chunk) and then materialise a bounded-
  rank representation at the chunk boundary.
-/
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Tactic.Abel
import StateDep.Scan

namespace StateDep.Delta

open Matrix

variable {R : Type*} [CommRing R] {d : ℕ}

/-- Multiplying two outer-product matrices collapses via the dot product:
    `(k₁k₁ᵀ)·(k₂k₂ᵀ) = (k₁·k₂)·k₁k₂ᵀ`.
This is the *one* nontrivial matrix identity underneath Sherman-Morrison. -/
theorem outer_mul_outer (k₁ k₂ : Fin d → R) :
    vecMulVec k₁ k₁ * vecMulVec k₂ k₂ = (k₁ ⬝ᵥ k₂) • vecMulVec k₁ k₂ := by
  rw [vecMulVec_mul_vecMulVec, vecMulVec_smul]

/-- **Sherman-Morrison-style composition.** The product of two rank-1
perturbations of the identity expands into three rank-1 pieces:
the originals, plus a single cross term of rank 1. The cross-term weight
is the dot product `k₁·k₂` times the product of the input weights.

This statement is the reason DeltaNet's scan combine-op has the shape
it does. -/
theorem rank1_comp (β₁ β₂ : R) (k₁ k₂ : Fin d → R) :
    ((1 : Matrix (Fin d) (Fin d) R) - β₁ • vecMulVec k₁ k₁) *
        (1 - β₂ • vecMulVec k₂ k₂)
      = 1 - β₁ • vecMulVec k₁ k₁ - β₂ • vecMulVec k₂ k₂
          + (β₁ * β₂ * (k₁ ⬝ᵥ k₂)) • vecMulVec k₁ k₂ := by
  simp only [sub_mul, mul_sub, smul_sub, one_mul, mul_one, smul_mul_assoc,
             mul_smul_comm, smul_smul, outer_mul_outer]
  have h : β₂ * (β₁ * (k₁ ⬝ᵥ k₂)) = β₁ * β₂ * (k₁ ⬝ᵥ k₂) := by ring
  rw [h]
  abel

/-- Parallel-scan correctness for matrix multiplication is already in
mathlib: `Matrix (Fin d) (Fin d) R` is a `Monoid` under `*`. So *any*
associative combine rule on matrices (including the rank-1 perturbation
one, once we've exhibited the closure theorem above) inherits parallel-
scan correctness from the abstract theorem. -/
example :
    ∀ t : StateDep.Scan.Tree (Matrix (Fin d) (Fin d) R),
      t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod

end StateDep.Delta
