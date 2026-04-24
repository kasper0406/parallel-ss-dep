/-
  StateDep.Signature

  The level-3 truncated tensor-algebra signature over `R^d` as a
  state-transition monoid.

  An element at level ≤ 3 is a 4-tuple
      (s, v, M, T) : R × (Fin d → R)
                       × Matrix (Fin d) (Fin d) R
                       × (Fin d → Fin d → Fin d → R)
  where `s` is the grade-0 scalar, `v` the grade-1 vector, `M` the
  grade-2 matrix, and `T` the grade-3 tensor.

  Composition is the graded tensor-algebra product truncated at level 3:

      (s₁,v₁,M₁,T₁) · (s₂,v₂,M₂,T₂) =
        ( s₁·s₂,
          s₁·v₂ + s₂·v₁,
          s₁·M₂ + s₂·M₁ + vecMulVec v₁ v₂,
          s₁·T₂ + s₂·T₁ + v₁ ⊗ M₂ + M₁ ⊗ v₂ )

  where `v₁ ⊗ M₂` at index `(i,j,k)` is `v₁ i · M₂ j k`, and
  `M₁ ⊗ v₂` at `(i,j,k)` is `M₁ i j · v₂ k`.

  Identity is `(1, 0, 0, 0)`. Associative but not commutative: the `M`
  slot alone is enough to see this (`vecMulVec v₁ v₂ ≠ vecMulVec v₂ v₁`
  in general).

  Once we show `Sig R d` is a `Monoid`, the abstract `Tree.eval_eq_prod`
  theorem in `StateDep.Scan` automatically gives parallel-scan
  correctness for any binary-tree re-association of a sequence of
  signature elements.
-/
import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Abel
import Mathlib.Algebra.Group.Defs
import StateDep.Scan

set_option linter.dupNamespace false

namespace StateDep.Signature

open Matrix

/-- Element of the level-3 truncated tensor-algebra signature over
`R^d`. The four slots carry the grade-0, grade-1, grade-2 and grade-3
tensors respectively. -/
@[ext]
structure Sig (R : Type*) (d : ℕ) where
  s : R
  v : Fin d → R
  M : Matrix (Fin d) (Fin d) R
  T : Fin d → Fin d → Fin d → R

variable {R : Type*} [CommRing R] {d : ℕ}

/-- Scalar multiplication on a vector, used in the grade-1 slot of the
product. We inline it as a function rather than going through `SMul`
to keep the component-wise proofs transparent. -/
def smulVec (s : R) (v : Fin d → R) : Fin d → R := fun i => s * v i

/-- Scalar multiplication on a matrix, used in the grade-2 slot of the
product. -/
def smulMat (s : R) (M : Matrix (Fin d) (Fin d) R) :
    Matrix (Fin d) (Fin d) R := fun i j => s * M i j

/-- Scalar multiplication on a 3-tensor, used in the grade-3 slot of
the product. -/
def smulTen (s : R) (T : Fin d → Fin d → Fin d → R) :
    Fin d → Fin d → Fin d → R := fun i j k => s * T i j k

@[simp] theorem smulVec_apply (s : R) (v : Fin d → R) (i : Fin d) :
    smulVec s v i = s * v i := rfl

@[simp] theorem smulMat_apply (s : R) (M : Matrix (Fin d) (Fin d) R)
    (i j : Fin d) : smulMat s M i j = s * M i j := rfl

@[simp] theorem smulTen_apply (s : R) (T : Fin d → Fin d → Fin d → R)
    (i j k : Fin d) : smulTen s T i j k = s * T i j k := rfl

/-- The grade-(1,2) outer product: `vecMat v M` at `(i,j,k) = v i · M j k`.
This is the `v₁ ⊗ M₂` tensor appearing in the `T` slot of the signature
product. -/
def vecMat (v : Fin d → R) (M : Matrix (Fin d) (Fin d) R) :
    Fin d → Fin d → Fin d → R :=
  fun i j k => v i * M j k

/-- The grade-(2,1) outer product: `matVec M v` at `(i,j,k) = M i j · v k`.
This is the `M₁ ⊗ v₂` tensor appearing in the `T` slot of the signature
product. -/
def matVec (M : Matrix (Fin d) (Fin d) R) (v : Fin d → R) :
    Fin d → Fin d → Fin d → R :=
  fun i j k => M i j * v k

@[simp] theorem vecMat_apply (v : Fin d → R) (M : Matrix (Fin d) (Fin d) R)
    (i j k : Fin d) : vecMat v M i j k = v i * M j k := rfl

@[simp] theorem matVec_apply (M : Matrix (Fin d) (Fin d) R) (v : Fin d → R)
    (i j k : Fin d) : matVec M v i j k = M i j * v k := rfl

/-- Linearity of `vecMat` in its first (vector) argument. -/
theorem add_vecMat (v₁ v₂ : Fin d → R) (M : Matrix (Fin d) (Fin d) R) :
    vecMat (v₁ + v₂) M = vecMat v₁ M + vecMat v₂ M := by
  funext i j k; simp [vecMat]; ring

/-- Linearity of `vecMat` in its second (matrix) argument. -/
theorem vecMat_add (v : Fin d → R) (M₁ M₂ : Matrix (Fin d) (Fin d) R) :
    vecMat v (M₁ + M₂) = vecMat v M₁ + vecMat v M₂ := by
  funext i j k; simp [vecMat, Matrix.add_apply]; ring

/-- Scalar pulling through `vecMat` on the left. -/
theorem smulVec_vecMat (s : R) (v : Fin d → R)
    (M : Matrix (Fin d) (Fin d) R) :
    vecMat (smulVec s v) M = smulTen s (vecMat v M) := by
  funext i j k; simp [vecMat, smulVec, smulTen]; ring

/-- Scalar pulling through `vecMat` on the right. -/
theorem vecMat_smulMat (s : R) (v : Fin d → R)
    (M : Matrix (Fin d) (Fin d) R) :
    vecMat v (smulMat s M) = smulTen s (vecMat v M) := by
  funext i j k; simp [vecMat, smulMat, smulTen]; ring

/-- Linearity of `matVec` in its first (matrix) argument. -/
theorem add_matVec (M₁ M₂ : Matrix (Fin d) (Fin d) R) (v : Fin d → R) :
    matVec (M₁ + M₂) v = matVec M₁ v + matVec M₂ v := by
  funext i j k; simp [matVec, Matrix.add_apply]; ring

/-- Linearity of `matVec` in its second (vector) argument. -/
theorem matVec_add (M : Matrix (Fin d) (Fin d) R) (v₁ v₂ : Fin d → R) :
    matVec M (v₁ + v₂) = matVec M v₁ + matVec M v₂ := by
  funext i j k; simp [matVec]; ring

/-- Scalar pulling through `matVec` on the left. -/
theorem smulMat_matVec (s : R) (M : Matrix (Fin d) (Fin d) R)
    (v : Fin d → R) :
    matVec (smulMat s M) v = smulTen s (matVec M v) := by
  funext i j k; simp [matVec, smulMat, smulTen]; ring

/-- Scalar pulling through `matVec` on the right. -/
theorem matVec_smulVec (s : R) (M : Matrix (Fin d) (Fin d) R)
    (v : Fin d → R) :
    matVec M (smulVec s v) = smulTen s (matVec M v) := by
  funext i j k; simp [matVec, smulVec, smulTen]; ring

/-- Scalar pulling through `vecMulVec` on the left. -/
theorem smulVec_vecMulVec (s : R) (v w : Fin d → R) :
    vecMulVec (smulVec s v) w = smulMat s (vecMulVec v w) := by
  ext i j; simp [vecMulVec_apply, smulVec, smulMat]; ring

/-- Scalar pulling through `vecMulVec` on the right. -/
theorem vecMulVec_smulVec (s : R) (v w : Fin d → R) :
    vecMulVec v (smulVec s w) = smulMat s (vecMulVec v w) := by
  ext i j; simp [vecMulVec_apply, smulVec, smulMat]; ring

instance : Mul (Sig R d) where
  mul x y := ⟨x.s * y.s,
              smulVec x.s y.v + smulVec y.s x.v,
              smulMat x.s y.M + smulMat y.s x.M + vecMulVec x.v y.v,
              smulTen x.s y.T + smulTen y.s x.T
                + vecMat x.v y.M + matVec x.M y.v⟩

instance : One (Sig R d) := ⟨⟨1, 0, 0, 0⟩⟩

@[simp] theorem mul_s (x y : Sig R d) : (x * y).s = x.s * y.s := rfl
@[simp] theorem mul_v (x y : Sig R d) :
    (x * y).v = smulVec x.s y.v + smulVec y.s x.v := rfl
@[simp] theorem mul_M (x y : Sig R d) :
    (x * y).M = smulMat x.s y.M + smulMat y.s x.M + vecMulVec x.v y.v := rfl
@[simp] theorem mul_T (x y : Sig R d) :
    (x * y).T = smulTen x.s y.T + smulTen y.s x.T
                  + vecMat x.v y.M + matVec x.M y.v := rfl
@[simp] theorem one_s : (1 : Sig R d).s = 1 := rfl
@[simp] theorem one_v : (1 : Sig R d).v = 0 := rfl
@[simp] theorem one_M : (1 : Sig R d).M = 0 := rfl
@[simp] theorem one_T : (1 : Sig R d).T = 0 := rfl

/-- `smulVec 1 = id`. -/
@[simp] theorem smulVec_one (v : Fin d → R) : smulVec 1 v = v := by
  funext i; simp [smulVec]

/-- `smulMat 1 = id`. -/
@[simp] theorem smulMat_one (M : Matrix (Fin d) (Fin d) R) :
    smulMat 1 M = M := by
  ext i j; simp [smulMat]

/-- `smulTen 1 = id`. -/
@[simp] theorem smulTen_one (T : Fin d → Fin d → Fin d → R) :
    smulTen 1 T = T := by
  funext i j k; simp [smulTen]

/-- `smulVec _ 0 = 0`. -/
@[simp] theorem smulVec_zero (s : R) : smulVec s (0 : Fin d → R) = 0 := by
  funext i; simp [smulVec]

/-- `smulMat _ 0 = 0`. -/
@[simp] theorem smulMat_zero (s : R) :
    smulMat s (0 : Matrix (Fin d) (Fin d) R) = 0 := by
  ext i j; simp [smulMat]

/-- `smulTen _ 0 = 0`. -/
@[simp] theorem smulTen_zero (s : R) :
    smulTen s (0 : Fin d → Fin d → Fin d → R) = 0 := by
  funext i j k; simp [smulTen]

/-- `smulVec 0 = 0`. -/
@[simp] theorem zero_smulVec (v : Fin d → R) : smulVec 0 v = 0 := by
  funext i; simp [smulVec]

/-- `smulMat 0 = 0`. -/
@[simp] theorem zero_smulMat (M : Matrix (Fin d) (Fin d) R) :
    smulMat 0 M = 0 := by
  ext i j; simp [smulMat]

/-- `smulTen 0 = 0`. -/
@[simp] theorem zero_smulTen (T : Fin d → Fin d → Fin d → R) :
    smulTen 0 T = 0 := by
  funext i j k; simp [smulTen]

/-- Compatibility of `smulVec` with multiplication of scalars. -/
theorem smulVec_smulVec (a b : R) (v : Fin d → R) :
    smulVec a (smulVec b v) = smulVec (a * b) v := by
  funext i; simp [smulVec]; ring

/-- Compatibility of `smulMat` with multiplication of scalars. -/
theorem smulMat_smulMat (a b : R) (M : Matrix (Fin d) (Fin d) R) :
    smulMat a (smulMat b M) = smulMat (a * b) M := by
  ext i j; simp [smulMat]; ring

/-- Compatibility of `smulTen` with multiplication of scalars. -/
theorem smulTen_smulTen (a b : R) (T : Fin d → Fin d → Fin d → R) :
    smulTen a (smulTen b T) = smulTen (a * b) T := by
  funext i j k; simp [smulTen]; ring

/-- `smulVec` distributes over addition of vectors. -/
theorem smulVec_add (s : R) (v w : Fin d → R) :
    smulVec s (v + w) = smulVec s v + smulVec s w := by
  funext i; simp [smulVec]; ring

/-- `smulMat` distributes over addition of matrices. -/
theorem smulMat_add (s : R) (M N : Matrix (Fin d) (Fin d) R) :
    smulMat s (M + N) = smulMat s M + smulMat s N := by
  ext i j; simp [smulMat, Matrix.add_apply]; ring

/-- `smulTen` distributes over addition of 3-tensors. -/
theorem smulTen_add (s : R) (T U : Fin d → Fin d → Fin d → R) :
    smulTen s (T + U) = smulTen s T + smulTen s U := by
  funext i j k; simp [smulTen]; ring

/-- **The truncated signature algebra is a monoid.** Associativity is
checked slot-by-slot: the scalar slot is a plain `ring`; the vector and
matrix slots reduce via the scalar-distributivity lemmas and
`abel`/`ring`; the 3-tensor slot is the hardest and needs the
three-tensor linearity lemmas `add_vecMat`, `vecMat_add`, `add_matVec`,
`matVec_add`, together with scalar pull-through on both outer products
and on `vecMulVec`. -/
instance : Monoid (Sig R d) where
  mul_assoc x y z := by
    apply Sig.ext
    · change x.s * y.s * z.s = x.s * (y.s * z.s); ring
    · -- vector slot
      change smulVec (x.s * y.s) z.v
              + smulVec z.s (smulVec x.s y.v + smulVec y.s x.v) =
             smulVec x.s (smulVec y.s z.v + smulVec z.s y.v)
              + smulVec (y.s * z.s) x.v
      rw [smulVec_add, smulVec_add, smulVec_smulVec, smulVec_smulVec,
          smulVec_smulVec, smulVec_smulVec]
      -- Now everything is of the form `smulVec (scalar) vec`; push scalars
      -- around by ring and equate.
      funext i
      simp [smulVec]
      ring
    · -- matrix slot
      change smulMat (x.s * y.s) z.M
              + smulMat z.s (smulMat x.s y.M + smulMat y.s x.M
                              + vecMulVec x.v y.v)
              + vecMulVec (smulVec x.s y.v + smulVec y.s x.v) z.v =
             smulMat x.s (smulMat y.s z.M + smulMat z.s y.M
                            + vecMulVec y.v z.v)
              + smulMat (y.s * z.s) x.M
              + vecMulVec x.v (smulVec y.s z.v + smulVec z.s y.v)
      rw [smulMat_add, smulMat_add, smulMat_add, smulMat_add,
          smulMat_smulMat, smulMat_smulMat, smulMat_smulMat,
          smulMat_smulMat,
          add_vecMulVec, vecMulVec_add,
          smulVec_vecMulVec, smulVec_vecMulVec,
          vecMulVec_smulVec, vecMulVec_smulVec]
      -- All `vecMulVec` outer products match; all `smulMat` terms are of
      -- the form `smulMat (product of x.s y.s z.s) _`; close by ext+ring.
      ext i j
      simp [smulMat, vecMulVec_apply]
      ring
    · -- 3-tensor slot: the hardest. Drop straight to pointwise and let
      -- `ring` handle the accumulated cubic expression.
      change smulTen (x.s * y.s) z.T
              + smulTen z.s (smulTen x.s y.T + smulTen y.s x.T
                              + vecMat x.v y.M + matVec x.M y.v)
              + vecMat (smulVec x.s y.v + smulVec y.s x.v) z.M
              + matVec (smulMat x.s y.M + smulMat y.s x.M
                          + vecMulVec x.v y.v)
                       z.v =
             smulTen x.s (smulTen y.s z.T + smulTen z.s y.T
                            + vecMat y.v z.M + matVec y.M z.v)
              + smulTen (y.s * z.s) x.T
              + vecMat x.v (smulMat y.s z.M + smulMat z.s y.M
                              + vecMulVec y.v z.v)
              + matVec x.M (smulVec y.s z.v + smulVec z.s y.v)
      funext i j k
      simp [smulTen, smulMat, smulVec, vecMat, matVec,
            vecMulVec_apply, Matrix.add_apply, Pi.add_apply]
      ring
  one_mul x := by
    apply Sig.ext
    · change 1 * x.s = x.s; ring
    · change smulVec 1 x.v + smulVec x.s (0 : Fin d → R) = x.v
      simp
    · change smulMat 1 x.M + smulMat x.s (0 : Matrix (Fin d) (Fin d) R)
              + vecMulVec (0 : Fin d → R) x.v = x.M
      simp [zero_vecMulVec]
    · change smulTen 1 x.T + smulTen x.s (0 : Fin d → Fin d → Fin d → R)
              + vecMat (0 : Fin d → R) x.M
              + matVec (0 : Matrix (Fin d) (Fin d) R) x.v = x.T
      have h1 : vecMat (0 : Fin d → R) x.M
              = (0 : Fin d → Fin d → Fin d → R) := by
        funext i j k; simp [vecMat]
      have h2 : matVec (0 : Matrix (Fin d) (Fin d) R) x.v
              = (0 : Fin d → Fin d → Fin d → R) := by
        funext i j k; simp [matVec]
      rw [h1, h2]
      simp
  mul_one x := by
    apply Sig.ext
    · change x.s * 1 = x.s; ring
    · change smulVec x.s (0 : Fin d → R) + smulVec 1 x.v = x.v
      simp
    · change smulMat x.s (0 : Matrix (Fin d) (Fin d) R) + smulMat 1 x.M
              + vecMulVec x.v (0 : Fin d → R) = x.M
      simp [vecMulVec_zero]
    · change smulTen x.s (0 : Fin d → Fin d → Fin d → R) + smulTen 1 x.T
              + vecMat x.v (0 : Matrix (Fin d) (Fin d) R)
              + matVec x.M (0 : Fin d → R) = x.T
      have h1 : vecMat x.v (0 : Matrix (Fin d) (Fin d) R)
              = (0 : Fin d → Fin d → Fin d → R) := by
        funext i j k; simp [vecMat]
      have h2 : matVec x.M (0 : Fin d → R)
              = (0 : Fin d → Fin d → Fin d → R) := by
        funext i j k; simp [matVec]
      rw [h1, h2]
      simp

/-- Explicit closed-form for a three-fold product at the two pure
"weight-1 per factor" inputs `(1, a, 0, 0)`, `(1, b, 0, 0)`,
`(1, c, 0, 0)`. The grade-3 slot `T` is the symmetric-product
accumulation of the three weight-1 vectors, reading three tensors:
`a⊗b`-with-`c`-from-the-right, `a⊗c`-from-`vecMat` of `M=a⊗b`, and
`b⊗c`-from the pairwise `y*z` step. At index `(i,j,k)` this reads
`a i · b j · c k`, the *ordered* triple product.

We state the three-fold product in closed form and let `ext; ring`
verify it component-wise. -/
example (a b c : Fin d → R) :
    (⟨1, a, 0, 0⟩ : Sig R d) * ⟨1, b, 0, 0⟩ * ⟨1, c, 0, 0⟩ =
      ⟨1,
       a + b + c,
       vecMulVec a b + vecMulVec a c + vecMulVec b c,
       fun i j k => a i * b j * c k⟩ := by
  apply Sig.ext
  · change 1 * 1 * 1 = 1; ring
  · -- vector slot
    change smulVec (1 * 1) c + smulVec 1 (smulVec 1 b + smulVec 1 a)
          = a + b + c
    funext i
    simp
    ring
  · -- matrix slot
    change smulMat (1 * 1) (0 : Matrix (Fin d) (Fin d) R)
            + smulMat 1 (smulMat 1 (0 : Matrix (Fin d) (Fin d) R)
                          + smulMat 1 (0 : Matrix (Fin d) (Fin d) R)
                          + vecMulVec a b)
            + vecMulVec (smulVec 1 b + smulVec 1 a) c
          = vecMulVec a b + vecMulVec a c + vecMulVec b c
    ext i j
    simp [vecMulVec_apply]
    ring
  · -- tensor slot
    change smulTen (1 * 1) (0 : Fin d → Fin d → Fin d → R)
            + smulTen 1 (smulTen 1 (0 : Fin d → Fin d → Fin d → R)
                          + smulTen 1 (0 : Fin d → Fin d → Fin d → R)
                          + vecMat a 0
                          + matVec 0 b)
            + vecMat (smulVec 1 b + smulVec 1 a) 0
            + matVec (smulMat 1 (0 : Matrix (Fin d) (Fin d) R)
                        + smulMat 1 (0 : Matrix (Fin d) (Fin d) R)
                        + vecMulVec a b) c
          = fun i j k => a i * b j * c k
    funext i j k
    simp [vecMat, matVec, vecMulVec_apply]

/-- **Parallel-scan correctness for the level-3 truncated tensor-algebra
signature, for free.** Any binary-tree re-association of a sequence of
signature elements yields the same product as the sequential left-fold. -/
example (t : StateDep.Scan.Tree (Sig R d)) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.Signature
