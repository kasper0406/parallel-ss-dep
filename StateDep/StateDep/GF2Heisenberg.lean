/-
  StateDep.GF2Heisenberg

  The GF(2)-Heisenberg / parity-capturing monoid.

  Instantiating the scalar Heisenberg group `H R` of `StateDep.Heisenberg`
  at the two-element field `R = ZMod 2` gives a three-bit state whose
  composition is

      (a₁, b₁, c₁) * (a₂, b₂, c₂)
        = (a₁ ⊕ a₂, b₁ ⊕ b₂, c₁ ⊕ c₂ ⊕ (a₁ ∧ b₂))

  where ⊕ is XOR and ∧ is AND. Feeding this monoid the per-step element
  `(xₜ, xₜ, 0)` for each input bit `xₜ ∈ {0, 1}` makes the `c` slot of
  the accumulated product equal to

      Σ_{i<j} xᵢ · xⱼ    (mod 2).

  Since every `xᵢ` is 0 or 1, `xᵢ · xⱼ = 1` iff both positions are 1s,
  so the sum counts *pairs of 1-positions*, giving

      c = (k choose 2)    (mod 2),    where k = popcount(x).

  `(k choose 2) mod 2` is the second-lowest bit of `k`; computing it
  requires resolving position-dependent pairs of bits and is provably
  *not* computable by a fixed-width linear RNN (cf. the bilinear-RNN
  analysis of Csordas 2025, Hahn 2020 "Theoretical limitations of
  self-attention", and Merrill–Sabharwal's log-precision Transformer
  limits). The GF(2)-Heisenberg monoid solves it with three bits of
  state and an O(1) associative combine.

  Parallel-scan correctness is inherited for free from
  `StateDep.Scan.Tree.eval_eq_prod` via the `Group (H (ZMod 2))`
  instance already proved in `StateDep.Heisenberg`.
-/
import StateDep.Heisenberg
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Nat.Choose.Basic

set_option linter.dupNamespace false

namespace StateDep.GF2Heisenberg

open StateDep.Heisenberg

/-- Shorthand for the GF(2)-Heisenberg element type. -/
abbrev GF2H := H (ZMod 2)

/-- Per-step input encoding: an input bit `x ∈ ZMod 2` becomes the
Heisenberg element `(x, x, 0)`. Feeding this into the scan makes the
running `a` and `b` slots both equal to the XOR-parity of the bits
seen so far, and the `c` slot equal to `Σ_{i<j} xᵢ xⱼ` (mod 2). -/
def parity_inputs (xs : List (ZMod 2)) : List GF2H :=
  xs.map (fun x => ⟨x, x, 0⟩)

@[simp] theorem parity_inputs_nil : parity_inputs [] = [] := rfl

@[simp] theorem parity_inputs_cons (x : ZMod 2) (xs : List (ZMod 2)) :
    parity_inputs (x :: xs) = ⟨x, x, 0⟩ :: parity_inputs xs := rfl

/-- Popcount (as a natural number) of a `List (ZMod 2)`: the number of
entries equal to `1`. -/
abbrev popcount (xs : List (ZMod 2)) : Nat :=
  (xs.filter (· = 1)).length

@[simp] theorem popcount_nil : popcount ([] : List (ZMod 2)) = 0 := rfl

theorem popcount_cons (x : ZMod 2) (xs : List (ZMod 2)) :
    popcount (x :: xs) = (if x = 1 then 1 else 0) + popcount xs := by
  unfold popcount
  by_cases hx : x = 1
  · simp [hx, Nat.add_comm]
  · simp [hx]

/-- Over `ZMod 2`, any element equals its own square: `x * x = x`. This
is because the two elements `0, 1` are each idempotent. -/
theorem sq_self (x : ZMod 2) : x * x = x := by
  fin_cases x <;> decide

/-- The `a`-slot of the scanned product is the XOR-parity of the inputs,
which in `ZMod 2` is the cast of the popcount. -/
theorem prod_a (xs : List (ZMod 2)) :
    ((parity_inputs xs).prod).a = (popcount xs : ZMod 2) := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp only [parity_inputs_cons, List.prod_cons, mul_a]
      rw [ih, popcount_cons]
      by_cases hx : x = 1
      · subst hx; push_cast; simp
      · have hx0 : x = 0 := by
          fin_cases x
          · rfl
          · exact absurd rfl hx
        subst hx0; simp

/-- The `b`-slot is symmetric with the `a`-slot: also the popcount
mod 2. -/
theorem prod_b (xs : List (ZMod 2)) :
    ((parity_inputs xs).prod).b = (popcount xs : ZMod 2) := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      simp only [parity_inputs_cons, List.prod_cons, mul_b]
      rw [ih, popcount_cons]
      by_cases hx : x = 1
      · subst hx; push_cast; simp
      · have hx0 : x = 0 := by
          fin_cases x
          · rfl
          · exact absurd rfl hx
        subst hx0; simp

/-- **Central theorem.** The `c`-slot of the scanned product is
`(popcount xs) choose 2`, viewed in `ZMod 2`.

Equivalently, this is the second-lowest bit of the popcount: a parity
function of the number of 1s that is provably not computable by a
fixed-width linear RNN, yet falls out of an O(1) associative combine
with a three-bit state.

*Proof idea.* Induction on `xs`. Writing `P` for the accumulated
product, the Heisenberg composition law gives
`P(x::xs).c = P(xs).c + x · P(xs).b`, and we already know
`P(xs).b = (popcount xs : ZMod 2)`. When `x = 0` nothing changes; when
`x = 1` the popcount grows by one and Pascal's identity
`(k+1).choose 2 = k.choose 2 + k.choose 1 = k.choose 2 + k` closes the
step, with the extra `k` contributed exactly by `x · P(xs).b`. -/
theorem prod_c_eq_popcount_choose_two (xs : List (ZMod 2)) :
    ((parity_inputs xs).prod).c = (popcount xs |>.choose 2 : ZMod 2) := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      have hb := prod_b xs
      simp only [parity_inputs_cons, List.prod_cons, mul_c]
      -- goal: 0 + (P xs).c + x * (P xs).b = ((popcount (x::xs)).choose 2 : ZMod 2)
      rw [ih, hb, popcount_cons]
      by_cases hx : x = 1
      · subst hx
        -- (1 + popcount xs).choose 2 = popcount xs + (popcount xs).choose 2 in ℕ
        have hpascal : (1 + popcount xs).choose 2 =
                 popcount xs + (popcount xs).choose 2 := by
          rw [Nat.add_comm 1 (popcount xs), Nat.choose_succ_succ,
              Nat.choose_one_right]
        simp only [if_true, hpascal]
        push_cast
        ring
      · have hx0 : x = 0 := by
          fin_cases x
          · rfl
          · exact absurd rfl hx
        subst hx0
        simp

/-- **Central theorem, re-stated with `List.filter` on the RHS** as in
the module's target signature. -/
theorem scan_c_eq_popcount_choose_two (xs : List (ZMod 2)) :
    ((parity_inputs xs).prod).c =
      ((xs.filter (· = 1)).length.choose 2 : ZMod 2) :=
  prod_c_eq_popcount_choose_two xs

/-- Smoke test: 4 ones → (4 choose 2) = 6 ≡ 0 (mod 2). -/
example : ((parity_inputs [1, 1, 1, 1]).prod).c = 0 := by decide

/-- Smoke test: 3 ones → (3 choose 2) = 3 ≡ 1 (mod 2). -/
example : ((parity_inputs [1, 1, 1]).prod).c = 1 := by decide

/-- Smoke test: 2 ones → (2 choose 2) = 1 ≡ 1 (mod 2). -/
example : ((parity_inputs [1, 1]).prod).c = 1 := by decide

/-- Smoke test: 6 ones → (6 choose 2) = 15 ≡ 1 (mod 2). -/
example : ((parity_inputs [1, 1, 1, 1, 1, 1]).prod).c = 1 := by decide

/-- **Parallel-scan correctness for GF(2)-Heisenberg, for free.** Any
binary-tree re-association of the per-step elements yields the same
accumulated state as the sequential left-fold. This is the instance of
`Tree.eval_eq_prod` at monoid `H (ZMod 2)` and is all that is needed to
license a parallel-scan kernel for the popcount-bit-2 primitive. -/
example (t : StateDep.Scan.Tree (H (ZMod 2))) :
    t.eval = t.toList.prod :=
  StateDep.Scan.Tree.eval_eq_prod t

end StateDep.GF2Heisenberg
