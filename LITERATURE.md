# LITERATURE — follow-up search on IDEAS.md candidates

## Methodology

For each top candidate I ran 3–5 targeted web queries (arXiv, proceedings,
blog posts) with 2024–2026 restrictions, then drilled into the full HTML
of the closest-match papers to verify the *recurrence structure* — i.e.
is the proposed algebraic object the actual carrier of the scan state,
or just a feature / gate / input? The critical novelty claim in
`IDEAS.md` is always "X as the monoid of a causal parallel scan inside
an LM layer", which most adjacent literature misses even when X is a
well-studied object. Where a paper looked close, I fetched it directly
to read the state-update equation; citations below come from those
primary verifications, not the snippet summaries.

---

## Per-candidate updates

### §2.1 Truncated tensor-algebra signature — revised verdict: 🟢 (strongly open)

**Adjacent work (closest):** [**SLiCEs / Structured Linear CDEs** (Walker
et al., NeurIPS 2025 spotlight, arXiv:2505.17761)](https://arxiv.org/abs/2505.17761)
and [**Log-NCDE** (Walker et al., ICML
2024, arXiv:2402.18512)](https://arxiv.org/abs/2402.18512). I fetched
both and read the state-update equations directly. **Neither uses the
tensor algebra as scan state.** In Log-NCDE, the log-signature is
computed over **fixed local windows** `[rᵢ, rᵢ₊₁]` and fed as a
*forcing term* into an ODE whose state `h ∈ ℝᵘ` is plain Euclidean.
SLiCEs makes this worse (for our purposes): the scan state is `ℝᵈʰ`,
combine is *matrix product* of block-diagonal / sparse / Walsh–Hadamard
transition matrices, and the tensor algebra only appears in the
expressivity proof as a theoretical ceiling. [**SigGate** (Genet &
Inzirillo, arXiv:2502.09318)](https://arxiv.org/abs/2502.09318) — which
is the hottest 2025 "signature × RNN" hit — uses signatures as a
**gate-input feature only**; the LSTM/GRU hidden state propagation is
unchanged. [**FRUITS / iterated-sums signature** (Diehl et al., DMKD
2024, arXiv:2311.14549)](https://arxiv.org/abs/2311.14549) computes ISS
features non-recurrently for a *linear classifier*.

**Citations the brainstorm missed:**
- [SLiCEs (NeurIPS '25 spotlight)](https://arxiv.org/abs/2505.17761) —
  *must cite*; closest published claim of "maximally expressive
  parallel-in-time via signature-flavored structure", but the state
  lives in ℝᵈʰ, not in T^{≤K}.
- [Log-NCDE (ICML '24)](https://arxiv.org/abs/2402.18512) — *must cite*;
  establishes that log-signatures + Lie brackets add expressivity, but
  via forcing, not via a Chen-product scan.
- [SigGate (2025)](https://arxiv.org/abs/2502.09318) — signature as
  gate feature, not state.
- [FRUITS / iterated-sums signature](https://arxiv.org/abs/2311.14549)
  — commutative semiring ISS, distinct from non-commutative Chen
  signature.

**What's still open:** The specific construction "scan state ∈ T^{≤K}(ℝᵈ),
combine = tensor-algebra Chen product (concatenation), so the scan
*is the signature of the path concatenation*" remains unpublished. The
associativity is Chen's identity, which — amazingly — no one seems to
have reformulated as a scan monoid for an LM.

### §2.22 Dyck / parenthesis-balance monoid — revised verdict: 🟢 (open)

**Adjacent work:** The Dyck limitation literature for Transformers /
SSMs is enormous and the brainstorm undercited it. Key papers:
[**Hahn 2020**](https://aclanthology.org/2020.tacl-1.11) (hard-attention
cannot recognize parity or Dyck-1),
[**Merrill–Sabharwal "Illusion of State"** (ICML
2024, arXiv:2404.08819)](https://arxiv.org/abs/2404.08819) (SSMs are in
TC⁰ and cannot do permutation composition, Dyck-2),
[**Strobl et al. TACL 2024
survey**](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00663/120983)
(consolidated Transformer-formal-language limits).
[**Hewitt et al. "RNNs can generate bounded hierarchical languages
with optimal memory"**](https://arxiv.org/abs/2010.07515) proves the
`O(m log k)` memory bound for Dyck-(k, m). These establish the problem;
nobody proposes a bracket-balance *monoid scan*. The closest related
work is the stack-RNN / pushdown-RNN literature (Joulin–Mikolov 2015,
Suzgun et al. 2019) which uses a **non-associative** stack; and
[**SD-SSM** (Hersche et al. AAAI
2025, arXiv:2412.19350)](https://arxiv.org/abs/2412.19350) plus
[**PD-SSM** (IBM, NeurIPS 2025 spotlight,
arXiv:2509.22284)](https://arxiv.org/abs/2509.22284), which track
*finite* automata via selected dense / sparse-plus-diagonal transition
matrices. Finite-state ≠ Dyck (which needs context-free counting).

**Citations the brainstorm missed:**
- [Merrill & Sabharwal, "The Illusion of State in SSMs" (ICML
  2024)](https://arxiv.org/abs/2404.08819) — *must cite*; pinpoints
  the TC⁰ limit that motivates §2.22.
- [Strobl et al. TACL
  2024](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00663/120983)
  — *must cite*; the canonical survey of Transformer formal-language
  results.
- [Hewitt et al., "RNNs can generate bounded hierarchical languages
  with optimal memory" (EMNLP
  2020)](https://arxiv.org/abs/2010.07515) — memory lower bound for
  Dyck-(k, m); our `(l, r)` pair is the minimal O(1)-state proxy.
- [PD-SSM (NeurIPS '25 spotlight)](https://arxiv.org/abs/2509.22284)
  and [SD-SSM (AAAI
  '25)](https://arxiv.org/abs/2412.19350) — closest parallel-scan
  models that *track finite automata*, but not Dyck.

**What's still open:** The two-counter bracket-balance monoid
`(l, r) * (l', r') = (l + (l' − r)⁺, r' + (r − l')⁺)` with relaxation
to ℝ² is, as far as I can find, genuinely unpublished as an NN scan
primitive. Its main selling point — O(1) state that provably reduces a
Dyck-1 substring correctly under divide-and-conquer — is not inherited
from any existing cell.

### §2.18 GF(2)-Heisenberg / parity — revised verdict: 🟡 (partially covered; the *mechanism* is new but the *goal* is now crowded)

**Adjacent work:** The "solve parity inside a linear-RNN scan" niche
was occupied in 2024–2025 by several papers that I missed in the
brainstorm. [**Grazzi et al. "Unlocking State-Tracking in Linear RNNs
Through Negative Eigenvalues"** (ICLR
2025, arXiv:2411.12537)](https://arxiv.org/abs/2411.12537) shows the
parity failure mode follows from the [0, 1] eigenvalue constraint on
diagonal SSMs, and that allowing `[−1, 1]` eigenvalues (equivalently a
sign-flip generator) suffices for parity and any 2-element permutation
group. [**DeltaProduct** (Siems et al., NeurIPS
2025, arXiv:2502.10297)](https://arxiv.org/abs/2502.10297) generalises
further via products of Householder matrices and handles A_5.
[**PD-SSM** (IBM,
arXiv:2509.22284)](https://arxiv.org/abs/2509.22284) does any FSA via
permutation × diagonal. [**Implicit LM = RNN** (Schöne et al., ICLR
2025, arXiv:2502.07827)](https://arxiv.org/abs/2502.07827) uses
fixed-point iteration. All of these solve parity *by construction*
inside a parallel scan — GF(2)-Heisenberg no longer has the
parity-primacy monopoly the brainstorm implied.

**That said**, none of them use GF(2) / modular arithmetic as the
scalar ring. They work in ℝ with a sign flip. The GF(2)-Heisenberg
angle — parity *of ordered bit-pairs*, not just parity — is still
unique, and the mod-p bit-efficient state is still unexplored.

**Citations the brainstorm missed:**
- [Grazzi et al. (ICLR '25)](https://arxiv.org/abs/2411.12537) —
  *must cite*; now the canonical "parity fix for linear RNNs".
- [DeltaProduct (NeurIPS
  '25)](https://arxiv.org/abs/2502.10297) — *must cite*; Householder
  product is a direct competitor on the state-tracking axis.
- [Sarrof et al. 2024 (same thread)](https://aclanthology.org/2024.naacl-short.4.pdf)
  — "Advancing Regular Language Reasoning in Linear RNNs", NAACL 2024,
  earliest version of the eigenvalue-sign argument.

**What's still open:** GF(2) / ZMod(p) Heisenberg with *ordered
bit-pair* statistic, bit-packed state. The "ordered-pair parity"
statistic in §2.18 is not subsumed by any of the above; they all give
single-index parity.

### §2.3 Truncated BCH / free-nilpotent — revised verdict: 🟢 (open as scan monoid; well-known as numerical integrator)

**Adjacent work:** Magnus / BCH integrators are standard in numerical
ODE ([Iserles–Munthe-Kaas–Nørsett–Zanna, *Lie-group methods*, Acta
Numerica 2000]) and appear in rough-path numerics. **Log-NCDE** (above)
comes *closest* — it uses log-signatures (i.e. free-Lie-algebra
elements) as the control signal, and the Log-ODE method is a Magnus /
Chen-Strichartz approximation. But again, the scan state is ℝᵘ, and
the Lie algebra lives only in the *vector-field input*. [**MEA (Matrix
Exponential Attention)** (Zhang, arXiv Dec 2025, based on HLA
2510.27258)](https://yifanzhang-pro.github.io/MEA/) approximates a
matrix exponential by truncated Taylor, which is orthogonal — no BCH,
just polynomial-in-attention-matrix.

**Citations the brainstorm missed:**
- [Log-NCDE](https://arxiv.org/abs/2402.18512) — the closest "Lie
  brackets in sequence modelling" paper; framing is different.
- [MEA (2025)](https://yifanzhang-pro.github.io/MEA/) — matrix-exp
  attention but *not* via BCH; confirms matrix-exp is in the air but
  the commutator-correction angle is not.

**What's still open:** "Scan state ∈ free-nilpotent Lie group of depth
K, combine = truncated BCH" is unpublished. The §2.25 variant
`(A₁, B₁) * (A₂, B₂) = (A₁ + A₂, B₁ + B₂ + [A₁, A₂]/2)` at K=2 is
genuinely novel and small enough to be the first Triton target after
multi-d Heisenberg.

### §2.7 Möbius / PGL₂ — revised verdict: 🟡 (the 1-D case is covered by KLA; multi-head / multi-d variants open)

**Adjacent work:** [**Kalman Linear Attention** (Shukla et al., arXiv
Feb 2026, 2602.10743)](https://arxiv.org/abs/2602.10743) is exactly
the 1-D Möbius scan; the README already cites it. The multi-head and
multi-dimensional PGL₂ variants (e.g. a per-head different Möbius
subgroup, or PGL₂(ℂ) with a hyperbolic interpretation) are not
published. [**MöbiusAttention** (Deng et al.,
arXiv:2409.12175)](https://arxiv.org/abs/2409.12175) uses Möbius maps
inside attention but **not as a scan monoid** — it reparametrises
query/key interactions in a static attention kernel.

**Citations:**
- [KLA](https://arxiv.org/abs/2602.10743) — already cited; the direct
  occupant of the 1-D slot.
- [MöbiusAttention (2024)](https://arxiv.org/abs/2409.12175) — *new
  citation*; occupies "Möbius as attention kernel", not as scan.

**What's still open:** Multi-head heterogeneous Möbius scan; hyperbolic
PGL₂(ℝ) with rapidity parameter; SL₂(ℂ) at complex parameters. The
§2.8 Lorentz scan is essentially in the same bucket.

### §2.10 Braid group scan — revised verdict: 🟢 (open)

**Adjacent work:** BraidNet / braid-arrangement NN use braids as
*static* structure (already noted in IDEAS.md). Nothing new found.
Closest dynamic work is **DeltaProduct** (Householder products =
elementary reflections in O(n), a *different* group from B_n).
[**BRAID**
(Saxena et al., Nat. Methods 2024, dead-ringer
name)](https://openreview.net/forum?id=3usdM1AuI3) is input-driven
nonlinear dynamics for neural data — unrelated to braid groups.

**Citations:** none new.

**What's still open:** Everything. Braid group normal-form scan is
genuinely unexplored. However, the Lean cost is large (normal-form
rewriter is not in mathlib), so this remains a great-idea-but-high-cost
candidate.

### §2.6 Count-min sketch monoid — revised verdict: 🟡 (algebra trivial, learned-sketch literature is large)

**Adjacent work:** [**Partitioned Learned Count-Min Sketch** (Aamand et
al., ICLR
2019+)](https://openreview.net/forum?id=7W4boWjb3Q), [**UCL-sketch**
(Zhang et al., arXiv:2412.03611)](https://arxiv.org/html/2412.03611v2),
[**Meta-Sketch** (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/25846).
All of these *learn* a sketch but treat it as a frequency-estimation
data structure, not as a scan primitive inside an LM layer. **The
reframing as a neural-scan monoid** is still what's novel here — the
algebra itself (matrix addition over hashed counters) is textbook.

**Citations:**
- [Partitioned Learned
  Count-Min](https://openreview.net/forum?id=7W4boWjb3Q) — *new
  citation* for "learned sketch" context.

**What's still open:** Count-min *decoded with min (tropical)*, **inside
a causal LM layer**, with gradient flowing through the nonlinear
readout. This is the distinguishing claim.

### §2.9 p-adic digit monoid — revised verdict: 🟢 (open, very wild)

**Adjacent work:** The only hits are theoretical — p-adic cellular NNs
(Zúñiga-Galindo), a p-adic-deep-learning blogpost, and recent work on
[**ultrametric / p-adic norm-based operations** (Nuñez & Murtagh, MDPI
Mathematics 2025)](https://www.mdpi.com/2227-7390/14/8/1284). Nothing
uses p-adic arithmetic as the scan state of an LM cell.

**Citations:** none worth adding beyond what the brainstorm already
had.

**What's still open:** Everything. The speculative high-wildness rating
in IDEAS.md stands.

### §2.12 Tropical × Heisenberg (max-plus Heisenberg)
— revised verdict: 🟡 (tropical-attention literature is now hot)

This wasn't on the priority list but the search turned up material that
forces a reranking. [**Tropical Attention** (Hashemi et al., NeurIPS
2025, arXiv:2505.17190)](https://arxiv.org/abs/2505.17190) builds an
attention mechanism in tropical projective space for neural algorithmic
reasoning (NP-hard combinatorial problems). [**The Geometry of Thought:
Transformer as Tropical Polynomial Circuit** (arXiv
2601.09775)](https://arxiv.org/abs/2601.09775) shows the high-confidence
softmax limit *is* a tropical matrix product — making attention
implicitly a Bellman-Ford scan. These don't cover §2.12 exactly
(neither uses the max-plus *Heisenberg* specifically), but the tropical
monoid niche is busier than IDEAS.md suggests.

**Citations to add:**
- [Tropical Attention (NeurIPS
  '25)](https://arxiv.org/abs/2505.17190)
- [Geometry of Thought (2026)](https://arxiv.org/abs/2601.09775)

**What's still open:** The *ordered-pair* max `c = max_{i≤j}(aᵢ + bⱼ)`
as Heisenberg's c-coordinate in the tropical semiring is *not* what
Tropical Attention does. Still novel as a cell.

---

## New candidates surfaced by the search

### 2.26 — Iterated-sums signature (ISS) over a commutative semiring as a scan monoid

- **Math.** For a commutative semiring `(S, ⊕, ⊙)`, the ISS of a
  sequence `(x₁, …, xₜ)` at word `w = (i₁, …, iₖ)` is
  `ISS_w = ⊕_{j₁<…<jₖ≤t} ⊙ₗ xⱼ_l^{iₗ}` (Diehl–Ebrahimi-Fard–Tapia 2020).
  This *is* a monoid under the quasi-shuffle law, giving an associative
  parallel scan. Dual to §2.1 (continuous signature) but discrete- and
  semiring-native.
- **State dim & combine.** Same `Σ dᵏ` for k ≤ K. Combine is the
  quasi-shuffle / stuffle product, `O(dᴷ · 2ᴷ)`.
- **Known?** [FRUITS](https://arxiv.org/abs/2311.14549) computes ISS
  features for classification (non-recurrent). No LM scan framing.
- **Wildness.** 3.

### 2.27 — PD-structure scan (permutation × diagonal)

- **Math.** State transition is `A_t = P_t · D_t` where `P_t` is a
  column one-hot permutation and `D_t` is complex-diagonal. Composition
  `P₁D₁ · P₂D₂ = (P₁P₂)(P₂⁻¹D₁P₂ · D₂)` stays PD *because P₂⁻¹D₁P₂ is
  still diagonal*.
- **Known?** This is
  [**PD-SSM** (NeurIPS '25 spotlight)](https://arxiv.org/abs/2509.22284).
  Already published. Listed here as a warning: any "sparse + diagonal"
  proposal we make must differentiate from PD-SSM.
- **Wildness.** 2 (now that it's published).

### 2.28 — Tropical projective scan (Tropical Attention as scan monoid)

- **Math.** State ∈ `(ℝ ∪ {−∞})ⁿ` projective, combine is tropical
  matrix-vector product. Associative because (min-plus, +) is a
  semiring.
- **Known?** [Tropical
  Attention](https://arxiv.org/abs/2505.17190) occupies this exactly.
  Included for reranking honesty.
- **Wildness.** 2 (published).

---

## Updated top-3 picks for Lean

I'd revise the IDEAS.md §3 picks as follows:

1. **§2.1 Truncated tensor-algebra signature with Chen-product combine.**
   *Still Pick A.* The literature search confirmed that SLiCEs,
   Log-NCDE, SigGate, FRUITS, and the entire rough-paths-ML lineage
   treat signatures as **forcing / features / gates**, never as the
   scan state under Chen's identity. The novelty window is still open
   and the payoff is large: proving Heisenberg and Unipotent U_n are
   explicit projections of the grade-2 / triangular fibre of
   `T^{≤K}(ℝᵈ)`. This is the *most* de-risked pick after the search.

2. **§2.3 / §2.25 BCH / free-nilpotent Lie group at K = 2.**
   *Promote from Pick C to Pick B.* The state-tracking literature boom
   (DeltaProduct, PD-SSM, Grazzi et al., Structured Linear CDEs) has
   re-validated "non-commutative / Lie-structured transitions"
   empirically. None of them use a Magnus / BCH commutator correction
   explicitly — they all work in concrete matrix groups (diagonal +
   Householder, permutation × diagonal, block-diagonal). The §2.25
   K=2 BCH cell — state `(A, B)`, combine
   `(A+A', B+B' + ½[A, A'])` — is a drop-in generalisation that
   dominates Heisenberg (recovers it at `A` scalar), has a closed-form
   associativity proof, and sits in unclaimed territory.

3. **§2.22 Dyck bracket-balance monoid.**
   *Demote from Pick B to Pick C, but keep.* Still novel as a scan
   primitive. The Merrill–Sabharwal "Illusion of State" bound gives
   it a clean publishable motivation (SSMs *cannot* do Dyck; here's a
   2-scalar monoid that does Dyck-1 in O(1) state and Dyck-k in O(k)).
   The Lean proof is short (a case split on `sign(l' − r)`) and
   high-pedagogical-value. The reason to demote is competition: PD-SSM
   and DeltaProduct already solve Dyck-like regular tasks; the story
   needs to pin down "context-free depth, not just finite-state".

A fourth candidate worth a small Lean file for future-proofing:

4. **§2.18 GF(2)-Heisenberg with ordered-bit-pair parity.**
   The single-index-parity fix is crowded (Grazzi, DeltaProduct), but
   the *bit-pair-parity* statistic `c_{ij} = XOR over pairs (aᵢ AND
   bⱼ), i < j` is still unoccupied and is the GF(2)-native analogue of
   our multi-d Heisenberg ordered-pair claim. Same proof skeleton as
   `HeisenbergD` over `ZMod 2`.

---

## Sources (representative; not exhaustive)

- [SLiCEs — Structured Linear CDEs (NeurIPS '25)](https://arxiv.org/abs/2505.17761)
- [Log-NCDE (ICML '24)](https://arxiv.org/abs/2402.18512)
- [SigGate (2025)](https://arxiv.org/abs/2502.09318)
- [FRUITS / ISS (2024)](https://arxiv.org/abs/2311.14549)
- [Illusion of State in SSMs (ICML '24)](https://arxiv.org/abs/2404.08819)
- [Strobl et al. TACL 2024 (formal-language survey)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00663/120983)
- [Hewitt et al. (Dyck memory bound, EMNLP '20)](https://arxiv.org/abs/2010.07515)
- [Grazzi et al. (negative eigenvalues, ICLR '25)](https://arxiv.org/abs/2411.12537)
- [Sarrof et al. (NAACL '24)](https://aclanthology.org/2024.naacl-short.4.pdf)
- [DeltaProduct (NeurIPS '25)](https://arxiv.org/abs/2502.10297)
- [PD-SSM (NeurIPS '25 spotlight)](https://arxiv.org/abs/2509.22284)
- [SD-SSM (AAAI '25)](https://arxiv.org/abs/2412.19350)
- [Implicit LMs are RNNs (ICLR '25)](https://arxiv.org/abs/2502.07827)
- [Prefix-Scannable Models (2025)](https://arxiv.org/abs/2506.10918)
- [PaTH Attention (2025)](https://arxiv.org/abs/2505.16381)
- [Kalman Linear Attention (2026)](https://arxiv.org/abs/2602.10743)
- [MöbiusAttention (2024)](https://arxiv.org/abs/2409.12175)
- [MEA — Matrix Exponential Attention (2025)](https://yifanzhang-pro.github.io/MEA/)
- [Tropical Attention (NeurIPS '25)](https://arxiv.org/abs/2505.17190)
- [Geometry of Thought — tropical Transformer (2026)](https://arxiv.org/abs/2601.09775)

---

**Confirmation:** wrote `/Volumes/git/state-dep-parallel/LITERATURE.md`.
Biggest verdict revisions: §2.1 signature-as-scan-state confirmed 🟢
open after direct verification that SLiCEs and Log-NCDE use signatures
only as forcing; §2.18 GF(2)-Heisenberg softened from 🟢 to 🟡 because
Grazzi et al. (ICLR '25), DeltaProduct, and PD-SSM solve the parity /
state-tracking problem inside parallel scans by 2025, though the
ordered-bit-pair statistic survives; §2.12 tropical-Heisenberg
softened to 🟡 because Tropical Attention (NeurIPS '25) and "Geometry of
Thought" (2026) now occupy much of the tropical-scan niche. Top-3 Lean
picks revised: §2.1 stays, §2.25 (BCH K=2) promoted to #2, §2.22 Dyck
demoted to #3, §2.18 added as a #4 future-proofing target.
