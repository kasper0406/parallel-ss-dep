# Literature / novelty assessment — latent execution-trace reasoning (2026-07-13)

Adversarial literature sweep (agent-run, arXiv/Scholar through 2026-07-13) on
whether the Stage-A/Stage-B exec-trace program is novel. Full verdicts below;
read SIM-CoT, 2606.20075, 2602.01148, the CWM report, and survey 2505.16782 in
full before any writeup.

## Verdict table

| Claim | Verdict | Closest paper |
|---|---|---|
| A. Curriculum text→latent replacement | **Already done** | Coconut (arXiv 2412.06769, COLM 2025) |
| B. Per-step supervision of latent thoughts | **Incremental** — concept published; our delta = ground-truth INTERPRETER state via the shared LM head (they use teacher-text CoT via an aux decoder) | SIM-CoT (2509.20317, ICLR 2026) |
| C. Execution traces as supervision | Text version **already done** (Meta CWM, 32B, Sep 2025 — Stage A alone has ZERO novelty); **exec-trace-supervised LATENT thoughts = unclaimed combination** | CWM / SIM-CoT |
| D. Staging necessity | **Incremental-but-valuable** — theoretically predicted (2602.01148 proves curriculum necessity); our controlled cell (dense per-step supervision does NOT rescue latent-first; failure pinned at value prior) is missing from the literature (2606.20075 explicitly lacks the no-curriculum arm; SIM-CoT never trains from a token-incompetent base) | Capabilities & Fundamental Limits of Latent CoT (2602.01148) |
| E. Continuous thoughts in a bounded-state linear RNN + zero-context-cost framing | **New as combination/framing** ("nothing prevented Coconut-on-Mamba" — pair with the decode-cost bench to make substantive) | Huginn (2502.05171) in spirit; Tiny Recursive Mamba-2 (2602.12078) in architecture |

## What IS claimable (if Stage B holds)

1. **Main claim (C∩B): "latent execution"** — compressing verified interpreter
   traces into continuous thoughts with each latent step supervised against
   the ground-truth machine state. Both parent lines leave this hole open:
   execution-trace work keeps state in TEXT (CWM, Neural Debugger 2603.09951,
   Code-Exec-as-Grounded-Supervision 2506.10343, SemCoder); latent-step-
   supervision work uses model/teacher TEXT as target (SIM-CoT). The
   interpreter gives free, dense, VERIFIABLE process supervision — answering
   the "latent supervision targets are unknowable" critique (2602.08332,
   2606.20075).
2. **The staging cell**: per-step supervision is NOT sufficient; prior
   token-space competence is necessary (hop CE pinned at ln 10 in the
   latent-first arm vs decisive success text-first). Sharpest empirical
   statement of the 2602.01148 prediction; disconfirms the natural reading of
   SIM-CoT ("just supervise the steps"). Frame as confirming+sharpening, not
   discovering.
3. **Depth-true signature** (R=K works; R=1 AND R=K+4 collapse; 0.84 per-hop
   decode) — rebuts the "latent tokens are pseudo-reasoning placeholders"
   line (Do Latent Tokens Think? 2512.21711; 2411.15862).
4. **Bounded-state angle**: first continuous-thought reasoning in a
   DeltaNet-class linear RNN; latent deliberation costs zero context in an
   O(1)-decode model (the agent-economics framing).

## NOT claimable

- Coconut mechanism / "latent reasoning works" (A).
- "Per-step supervision stabilizes latent reasoning" in general (SIM-CoT's
  headline; 2606.20075's theorem: outcome-only latent supervision collapses).
- "LMs can simulate Python execution / predict variable states" — CWM at 32B
  in text. **Stage A alone = small-scale CWM replication.**
- "Curriculum is necessary for latent CoT" as a bare claim (2602.01148).

## Weak points to fix before any writeup

(a) synthetic-programs-only → need the CRUXEval-O-style transfer probe;
(b) resolve the ~6-hop horizon (curriculum-exposure vs structural; relate to
Depth Ceiling 2604.06427 — latent planning capacity barely scales — and
SIM-CoT's latent collapse at higher latent counts; Soft-Tokens-Hard-Truths
notes a ~6-continuous-token practical limit for Coconut — suspiciously close
to our horizon);
(c) single model / single seed at 402M — SIM-CoT shows GPT-2-scale results can
invert at 8B; state scale limits;
(d) latent-first arm confound ("maybe that base is just weaker") — pre-empt
with a matched-capability control.

## Suggested positioning (agent's draft, kept verbatim)

> Prior work supervises latent chain-of-thought steps against model-generated
> text rationales (SIM-CoT), and separately trains language models on
> interpreter execution traces rendered as text (Code World Model). We combine
> the two: a program interpreter provides free, dense, verifiable per-step
> ground truth for continuous thoughts, letting a 402M bounded-state (DeltaNet)
> LM internalize program execution into latent space — each latent step decodes
> to the true machine state, and the computation is depth-true. In a controlled
> comparison we further show that dense per-step supervision is not sufficient
> for latent-first training: compression into latent space succeeds only when
> the computation already exists in token space — the sharpest empirical
> evidence to date for the theoretically-predicted necessity of externalization
> before internalization. Because latent steps consume no context and the
> model's decode state is O(1), internalized execution provides
> zero-context-cost deliberation for long-horizon coding agents.

Title direction: "Latent Execution: Internalizing Interpreter Traces into
Continuous Thoughts"; second headline = "externalize before you internalize".

## Key references

Coconut 2412.06769 · Stepwise Internalization 2405.14838 · CODI 2502.21074 ·
SIM-CoT 2509.20317 · Info-theoretic latent supervision 2606.20075 ·
Fundamental Limits of Latent CoT 2602.01148 · Thinking States 2602.08332 ·
CWM (Meta FAIR, Sep 2025, github.com/facebookresearch/cwm) · Neural Debugger
2603.09951 · Code Exec as Grounded Supervision 2506.10343 · SemCoder
2406.01006 · LaSynth 2107.00101 · NPI (Reed & de Freitas 2016) · Jin & Rinard
2305.11169 · CRUXEval 2401.03065 · Soft Tokens Hard Truths 2509.19170 · Token
Assorted 2502.03275 · Huginn 2502.05171 · Tiny Recursive Mamba-2 2602.12078 ·
Do Latent Tokens Think? 2512.21711 · Depth Ceiling 2604.06427 · surveys
2505.16782, 2509.02350.
