# Lens: Learning-dynamics science (verbatim agent report, 2026-07-13)

# Learning-Dynamics Ideation: 5 ideas, ranked by expected value per GPU-hour

Context I'm exploiting: you hold an unusually coherent set of dynamics facts — (a) **externalize-before-internalize** (latent-first pins to the marginal-distribution plateau; text-first-then-wean breaks through), (b) **converged-base attach pathology** (post-hoc features inert; day-1 co-training load-bears; frozen-trunk pre-warm rescues), (c) **residual-stream collapse under WD=0.1**, (d) **curricula must fit in ≤40% of the run**, (e) **WSD-plateau SWA = free ~3%**, (f) **RL sharpens greedy but the pass@k envelope is base-set-bound; round 2 saturates**. Each idea below is an attempt to convert one of these from "observation" into "lever."

---

## Idea 1 — The Skill Tower: staged externalize→wean curriculum with a *decodability predictor* (RANK 1)

**Mechanism + phenomenon.** Your staging finding generalizes into a testable theory of *why* latent-first fails: latent supervision is only informative once the intermediate quantities of the skill are already *represented* somewhere in the network's state — token-space training is what creates those representations; the wean then re-routes them off the output channel. This yields a predictive criterion: **a skill needs externalization first iff a linear probe on the base model's hidden states cannot decode the skill's intermediate states above chance.** Probe-decodable skills should compress latent-first directly; non-decodable ones should show exactly your plateau. If the criterion holds, you get a *tower*: externalize primitive k in text → wean to latent → the compressed latent skill becomes a decodable substrate for primitive k+1 (whose intermediates are now probe-visible), so k+1 needs a shorter or no externalization phase. This composes with your depth-via-iteration finding: heterogeneous composition failed when trained monolithically — the tower is the staged alternative.

**Why our evidence supports it.** (i) The staging result itself; (ii) attach-pathology is the same phenomenon at module level (a feature can't learn from a signal the trunk doesn't yet represent — frozen-trunk pre-warm is "externalization" for modules); (iii) latent per-hop supervision only worked once per-hop states existed (pointer-chase R=K per-hop thread); (iv) heterogeneous-composition collapse (2026-06-25) is what un-staged training of a tower looks like.

**Cheapest decisive kill-test (~1 day, 1 GPU).** Three skill families on the small synthetic rig: pointer-chase f^K (known: needs externalization), modular-arithmetic chain (known learnable), and 2-op heterogeneous composition (known to collapse). Before any skill training, fit linear probes for each family's intermediate states on trunk hiddens. Then run latent-first vs text-first-wean per family. **Decision rule:** probe accuracy at init rank-orders which families show the latent-first plateau. Second, decisive tower test: after weaning skill A, re-probe — do A's intermediates become decodable, and does skill B (composing A) now train latent-first where it previously couldn't? That single 2×2 (B latent-first, with/without compressed-A prefix training) is the whole methodology's go/no-go, and it directly de-risks the Stage-A→Stage-B plan of the live exec-trace program.

**Honest expected effect.** High-information, medium capability: it won't move HumanEval by itself, but it converts a one-off finding into the training recipe for the exec-trace latent interpreter (your current bet), and tells you *in advance* which stages the staged path needs — the difference between a 2-stage and a 5-stage curriculum under the ≤40%-of-run constraint.

**Novelty check.** Closest: [Stepwise Internalization / ICoT-SI](https://openreview.net/pdf?id=fRPmc94QeH) (curriculum token removal, incl. removal-smoothing) and Coconut's stage curriculum — both are single-skill weaning schedules. Recent latent-CoT surveys ([review](https://www.themoonlight.io/en/review/reasoning-beyond-language-a-comprehensive-survey-on-latent-chain-of-thought-reasoning)) contain no *predictive criterion* for which skills require externalization, and no compositional tower where a compressed skill becomes the substrate for the next. The probe-decodability predictor appears novel.

---

## Idea 2 — SAM only during WSD decay, stacked on plateau-SWA (RANK 2)

**Mechanism + phenomenon.** ICLR-2025 theory shows SAM applied late escapes the SGD minimum exponentially fast and settles into a flatter minimum *in the same valley* — you don't need SAM's 2× cost for the whole run, only during the decay tail ([OpenReview](https://openreview.net/forum?id=aD2uwhLbnA)). 2026 work implements exactly this on WSD: AdamW through warmup+plateau, SAM for the ~10% decay, and reports **31% less forgetting after SFT and 40% after 4-bit quantization** on OLMo-2-1B ([Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting](https://arxiv.org/abs/2605.02105)). Flat-minimum selection and SWA variance-reduction are complementary interventions on the same plateau geometry — your own note says SWA "proves the plateau learns a real center"; SAM biases *which* center.

**Why our evidence supports it.** You already cash out geometry-level free wins (SWA +3%); your weight-teleportation post-mortem explicitly named SAM as the working family member you hadn't adopted; and your pipeline's chronic failure mode is exactly what decay-SAM protects — SFT/RL degrading pretrain skills (dirty-SFT regressions, RL format collapse), plus you plan quantized O(1)-decode deployment where the 40%-less-quantization-damage result is directly on-thesis.

**Cheapest decisive kill-test (~1 day, 1 GPU).** Take any existing WSD plateau checkpoint (e.g. wide-10L). Run the decay leg twice from the same state: control vs SAM (ρ ≈ 0.01–0.05, ascent step on the Muon+AdamW composite; 2× cost only on ~10% of steps ≈ +10% total). Compare: stratified per-source CE, SWA-vs-SAM-vs-both, then a short standard SFT on top of each and measure base-skill regression (the dep-distance-stratified probe + recall kill-gates). Decision: both-stack ≥ SWA-alone on CE *and* less SFT-induced regression.

**Honest expected effect.** Small but nearly free: 0.5–1.5% CE-equivalent, with the real payoff being SFT/RL robustness and quantization tolerance. High confidence — externally replicated at your scale class.

**Novelty check.** Low novelty by design — [the recipe exists](https://openreview.net/pdf/2e91997f1bd7a2c7457aa66b32f048edae2c4e84.pdf); the SAM×SWA×Muon stack on a linear-RNN trunk is the only unexplored cell. This is an *adoption* play; that's why it ranks 2nd on EV/GPU-hour despite modest ceiling.

---

## Idea 3 — Critical-period placement of scratchpad data: early-circuit vs decay-anneal (RANK 3)

**Mechanism + phenomenon.** Two literatures collide: (a) critical-period/attach evidence (yours is among the strongest anywhere: post-hoc modules inert, day-1 co-training load-bearing) says skill-forming data must come **early**, while the circuit is being laid down; (b) the mid-training/annealing literature (MiniCPM, OLMo-2, and [NVIDIA's Front-Loading Reasoning](https://research.nvidia.com/labs/adlr/files/Front_Loading_Reasoning_The_Synergy_between_Pretraining_and_Post_Training_Data.pdf)) says high-quality reasoning data is best *injected during LR decay* where it's retained rather than overwritten. These make **opposite predictions** for your Stage-A text-scratchpad corpus, and the answer determines the layout of every future pretrain under the ≤40%-curriculum constraint. The reconciling hypothesis worth testing: *circuit-forming* data (exec traces, scratchpads — skills the model can't do at all) is critical-period-early, while *knowledge* data (rare APIs, tail docs) is anneal-late. Placement should sort by whether the data teaches a computation or a fact.

**Why our evidence supports it.** Cold latent co-train destabilizes early (v12) but attach-late is inert — the window is neither t=0 nor t=end, i.e., there really is a period. The recall-stream and namekey results show day-1 co-training is what made WM load-bear. Meanwhile your token-poverty diagnosis says knowledge is the binding constraint — so getting *placement* right is one of the few free levers left on a fixed 5B budget.

**Cheapest decisive kill-test (~1.5–2 days, 2 GPUs).** Small config (10L), common 300M-token seed, then three 700M-token continuations with identical totals: scratchpad data (10% of mix) placed early / uniform / concentrated-in-decay. Score: stage-A executor eval + stratified CE + a post-hoc latent-wean attempt on each (does early placement make the *wean* work better — connecting to Idea 1?). The wean-success difference is the decisive readout, not raw CE.

**Honest expected effect.** Raw CE differences will be small (order effects usually are); the strategic information — "computation-data early, knowledge-data late" as a mix-design rule — is worth more than the delta itself. Real risk: 1B tokens is too short to see it; mitigate by evaluating the skill (executor accuracy), which has a much lower noise floor than CE.

**Novelty check.** Continued-pretraining critical periods documented for language transfer ([ACL 2025](https://aclanthology.org/2025.acl-long.1547/)); front-loading-vs-annealing for reasoning data studied at 7B+ by NVIDIA; nobody has run the placement question against a *latent-compression* downstream criterion (does placement determine wean-ability). That link is yours alone.

---

## Idea 4 — Verifier-gated self-distillation ring: test whether SFT-on-verified-samples grows the envelope where RL couldn't (RANK 4)

**Mechanism + phenomenon.** Your pass@k post-mortem: RL sharpens greedy (3→7) but the pass@30 envelope (~18–21) is base-set-bound and round 2 saturated. The dynamics question left open: RL's on-policy KL-anchored updates *can't* consolidate rare successes into the base set, but **off-policy SFT on execution-verified best-of-N outputs** is a different operator — it's self-distillation of the model's own tail into its mode, and 2025 theory says verification provably prevents the collapse that plagues unverified recursive training ([V-STaR](https://arxiv.org/abs/2402.06457); [self-improving transformers overcoming easy-to-hard generalization](https://arxiv.org/pdf/2502.01612)). The kill-question at 400M: does round-over-round verified SFT *grow* pass@30 (compounding) or merely convert envelope→greedy (your RL result, saturating)?

**Why our evidence supports it.** All infrastructure exists (`gen_rejection_data.py --keep_all`, `code_grader` dense tiers, DPO trainer); the grader is a real execution verifier, which is precisely the collapse-prevention condition in the theory; and your STaR-adjacent path was never run as a multi-round loop — v-STaR-style use of *failures* as DPO negatives is also unexploited (you have `error_text` for free).

**Cheapest decisive kill-test (~1–1.5 days, 2 GPUs).** Two rounds: sample 64@τ=0.8 on MBPP-train → grade → SFT on passes + DPO on (pass, near-miss) pairs → re-measure greedy and pass@30 on the held-out grid after each round. Decision: pass@30 round-2 > round-0 by more than the bootstrap CI = compounding exists at 400M; flat = your "envelope is knowledge-bound" claim is confirmed against a second, independent operator.

**Honest expected effect.** Most likely outcome given your own evidence: +3–5 greedy in round 1, envelope flat, round 2 saturates — i.e., a *confirmation*, not a lever. Ranked 4th for exactly that reason; it's cheap because infra exists and it closes a loop your strategy doc left ajar.

**Novelty check.** V-STaR/STaR are established; compounding-vs-saturation as a function of model scale is the open bit ([task-centric theory of iterative self-improvement](https://arxiv.org/pdf/2602.10014) is theory-only) — a clean 400M data point is publishable but not a capability breakthrough.

---

## Idea 5 — Grok-injection micro-phases for code primitives (RANK 5)

**Mechanism + phenomenon.** Grokking = generalizing-circuit formation winning over memorization under data restriction + weight-norm pressure; recent work shows the regime can be *engineered* (weight-decay regime diagnostics: [arXiv 2605.20441](https://arxiv.org/html/2605.20441); grokking transferred below the data threshold by distilling from an already-grokked model: [When Data Falls Short](https://openreview.net/pdf/7991ee750af386d28ca36a3a3c932c509bfc7760.pdf); embedding transfer from a weaker grokked model accelerating grokking: [arXiv 2504.13292](https://arxiv.org/pdf/2504.13292)). Token-poor means you can't buy knowledge, but *algorithmic* code primitives (index arithmetic, scope resolution, bracket/paren tracking, small-int evaluation) are grokkable from tiny synthetic sets repeated to high epoch counts. Proposal: grok the primitive in a **separate tiny model or adapter** under aggressive WD, then transfer (KD on hidden states / embedding transfer) into the main trunk — never raise trunk WD, which your residual-collapse result forbids.

**Why our evidence supports it — and why it's ranked last.** Supports: the exec-trace program needs exactly these primitives; distill-from-grokked sidesteps the WD-collapse conflict; length-generalization (the signature grokking payoff) is your known weakness (K=4→10 worked in latent thinking, but code-primitive length-gen is untested). Against: your features-null forensics shows the trunk likely already has the easy primitives; the transfer step re-enters attach-pathology territory (grokked-circuit KD into a converged trunk may go inert — frozen-trunk pre-warm would be mandatory); and the honest prior on HumanEval movement is near zero.

**Cheapest decisive kill-test (~1 day, 1 GPU).** One primitive (multi-digit index arithmetic with distractors). (a) Confirm the base 287M fails length-generalization on it; if it doesn't fail, kill immediately. (b) Grok a 2L model on 1k examples (WD 0.5–1.0, watch for the delayed val-accuracy jump); (c) KD its hiddens into a frozen-trunk adapter; (d) test length-gen transfer. Any break in the chain kills the idea for ~$0.

**Honest expected effect.** Low — a rigor/length-generalization play, not a headline mover. Included because it's the only recipe class that manufactures *new circuits* from data you can synthesize for free, which is the correct shape of answer to token-poverty even if this instance fails.

**Novelty check.** Grokking-transfer-by-distillation exists (2025); *injecting grokked primitive circuits into an LM pretrain as a curriculum stage* does not appear in the literature — but that's partly because the prior is unfavorable.

---

## Single best bet

**Idea 1, the Skill Tower with the probe-decodability predictor.** It is the only idea that (a) grows directly out of your own strongest and most novel finding rather than importing one, (b) has a same-week synthetic kill-test whose *negative* result is also valuable (it would say the staging finding is task-specific, capping how much to invest in staged curricula), and (c) feeds the live exec-trace latent-interpreter program immediately — the decodability probe tells you where Stage-B weaning will stall *before* you spend the GPU-days, and the tower composition test is the first real attack on the heterogeneous-composition collapse that currently bounds latent thinking's ceiling. Pair it with Idea 2 in the same week (Idea 2 runs unattended on the second GPU): one is your discovery compounding, the other is externally-validated free money.

Sources:
- [Learning to Internalize CoT Step by Step (ICoT-SI)](https://openreview.net/pdf?id=fRPmc94QeH)
- [Latent CoT reasoning survey](https://www.themoonlight.io/en/review/reasoning-beyond-language-a-comprehensive-survey-on-latent-chain-of-thought-reasoning)
- [SAM Efficiently Selects Flatter Minima Late in Training (ICLR 2025)](https://openreview.net/forum?id=aD2uwhLbnA)
- [Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting](https://arxiv.org/abs/2605.02105) ([tiny-paper version](https://openreview.net/pdf/2e91997f1bd7a2c7457aa66b32f048edae2c4e84.pdf))
- [Training Dynamics of the WSD Cooldown Stage](https://arxiv.org/pdf/2508.01483)
- [Front-Loading Reasoning: Synergy between Pretraining and Post-Training Data (NVIDIA)](https://research.nvidia.com/labs/adlr/files/Front_Loading_Reasoning_The_Synergy_between_Pretraining_and_Post_Training_Data.pdf)
- [Emergent Abilities under Continued Pre-training for Language Adaptation (ACL 2025, critical period)](https://aclanthology.org/2025.acl-long.1547/)
- [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457)
- [Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization](https://arxiv.org/pdf/2502.01612)
- [A Task-Centric Theory for Iterative Self-Improvement](https://arxiv.org/pdf/2602.10014)
- [Weight Decay Regimes in Grokking Transformers](https://arxiv.org/html/2605.20441)
- [When Data Falls Short: Grokking Below the Critical Threshold](https://openreview.net/pdf/7991ee750af386d28ca36a3a3c932c509bfc7760.pdf)
- [Let Me Grok for You: Embedding Transfer from a Weaker Model](https://arxiv.org/pdf/2504.13292)
- [A Study on Hidden Layer Distillation for LLM Pre-Training](https://arxiv.org/html/2605.11513v1)
