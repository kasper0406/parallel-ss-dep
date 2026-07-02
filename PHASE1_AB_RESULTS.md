# Phase-1 A/B — Do OUR features help on a COMPETENT, INHERITED base?

**Question.** Architectural add-ons (FiLM feedback, WorkingMemory, PKM, latent
thinking, output-gate) have repeatedly looked load-bearing on a *from-scratch*
287 M base. Do they still add value when bolted onto a **competent inherited
base** — the linearized 402 M DeltaNet that inherited SmolLM2-360M's coding
knowledge (teacher-forced HumanEval-solution CE = 0.759, vs 0.97 from-scratch /
0.614 donor)? Or is the inheritance itself the whole story?

**Design (scrupulously controlled).** Both arms start from the SAME
`checkpoints/linearize/linearized_stage3.pt`, train on the SAME data
(`configs/pretrain_mix_v18_arxiv.yaml`), for the SAME token budget, with the
SAME optimizer/schedule/seed. **The only difference is features off vs on.**

| | base / control trunk | features |
|---|---|---|
| **Arm A (control)** | DeltaNet d960 × 32L × 15h × d_head64, d_ff2560, TIED 49152, RMSNorm, feedback=none | **none** |
| **Arm B (treatment)** | same trunk, same weights at step 0 | FiLM(reverse fan-in, 4 pairs)+K=3 self-feed · WorkingMemory(ctx_namekey + mem_always_read + copy-head + ctx-addr aux) · PKM(after L16) · latent(LatentFeedbackAdapter + latent_reasoning aux) · output_gate + gate_entropy_aux · trunk gist loss |

Every Arm-B feature module inits **near-no-op** (FiLM α=0, latent proj
zero-init, PKM α-floor curriculum, WM read_alpha frozen 0 + copy-gate bias −6),
so the inherited base is byte-preserved at step 0 (confirmed: Arm B step-3
VAL ppl = 4.58 = Arm A step-3 VAL ppl = base).

## Budget / shared config

- 1900 steps × batch2 × grad_accum32 × T2048 = **131072 tok/step → 249.0 M
  tokens per arm** (identical A/B). batch=2 chosen so Arm B's K=3 self-feed +
  latent stack fits 32 GB; total compute for a fixed token budget is invariant
  to the batch/grad-accum split. (Reduced from the ~300 M target purely for
  wall-clock: K=3 self-feed + latent aux on the 479 M Arm-B model run ~12 s/step;
  identical budget across arms preserves the comparison.)
- `--optimizer muon --lr 3e-4 --lr_muon 1e-3 --lr_schedule wsd --warmup_steps 100
  --lr_decay_frac 0.15 --wd 0.01 --alpha_wd 0.0 --grad_clip 1.0 --z_loss 1e-4`
- `--bf16 --tf32 --no-compile --bf16_optim_state --activation_checkpointing`
- Plain DeltaNet (working Blackwell kernel). Single GPU (CUDA_VISIBLE_DEVICES=1).

## Correctness — Arm B does NOT resize the inherited tied 49152 embedding

`train_lm`'s `--data_mix` path normally sets model_vocab = round64(max(vocab,len)+1)
= **49216**, which would shape-mismatch the base's 49152 tied embed/lm_head on
load. Fix = new guarded flag **`--keep_base_vocab 49152`** (default 0 = off,
byte-identical legacy behavior): keeps model vocab at the base size and aliases
the discrete think token to an **in-range** id (EOS=0). Safe because
`--think_burst_prob 0` + `--mem_always_read` mean the discrete think token is
never emitted into the stream (the flag asserts `think_burst_prob==0`). Code:
`experiments/train_lm.py` (vocab block) + `experiments/train_lm_args.py`.

### Arm-B load manifest (pre-flight, asserted before training)

```
== vocab resolution ==
  tok.vocab_size=49152 len(tok)=49152 eos=0
  model_vocab_size=49152  thinking_token_id=0
== base ckpt: checkpoints/linearize/linearized_stage3.pt ==  step=29295  n_base_params=451
== load manifest ==
  inherited keys present in BOTH model & ckpt : 451
  ...of which load with MATCHING shape        : 451
  ...of which SHAPE-MISMATCH (MUST be 0)       : 0
  base ckpt keys with NO model slot (dropped)  : 0
  freshly-initialized model params (FEATURES)  : 53
  OK  embed.weight:   (49152, 960)  (inherited, no resize)
  OK  lm_head.weight: (49152, 960)  (inherited, no resize)
  fresh feature groups: memory.* (ctxkey/W_*/copy), pkm_layer.*, sparse_feedback.{0,4,8,12},
                        latent_feedback_adapter.*, gate_head.*, gist heads
== strict=False load: missing=53 (fresh feature params)  unexpected=0
== VERDICT: PASS (zero shape-mismatch, embedding inherited) ==
```

All 451 inherited weights (incl. the tied 49152×960 embedding & lm_head) load
with **zero shape-mismatch**; the only fresh params are the 53 feature-module
tensors. Arm A loads with `missing=0 unexpected=0` (pure continuation).

### Data identity (the "same data" claim)

Verified: with the same seed, the `(x, y)` token stream is **bit-identical**
whether `--emit_read_mask` is off (Arm A, 3-tuple) or on (Arm B, 4-tuple) — the
read-mask is an extra channel consumed only by WM and does not perturb the
token stream. (`torch.equal` over the first 12 sequences → identical.)

---

## RESULTS

### Primary — teacher-forced HumanEval-solution CE (lower = better)

Probe: `/tmp/probe_he_ce.py <ckpt>` (CE on the canonical-solution tokens of all
164 HumanEval problems; 9513 solution tokens). Anchors: from-scratch 0.97,
donor SmolLM2-360M 0.614.

| checkpoint | HumanEval-solution CE | Δ vs base | Δ vs Arm A |
|---|---|---|---|
| base (linearized_stage3) | **0.7585** | — | — |
| Arm A (control, features off) | _PENDING_ | | — |
| Arm B (treatment, features on) | _PENDING_ | | _PENDING_ |

### Secondary — HumanEval pass@1 (greedy, identical config both arms)

| checkpoint | pass@1 | _PENDING_ |
|---|---|---|

### Feature probes

_PENDING_ (WM leak-free recall kill-gate; in-training feature engagement
diagnostics: PKM αeff/row, WM copy-gate, latent reason-loss).

### VERDICT

_PENDING_

---

## Exact re-run commands

```bash
# Both arms (sequential, GPU1). Arm B runs a load-manifest pre-flight first.
cd /home/knielsen/ml/parallel-ss-dep
GPU=1 bash launch_phase1_ab_A.sh      # control  -> checkpoints/phase1_ab_A.pt
GPU=1 bash launch_phase1_ab_B.sh      # treatment -> checkpoints/phase1_ab_B.pt

# Primary eval (run on each ckpt):
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/probe_he_ce.py checkpoints/linearize/linearized_stage3.pt
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/probe_he_ce.py checkpoints/phase1_ab_A.pt
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/probe_he_ce.py checkpoints/phase1_ab_B.pt
```

The full feature flag lists are in `launch_phase1_ab_A.sh` / `launch_phase1_ab_B.sh`.
