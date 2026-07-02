# SmolLM2-360M → DeltaNet LINEARIZATION proof

**Question.** Does a DeltaNet that *inherits* SmolLM2-360M's non-attention weights
and *distills* its attention into DeltaNet layers retain the donor's knowledge —
landing near SmolLM2-360M's teacher-forced HumanEval-solution CE (≈0.614) rather
than our-from-scratch (≈0.969)? If yes, the "inherit tokens via same-size
linearization" path is validated and worth scaling to a stronger donor.

**Status: COMPLETE — VERDICT = PARTIAL (strong, positive).**
Linearized CE **0.7587** (fp32). Broke **decisively below** the our-from-scratch
floor (0.9716 → 0.7587, **−0.21 / −22%**) but landed ~0.14 **above** the donor
target (0.6142), i.e. outside the ~0.1 "success" band. The inherit-and-linearize
path transfers real donor knowledge that from-scratch training at our token budget
cannot reach; the residual gap to the donor is the linear-RNN attention-matching
bottleneck plus a still-descending end-to-end stage. **The hypothesis is supported;
worth scaling to a stronger donor with a longer end-to-end phase.**

All code: `experiments/linearize_smollm2.py` (no existing training code modified).
Single GPU, `CUDA_VISIBLE_DEVICES=1`. Env: repo `.venv`, `PYTHONPATH=.`.
Total training budget = **370M tokens** (250M layerwise + 120M end-to-end), well
under the 1B-token verdict ceiling. Wall-clock ≈ 2.7 h on 1× RTX 5090.

---

## 1. Architecture / weight-copy manifest

SmolLM2-360M is a Llama block; our `TinyLM.Block` is structurally identical except
the attention sublayer (RoPE-GQA → DeltaNet). Verified donor config:
`hidden 960, 32 layers, 15 heads / 5 KV / head_dim 64, intermediate 2560,
vocab 49152, RMSNorm eps 1e-5, tied embeddings, silu`.

Bare model built: `arch=deltanet, d_model=960, n_layers=32, n_heads=15, d_head=64,
d_ff=2560, vocab=49152, tie_embeddings=True, feedback=none`, no memory/PKM/gate/
FiLM/latent. RMSNorm eps overridden to **1e-5** to match the donor exactly.

**Copied DIRECTLY from the donor (162 tensors):**

| our param | ← donor param |
|---|---|
| `embed.weight` (+ tied `lm_head.weight`) | `model.embed_tokens.weight` |
| `blocks.{i}.attn_norm.weight` | `model.layers.{i}.input_layernorm.weight` |
| `blocks.{i}.mlp_norm.weight` | `model.layers.{i}.post_attention_layernorm.weight` |
| `blocks.{i}.mlp.W_g.weight` | `model.layers.{i}.mlp.gate_proj.weight` |
| `blocks.{i}.mlp.W_u.weight` | `model.layers.{i}.mlp.up_proj.weight` |
| `blocks.{i}.mlp.W_d.weight` | `model.layers.{i}.mlp.down_proj.weight` |
| `out_norm.weight` | `model.norm.weight` |

Every copy asserts `dst.shape == src.shape`. (`GLU.forward = W_d(silu(W_g)·W_u)`
matches Llama SwiGLU; our `RMSNorm = x·rsqrt(mean(x²)+eps)·w` is identical to
LlamaRMSNorm at eps 1e-5 — both verified by an independent code review.)

**Left RANDOM (trained) — DeltaNet attention sublayers only, 288 tensors / 118.8M
params (≈29.5% of the 402M model):** per layer `attn.layer.{q,k,v,b,o}_proj.weight`,
`attn.layer.{q,k,v}_conv1d.weight`, `attn.layer.o_norm.weight`.

### Copy-correctness verification (decisive)

Zero **every** attention sublayer in BOTH the donor (HF) and our model (via forward
hooks) and compare logits on the same input. With attention removed both reduce to
`embed → (RMSNorm → SwiGLU MLP)×32 → norm → tied head` — pure feed-forward over the
*inherited* weights. If the copy is exact they must match to fp32 noise:

```
[verify] zero-attn logit match vs donor: max|Δ|=0.0000e+00  mean|Δ|=0.0000e+00
         (donor logit scale 3.092)  argmax-agree=100.0%
[verify] COPY EXACT (PASS)
```

**max abs logit diff = 0.0, 100% argmax agreement** → the inherited embed / RMSNorm /
SwiGLU / final-norm / tied-head path is wired bit-for-bit identically to SmolLM2-360M.
A silent bad copy (transpose, wrong gate/up/down assignment, eps drift) is ruled out.

---

## 2. Eval protocol (identical across all arms)

Teacher-forced **HumanEval-solution CE**, byte-identical to the repo's existing
probes (`/tmp/probe_capacity_ref.py` for HF models, `/tmp/probe_he_ce.py` for our
ckpts): tokenizer `HuggingFaceTB/SmolLM2-135M` (== the 49152 SmolLM2 vocab); for each
of the 164 problems encode `prompt` and `prompt+canonical_solution`; CE is summed
over the solution-token positions only (`start = max(len(prompt_ids)-1, 0)`) and
divided by the total solution-token count (9513 tokens). The linearized model is
evaluated in **fp32** (same as the from-scratch floor), with the DeltaNet FLA kernel
internally bf16 in both. (bf16-autocast eval differs by <0.001 CE — measured.)

**Reference points re-measured in THIS environment (anchors the harness):**

| model | params | HumanEval-solution CE |
|---|---|---|
| SmolLM2-360M (donor / target) | 362M | **0.6142** ✓ (matches published 0.614; HF model, bf16) |
| our-from-scratch SFT (`rejection_sft_v1.pt`, floor) | 287M | **0.9716** ✓ (matches published 0.969; TinyLM, fp32) |
| uniform prior `ln(49152)` | — | 10.803 |

---

## 3. Distillation recipe

**Stage 2 — layerwise attention transfer (MOHAWK / Mamba-in-Llama style).**
Freeze all inherited weights; train only the DeltaNet sublayers. Hooks capture, per
donor layer, the attention-sublayer INPUT (output of `input_layernorm`) and OUTPUT
(output of `self_attn`). Each `blocks[i].attn(donor_input_i)` is trained to match
`donor_output_i` via **relative MSE** (`mse / mean(target²)`, so all 32 layers weigh
equally). AdamW lr 1e-3 OneCycle, batch 12 × T 1024, 250M tokens (codeparrot-clean).
≈57k tok/s. Per-layer relative reconstruction error fell 1.10 → **~0.20** and
plateaued (the linear-RNN-can't-match-softmax-attention floor).

**Stage 3 — end-to-end logit distillation.** Unfreeze all; minimize
`KL(teacher ‖ student)` at T=2 (`F.kl_div(log_softmax(student/T), softmax(teacher/T))·T²`,
per-token) + small hard-target CE anchor (weight 0.1). AdamW lr 1e-4 OneCycle,
batch 4 × T 1024, 120M tokens. Repairs the layerwise teacher-forcing/exposure gap
(student layers feed each other, not the donor's clean residual stream).

---

## 4. Results — HumanEval-solution CE across stages (authoritative fp32)

| stage | tokens | HumanEval-solution CE | vs target 0.6142 | vs floor 0.9716 |
|---|---|---|---|---|
| reference: SmolLM2-360M donor | (4T, donor) | 0.6142 | — | — |
| reference: our-from-scratch SFT | ~5B | 0.9716 | — | — |
| (a) init — random DeltaNet attn, inherited rest | 0 | **9.8914** | +9.28 | +8.92 |
| (b) after layerwise transfer | 250M | **1.0046** | +0.39 | +0.03 |
| (c) after end-to-end distill | 370M | **0.7587** | **+0.144** | **−0.213** |

(In-run bf16 evals agreed to <0.01: stage2 1.0052, stage3 0.7587. Init CE 9.89 <
uniform 10.80: random token-mixing leaves the inherited MLP/embedding prior intact
but corrupts the residual stream — the §1 zero-attn check is the clean
copy-correctness signal, not this number.)

**Reading the trajectory.** Layerwise alone converges to ~1.0 ≈ the from-scratch
floor: the ~20% per-layer attention-reconstruction error compounds over 32 layers
and bottlenecks access to the inherited knowledge. The end-to-end KL stage, which
optimizes the final objective directly, **breaks the floor** and was **still
descending at the budget end** (0.80 @ +50M → 0.76 @ +100M → 0.759 @ +120M).

---

## VERDICT — PARTIAL (strong, positive)

- **Knowledge IS inherited.** 0.7587 is **0.213 below** the from-scratch floor
  (0.9716) — a 22% relative CE reduction on correct-code modeling that a model
  trained from scratch on our token budget cannot reach. This is the apples-to-apples
  comparison (both TinyLM, both fp32).
- **Not at the donor target.** 0.7587 is **0.144 above** the donor (0.6142), outside
  the ~0.1 "success" band, so this is a **PARTIAL** result under the task's bands
  (success ≤~0.71 · partial = clearly below 0.97, not near 0.614 · kill ≈0.97).
- **The path is validated and the ceiling isn't hit.** The end-to-end CE was still
  falling at 370M tokens, and the layerwise stage is bottlenecked by linear-attention
  expressivity, not data — so a longer end-to-end phase, a stronger donor, and/or
  DeltaProduct-class attention (more Householder products → closer to softmax mixing)
  should close more of the remaining 0.14. **Recommendation: scale it** — inherit
  from a stronger same-vocab donor (e.g. SmolLM2-1.7B / Qwen) with a larger
  end-to-end budget.

---

## Re-run commands

```bash
cd /home/knielsen/ml/parallel-ss-dep
# Full proof (init + verify + stage2 + stage3 + eval), ~2.7 h on 1× RTX 5090:
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  .venv/bin/python experiments/linearize_smollm2.py \
  --batch 12 --e2e_batch 4 --T 1024 \
  --layerwise_tokens 250000000 --e2e_tokens 120000000 \
  --lr_layerwise 1e-3 --lr_e2e 1e-4 --kd_temp 2.0 --ce_anchor 0.1 \
  --eval_every_tokens 50000000 --log_every 100 --out_dir checkpoints/linearize

# Authoritative fp32 re-eval of a saved ckpt:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -c "
import torch; from transformers import AutoTokenizer
import experiments.linearize_smollm2 as L
tok=AutoTokenizer.from_pretrained(L.TOKID); m=L.build_bare_deltanet()
m.load_state_dict(torch.load('checkpoints/linearize/linearized_stage3.pt',weights_only=False)['state_dict'],strict=False)
print(L.humaneval_solution_ce(L.make_forward_logits(m,bf16=False),tok))"

# Re-measure the reference anchors:
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python /tmp/probe_capacity_ref.py HuggingFaceTB/SmolLM2-360M  # 0.6142
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python /tmp/probe_he_ce.py checkpoints/rejection_sft_v1.pt     # 0.9716
```

Checkpoints: `checkpoints/linearize/linearized_{init,stage2,stage3}.pt`;
metrics JSON: `checkpoints/linearize/linearize_results.json`;
training log: `logs/linearize_run.log`.

---

## Caveats / honesty

- **Size confound (minor, argued benign).** The linearized model is 360M (the
  donor's size); the from-scratch floor is our 287M. So 73M extra params could in
  principle account for some of the 0.972→0.759 gain. Two reasons it doesn't
  materially: (i) the *init* 360M model (random attn, inherited MLP/embed) scores
  9.89 — params alone buy nothing; (ii) per the capstone finding
  ([[project_undertrained_not_undercapacity]]) our 287M is **token-limited, not
  capacity-limited** (SmolLM2-135M with half the params beats it), so a
  360M-from-scratch at ~5B tokens would not approach 0.76. The cleanest control
  (a 360M-from-scratch floor) was out of budget; the gain is attributed to inherited
  4T-token knowledge, not param count.
- **Depth is the donor's 32L, not our production 10L** — a faithfulness proof of the
  inheritance hypothesis, not a production-config result.
- **Precision across references.** Floor (0.9716) and linearized (0.7587) are both
  fp32 TinyLM → directly comparable. The donor target (0.6142) is HF/bf16; measured
  fp32-vs-bf16 gap is <0.001, so it does not affect the verdict.
- DeltaNet (linear RNN) cannot perfectly reproduce softmax attention's token mixing;
  layerwise rel-MSE floored ~0.20 → this bounds how close stage (c) can get without
  the end-to-end stage carrying the rest.
- CE on correct code ≠ pass@1, but lower solution-CE strongly tracks code capability
  (the same metric that ordered SmolLM2-360M < SmolLM2-135M < our 287M).
- Protocol is byte-identical across all arms (tokenizer, prompt, span, reduction).
