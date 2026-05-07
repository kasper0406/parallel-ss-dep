# DN-4B Distillation Pilot — Full-Pilot Report

**Status:** _IN PROGRESS — Phase A pending GPU 0 availability_
**Start date:** 2026-04-30
**Worktree:** `/home/knielsen/ml/parallel-ss-dep-distill`
**Branch:** `main`
**Operator GPU:** GPU 0 only. (GPU 1 reserved for parallel surprise-PoC agent.)

This document covers the **full multi-day pilot** that scales the validated
α=0.9 KL+CE recipe (see `DISTILL_PILOT_REPORT.md`) to a ~3 B plain-DN
student trained on a ~30 M-token Qwen3.6-self-generated corpus.

## Objective

Replicate the validation pilot's recipe at scale and produce an honest
evaluation:

- Architecture: **plain DN, no FiLM, no K=3 self-feeding, no L_sem**.
  Architectural variable kept fixed.
- Loss: KL+CE at α = 0.9 (90 % CE + 10 % KL), top-K = 20 teacher logprobs.
- Optimizer: Muon for ≥2D matrices + AdamW for embed/lm_head/1D.
- Data: Qwen3.6-35B-A3B-AWQ self-generated code corpus (~30 M tokens).
- Eval: codeparrot val tail PPL/BPB, plain-DN-708M-baseline BPB, optional
  HumanEval pass@1.

## GPU contention encountered

At kickoff (2026-04-30 ≈ 16:30), GPU 0 was contested by the parallel
structural-surprise multi-seed run — both GPUs were occupied by
`experiments/train_lm.py` jobs from `/home/knielsen/ml/parallel-ss-dep`.
Per the brief I paused rather than fighting for the GPU, prepared the
infrastructure (Phase A and Phase B scripts, eval scripts, prompt
diversity expansion), and waited for GPU 0 to free up.

The contention is documented; the brief explicitly anticipated this and
instructs "pause and report rather than fighting for it." A monitor
polled GPU 0 memory until it dropped below 1 GB, at which point Phase A
launched.

## Setup

[TODO: fill in after run]

## Phase A: Teacher corpus generation

[TODO: fill in after run]

## Phase B: Student training

[TODO: fill in after run]

## Evaluation

[TODO: fill in after run]

## Conclusions and next steps

[TODO: fill in after run]
