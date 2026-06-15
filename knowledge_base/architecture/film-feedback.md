# FiLM sparse feedback (the headline finding)

## Summary
A **single sparse late→early FiLM connection** in a DeltaNet stack gives a robust **−3 to −5 % PPL lift** that survives a 3.3× param scale-up and an optimizer change. One late-layer output (lagged 1 token) modulates an early layer via FiLM with one learnable scalar α (+0.3 % params). This is the project's strongest, most reproducible architectural result. Source: `README.md`, `CLAUDE.md`.

## Numbers
| Setup | DN baseline | + Sparse FiLM | Δ |
|---|---|---|---|
| 217 M / AdamW / 5 K | 51.00 | 49.40 (2,28) | **−3.1 %** |
| 360 M / Muon / 15 K | 22.79 | 21.57 (2,28) | **−5.4 %** |
| 708 M / Muon / 15 K | 35.38 | 34.26 (2,34) | **−3.2 %** |

3-seed reproducibility at 217 M: 49.40 ± 0.31 (σ < 1 %).

## Mechanism (20+ controlled ablations)
The lift comes from a **negative-α subtractive basin** reachable *iff*:
1. modulation is **multiplicative** (FiLM-form, not Q-K-V additive), AND
2. cross-source aggregation is **non-softmax** (sum or sigmoid; softmax dilutes by 1/K).

Either condition failing breaks the basin. Don't apply weight decay to α (`--alpha_wd 0.0`) — the WD-equilibrium probe showed the gradient consistently wants α higher; WD was the brake, not a loss-flat ceiling.

## K=3 self-feeding
The FiLM source is normally a lagged (previous-token) late-layer state. **K=3 self-feeding** runs the block stack 3× so deployment uses a single forward at **1× decode cost** (closes the train/inference gap) → −1.5 % lift at 1× decode. Enable `--feedback film --feedback_pairs "2,28" --feedback_self_k 3`.

- K=3 runs every block K times → peak activation memory scales ~with K. At `--batch 20 --activation_checkpointing --feedback_self_k 3` it OOMs when K kicks in; use `--batch 14` whenever K=3 will engage.
- **K-self-feed warmup bypass** (`--feedback_self_k_warmup_steps N`): the plain 1-pass loop before step N (early gradient is noise) — measured **+40 %** speed; stacks with `--compile` to +57 %.

## Variants
- **Shallow-wide** uses 5 dense reverse-FiLM pairs `0,5;1,6;2,7;3,8;4,9` — see [[shallow-wide-trunk]].
- **Cross-layer attention** (`--feedback_xattn`) is a more-expressive sister to FiLM; converges to similar VAL ppl at ~equal step cost after the no_grad-pass-1 fix. FiLM is the cheaper-per-step default.

## Honest framing
Cross-architecture (vs Transformer) the comparison is scale/optimizer-dependent (Transformer wins at 360 M/Muon). What's robust is the lift *within* the linear-RNN family. The top-down-feedback idea isn't new (GF-RNN 2015, BRIMs 2020); the contribution is the minimal-form demonstration + mechanism.

## Related
[[deltanet-backbone]] · [[shallow-wide-trunk]] · [[training-and-optim-knobs]] · #architecture
