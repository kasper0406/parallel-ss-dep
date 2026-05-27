"""Why does FIX A (write-only-at-think) HURT pass@1 when forced at
training time, but HELP at inference-time-only?

Hypothesis (2026-05-19): the [THINKING] token has a single learned
input embedding. When the model emits a burst of think tokens, every
think position sees the SAME input embedding. The hidden states at
those think positions are driven by similar inputs (modulo accumulated
trunk context), so they end up correlated. With FIX A, the WM buffer
is filled with these correlated states — sharp read queries can only
retrieve from a low-rank pool, defeating the sharp-retrieval advantage.

This probe measures:
  1. Per-position hidden-state norms at think vs emit
  2. Pairwise cosine similarity within think vs within emit
  3. Effective rank of think-state matrix vs emit-state matrix
  4. Write-gate g values at think vs emit (should be ~uniform per the
     uniform-baseline finding)
  5. WM read-attention sharpness at the LAST think position when the
     buffer is forced to all-think vs all-emit (sanity probe for the
     FIX A inference-vs-training divergence)

If think hidden states are highly correlated → FIX A's buffer is
low-rank → sharp reads retrieve from low-information pool → hurts.

Usage:
  PYTHONPATH=. .venv/bin/python experiments/diag_think_position_diversity.py \\
      --ckpt checkpoints/sft_v7_pkm_film_combined.pt --n_problems 5
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def _pairwise_cos(x: torch.Tensor) -> torch.Tensor:
    """x: (N, d). Returns (N, N) cosine similarity matrix."""
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return xn @ xn.T


def _effective_rank(x: torch.Tensor, eps: float = 1e-9) -> float:
    """Effective rank from singular value entropy: exp(H(p_i)) where
    p_i = σ_i / Σ σ_i. Equals true rank if uniform, 1 if rank-1."""
    if x.shape[0] < 2:
        return float(x.shape[0])
    # x.float() — bf16 lacks the precision for svd
    s = torch.linalg.svdvals(x.float())
    p = s / s.sum().clamp_min(eps)
    H = -(p * (p.clamp_min(eps).log())).sum()
    return float(H.exp().item())


def _capture_hidden_state(model, h_capture_list):
    """Register a forward hook on model.out_norm to capture the
    post-norm hidden state at the OUTPUT of the trunk, which is the
    `h` fed to WM and lm_head."""
    def hook(_module, _input, output):
        h_capture_list.append(output.detach())
    return model.out_norm.register_forward_hook(hook)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_problems", type=int, default=5)
    p.add_argument("--max_gen", type=int, default=300)
    args = p.parse_args()

    from experiments.eval_bracket_structure import build_model_from_ckpt
    from experiments.eval_humaneval import generate
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.cuda().eval()
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer",
                                                 "HuggingFaceTB/SmolLM2-135M"))
    think_id = int(cfg["thinking_token_id"])
    ds = load_dataset("openai_humaneval", split="test")

    print(f"\nRunning {args.n_problems} HumanEval problems, capturing "
          f"trunk-output hidden states at think and emit positions.\n")

    all_think_h = []  # list of (n_think_in_seq, d) tensors per problem
    all_emit_h = []

    for i in range(args.n_problems):
        prob = ds[i]
        wrapped = "# Complete the following Python function.\n" + prob["prompt"]
        prompt_ids = tok.encode(wrapped, add_special_tokens=False)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                 device="cuda").unsqueeze(0)
        # Generate. Capture hidden states for ALL forward passes; we'll
        # take the FINAL forward's output (which covers the entire
        # prompt+completion sequence) as our snapshot.
        captures = []
        hook = _capture_hidden_state(model, captures)
        try:
            out, diag = generate(
                model, prompt_t, max_gen=args.max_gen, temperature=0.0,
                use_thinking=True, max_think_per_step=8,
                total_think_budget=200, emit_threshold=0.5,
                gate_floor=0.0, min_emit_before_eos=30,
                thinking_token_id=think_id,
            )
        finally:
            hook.remove()
        # The last forward's hidden state covers the full sequence so far.
        # captures[-1] shape (1, T_last, d).
        h_full = captures[-1][0]                              # (T, d)
        ids_full = out[0, :h_full.shape[0]].cpu().tolist()    # align to h
        is_think = torch.tensor([t == think_id for t in ids_full])
        think_h = h_full[is_think].float().cpu()
        emit_h = h_full[~is_think].float().cpu()
        all_think_h.append(think_h)
        all_emit_h.append(emit_h)
        print(f"[{i}] {prob['task_id']:>15}  "
              f"n_think={think_h.shape[0]:>3}  n_emit={emit_h.shape[0]:>3}")

    print()
    print("=" * 78)
    print("PER-PROBLEM diversity measurements")
    print("=" * 78)
    print(f"{'problem':<8} {'pos type':<6} {'n':>4} {'mean ||h||':>11} "
          f"{'median cos':>12} {'eff_rank':>10}")
    for i in range(args.n_problems):
        for label, h in [("think", all_think_h[i]), ("emit", all_emit_h[i])]:
            if h.shape[0] < 2:
                print(f"  P{i:<6} {label:<6} {h.shape[0]:>4} (too few — skipping)")
                continue
            mean_norm = h.norm(dim=-1).mean().item()
            cos = _pairwise_cos(h)
            # Strip diagonal (always 1.0), take median of off-diag
            mask = ~torch.eye(cos.shape[0], dtype=torch.bool)
            med_cos = cos[mask].median().item()
            erank = _effective_rank(h)
            print(f"  P{i:<6} {label:<6} {h.shape[0]:>4} "
                  f"{mean_norm:>11.3f} {med_cos:>12.3f} {erank:>10.2f}")

    # Aggregated: pool all problems' hidden states by position type
    print()
    print("=" * 78)
    print("AGGREGATED across problems")
    print("=" * 78)
    for label, hs in [("THINK positions", all_think_h),
                       ("EMIT positions", all_emit_h)]:
        h_concat = torch.cat([h for h in hs if h.shape[0] > 0], dim=0)
        if h_concat.shape[0] < 2:
            print(f"{label}: too few ({h_concat.shape[0]}) for stats")
            continue
        mean_norm = h_concat.norm(dim=-1).mean().item()
        cos = _pairwise_cos(h_concat)
        mask = ~torch.eye(cos.shape[0], dtype=torch.bool)
        med_cos = cos[mask].median().item()
        # eff rank can be expensive — cap at 2000 rows
        if h_concat.shape[0] > 2000:
            idx = torch.randperm(h_concat.shape[0])[:2000]
            erank = _effective_rank(h_concat[idx])
            erank_note = f" (sampled 2000 of {h_concat.shape[0]})"
        else:
            erank = _effective_rank(h_concat)
            erank_note = ""
        print(f"{label:<20}  n={h_concat.shape[0]:>5}  "
              f"mean ||h||={mean_norm:.3f}  "
              f"median pairwise cos={med_cos:+.3f}  "
              f"effective rank={erank:.1f}{erank_note}")

    print()
    print("INTERPRETATION:")
    print("  If THINK median cos >> EMIT median cos → think hidden states are")
    print("  highly correlated. A FIX A buffer (think-only) would be low-rank.")
    print("  If THINK effective rank << EMIT effective rank → same conclusion.")


if __name__ == "__main__":
    sys.exit(main())
