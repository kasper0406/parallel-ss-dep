"""Diagnose what the thinking machinery (WorkingMemory + PKM) ACTUALLY does
at think positions vs emit positions on the distilled SFT ckpt.

The A/B HumanEval result (pass@1 6/164 with thinking == 6/164 without) shows
thinking is decorative — the model emits the same code with or without it.
This probe answers WHY: is the thinking infrastructure carrying real
information, or is it inert?

Five things we measure on every HumanEval forward:

  1. WM write-gate distribution
     `g_t = σ(W_write(h_t))` at each position. Healthy: writes concentrate
     on a few salient positions (high variance, sparse spikes). Dead:
     uniform around 0.5 (no preference).

  2. WM top-K buffer composition
     Which positions end up in the K-slot buffer? Healthy: includes
     think positions where the model "computed" something. Dead: just
     the highest-magnitude tokens regardless of position.

  3. WM read attention at think positions
     Softmax-attention over buffer slots when a query happens at a
     think token. Healthy: sharp, concentrated on a few buffer slots.
     Dead: near-uniform (no information retrieval).

  4. WM injection magnitude
     ||W_proj(read)|| at think vs emit positions. Should be much larger
     at think positions (since `inj` is gated by the read_mask). If
     similar magnitude across positions, the architectural advantage
     of "memory injection only at think" is being wasted.

  5. PKM slot-routing differences think vs emit
     Per-head top-k slot indices at think positions vs emit positions.
     Healthy: think-position routing differs from emit-position routing
     (think positions retrieve different information). Dead: same slot
     distribution regardless of position type → PKM is position-blind.

Usage:
    PYTHONPATH=. .venv/bin/python experiments/diag_thinking_machinery.py \\
        --ckpt checkpoints/sft_v7_pkm_film_distilled.pt --n_problems 5
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Monkey-patches to capture internals during forward
# ---------------------------------------------------------------------------

class _Recorder:
    """Captures per-forward intermediates that the model doesn't expose."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.wm_attn = None         # last WM softmax attention (B, T, K)
        self.wm_injection_norm = None  # ||injection||_2 per position (B, T)
        self.wm_top_idx = None      # positions chosen for the K-slot buffer
        self.pkm_slot_idx = None    # PKM slot indices per (B, T, head, tk)
        self.pkm_weights = None     # PKM weights per (B, T, head, tk)


def _patch_wm(wm_module, recorder: _Recorder):
    """Wrap WorkingMemory.forward to stash attn, top_idx, injection norm."""
    original_forward = wm_module.forward.__func__

    def forward_with_record(self, h, input_ids, read_mask=None, doc_ids=None):
        B, T, _ = h.shape
        device = h.device
        # Copy of the body but stashing intermediates. Lazy approach:
        # just call original and patch the returned tensor. We can't get
        # attn/injection without re-doing the math; so we do it here too.
        write_logits = self.W_write(h).squeeze(-1)
        g = torch.sigmoid(write_logits)
        v = self.W_v(h)
        if self.pad_token_id is not None:
            is_pad = input_ids == int(self.pad_token_id)
            g = g.masked_fill(is_pad, 0.0)
        K_eff = min(T, self.mem_size)
        _, top_idx = torch.topk(g, k=K_eff, dim=-1)
        gather_idx_v = top_idx.unsqueeze(-1).expand(-1, -1, self.d_mem)
        buf_v = torch.gather(v, dim=1, index=gather_idx_v)
        buf_g = torch.gather(g, dim=1, index=top_idx)
        q = self.W_q(h)
        import math as _m
        scale = 1.0 / _m.sqrt(self.d_mem)
        scores = torch.einsum("btd,bkd->btk", q, buf_v) * scale
        scores = scores + torch.log(buf_g.clamp_min(1e-6)).unsqueeze(1)
        pos = torch.arange(T, device=device).view(1, T, 1)
        src_pos = top_idx.unsqueeze(1)
        causal_mask = src_pos >= pos
        scores = scores.masked_fill(causal_mask, float("-inf"))
        blocked = causal_mask
        if doc_ids is not None:
            buf_doc = torch.gather(doc_ids, 1, top_idx)
            doc_mask = buf_doc.unsqueeze(1) != doc_ids.unsqueeze(-1)
            scores = scores.masked_fill(doc_mask, float("-inf"))
            blocked = causal_mask | doc_mask
        all_masked = blocked.all(dim=-1, keepdim=True)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(all_masked, torch.zeros_like(attn), attn)
        read = torch.einsum("btk,bkd->btd", attn, buf_v)
        injection = self.W_proj(read)
        # Stash
        recorder.wm_attn = attn.detach().float().cpu()                  # (B,T,K)
        recorder.wm_top_idx = top_idx.detach().cpu()                    # (B,K)
        recorder.wm_injection_norm = injection.norm(dim=-1).detach().float().cpu()  # (B,T)
        # Now do the injection-mask gating to match the real behaviour
        if read_mask is None:
            inj_mask = (input_ids == self.thinking_token_id).to(h.dtype).unsqueeze(-1)
        else:
            inj_mask = read_mask.to(h.dtype).unsqueeze(-1)
        return h + injection * inj_mask

    wm_module.forward = forward_with_record.__get__(wm_module, type(wm_module))


def _patch_pkm(pkm_module, recorder: _Recorder):
    """Wrap PKMLayer to stash slot indices and weights."""
    original_forward = pkm_module.forward.__func__

    def forward_with_record(self, h):
        out = original_forward(self, h)
        # PKM internally stashes _last_slot_idx and _last_weights (per FIX 5)
        if hasattr(self, "_last_slot_idx"):
            recorder.pkm_slot_idx = self._last_slot_idx.detach().cpu()
            recorder.pkm_weights = self._last_weights.detach().cpu()
        return out

    pkm_module.forward = forward_with_record.__get__(pkm_module, type(pkm_module))


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _report_one(recorder: _Recorder, input_ids: torch.Tensor,
                thinking_token_id: int, label: str):
    """Print per-forward summary for one problem.
    `input_ids` here is the full out tensor from generate(), which
    includes the just-sampled final token. The recorder reflects the
    last forward (input to that sample step), so we drop the trailing
    token to align dims with recorder.wm_attn / wm_injection_norm /
    pkm_slot_idx (all of which have shape (1, T_input, ...))."""
    # Drop trailing token to align with recorder snapshot.
    if recorder.wm_injection_norm is not None:
        T_rec = recorder.wm_injection_norm.shape[-1]
    elif recorder.pkm_slot_idx is not None:
        T_rec = recorder.pkm_slot_idx.shape[0]
    else:
        T_rec = input_ids.shape[1]
    ids = input_ids[0, :T_rec].cpu().tolist()
    is_think = torch.tensor([t == thinking_token_id for t in ids])
    T = len(ids)
    n_think = int(is_think.sum())
    n_emit = T - n_think
    print(f"\n=== {label}  T={T}  n_think={n_think}  n_emit={n_emit} ===")
    if n_think == 0:
        print("  (no think tokens — skipping think/emit comparisons)")
        return

    # --- WM injection magnitude at think vs emit ----------------------------
    if recorder.wm_injection_norm is not None:
        inj = recorder.wm_injection_norm[0]  # (T,)
        inj_think = inj[is_think]
        inj_emit = inj[~is_think]
        print(f"  WM ||W_proj(read)||:")
        print(f"    at think  positions: mean={inj_think.mean().item():.3f}  "
              f"max={inj_think.max().item():.3f}")
        print(f"    at emit   positions: mean={inj_emit.mean().item():.3f}  "
              f"max={inj_emit.max().item():.3f}")
        print(f"    (only think contributes to h; emit's W_proj(read) is unused.)")

    # --- WM read attention sharpness at think positions ---------------------
    if recorder.wm_attn is not None:
        attn = recorder.wm_attn[0]  # (T, K)
        # Effective number of attended slots = exp(entropy(attn))
        eps = 1e-9
        ent = -(attn * (attn.clamp_min(eps).log())).sum(dim=-1)  # (T,)
        eff_k = ent.exp()
        # Filter to think positions where attn isn't all-zero
        valid = attn.sum(dim=-1) > 0.5  # exclude positions with all-masked attn
        if (is_think & valid).any():
            eff_k_think = eff_k[is_think & valid]
            print(f"  WM read attention sharpness at think positions:")
            print(f"    effective # buffer slots attended (lower = sharper):")
            print(f"    median={eff_k_think.median().item():.2f}  "
                  f"min={eff_k_think.min().item():.2f}  "
                  f"max={eff_k_think.max().item():.2f}")
            print(f"    (buffer has K={attn.shape[1]} total slots)")
        if (~is_think & valid).any():
            eff_k_emit = eff_k[~is_think & valid]
            print(f"    (for comparison, at emit positions: "
                  f"median={eff_k_emit.median().item():.2f})")

    # --- WM top-K buffer composition ----------------------------------------
    if recorder.wm_top_idx is not None:
        top = recorder.wm_top_idx[0]  # (K,)
        n_in_buf_think = sum(1 for i in top.tolist() if is_think[i].item())
        n_in_buf_total = top.shape[0]
        print(f"  WM buffer composition: {n_in_buf_think}/{n_in_buf_total} "
              f"({100*n_in_buf_think/n_in_buf_total:.0f}%) of K-slots are "
              f"think positions  (baseline: {100*n_think/T:.0f}% of T are think)")
        if n_in_buf_think / n_in_buf_total > 1.2 * n_think / T:
            print(f"    → enriched for think positions (good)")
        elif n_in_buf_think / n_in_buf_total < 0.8 * n_think / T:
            print(f"    → depleted of think positions (suspicious)")
        else:
            print(f"    → similar to baseline (think writes aren't preferred)")

    # --- PKM slot routing think vs emit -------------------------------------
    if recorder.pkm_slot_idx is not None:
        slot = recorder.pkm_slot_idx[0]  # (T, n_heads, top_k)
        slot_think = slot[is_think]      # (n_think, H, k)
        slot_emit = slot[~is_think]      # (n_emit, H, k)
        # Per-head: count unique slots hit by think vs emit, fraction overlap
        H = slot.shape[1]
        for h_i in [0, H // 2, H - 1]:
            think_slots = set(slot_think[:, h_i, :].flatten().tolist())
            emit_slots = set(slot_emit[:, h_i, :].flatten().tolist())
            inter = think_slots & emit_slots
            union = think_slots | emit_slots
            print(f"  PKM head {h_i:>2}: think hit {len(think_slots)} unique slots, "
                  f"emit hit {len(emit_slots)}, overlap "
                  f"{len(inter)}/{len(union)} ({100*len(inter)/max(1,len(union)):.0f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n_problems", type=int, default=5)
    p.add_argument("--max_gen", type=int, default=400)
    args = p.parse_args()

    print(f"Loading {args.ckpt}")
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.cuda().eval()
    thinking_token_id = int(cfg["thinking_token_id"])

    recorder = _Recorder()
    _patch_wm(model.memory, recorder)
    if hasattr(model, "pkm_layer"):
        _patch_pkm(model.pkm_layer, recorder)
    print("  patched WM + PKM for instrumentation")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer",
                                                 "HuggingFaceTB/SmolLM2-135M"))

    ds = load_dataset("openai_humaneval", split="test")
    print(f"\nRunning {args.n_problems} HumanEval problems with thinking ON, "
          f"sft_comment prompt, max_gen={args.max_gen}")

    for i in range(args.n_problems):
        prob = ds[i]
        wrapped = "# Complete the following Python function.\n" + prob["prompt"]
        prompt_ids = tok.encode(wrapped, add_special_tokens=False)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long,
                                 device="cuda").unsqueeze(0)
        recorder.reset()
        out, diag = generate(
            model, prompt_t, max_gen=args.max_gen,
            temperature=0.0,
            use_thinking=True,
            max_think_per_step=8,
            total_think_budget=200,
            emit_threshold=0.5,
            gate_floor=0.0,
            min_emit_before_eos=30,
            thinking_token_id=thinking_token_id,
        )
        # The LAST forward inside generate() processed `out` up to its
        # final token — recorder holds that snapshot, which includes the
        # entire prompt + generated trace. That's what we want.
        _report_one(recorder, out, thinking_token_id,
                    f"problem={prob['task_id']}  "
                    f"emit={diag['emit_count']}  "
                    f"think_total={diag['think_total']}")


if __name__ == "__main__":
    sys.exit(main())
