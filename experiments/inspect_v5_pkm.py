"""Inspect what the v5-pkm model actually learned about thinking and memory.

Runs greedy decoding with the thinking gate engaged on a small held-out
prompt set, and captures per-step diagnostics from three sources:

1. **Output gate** (`model._last_gate`): per-token σ(gate_head(h)). When
   does the model decide to emit vs think? What's the gate distribution?

2. **WorkingMemory** (`model.memory`): which past source positions does
   the read-side attend to at think positions? Sharp (lookup-like) or
   diffuse (averaging-like)?

3. **PKM** (`model.pkm_layer`): which slots in the 262 k effective table
   get hit on each token? Is it heavy-tail (a few "hot" slots) or
   distributed? Per-head specialisation?

Inspection is done by monkey-patching forward() on the two memory
modules to stash extra tensors as `._last_*` attributes. The model code
itself is untouched.

Output: prints a human-readable summary and saves a JSON to
`runs/inspect_v5_pkm.json` for downstream plotting / RL-ckpt
comparison.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \\
        experiments/inspect_v5_pkm.py \\
        --ckpt checkpoints/pretrain_mix_v5_pkm.pt \\
        --max_gen 80
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.eval_bracket_structure import build_model_from_ckpt


# ---------------- Held-out prompt set ----------------------------------------

PROMPTS = [
    {
        "name": "humaneval_0_close_elements",
        "kind": "code",
        "text": (
            "from typing import List\n\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer\n'
            "    to each other than given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
    },
    {
        "name": "wikipedia_factual_recall",
        "kind": "wiki",
        "text": "The capital of France is",
    },
    {
        "name": "code_completion_fibonacci",
        "kind": "code",
        "text": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    ",
    },
    {
        "name": "natural_text_continuation",
        "kind": "nat",
        "text": (
            "The Roman Empire reached its greatest extent under emperor Trajan in"
        ),
    },
]


# ---------------- Monkey-patches to capture internals ------------------------

def _patch_working_memory(mem) -> None:
    """Wrap mem.forward to stash `_last_read_attn` and `_last_top_idx`."""
    orig_forward = mem.forward

    def patched(h, input_ids, read_mask=None, doc_ids=None):
        B, T, _ = h.shape
        device = h.device

        # Recompute write-side bits we need (cheap; matches mem.forward).
        write_logits = mem.W_write(h).squeeze(-1)
        g = torch.sigmoid(write_logits)
        if mem.pad_token_id is not None:
            is_pad = input_ids == int(mem.pad_token_id)
            g = g.masked_fill(is_pad, 0.0)
        v = mem.W_v(h)
        K_eff = min(T, mem.mem_size)
        _, top_idx = torch.topk(g, k=K_eff, dim=-1)
        gather_v = top_idx.unsqueeze(-1).expand(-1, -1, mem.d_mem)
        buf_v = torch.gather(v, dim=1, index=gather_v)
        buf_g = torch.gather(g, dim=1, index=top_idx)
        q = mem.W_q(h)
        scale = 1.0 / math.sqrt(mem.d_mem)
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

        # Stash the bits we want to inspect.
        mem._last_read_attn = attn.detach()       # (B, T, K)
        mem._last_top_idx = top_idx.detach()      # (B, K)
        mem._last_write_gate = g.detach()         # (B, T)

        # Then run the real forward (we've duplicated work, but it's the
        # cleanest way to avoid mutating model.py for this read-only probe).
        return orig_forward(h, input_ids, read_mask=read_mask, doc_ids=doc_ids)

    mem.forward = patched


def _patch_pkm(pkm) -> None:
    """Wrap pkm.forward to stash `_last_slot_idx` and `_last_weights`."""
    orig_forward = pkm.forward

    def patched(h):
        B, T, d = h.shape
        H, K, kd, tk = pkm.n_heads, pkm.n_keys, pkm.k_dim, pkm.top_k

        h_n = pkm.norm(h)
        q = pkm.query_proj(h_n).float().view(B, T, H, 2, kd)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        sk1 = pkm.subkeys[:, 0].float()
        sk2 = pkm.subkeys[:, 1].float()
        s1 = torch.einsum("bthk,hnk->bthn", q1, sk1).reshape(-1, K)
        s2 = torch.einsum("bthk,hnk->bthn", q2, sk2).reshape(-1, K)
        s1 = pkm.bn_s1(s1).reshape(B, T, H, K)
        s2 = pkm.bn_s2(s2).reshape(B, T, H, K)
        s1_top, i1 = s1.topk(tk, dim=-1)
        s2_top, i2 = s2.topk(tk, dim=-1)
        scores = (s1_top.unsqueeze(-1) + s2_top.unsqueeze(-2))
        scores_flat = scores.view(B, T, H, tk * tk)
        final_scores, final_idx = scores_flat.topk(tk, dim=-1)
        i_in_top1 = final_idx // tk
        j_in_top2 = final_idx % tk
        sel1 = torch.gather(i1, dim=-1, index=i_in_top1)
        sel2 = torch.gather(i2, dim=-1, index=j_in_top2)
        slot_idx = sel1 * K + sel2
        weights = F.softmax(final_scores, dim=-1)

        pkm._last_slot_idx = slot_idx.detach()    # (B, T, H, tk)
        pkm._last_weights = weights.detach()      # (B, T, H, tk)

        return orig_forward(h)

    pkm.forward = patched


# ---------------- Generation with diagnostics --------------------------------

def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy in nats."""
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=dim)


def generate_with_probes(model, tokenizer, prompt: str, *,
                          max_gen: int = 80,
                          thinking_token_id: int,
                          emit_threshold: float = 0.5,
                          gate_floor: float = 0.5,
                          max_think_per_step: int = 8,
                          ) -> dict:
    """Greedy decode with the gate engaged; record per-step diagnostics.

    Mirrors eval_humaneval.generate but logs the internal state we just
    patched in. Returns a dict ready to json.dump.
    """
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    eos_id = tokenizer.eos_token_id

    # Per-emit-step records.
    steps = []
    think_total = 0
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.no_grad(), autocast:
        for emit_idx in range(max_gen):
            thinks_this_step = 0
            think_gate_vals = []
            think_mem_diags = []
            think_pkm_diags = []
            while True:
                logits = model(ids)
                gate_t = getattr(model, "_last_gate", None)
                if gate_t is None:
                    gate_val = 1.0
                else:
                    gate_val = float(gate_t[0, -1].item())

                # Snapshot memory + PKM at this position (last token of `ids`).
                mem = getattr(model, "memory", None)
                pkm = getattr(model, "pkm_layer", None)
                pos = ids.shape[1] - 1

                this_mem = None
                if mem is not None and hasattr(mem, "_last_read_attn"):
                    attn = mem._last_read_attn[0, pos]              # (K,)
                    if attn.sum() > 0:
                        top_w, top_k = attn.topk(min(5, attn.shape[0]))
                        # source positions of each buffer slot
                        src_pos = mem._last_top_idx[0]              # (K,)
                        # Top sources by attention weight:
                        top_sources = src_pos[top_k]
                        this_mem = {
                            "attn_entropy_nats": float(_entropy(attn).item()),
                            "top_src_pos": top_sources.cpu().tolist(),
                            "top_src_weights": top_w.cpu().tolist(),
                            "top_src_rel_to_cur": (
                                (pos - top_sources).cpu().tolist()
                            ),
                        }

                this_pkm = None
                if pkm is not None and hasattr(pkm, "_last_slot_idx"):
                    slots = pkm._last_slot_idx[0, pos]              # (H, tk)
                    pkm_w = pkm._last_weights[0, pos]               # (H, tk)
                    per_head = []
                    for h in range(slots.shape[0]):
                        top_w, top_k_idx = pkm_w[h].topk(3)
                        per_head.append({
                            "top_slots": slots[h, top_k_idx].cpu().tolist(),
                            "top_weights": top_w.cpu().tolist(),
                            "entropy_nats": float(_entropy(pkm_w[h]).item()),
                        })
                    this_pkm = {"per_head": per_head}

                # Decide emit vs think.
                gate_clamped = max(gate_val, gate_floor) if gate_floor > 0 else gate_val
                force_emit = thinks_this_step >= max_think_per_step
                if gate_clamped >= emit_threshold or force_emit:
                    break
                # Else: append a think token and loop.
                think_tok = torch.full((ids.shape[0], 1), thinking_token_id,
                                        dtype=ids.dtype, device=ids.device)
                ids = torch.cat([ids, think_tok], dim=1)
                thinks_this_step += 1
                think_total += 1
                think_gate_vals.append(gate_val)
                think_mem_diags.append(this_mem)
                think_pkm_diags.append(this_pkm)

            # Emit the next token.
            next_logits = logits[:, -1, :].clone()
            next_logits[..., thinking_token_id] = -float("inf")
            next_tok_id = int(next_logits.argmax(dim=-1).item())
            ids = torch.cat([ids, torch.tensor([[next_tok_id]], device=device,
                                                dtype=ids.dtype)], dim=1)
            steps.append({
                "emit_idx": emit_idx,
                "token": tokenizer.decode([next_tok_id]),
                "token_id": next_tok_id,
                "gate_at_emit": gate_val,
                "thinks_before_emit": thinks_this_step,
                "think_gate_vals": think_gate_vals,
                "mem_at_emit": this_mem,
                "pkm_at_emit": this_pkm,
                "mem_during_thinks": think_mem_diags,
                "pkm_during_thinks": think_pkm_diags,
            })
            if eos_id is not None and next_tok_id == eos_id:
                break

    # Strip thinking tokens for clean text view.
    out_ids = ids[0].cpu().tolist()
    out_no_think = [t for t in out_ids[prompt_len:] if t != thinking_token_id]
    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "out_text": tokenizer.decode(out_no_think),
        "steps": steps,
        "think_total": think_total,
        "emit_total": len(steps),
    }


# ---------------- Aggregate summary ------------------------------------------

def summarise(result: dict) -> dict:
    """Compute headline numbers per prompt."""
    n = result["emit_total"]
    thinks = [s["thinks_before_emit"] for s in result["steps"]]
    gates = [s["gate_at_emit"] for s in result["steps"]]
    n_with_think = sum(1 for t in thinks if t > 0)

    # Memory: at emit positions where read_mask fires (i.e. think tokens),
    # the read attention was computed. Easier to summarise per think step.
    mem_entropies = []
    mem_rel_pos = []
    for s in result["steps"]:
        for m in s["mem_during_thinks"]:
            if m is not None:
                mem_entropies.append(m["attn_entropy_nats"])
                if m["top_src_rel_to_cur"]:
                    mem_rel_pos.append(m["top_src_rel_to_cur"][0])

    # PKM: per-head entropy at each emit step (we ran PKM on every forward).
    pkm_entropies_per_head = {h: [] for h in range(4)}
    pkm_top_slot_freq = {h: {} for h in range(4)}
    for s in result["steps"]:
        if s["pkm_at_emit"] is not None:
            for h_idx, h in enumerate(s["pkm_at_emit"]["per_head"]):
                pkm_entropies_per_head[h_idx].append(h["entropy_nats"])
                for slot in h["top_slots"]:
                    pkm_top_slot_freq[h_idx][slot] = (
                        pkm_top_slot_freq[h_idx].get(slot, 0) + 1
                    )

    return {
        "n_emit": n,
        "n_think_steps": result["think_total"],
        "think_rate": n_with_think / max(1, n),
        "mean_thinks_per_emit": sum(thinks) / max(1, n),
        "gate_mean": sum(gates) / max(1, len(gates)),
        "gate_min": min(gates) if gates else None,
        "gate_max": max(gates) if gates else None,
        "memory": {
            "n_reads": len(mem_entropies),
            "mean_attn_entropy_nats": (
                sum(mem_entropies) / len(mem_entropies)
                if mem_entropies else None
            ),
            "max_uniform_entropy_nats": (
                math.log(1024) if mem_entropies else None
            ),
            "mean_top_src_rel_pos": (
                sum(mem_rel_pos) / len(mem_rel_pos)
                if mem_rel_pos else None
            ),
        },
        "pkm": {
            "per_head_mean_entropy_nats": {
                h: (sum(v) / len(v) if v else None)
                for h, v in pkm_entropies_per_head.items()
            },
            "max_uniform_entropy_nats_per_head": math.log(32),
            "per_head_top_5_hottest_slots": {
                h: sorted(freqs.items(), key=lambda kv: -kv[1])[:5]
                for h, freqs in pkm_top_slot_freq.items()
            },
        },
    }


# ---------------- Main -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--max_gen", type=int, default=80)
    ap.add_argument("--gate_floor", type=float, default=0.5,
                    help="Match training's --gate_floor_min.")
    ap.add_argument("--emit_threshold", type=float, default=0.5)
    ap.add_argument("--out_json", default="runs/inspect_v5_pkm.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model, cfg = build_model_from_ckpt(args.ckpt)
    model = model.to("cuda").eval()

    thinking_token_id = int(cfg.get("thinking_token_id"))
    print(f"[inspect] ckpt={args.ckpt}")
    print(f"[inspect] thinking_token_id={thinking_token_id}  "
          f"gate_floor={args.gate_floor}  emit_threshold={args.emit_threshold}")

    # Patch the internals we want to inspect.
    if hasattr(model, "memory") and model.memory is not None:
        _patch_working_memory(model.memory)
        print(f"[inspect] WorkingMemory patched (mem_size={model.memory.mem_size})")
    if hasattr(model, "pkm_layer") and model.pkm_layer is not None:
        _patch_pkm(model.pkm_layer)
        print(f"[inspect] PKMLayer patched "
              f"({model.pkm_layer.n_heads}h × {model.pkm_layer.n_keys}² = "
              f"{model.pkm_layer.n_heads * model.pkm_layer.n_keys ** 2} effective slots)")

    out = {"prompts": [], "summaries": []}
    for p in PROMPTS:
        print(f"\n[inspect] === {p['name']} ({p['kind']}) ===")
        print(f"PROMPT: {p['text']!r}")
        result = generate_with_probes(
            model, tok, p["text"],
            max_gen=args.max_gen,
            thinking_token_id=thinking_token_id,
            emit_threshold=args.emit_threshold,
            gate_floor=args.gate_floor,
        )
        summary = summarise(result)

        print(f"GENERATED ({summary['n_emit']} emit, "
              f"{summary['n_think_steps']} think, "
              f"think_rate={summary['think_rate']:.2f}):")
        print(f"  > {result['out_text']!r}")
        print(f"  gate mean/min/max = "
              f"{summary['gate_mean']:.3f} / "
              f"{summary['gate_min']:.3f} / {summary['gate_max']:.3f}")
        if summary["memory"]["n_reads"] > 0:
            mu = summary["memory"]["mean_attn_entropy_nats"]
            maxu = summary["memory"]["max_uniform_entropy_nats"]
            print(f"  WM read attn entropy: {mu:.2f} nats  "
                  f"(uniform-over-1024 = {maxu:.2f}, "
                  f"sharpness = {1 - mu/maxu:+.0%})")
            print(f"  WM mean top-src rel pos: "
                  f"-{summary['memory']['mean_top_src_rel_pos']:.1f} "
                  f"(token-distance back to attended source)")
        else:
            print("  WM no read events (no think tokens emitted)")
        if summary["pkm"]["per_head_mean_entropy_nats"]:
            print(f"  PKM per-head entropy (max-uniform-over-32 = "
                  f"{summary['pkm']['max_uniform_entropy_nats_per_head']:.2f}):")
            for h, e in summary["pkm"]["per_head_mean_entropy_nats"].items():
                if e is not None:
                    sharp = 1 - e / summary["pkm"]["max_uniform_entropy_nats_per_head"]
                    print(f"    head {h}: {e:.2f} nats (sharpness {sharp:+.0%})  "
                          f"top-5 slots: "
                          f"{summary['pkm']['per_head_top_5_hottest_slots'][h]}")

        out["prompts"].append({"name": p["name"], "kind": p["kind"],
                               "prompt": p["text"], "result": result})
        out["summaries"].append({"name": p["name"], "kind": p["kind"],
                                  "summary": summary})

    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[inspect] wrote {out_path}  "
          f"({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
