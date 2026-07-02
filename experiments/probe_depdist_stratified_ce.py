"""Dependency-distance-stratified per-position CE on natural long Python files.

THE decisive probe for "do the mechanisms (FiLM / WM / PKM) help code where
they SHOULD" — long-range token reuse. Per-position next-token CE on natural
codeparrot files (>= T+1 tokens), stratified by DEPENDENCY DISTANCE = #tokens
since the PREVIOUS occurrence of the target token in the same file:
    first-occurrence / <64 / 64-255 / 256-1023 / 1024-2047 / >=2048
and split by token class (identifier-like = decoded token is [A-Za-z0-9_]+,
the recall-bearing class, vs other).

Mechanism toggles (paired per-token, same data for every config):
    base      : ckpt exactly as trained/deployed (mem_read_mask=None)
    pkm_off   : pkm_layer.out_alpha zeroed (the kill-gate toggle)
    film_off  : model._film_bypass = True (the trainer's K=1 "feedback none" path)
    wm_on     : mem_read_mask = all-True (engages WM read + copy head, the way
                recall streams engage it in training; on natural text the WM is
                structurally inert without this because read_alpha==0 and the
                copy head requires an explicit mask)
    all_off   : pkm_off + film_off

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      experiments/probe_depdist_stratified_ce.py \
      --ckpt checkpoints/phase1_10L_A.pt --label 10L_A \
      --n_seqs 700 --T 3072 --skip 50000 \
      --data_cache /tmp/scratch/depdist_data_T3072.pt \
      --out /tmp/scratch/depdist_10L_A.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

STRATA = ["first", "<64", "64-255", "256-1023", "1024-2047", ">=2048"]
CLASSES = ["ident", "other"]


def collect_data(tokenizer_name: str, T: int, n_seqs: int, skip: int,
                 cache_path: str) -> torch.Tensor:
    """Stream codeparrot-clean, skip `skip` examples, keep files that tokenize
    to >= T+1 tokens, truncate to T+1. Returns LongTensor (N, T+1). Cached."""
    if cache_path and os.path.exists(cache_path):
        seqs = torch.load(cache_path, weights_only=True)
        assert seqs.shape[1] == T + 1, f"cache T mismatch {seqs.shape}"
        assert seqs.shape[0] >= n_seqs, f"cache too small {seqs.shape}"
        print(f"[data] loaded cache {cache_path} shape={tuple(seqs.shape)}")
        return seqs[:n_seqs]
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("codeparrot/codeparrot-clean", split="train",
                      streaming=True)
    if skip > 0:
        ds = ds.skip(skip)
    out, seen, t0 = [], 0, time.time()
    for ex in ds:
        seen += 1
        txt = ex.get("content", "")
        if len(txt) < 4 * T:      # cheap pre-filter: need >= T+1 tokens
            continue
        ids = tok.encode(txt, add_special_tokens=False)
        if len(ids) < T + 1:
            continue
        out.append(torch.tensor(ids[:T + 1], dtype=torch.long))
        if len(out) % 100 == 0:
            print(f"[data] {len(out)}/{n_seqs} kept ({seen} scanned, "
                  f"{time.time()-t0:.0f}s)")
        if len(out) >= n_seqs:
            break
    seqs = torch.stack(out)
    if cache_path:
        torch.save(seqs, cache_path)
        print(f"[data] cached {tuple(seqs.shape)} -> {cache_path}")
    return seqs


def dep_distance(ids: torch.Tensor) -> torch.Tensor:
    """Distance (in tokens) from target position i to the previous occurrence
    of token ids[i] within ids[0:i]. -1 = first occurrence. Returns (L-1,)
    int32 for targets at positions 1..L-1."""
    last: dict[int, int] = {}
    L = ids.shape[0]
    out = torch.full((L - 1,), -1, dtype=torch.int32)
    arr = ids.tolist()
    last[arr[0]] = 0
    for i in range(1, L):
        t = arr[i]
        j = last.get(t)
        if j is not None:
            out[i - 1] = i - j
        last[t] = i
    return out


def stratum_index(dist: torch.Tensor) -> torch.Tensor:
    """Map distance array to stratum index 0..5 (STRATA)."""
    s = torch.zeros_like(dist)
    s[(dist >= 1) & (dist < 64)] = 1
    s[(dist >= 64) & (dist < 256)] = 2
    s[(dist >= 256) & (dist < 1024)] = 3
    s[(dist >= 1024) & (dist < 2048)] = 4
    s[dist >= 2048] = 5
    s[dist < 0] = 0
    return s


def ident_table(tokenizer_name: str, vocab_size: int) -> torch.Tensor:
    """Bool table (vocab,) — token decodes (mod leading space) to [A-Za-z0-9_]+."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    tbl = torch.zeros(vocab_size, dtype=torch.bool)
    pat = re.compile(r"[A-Za-z0-9_]+")
    toks = tok.convert_ids_to_tokens(list(range(len(tok))))
    for i, piece in enumerate(toks):
        if piece is None or i >= vocab_size:
            continue
        core = piece[1:] if piece.startswith("Ġ") else piece  # strip Ġ
        if core and pat.fullmatch(core):
            tbl[i] = True
    return tbl


def make_configs(model, want: list[str]) -> list[str]:
    has_pkm = hasattr(model, "pkm_layer") and model.pkm_layer is not None
    has_film = bool(getattr(model, "feedback_pairs", ()) and
                    getattr(model, "feedback_mode", "none") != "none")
    has_mem = bool(getattr(model, "use_memory", False))
    cfgs = ["base"]
    if "pkm_off" in want and has_pkm:
        cfgs.append("pkm_off")
    if "film_off" in want and has_film:
        cfgs.append("film_off")
    if "wm_on" in want and has_mem and getattr(model, "use_copy_head", False):
        cfgs.append("wm_on")
    if "all_off" in want and (has_pkm or has_film):
        cfgs.append("all_off")
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n_seqs", type=int, default=700)
    ap.add_argument("--T", type=int, default=3072)
    ap.add_argument("--skip", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--data_cache", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--configs", default="pkm_off,film_off,wm_on,all_off")
    args = ap.parse_args()

    seqs = collect_data(args.tokenizer, args.T, args.n_seqs, args.skip,
                        args.data_cache)
    N, Lp1 = seqs.shape
    T = Lp1 - 1
    print(f"[probe] {args.label}: {N} seqs x {T} scored positions "
          f"= {N*T/1e6:.2f}M tokens")

    # Precompute per-seq strata + token-class (CPU, once — identical all arms).
    print("[probe] computing dependency distances ...")
    t0 = time.time()
    strat = torch.stack([stratum_index(dep_distance(seqs[i]))
                         for i in range(N)])            # (N, T) int32 0..5
    print(f"[probe] dep distances done in {time.time()-t0:.0f}s")

    from experiments.eval_bracket_structure import build_model_from_ckpt
    model, cfg = build_model_from_ckpt(args.ckpt)
    model.eval()
    vocab = cfg["vocab_size"]
    ident = ident_table(args.tokenizer, vocab)          # (vocab,) bool CPU

    configs = make_configs(model, args.configs.split(","))
    print(f"[probe] configs: {configs}")

    pkm = getattr(model, "pkm_layer", None)
    pkm_alpha = pkm.out_alpha.data.clone() if (pkm is not None and
                                               hasattr(pkm, "out_alpha")) else None

    # accumulators: [config][stratum 0..5][class 0..1] -> (sum_ce, n)
    sums = {c: torch.zeros(6, 2, dtype=torch.float64) for c in configs}
    cnts = {c: torch.zeros(6, 2, dtype=torch.int64) for c in configs}
    # per-seq totals for paired SE of deltas: (n_cfg, N, 6) sums + shared counts
    seq_sums = {c: torch.zeros(N, 6, dtype=torch.float64) for c in configs}
    seq_cnts = torch.zeros(N, 6, dtype=torch.int64)
    copy_gate_sum, copy_gate_n = 0.0, 0

    def set_config(c: str):
        model._film_bypass = c in ("film_off", "all_off")
        if pkm_alpha is not None:
            if c in ("pkm_off", "all_off"):
                pkm.out_alpha.data.zero_()
            else:
                pkm.out_alpha.data.copy_(pkm_alpha)

    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, N, args.batch):
            xb_full = seqs[b0:b0 + args.batch].cuda()      # (B, T+1)
            x, y = xb_full[:, :-1], xb_full[:, 1:]         # (B, T)
            st = strat[b0:b0 + args.batch]                 # (B, T) cpu
            cls = (~ident[y.cpu()]).long()                 # 0=ident, 1=other
            for c in configs:
                set_config(c)
                rm = None
                if c == "wm_on":
                    rm = torch.ones_like(x, dtype=torch.bool)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x, mem_read_mask=rm)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                ce = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    y.reshape(-1), reduction="none").view(y.shape).cpu()  # (B,T)
                if c == "wm_on":
                    g = getattr(model, "_last_copy_gate_eff", None)
                    if g is not None:
                        copy_gate_sum += float(g.float().sum())
                        copy_gate_n += g.numel()
                for s_i in range(6):
                    for cl in range(2):
                        m = (st == s_i) & (cls == cl)
                        if m.any():
                            sums[c][s_i, cl] += float(ce[m].sum())
                            cnts[c][s_i, cl] += int(m.sum())
                    ms = (st == s_i)
                    seq_sums[c][b0:b0 + x.shape[0], s_i] += \
                        (ce * ms.float()).sum(dim=1).double()
                    if c == configs[0]:
                        seq_cnts[b0:b0 + x.shape[0], s_i] += ms.sum(dim=1)
            if (b0 // args.batch) % 20 == 0:
                el = time.time() - t0
                done = b0 + args.batch
                print(f"[probe] {done}/{N} seqs  {el:.0f}s "
                      f"(eta {el/max(done,1)*(N-done):.0f}s)", flush=True)

    set_config("base")
    res = {"label": args.label, "ckpt": args.ckpt, "n_seqs": N, "T": T,
           "configs": configs, "strata": STRATA,
           "copy_gate_mean": (copy_gate_sum / copy_gate_n
                              if copy_gate_n else None),
           "table": {}}
    for c in configs:
        res["table"][c] = {
            "ce": (sums[c] / cnts[c].clamp_min(1)).tolist(),
            "n": cnts[c].tolist(),
            "total_ce": float(sums[c].sum() / cnts[c].sum()),
        }
    # paired per-seq delta SE vs base, per stratum (both classes pooled)
    res["delta_se"] = {}
    base_seq = seq_sums["base"] / seq_cnts.clamp_min(1)
    for c in configs[1:]:
        d = (seq_sums[c] / seq_cnts.clamp_min(1)) - base_seq   # (N, 6)
        valid = seq_cnts > 0
        se = []
        for s_i in range(6):
            v = d[:, s_i][valid[:, s_i]]
            se.append(float(v.std() / max(v.shape[0], 1) ** 0.5)
                      if v.shape[0] > 1 else float("nan"))
        res["delta_se"][c] = se

    # print table
    print(f"\n===== {args.label} ({args.ckpt}) =====")
    print(f"total tokens scored: {int(cnts['base'].sum())}")
    hdr = f"{'stratum':>10} {'class':>6} {'n_tok':>9}" + \
        "".join(f" {c:>10}" for c in configs) + \
        "".join(f" {'d_'+c:>10}" for c in configs[1:])
    print(hdr)
    for s_i, s in enumerate(STRATA):
        for cl, cname in enumerate(CLASSES):
            n = int(cnts["base"][s_i, cl])
            if n == 0:
                continue
            ces = [float(sums[c][s_i, cl] / n) for c in configs]
            ds = [ces[i] - ces[0] for i in range(1, len(configs))]
            print(f"{s:>10} {cname:>6} {n:>9}" +
                  "".join(f" {v:>10.4f}" for v in ces) +
                  "".join(f" {v:>+10.4f}" for v in ds))
    for c in configs:
        print(f"TOTAL {c:>10}: CE={res['table'][c]['total_ce']:.4f}")
    if copy_gate_n:
        print(f"wm_on mean effective copy gate: {copy_gate_sum/copy_gate_n:.5f}")
    for c in configs[1:]:
        print(f"delta SE ({c} - base) per stratum: "
              + ", ".join(f"{s}={v:.4f}" for s, v in
                          zip(STRATA, res["delta_se"][c])))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
