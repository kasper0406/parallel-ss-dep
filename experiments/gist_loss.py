"""Trunk multi-horizon gist loss — shared by sft_code.py and train_lm.py.

The trunk's "high-level direction" objective: at position t, a per-horizon
head predicts the GIST of the upcoming window — the mean-pooled hidden
state over h[t+1 : t+1+K], stop-grad'd. Because the trunk is causal each
h[t] is a running contextualised summary, so the windowed mean is a
genuine "where this is going" vector. Multi-horizon K gives local tactic
+ mid plan + global direction.

History (see GEMINI.md): v5 supervised the WM read to predict a single
future input embedding (context-free lexical); v6 supervised the WM read
to predict this gist (a blurry target routed through the recall path —
broke precise recall, 99%→61%). v7 moved the gist target to the TRUNK and
left WM free to learn precise retrieval. This module is that v7 mechanism,
factored out so the pretrain trainer can bake it in from step 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_horizons(s: str) -> list[int]:
    """Parse a comma-separated horizon string ('16,64,256') into a sorted
    unique list of positive ints."""
    hs = sorted({int(k) for k in s.split(",") if k.strip()})
    if not hs or any(k <= 0 for k in hs):
        raise ValueError(f"invalid gist horizons: {s!r}")
    return hs


def windowed_future_gist(h: torch.Tensor, K: int):
    """Mean-pooled hidden state over the K positions FOLLOWING each source
    position — the gist-prediction target.

    Given hidden states h of shape (B, T, d), returns (gist, valid_len):
      gist[:, t] = mean(h[:, t+1 : t+1+K])   for source t in [0, T-1-K]
      valid_len  = T - K  (number of source positions with a full window)

    Returns (None, 0) when K >= T (no full window). Computed in fp32 via a
    cumulative sum so it is O(T) not O(T*K). NOT detached — the caller
    detaches it (the trunk hidden states stay supervised only by the main
    loss; the gist is a stop-grad target)."""
    T = h.shape[1]
    if K >= T:
        return None, 0
    cumh = torch.cumsum(h.float(), dim=1)            # (B, T, d)
    # sum h[t+1 .. t+K] = cumh[t+K] - cumh[t]
    win_sum = cumh[:, K:] - cumh[:, :-K]             # (B, T-K, d)
    return win_sum / float(K), T - K


def build_gist_heads(d_model: int, horizons: list[int]) -> nn.ModuleDict:
    """One bias-free Linear(d_model, d_model) prediction head per horizon,
    keyed by str(K). Small init so the aux loss starts gentle."""
    heads = nn.ModuleDict({
        str(k): nn.Linear(d_model, d_model, bias=False) for k in horizons
    })
    for head in heads.values():
        nn.init.normal_(head.weight, std=0.02 / math.sqrt(2))
    return heads


def trunk_gist_loss(h: torch.Tensor, heads: nn.ModuleDict,
                    horizons: list[int]) -> torch.Tensor:
    """Multi-horizon trunk gist loss: for each horizon K, predict
    windowed_future_gist(h, K) from h[t] via head_K, score by 1 - cosine,
    average over positions and over the horizons that produced a valid
    window. The gist target is stop-grad'd; gradient flows to the trunk
    only through the prediction path."""
    acc = h.new_zeros(())
    n_used = 0
    for K in horizons:
        gist, vlen = windowed_future_gist(h, K)
        if gist is None:
            continue
        pred = heads[str(K)](h[:, :vlen].contiguous())
        cos = F.cosine_similarity(pred.float(), gist.detach(), dim=-1)
        acc = acc + (1.0 - cos).mean()
        n_used += 1
    if n_used == 0:
        return h.new_zeros(())
    return acc / n_used
