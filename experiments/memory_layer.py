"""Product-Key Memory layer (Lample et al. 2019; Berges et al. 2024).

A learned sparse key/value table dropped in as a residual side-module:

    h_out = h + PKM(norm(h))

Each query splits into two halves; each half does a top-k over its own
sub-key matrix; the cartesian product of the two top-k sets gives a
candidate set of `top_k_per_side ** 2` entries from the full
`n_keys ** 2` value table, from which the final top-k is selected.

This is sub-linear in the table size: query cost is O(n_keys · k_dim) vs
O(n_keys^2 · k_dim) for naive lookup, so a 65 k or 1 M-slot table costs
roughly the same as a 256-slot one per query.

Multi-head: each head has independent sub-keys *and* its own value
sub-table, so the final memory contribution is the concat of per-head
weighted-sum-of-values.

Cold-start fix: BatchNorm over the per-side scores (PKM warmup trick).
Without it only a handful of sub-keys win every lookup and the rest
never receive gradient.

Storage: values are stored bf16 to halve persistent memory footprint;
math runs in fp32 (cast on retrieval). Sub-keys / projections are fp32
(small, no real saving).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PKMLayer(nn.Module):
    """Product-key memory layer.

    Args:
        d_model: residual stream dim (input/output channels).
        n_heads: number of independent (sub-key, value-table) pairs.
        n_keys: per-sub-key-set size; total slots per head = n_keys ** 2.
        k_dim: per-sub-key-half query dim (so per-head query dim = 2 * k_dim).
        top_k: number of values retrieved per query (after the outer-product
            top-k of the two side top-ks).
        v_dim_per_head: per-head value dim. Default = d_model // n_heads
            so concatenated heads land back at d_model with no extra proj.
        value_bf16: store the big value table in bf16 (math still fp32).
        bn_eps / bn_momentum: standard BatchNorm1d kwargs for the score BN.
        norm: pre-PKM LayerNorm vs RMSNorm. Defaults to LayerNorm to match
            the Lample reference; the rest of the model uses RMSNorm but
            PKM scores are sensitive to centring.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_keys: int = 256,
        k_dim: int = 128,
        top_k: int = 32,
        v_dim_per_head: int | None = None,
        value_bf16: bool = True,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        if v_dim_per_head is None:
            assert d_model % n_heads == 0, (
                f"d_model={d_model} not divisible by n_heads={n_heads}"
            )
            v_dim_per_head = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_keys = n_keys
        self.k_dim = k_dim
        self.top_k = top_k
        self.v_dim_per_head = v_dim_per_head
        self.value_bf16 = value_bf16

        # Pre-PKM normalisation (Lample uses LayerNorm; we keep that).
        self.norm = nn.LayerNorm(d_model)

        # Query projection: d_model -> n_heads * 2 * k_dim
        self.query_proj = nn.Linear(d_model, n_heads * 2 * k_dim, bias=True)

        # Sub-keys: (n_heads, 2, n_keys, k_dim) — one matrix per head per side.
        self.subkeys = nn.Parameter(
            torch.empty(n_heads, 2, n_keys, k_dim)
        )
        nn.init.normal_(self.subkeys, mean=0.0, std=1.0 / math.sqrt(k_dim))

        # BatchNorm on per-side scores. We BN over the n_keys feature dim,
        # batched over (B*T*n_heads). One BN per side (2 total) — they
        # see different score distributions.
        self.bn_s1 = nn.BatchNorm1d(n_keys, eps=bn_eps, momentum=bn_momentum)
        self.bn_s2 = nn.BatchNorm1d(n_keys, eps=bn_eps, momentum=bn_momentum)

        # Value table per head, as nn.Embedding for fast gather.
        # n_keys ** 2 rows, v_dim_per_head channels, one Embedding per head.
        # (Stacking heads into one Embedding would force a shared lookup;
        # per-head separate tables = independent memory per head.)
        dtype = torch.bfloat16 if value_bf16 else torch.float32
        self.values = nn.ModuleList([
            nn.Embedding(n_keys * n_keys, v_dim_per_head)
            for _ in range(n_heads)
        ])
        for emb in self.values:
            nn.init.normal_(emb.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
            if value_bf16:
                emb.weight.data = emb.weight.data.to(torch.bfloat16)

        # Output is concat of heads (n_heads * v_dim_per_head). If that
        # equals d_model exactly (default), no output projection is
        # needed — the residual addition is dimensionally clean.
        # If not, project to d_model.
        if n_heads * v_dim_per_head != d_model:
            self.out_proj = nn.Linear(n_heads * v_dim_per_head, d_model,
                                       bias=False)
        else:
            self.out_proj = None

    @property
    def n_value_params(self) -> int:
        """Number of params in the value table (the dominant cost)."""
        return self.n_heads * self.n_keys * self.n_keys * self.v_dim_per_head

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d_model) -> (B, T, d_model). Caller does residual add."""
        B, T, d = h.shape
        H, K, kd, tk = self.n_heads, self.n_keys, self.k_dim, self.top_k

        # 1. Norm + query projection.
        h_n = self.norm(h)
        # cast to fp32 for the score math (autocast may have us in bf16).
        q = self.query_proj(h_n).float()
        # (B, T, H, 2, k_dim)
        q = q.view(B, T, H, 2, kd)
        q1 = q[:, :, :, 0]   # (B, T, H, kd)
        q2 = q[:, :, :, 1]   # (B, T, H, kd)

        # 2. Per-side scores. subkeys[:, 0] is (H, K, kd), broadcast over BT.
        # Score = q · key^T per head.
        sk1 = self.subkeys[:, 0].float()   # (H, K, kd)
        sk2 = self.subkeys[:, 1].float()   # (H, K, kd)
        # einsum over the k_dim axis.
        s1 = torch.einsum("bthk,hnk->bthn", q1, sk1)   # (B, T, H, K)
        s2 = torch.einsum("bthk,hnk->bthn", q2, sk2)   # (B, T, H, K)

        # 3. BatchNorm over the K feature dim, batched over (B*T*H).
        s1_flat = s1.reshape(-1, K)
        s2_flat = s2.reshape(-1, K)
        s1 = self.bn_s1(s1_flat).reshape(B, T, H, K)
        s2 = self.bn_s2(s2_flat).reshape(B, T, H, K)

        # 4. Per-side top-k. We use top_k per side; the outer product gives
        # top_k**2 candidates from which we pick the final top_k.
        topk_per_side = tk
        s1_top, i1 = s1.topk(topk_per_side, dim=-1)   # (B, T, H, tk)
        s2_top, i2 = s2.topk(topk_per_side, dim=-1)   # (B, T, H, tk)

        # 5. Outer-sum scores: (B, T, H, tk, tk)
        scores = s1_top.unsqueeze(-1) + s2_top.unsqueeze(-2)
        # Flatten the candidate axis: (B, T, H, tk*tk)
        scores_flat = scores.view(B, T, H, topk_per_side * topk_per_side)

        # 6. Final top-k from tk*tk candidates.
        final_scores, final_idx_in_grid = scores_flat.topk(tk, dim=-1)
        # final_idx_in_grid: indices into the (tk*tk) outer-product grid.
        # Decode back into (i,j) on top1, top2.
        i_in_top1 = final_idx_in_grid // topk_per_side   # (B, T, H, tk)
        j_in_top2 = final_idx_in_grid %  topk_per_side   # (B, T, H, tk)
        # Map back to indices in [0, K).
        sel1 = torch.gather(i1, dim=-1, index=i_in_top1)   # (B, T, H, tk)
        sel2 = torch.gather(i2, dim=-1, index=j_in_top2)   # (B, T, H, tk)
        # Combined index into the n_keys**2 value table.
        slot_idx = sel1 * K + sel2     # (B, T, H, tk)

        # 7. Softmax over the final top-k scores.
        weights = F.softmax(final_scores, dim=-1)  # (B, T, H, tk), fp32

        # 8. Gather values per head and weighted-sum.
        # Per-head Embedding lookup: indices (B, T, tk) -> (B, T, tk, v_dim).
        outs = []
        for h_idx, emb in enumerate(self.values):
            idx_h = slot_idx[:, :, h_idx]                  # (B, T, tk)
            v = emb(idx_h.reshape(-1)).view(B, T, tk, -1)  # (B, T, tk, v_dim)
            v = v.float()   # bf16 storage -> fp32 math
            w = weights[:, :, h_idx]                       # (B, T, tk)
            o = (v * w.unsqueeze(-1)).sum(dim=-2)          # (B, T, v_dim)
            outs.append(o)

        out = torch.cat(outs, dim=-1)  # (B, T, n_heads * v_dim_per_head)
        if self.out_proj is not None:
            out = self.out_proj(out)
        # Cast back to input dtype (bf16 in autocast contexts).
        return out.to(h.dtype)
