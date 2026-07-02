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
        # ---------- v7 PKM-bootstrap-fix package (2026-05-17) ----------
        # The v5-pkm probe found: 97% of value rows still at init, only
        # 4% of slots ever hit, residual contribution ~1%. Five fixes:
        score_norm: str = "layer",          # FIX 4: "batch" (Lample) or
                                            #        "layer" (default, ours).
        value_init_std: float = 1.0,         # FIX 3: init values at
                                            # residual-stream magnitude
                                            # instead of 1/sqrt(d_model).
        use_output_gate: bool = True,        # FIX 1: scalar α (init 0)
                                            # multiplies PKM output. Lets
                                            # the model gradually trust
                                            # the table — mirrors FiLM α.
        out_alpha_init: float = 0.0,         # Warm-start for α (2026-07-02,
                                            # converged-base pre-warm probe):
                                            # a POSITIVE init keeps the
                                            # sign-preserving floor from
                                            # flipping sign while αL noisy
                                            # around 0 (the pilot-B pattern).
                                            # 0.0 = legacy byte-identical.
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

        # Score normalisation (FIX 4): BatchNorm or LayerNorm.
        # BatchNorm-on-scores (Lample) accumulates running stats over batches
        # but at our token scale those stats are noisy early in training,
        # which destabilises the score distribution and breaks slot selection.
        # LayerNorm computes per-position stats from the current input, no
        # running stats, no early-training drift. Default = "layer".
        self.score_norm_kind = score_norm
        if score_norm == "batch":
            self.bn_s1 = nn.BatchNorm1d(n_keys, eps=bn_eps, momentum=bn_momentum)
            self.bn_s2 = nn.BatchNorm1d(n_keys, eps=bn_eps, momentum=bn_momentum)
        elif score_norm == "layer":
            # LayerNorm over the n_keys feature dim. Per-side.
            self.ln_s1 = nn.LayerNorm(n_keys, eps=bn_eps)
            self.ln_s2 = nn.LayerNorm(n_keys, eps=bn_eps)
        else:
            raise ValueError(f"score_norm must be 'batch' or 'layer', got {score_norm!r}")

        # Value table per head, as nn.Embedding for fast gather.
        # n_keys ** 2 rows, v_dim_per_head channels, one Embedding per head.
        # (Stacking heads into one Embedding would force a shared lookup;
        # per-head separate tables = independent memory per head.)
        # FIX 3: init at `value_init_std` instead of the original 1/sqrt(d_model)
        # (= ~0.04 for d=576). v5-pkm probe showed that small init gives PKM
        # output magnitude ~1% of the residual stream, so it can't meaningfully
        # contribute even when slots route well. value_init_std=1.0 puts the
        # PKM output on the same scale as the residual stream from step 0.
        dtype = torch.bfloat16 if value_bf16 else torch.float32
        self.values = nn.ModuleList([
            nn.Embedding(n_keys * n_keys, v_dim_per_head)
            for _ in range(n_heads)
        ])
        self.value_init_std = float(value_init_std)
        for emb in self.values:
            nn.init.normal_(emb.weight, mean=0.0, std=self.value_init_std)
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

        # FIX 1: output gate α. Single learnable scalar, init at 0 so PKM
        # contributes ZERO at step 1 (the model can't be poisoned by random
        # initial output) and the gradient grows α naturally if PKM is
        # useful. Mirrors the FiLM α curriculum that worked in v3a/v4/v5.
        # When False, PKM output is added unscaled (the v5-pkm behaviour).
        self.use_output_gate = bool(use_output_gate)
        if self.use_output_gate:
            self.out_alpha = nn.Parameter(torch.full((1,), float(out_alpha_init)))
        # FIX 1B (added 2026-05-17 after v7 step-440 α-decay observation):
        # `alpha_floor` is a sign-preserving additive minimum magnitude that
        # the trainer sets per-step (annealed from `pkm_alpha_floor_start`
        # to 0 over `pkm_alpha_floor_warmup_steps`, synced to ε-greedy).
        # Effective gate is `α_eff = α_learned + sign(α)·α_floor`.
        # During warmup this *guarantees* a non-trivial PKM contribution so
        # value rows receive meaningful gradient — breaking the
        # bootstrap deadlock the v7 step-280 → step-440 trace exposed
        # (α grew 0 → 0.085 then *shrank* back to 0.04 because PKM output
        # was structured noise, while v_std stayed glued at 1.000).
        # α itself still receives the usual gradient so it can settle to
        # the right post-warmup magnitude.
        self.alpha_floor = 0.0
        # FIX 2: ε for random-slot exploration. Set externally by the
        # trainer per step (curriculum); defaults to 0 (no exploration).
        # When ε > 0, at training time each top-k retrieval has prob ε of
        # being replaced by a uniformly random slot, ensuring every slot
        # receives at least some gradient even before the learned router
        # would have selected it. The v5-pkm probe found only 4% of slots
        # ever fired — this is the direct fix.
        self.random_slot_epsilon = 0.0
        # FIX 5: stash the per-step slot distribution so the trainer can
        # compute a diversity-entropy auxiliary loss (encourages spreading
        # retrievals across the full table, penalises the head_2/slot_8824
        # hot-spot pattern observed in v5-pkm).
        self._last_slot_idx = None
        self._last_weights = None
        # v7.1 follow-up: register the EXPECTED init row-norm so the trainer
        # diagnostic can report drift from init. (The previous `v_std` diag
        # was misleading — variance-over-all-elements is invariant under
        # gradient updates that preserve approximate Gaussian distribution,
        # which is exactly what happens to centred-init values under SGD.
        # Mean row-norm DOES detect drift — it's the metric the v5-pkm
        # probe used to confirm the table was frozen.)
        # Expected init row-norm: sqrt(v_dim_per_head) * value_init_std.
        self.register_buffer(
            "_expected_init_row_norm",
            torch.tensor(math.sqrt(v_dim_per_head) * self.value_init_std),
            persistent=False,
        )

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

        # 3. Score normalisation (BN or LN — see FIX 4 in __init__).
        if self.score_norm_kind == "batch":
            s1_flat = s1.reshape(-1, K)
            s2_flat = s2.reshape(-1, K)
            s1 = self.bn_s1(s1_flat).reshape(B, T, H, K)
            s2 = self.bn_s2(s2_flat).reshape(B, T, H, K)
        else:  # "layer"
            s1 = self.ln_s1(s1)
            s2 = self.ln_s2(s2)

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

        # FIX 2: ε-greedy random-slot exploration. At training time, with
        # probability ε we replace a learned-router pick with a uniform
        # random slot. Gradient still flows through the embedding lookup at
        # the randomised index, so every slot gets some training signal
        # even before the router would have selected it. Anneal ε → 0 via
        # the trainer-driven curriculum (set self.random_slot_epsilon).
        if self.training and self.random_slot_epsilon > 0.0:
            rand_mask = (torch.rand_like(slot_idx, dtype=torch.float32)
                         < self.random_slot_epsilon)
            rand_slots = torch.randint(
                0, K * K, slot_idx.shape, device=slot_idx.device,
                dtype=slot_idx.dtype,
            )
            slot_idx = torch.where(rand_mask, rand_slots, slot_idx)

        # 7. Softmax over the final top-k scores.
        weights = F.softmax(final_scores, dim=-1)  # (B, T, H, tk), fp32

        # FIX 5: stash for diversity-bonus computation by the trainer.
        # Detached so trainer aux-loss does NOT backpropagate through the
        # router; we only want the entropy on selection statistics.
        self._last_slot_idx = slot_idx.detach()
        self._last_weights = weights.detach()

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
        # FIX 1 + 1B: gate the output by learned α with sign-preserving
        # additive floor. Init α=0 + floor=0 → contribution = 0 at step 1.
        # During warmup (floor > 0): α_eff = α + sign(α)·floor so PKM
        # contribution magnitude ≥ floor, ensuring value rows get
        # meaningful gradient even before α has grown. Sign defaults to +1
        # when α is essentially zero (the cold-start convention).
        # `self.training` gate (2026-07-02 design review FIX 1, real bug):
        # the floor is a TRAINING-ONLY bootstrap crutch — it must never fire
        # during eval/VAL, else VAL includes a floor-forced random-value
        # contribution that doesn't reflect the learned model (train/eval
        # mismatch; this polluted Phase-1 arm B's VAL, SESSION_FINDINGS.md
        # 2026-07-02). `alpha_floor` itself is set externally per-STEP by the
        # trainer curriculum (not per-forward-call), so gating its
        # application here doesn't disturb the warmup bookkeeping — eval
        # forwards interleaved mid-training (e.g. the VAL-loss `model.eval()`
        # window in train_lm.py) simply skip the floor for that forward and
        # resume floor-application on the very next training-mode forward,
        # with the curriculum's step-driven value unchanged. The `αeff`
        # console/TB diagnostics in train_lm.py recompute αeff independently
        # from `pkm.out_alpha` / `pkm.alpha_floor` (not from a value stashed
        # by this forward), so they continue to report the training-time
        # value regardless of the model's current train/eval mode.
        if self.use_output_gate:
            if self.training and self.alpha_floor > 0.0:
                α = self.out_alpha
                sign = torch.where(
                    α.detach().abs() > 1e-3,
                    torch.sign(α.detach()),
                    torch.ones_like(α),
                )
                α_eff = α + sign * self.alpha_floor
            else:
                α_eff = self.out_alpha
            out = α_eff * out
        # Cast back to input dtype (bf16 in autocast contexts).
        return out.to(h.dtype)
