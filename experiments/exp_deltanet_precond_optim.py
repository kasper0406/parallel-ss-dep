"""DeltaNet-TAILORED matrix optimizer (standalone, does NOT touch optim_utils).

Muon = steepest descent under the spectral (RMS->RMS operator) norm via
Newton-Schulz orthogonalization. It is the modular-norm *dualizer* for a
GENERIC linear layer, and is architecture-naive to DeltaNet's structure:

  S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T,   o_t = S_t q_t

  * q_proj / k_proj / v_proj produce n_heads INDEPENDENT associative memories
    (no cross-head mixing in the recurrence). The modular-norm-correct dual
    of a multi-head linear op is the PER-HEAD spectral norm -> Newton-Schulz
    each head's output slice independently, not the whole (d_model x d_model)
    matrix as one operator. (Bernstein/Newhouse modular norm; head-structured
    dualizer.)
  * k enters only via k k^T and pairs with v via v k^T; q reads via S q. The
    loss sees only relative q/k geometry -> a per-head O(d_head) key-space
    gauge shared by q_proj and k_proj (q->Uq, k->Uk leaves outputs invariant).
    Newton-Schulz is equivariant under left-orthogonal mult, so per-head NS on
    q and k SEPARATELY already commutes with this gauge. The 'qk_coupled' mode
    goes further: it stacks [g_q[h]; g_k[h]] per head and NS's them JOINTLY,
    forcing the query and key bases to span complementary residual directions
    (orthogonalize q against k) -- a stronger, DeltaNet-specific inductive
    bias to test.
  * qk_norm='l2' on q,k adds a per-head RADIAL gauge (vector magnitude is
    killed by the L2 norm). per-head NS already produces a (semi-)orthogonal
    update whose rows are balanced, which is consistent with that radial gauge.

This file implements `DeltaNetProjMuon`, a torch.optim.Optimizer applied to
ONLY the DeltaNet q/k/v projections (+ the beta/b_proj). It REUSES torch's
exact Newton-Schulz (`_zeropower_via_newtonschulz`) and Muon momentum/nesterov/
decoupled-WD/LR-adjust conventions, so the ONLY thing that differs from plain
Muon is WHERE the orthogonalization boundary is drawn (per-head / qk-coupled
vs whole-matrix). o_proj is intentionally NOT tailored (it MIXES heads -- the
one place heads interact -- so the whole-matrix Muon dual is correct there);
keep o_proj + MLP on standard Muon.

RMS-consistency (why the same LR range is fair across arms): a (d_head x d_in)
row-orthonormal slice has Frobenius^2 = d_head; summed over n_heads that is
n_heads*d_head = d_model = the Frobenius^2 of the whole (d_model x d_in)
orthogonal Muon update. So per-head and whole-matrix updates have the SAME
total norm -> the same `lr` produces the same effective step size. (Verified
numerically in the harness self-test.)
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

# Reuse torch's EXACT Newton-Schulz + LR-adjust so the only moving part is the
# orthogonalization boundary (per-head vs whole-matrix), not the NS polynomial.
from torch.optim._muon import _zeropower_via_newtonschulz, _adjust_lr, DEFAULT_A, DEFAULT_B, DEFAULT_C

_NS = (DEFAULT_A, DEFAULT_B, DEFAULT_C)
_EPS = 1e-7


def _ns(g: torch.Tensor, steps: int) -> torch.Tensor:
    return _zeropower_via_newtonschulz(g, _NS, steps, _EPS)


def _ns_batched(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Batched Newton-Schulz over a stack of slices G of shape (H, r, c) with
    r <= c. Numerically IDENTICAL to calling torch's `_zeropower_via_newtonschulz`
    on each slice independently (same quintic coeffs, same bf16, same per-slice
    Frobenius normalization, same addmm structure) -- just vectorized with bmm so
    the per-head dual costs ~one batched matmul instead of H kernel launches.
    All tailored DeltaNet slices have rows <= cols (dh<=d_in, 2dh<=d_in,
    1<=d_in), so no per-slice transpose branch is needed."""
    a, b, c = _NS
    X = G.bfloat16()
    n = X.norm(dim=(1, 2), keepdim=True).clamp(min=_EPS)
    X = X / n
    for _ in range(steps):
        A = X @ X.transpose(1, 2)          # (H, r, r)
        B = b * A + c * (A @ A)            # (H, r, r)
        X = a * X + B @ X                  # (H, r, c)
    return X


def build_units_from_model(model, *, mode: str):
    """Scan the model and return (tailored_params, tailored_units, other_2d_params).

    tailored set = q_proj / k_proj / v_proj / b_proj of every DeltaNet block.
    o_proj is EXCLUDED (mixes heads -> whole-matrix Muon dual is correct).
    other_2d_params = every OTHER 2D hidden matrix (MLP, o_proj) -> standard Muon.
    Embeddings / lm_head / 1D / 3D conv -> caller routes to AdamW (unchanged).
    """
    # Discover per-block (n_heads, d_head) from the attention wrapper.
    by_block: dict[str, dict] = {}
    name_of = {id(p): n for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        if ".attn.layer." not in n or not n.endswith(".weight"):
            continue
        prefix = n.split(".attn.layer.")[0]          # e.g. "blocks.0"
        leaf = n.split(".attn.layer.")[1]            # e.g. "q_proj.weight"
        role = leaf.split(".")[0]                     # q_proj / k_proj / ...
        by_block.setdefault(prefix, {})[role] = p

    # n_heads / d_head from the wrapper module.
    head_meta: dict[str, tuple[int, int]] = {}
    for mod_name, mod in model.named_modules():
        inner = getattr(mod, "layer", None)
        if inner is None:
            continue
        nh = getattr(inner, "num_heads", None)
        if nh is None:
            continue
        # FLA DeltaNet: key_dim = num_heads * head_k_dim. q_proj out = key_dim.
        qp = getattr(inner, "q_proj", None)
        if qp is None:
            continue
        d_head = qp.weight.shape[0] // nh
        head_meta[mod_name] = (int(nh), int(d_head))

    tailored_params, units = [], []
    other_2d = []
    used = set()
    for prefix, roles in by_block.items():
        attn_mod = f"{prefix}.attn"
        nh, dh = head_meta[attn_mod]
        qp, kp, vp = roles.get("q_proj"), roles.get("k_proj"), roles.get("v_proj")
        bp = roles.get("b_proj")
        op = roles.get("o_proj")
        if mode == "qk_coupled" and qp is not None and kp is not None:
            units.append({"kind": "qk_coupled", "q": qp, "k": kp,
                          "n_heads": nh, "d_head": dh, "params": [qp, kp]})
            used.update({id(qp), id(kp)})
        else:
            for pp in (qp, kp):
                if pp is not None:
                    units.append({"kind": "matrix_perhead", "param": pp,
                                  "n_heads": nh, "d_head": dh, "params": [pp]})
                    used.add(id(pp))
        if vp is not None:
            units.append({"kind": "matrix_perhead", "param": vp,
                          "n_heads": nh, "d_head": dh, "params": [vp]})
            used.add(id(vp))
        if bp is not None:
            units.append({"kind": "rownorm", "param": bp, "params": [bp]})
            used.add(id(bp))
        # o_proj is NOT tailored.
        if op is not None:
            other_2d.append(op)
            used.add(id(op))
    for u in units:
        tailored_params.extend(u["params"])

    # All remaining 2D hidden matrices (MLP weights, etc.) -> standard Muon.
    for n, p in model.named_parameters():
        if not p.requires_grad or p.ndim != 2 or id(p) in used:
            continue
        if n in {"embed.weight", "pos_embed.weight", "lm_head.weight"}:
            continue  # embedding-like -> AdamW
        other_2d.append(p)
        used.add(id(p))
    return tailored_params, units, other_2d


class DeltaNetProjMuon(Optimizer):
    """Muon with the orthogonalization done PER-HEAD (and optionally QK-coupled)
    on DeltaNet q/k/v/b projections. Identical momentum / nesterov / decoupled
    WD / LR-adjust to torch.optim.Muon -- only the NS boundary differs."""

    def __init__(self, units, *, lr: float, momentum: float = 0.95,
                 weight_decay: float = 0.0, nesterov: bool = True,
                 ns_steps: int = 5):
        params = []
        for u in units:
            params.extend(u["params"])
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay,
                                      nesterov=nesterov, ns_steps=ns_steps))
        self._units = units

    def _upd(self, p, mom, nesterov):
        g = p.grad
        st = self.state[p]
        if "momentum_buffer" not in st:
            st["momentum_buffer"] = torch.zeros_like(g)
        buf = st["momentum_buffer"]
        buf.lerp_(g, 1 - mom)                       # buf = mom*buf + (1-mom)*g
        return g.lerp(buf, mom) if nesterov else buf  # nesterov look-ahead

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        grp = self.param_groups[0]
        lr, mom, wd = grp["lr"], grp["momentum"], grp["weight_decay"]
        nesterov, ns_steps = grp["nesterov"], grp["ns_steps"]

        for u in self._units:
            kind = u["kind"]
            if kind == "qk_coupled":
                q, k = u["q"], u["k"]
                if q.grad is None or k.grad is None:
                    continue
                nh, dh = u["n_heads"], u["d_head"]
                d_in = q.shape[1]
                gq = self._upd(q, mom, nesterov).view(nh, dh, d_in)
                gk = self._upd(k, mom, nesterov).view(nh, dh, d_in)
                stacked = torch.cat([gq, gk], dim=1)         # (nh, 2*dh, d_in)
                o = _ns_batched(stacked, ns_steps).to(gq.dtype)
                oq, ok = o[:, :dh], o[:, dh:]
                # adjust_lr per the SLICE shape (here 2*dh x d_in -> ratio 1).
                alr = _adjust_lr(lr, None, (2 * dh, d_in))
                q.mul_(1 - lr * wd); q.add_(oq.reshape_as(q), alpha=-alr)
                k.mul_(1 - lr * wd); k.add_(ok.reshape_as(k), alpha=-alr)
            elif kind == "matrix_perhead":
                p = u["param"]
                if p.grad is None:
                    continue
                nh, dh = u["n_heads"], u["d_head"]
                d_in = p.shape[1]
                g = self._upd(p, mom, nesterov).view(nh, dh, d_in)
                out = _ns_batched(g, ns_steps).to(g.dtype)
                alr = _adjust_lr(lr, None, (dh, d_in))
                p.mul_(1 - lr * wd); p.add_(out.reshape_as(p), alpha=-alr)
            elif kind == "rownorm":   # b_proj: (n_heads, d_in), one row per head
                p = u["param"]
                if p.grad is None:
                    continue
                nh, d_in = p.shape
                g = self._upd(p, mom, nesterov).view(nh, 1, d_in)
                out = _ns_batched(g, ns_steps).to(g.dtype).view(nh, d_in)
                alr = _adjust_lr(lr, None, (1, d_in))
                p.mul_(1 - lr * wd); p.add_(out, alpha=-alr)
        return loss
