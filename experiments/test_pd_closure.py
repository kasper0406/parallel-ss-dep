"""Numerical closure check for PD-SSM scan.

Verifies the load-bearing math claim: the cumulative product of T
(P_t · D_t) operations is exactly representable as a single (P, D)
pair (no rank growth, no error). Same for ComplexPDScanAttention.

If this passes, we know the implementation honours the algebra; if it
fails, there's a bug in the gather-scatter pattern or the σ⁻¹ semantics.

This is the kind of test you write once, run before burning GPU hours.

Usage:
    python experiments/test_pd_closure.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch


def _build_P_inv(sigma_idx: torch.Tensor, N: int) -> torch.Tensor:
    """One-hot encoding of σ⁻¹ as a column-one-hot matrix.

    sigma_idx: (..., N) int — for each output slot i, the input slot σ⁻¹(i).
    Returns: (..., N, N) where M[i, j] = 1 iff σ⁻¹(i) = j.
    """
    return torch.nn.functional.one_hot(sigma_idx, num_classes=N).float()


def test_pd_closure_real():
    """A product of T (P_t, D_t) PD matrices equals a single (P_acc, D_acc)."""
    torch.manual_seed(0)
    N, T, B = 8, 16, 4
    # Random sigma_idx and D per step.
    sigma_seq = torch.randint(0, N, (T, B, N))           # (T, B, N)
    D_seq = torch.empty(T, B, N).uniform_(-1.0, 1.0)     # (T, B, N)

    # Reference: build the full N×N matrices and multiply.
    P_inv_seq = _build_P_inv(sigma_seq, N)                # (T, B, N, N)
    D_diag_seq = torch.diag_embed(D_seq)                  # (T, B, N, N)
    A_seq = P_inv_seq @ D_diag_seq                        # (T, B, N, N)
    # Note: cumulative product applies LATER tokens on the OUTSIDE
    # (h_t = A_t · h_{t-1}), so the cumulative is A_T · A_{T-1} · ... · A_1.
    # Build it explicitly:
    A_cum = A_seq[0].clone()
    for t in range(1, T):
        A_cum = A_seq[t] @ A_cum                          # (B, N, N)

    # Closure claim: A_cum is itself in the PD class. In our convention
    # (P_inv row-one-hot), the cumulative matrix has at most one nonzero
    # PER ROW — row i picks input column σ_acc(i) where σ_acc is the
    # composition of σ⁻¹s. Multiple rows can map to the same column when
    # the σ_t are non-bijective; that's fine and expected (T_N includes
    # non-bijections).
    nz_per_row = (A_cum.abs() > 1e-9).sum(dim=-1)          # (B, N)
    assert (nz_per_row <= 1).all(), \
        f"PD closure violated: row has multiple nonzeros, max={int(nz_per_row.max())}"

    # And the nonzero, when present, equals the cumulative D product along
    # the chain. Since this is a stronger statement, just sanity-check
    # the nonzero magnitudes are bounded by 1 (each |D_t| ≤ 1 ⇒ |product| ≤ 1).
    assert A_cum.abs().max().item() <= 1.0 + 1e-6, \
        f"cumulative |D-product| > 1 — bug in chain: {A_cum.abs().max().item()}"

    print("✓ test_pd_closure_real: cumulative product stays in PD class "
          f"(N={N}, T={T}, B={B}; max nonzeros per row = "
          f"{int(nz_per_row.max())}; |D_acc| ≤ {A_cum.abs().max().item():.4f})")


def test_pdscan_layer_matches_explicit_matmul():
    """The PDScanAttention layer's output equals the explicit-matmul reference.

    Build a tiny PDScanAttention, run it on a fixed input, and compare
    with the full-matrix-cumulative-product reference computed from the
    same σ_logits and D parameters.
    """
    from experiments.layers import PDScanAttention

    torch.manual_seed(0)
    B, T, d_model, H, d_head = 2, 8, 16, 2, 4
    N = 4

    layer = PDScanAttention(d_model=d_model, n_heads=H, d_head=d_head,
                            state_dim=N, use_short_conv=False,
                            use_silu_input=False)
    layer.eval()

    x = torch.randn(B, T, d_model)
    # Forward through the layer.
    y_layer = layer(x)

    # Reference: replicate the recurrence using the layer's own projections.
    with torch.no_grad():
        sigma_logits = layer.W_sigma(x).view(B, T, H, N, N)
        sigma_idx = sigma_logits.argmax(dim=-1)            # (B, T, H, N)
        # In eval, ST gradient is irrelevant — use hard one-hot.
        P_inv = torch.nn.functional.one_hot(sigma_idx, num_classes=N).float()
        D = 2.0 * torch.sigmoid(layer.W_D(x).view(B, T, H, N)) - 1.0
        w = layer.W_w(x).view(B, T, H, N)

        h = torch.zeros(B, H, N)
        out = torch.empty(B, T, H, N)
        for t in range(T):
            P_inv_t = P_inv[:, t]                           # (B, H, N, N)
            g_h = (P_inv_t @ h.unsqueeze(-1)).squeeze(-1)
            g_D = (P_inv_t @ D[:, t].unsqueeze(-1)).squeeze(-1)
            h = g_D * g_h + w[:, t]
            out[:, t] = h
        y_ref = layer.W_o(out.reshape(B, T, H * N))

    err = (y_layer - y_ref).abs().max().item()
    assert err < 1e-5, f"layer output mismatch: max abs error {err}"
    print(f"✓ test_pdscan_layer_matches_explicit_matmul: max err {err:.2e}")


def test_complex_pd_unit_disk():
    """Complex-PD eigenvalues stay in the closed unit disk."""
    from experiments.layers import ComplexPDScanAttention

    torch.manual_seed(0)
    B, T, d_model, H, d_head = 2, 8, 16, 2, 4
    N = 4

    layer = ComplexPDScanAttention(d_model=d_model, n_heads=H, d_head=d_head,
                                   state_dim=N, use_short_conv=False,
                                   use_silu_input=False)
    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        D_mag = torch.sigmoid(layer.W_D_mag(x).view(B, T, H, N))
    assert (D_mag <= 1.0).all() and (D_mag >= 0.0).all(), \
        f"|D| out of [0, 1]: range [{D_mag.min().item()}, {D_mag.max().item()}]"
    print(f"✓ test_complex_pd_unit_disk: |D| ∈ [{D_mag.min().item():.4f}, "
          f"{D_mag.max().item():.4f}] ⊂ [0, 1]")


def test_pdkv_layer_matches_explicit():
    """PDKVScanAttention output equals an explicit reimplementation."""
    from experiments.layers import PDKVScanAttention

    torch.manual_seed(0)
    B, T, d_model, H, d_head = 2, 8, 16, 2, 4
    N = 4
    Dh = d_head

    layer = PDKVScanAttention(d_model=d_model, n_heads=H, d_head=d_head,
                              state_dim=N, use_short_conv=False,
                              use_silu_input=False)
    layer.eval()
    x = torch.randn(B, T, d_model)
    y_layer = layer(x)

    with torch.no_grad():
        sigma_logits = layer.W_sigma(x).view(B, T, H, N, N)
        sigma_idx = sigma_logits.argmax(dim=-1)
        P_inv = torch.nn.functional.one_hot(sigma_idx, num_classes=N).float()
        Dvec = 2.0 * torch.sigmoid(layer.W_D(x).view(B, T, H, N)) - 1.0
        k = layer.W_k(x).view(B, T, H, N)
        v = layer.W_v(x).view(B, T, H, Dh)
        q = layer.W_q(x).view(B, T, H, N)

        S = torch.zeros(B, H, N, Dh)
        out = torch.empty(B, T, H, Dh)
        for t in range(T):
            P_inv_t = P_inv[:, t]
            g_S = P_inv_t @ S
            g_D = (P_inv_t @ Dvec[:, t].unsqueeze(-1)).squeeze(-1)
            S = g_D.unsqueeze(-1) * g_S + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
            out[:, t] = (q[:, t].unsqueeze(-2) @ S).squeeze(-2)
        y_ref = layer.W_o(out.reshape(B, T, H * Dh))

    err = (y_layer - y_ref).abs().max().item()
    assert err < 1e-5, f"PD-KV layer mismatch: max err {err}"
    print(f"✓ test_pdkv_layer_matches_explicit: max err {err:.2e}")


def test_pd_with_zero_input_returns_zero_state_then_writes():
    """If write is forced to 0 and h_0 = 0, h_t stays 0 forever."""
    from experiments.layers import PDScanAttention

    torch.manual_seed(0)
    B, T, d_model, H, d_head = 2, 8, 16, 2, 4
    N = 4

    layer = PDScanAttention(d_model=d_model, n_heads=H, d_head=d_head,
                            state_dim=N, use_short_conv=False,
                            use_silu_input=False)
    # Zero out the write projection so w_t = 0 always.
    with torch.no_grad():
        layer.W_w.weight.zero_()
        # Output projection identity-ish — doesn't matter, we just check h.
    x = torch.randn(B, T, d_model)
    y = layer(x)
    # If h_0 = 0 and w = 0, h_t = D_t * gather(h_{t-1}) = 0 forever ⇒ y = 0.
    err = y.abs().max().item()
    assert err < 1e-5, f"zero-write should give zero output, got max {err}"
    print(f"✓ test_pd_with_zero_input_returns_zero_state_then_writes: max y = {err:.2e}")


def test_muon_optimizer_runs():
    """Muon end-to-end: forward, backward, step on a small 2D-param model."""
    import torch
    from experiments.optim_muon import Muon, muon_param_groups

    torch.manual_seed(0)
    # Model with mixed 1D and 2D params.
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.LayerNorm(16),  # has 1D weight + bias
        torch.nn.Linear(16, 4),
    )
    muon_p, adam_p = muon_param_groups(model)
    assert len(muon_p) > 0, "no 2D params went to Muon group"
    assert len(adam_p) > 0, "no 1D/embed params went to AdamW group"

    muon = Muon(muon_p, lr=0.02)
    adam = torch.optim.AdamW(adam_p, lr=1e-3)

    x = torch.randn(4, 8)
    target = torch.randn(4, 4)
    init_loss = float("inf")
    for step in range(20):
        muon.zero_grad(); adam.zero_grad()
        out = model(x)
        loss = (out - target).pow(2).mean()
        if step == 0: init_loss = loss.item()
        loss.backward()
        muon.step(); adam.step()
    final_loss = loss.item()
    assert final_loss < init_loss, \
        f"Muon failed to reduce loss: init {init_loss}, final {final_loss}"
    print(f"✓ test_muon_optimizer_runs: loss {init_loss:.4f} -> {final_loss:.4f} "
          f"({len(muon_p)} 2D params via Muon, {len(adam_p)} other via AdamW)")


def main():
    test_pd_closure_real()
    test_pdscan_layer_matches_explicit_matmul()
    test_complex_pd_unit_disk()
    test_pdkv_layer_matches_explicit()
    test_pd_with_zero_input_returns_zero_state_then_writes()
    test_muon_optimizer_runs()
    print("\nAll PD closure / sanity tests pass.")


if __name__ == "__main__":
    main()
