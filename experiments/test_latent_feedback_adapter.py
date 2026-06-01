"""CPU-only tests for the LATENT-thinking input adapter + selective co-train
sampling (2026-06-01).

Two fixes that make latent (Coconut-style) thinking trainable on the real code
model:

  1. `LatentFeedbackAdapter` (RMSNorm -> zero-init Linear, identity-residual +
     learnable alpha) maps the fed-back out_norm hidden into the input-embedding
     manifold before it drives the next latent think-step. ZERO-init -> identity
     -> a fresh / untrained adapter is byte-identical to the no-adapter latent
     path. When its weights are nonzero it changes the latent Delta-logp path.

  2. `latent_cotrain_loss(..., selective=True)` samples co-train positions
     WEIGHTED toward where thinking should help (high gate-think / high no-think
     entropy) instead of uniform-random.

A tiny TinyLM with CPU-friendly SoftmaxAttention exercises every code path; no
CUDA / FLA / HF required.

Run (CPU-only, isolate from the CUDA suite):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \
      experiments/test_latent_feedback_adapter.py -v
"""
from __future__ import annotations

import copy

import torch

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM, LatentFeedbackAdapter
from experiments import thinking as T

THINK_ID = 5  # != PAD_ID (0)
PAD_ID = 0
EOS_ID = 1
VOCAB = 64
D_MODEL = 32


def _tiny_model(*, use_latent_feedback_adapter: bool, seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=2, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=256,
        output_gate=True, state_readonly_at_think=True,
        use_latent_feedback_adapter=use_latent_feedback_adapter,
    )
    m.thinking_token_id = THINK_ID
    return m.eval()


# ---------------------------------------------------------------------------
# 1. Identity at init: adapter forward + latent path byte-identical.
# ---------------------------------------------------------------------------

def test_adapter_module_identity_at_init():
    """A fresh LatentFeedbackAdapter is the identity (zero-init proj)."""
    torch.manual_seed(0)
    adapter = LatentFeedbackAdapter(D_MODEL)
    z = torch.randn(4, 1, D_MODEL)
    out = adapter(z)
    assert torch.equal(out, z), "untrained adapter must be the identity"


def test_apply_helper_noop_without_flag():
    """apply_latent_feedback_adapter is a pass-through when the flag is off."""
    m = _tiny_model(use_latent_feedback_adapter=False)
    assert not hasattr(m, "latent_feedback_adapter")
    z = torch.randn(3, 1, D_MODEL)
    assert m.apply_latent_feedback_adapter(z) is z


def test_latent_path_byte_identical_at_init():
    """With the adapter built but untrained (identity), the latent Delta-logp
    primitive matches the no-adapter model exactly."""
    m_off = _tiny_model(use_latent_feedback_adapter=False, seed=0)
    m_on = _tiny_model(use_latent_feedback_adapter=True, seed=0)
    # Copy the shared (non-adapter) weights so the two models are identical
    # apart from the (identity) adapter.
    shared = {k: v for k, v in m_off.state_dict().items()}
    m_on.load_state_dict(shared, strict=False)

    prefixes = torch.randint(2, VOCAB - 1, (3, 12))
    true_next = torch.randint(2, VOCAB - 1, (3,))
    lp_off = T.latent_think_logp(m_off, prefixes, true_next, R=3,
                                 thinking_token_id=THINK_ID, pad_id=PAD_ID)
    lp_on = T.latent_think_logp(m_on, prefixes, true_next, R=3,
                                thinking_token_id=THINK_ID, pad_id=PAD_ID)
    assert torch.allclose(lp_off, lp_on, atol=1e-6), \
        "identity-init adapter must not change the latent path"


# ---------------------------------------------------------------------------
# 2. Nonzero adapter weights change the latent Delta-logp path.
# ---------------------------------------------------------------------------

def test_nonzero_adapter_changes_latent_path():
    m_on = _tiny_model(use_latent_feedback_adapter=True, seed=0)
    prefixes = torch.randint(2, VOCAB - 1, (3, 12))
    true_next = torch.randint(2, VOCAB - 1, (3,))
    lp_identity = T.latent_think_logp(m_on, prefixes, true_next, R=3,
                                      thinking_token_id=THINK_ID, pad_id=PAD_ID)
    # Perturb the adapter off identity (proj weight + alpha both nonzero).
    with torch.no_grad():
        m_on.latent_feedback_adapter.proj.weight.normal_(0, 0.5)
        m_on.latent_feedback_adapter.alpha.fill_(1.0)
    lp_perturbed = T.latent_think_logp(m_on, prefixes, true_next, R=3,
                                       thinking_token_id=THINK_ID, pad_id=PAD_ID)
    assert not torch.allclose(lp_identity, lp_perturbed, atol=1e-4), \
        "a nonzero adapter must change the latent Delta-logp path"


def test_alpha_zero_keeps_identity_even_with_nonzero_proj():
    """alpha gates the whole adapter contribution: alpha=0 -> identity even if
    proj.weight is nonzero."""
    m_on = _tiny_model(use_latent_feedback_adapter=True, seed=0)
    z = torch.randn(4, 1, D_MODEL)
    with torch.no_grad():
        m_on.latent_feedback_adapter.proj.weight.normal_(0, 0.5)
        m_on.latent_feedback_adapter.alpha.zero_()
    out = m_on.apply_latent_feedback_adapter(z)
    assert torch.allclose(out, z, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. State-dict round-trip preserves the adapter.
# ---------------------------------------------------------------------------

def test_state_dict_round_trips_adapter():
    m = _tiny_model(use_latent_feedback_adapter=True, seed=0)
    with torch.no_grad():
        m.latent_feedback_adapter.proj.weight.normal_(0, 0.3)
        m.latent_feedback_adapter.proj.bias.normal_(0, 0.3)
        m.latent_feedback_adapter.alpha.fill_(0.7)
    sd = copy.deepcopy(m.state_dict())
    keys = [k for k in sd if k.startswith("latent_feedback_adapter.")]
    assert set(keys) == {
        "latent_feedback_adapter.norm.weight",
        "latent_feedback_adapter.proj.weight",
        "latent_feedback_adapter.proj.bias",
        "latent_feedback_adapter.alpha",
    }
    m2 = _tiny_model(use_latent_feedback_adapter=True, seed=1)
    m2.load_state_dict(sd, strict=False)
    for k in keys:
        assert torch.equal(m2.state_dict()[k], sd[k])


# ---------------------------------------------------------------------------
# 4. Selective co-train sampling biases toward high-(1-sigma)/high-entropy.
# ---------------------------------------------------------------------------

class _FakeGateModel:
    """Minimal model exposing the contract `_selective_position_weights` needs:
    forward(return_gate=True) -> (logits,) and a `_last_gate` (sigma=P(emit))
    side-effect. Lets us assert the weighting math without a real trunk."""

    def __init__(self, logits, gate):
        self._logits = logits          # (B, T, V)
        self._gate = gate              # (B, T) sigma in [0,1]
        self._last_gate = None
        self.memory = None

    def __call__(self, input_ids, *, return_hidden=False, return_gate=False,
                 **kw):
        self._last_gate = self._gate
        return (self._logits,)


def test_selective_weights_prefer_low_sigma_and_high_entropy():
    B, Tt, V = 1, 6, 8
    # Build logits: half the positions PEAKED (low entropy), half UNIFORM
    # (high entropy).
    logits = torch.zeros(B, Tt, V)
    peaked = torch.full((V,), -10.0)
    peaked[0] = 10.0
    for t in range(Tt):
        logits[0, t] = peaked if t < 3 else torch.zeros(V)   # uniform = zeros
    # Gate sigma: low (think-wanted) on the high-entropy half, high on the rest.
    gate = torch.tensor([[0.9, 0.9, 0.9, 0.1, 0.1, 0.1]])
    fake = _FakeGateModel(logits, gate)
    input_ids = torch.zeros(B, Tt, dtype=torch.long)
    bsel = torch.zeros(Tt, dtype=torch.long)
    tsel = torch.arange(Tt)
    w = T._selective_position_weights(fake, input_ids, bsel, tsel)
    # Positions 3..5 (low sigma AND high entropy) must outweigh 0..2.
    assert w[3:].mean() > w[:3].mean(), \
        "selective weights must favour low-sigma / high-entropy positions"
    assert (w > 0).all(), "every weight is floored positive"


def test_selective_sampling_draws_biased_positions():
    """multinomial over the weights concentrates draws on the useful half."""
    B, Tt, V = 1, 6, 8
    logits = torch.zeros(B, Tt, V)
    peaked = torch.full((V,), -10.0)
    peaked[0] = 10.0
    for t in range(Tt):
        logits[0, t] = peaked if t < 3 else torch.zeros(V)
    gate = torch.tensor([[0.95, 0.95, 0.95, 0.05, 0.05, 0.05]])
    fake = _FakeGateModel(logits, gate)
    input_ids = torch.zeros(B, Tt, dtype=torch.long)
    bsel = torch.zeros(Tt, dtype=torch.long)
    tsel = torch.arange(Tt)
    w = T._selective_position_weights(fake, input_ids, bsel, tsel)
    g = torch.Generator().manual_seed(0)
    draws = torch.multinomial(w, 3, replacement=True, generator=g)
    # The useful half is positions 3,4,5 -> expect a majority of draws there
    # over many samples.
    g2 = torch.Generator().manual_seed(1)
    many = torch.multinomial(w, 2000, replacement=True, generator=g2)
    frac_useful = float((many >= 3).float().mean())
    assert frac_useful > 0.6, \
        f"sampling should concentrate on useful positions, got {frac_useful}"


# ---------------------------------------------------------------------------
# 5. End-to-end: latent_cotrain_loss runs with selective=True on a tiny model.
# ---------------------------------------------------------------------------

def test_latent_cotrain_loss_selective_runs_and_restores_stash():
    m = _tiny_model(use_latent_feedback_adapter=True, seed=0)
    m.train()
    B, Tt = 2, 40
    input_ids = torch.randint(2, VOCAB - 1, (B, Tt))
    targets = torch.randint(2, VOCAB - 1, (B, Tt))
    # Plant a sentinel in the gate stash; the loss must restore it.
    sentinel = torch.full((B, Tt), 0.123)
    m._last_gate = sentinel
    gen = torch.Generator().manual_seed(0)
    out = T.latent_cotrain_loss(
        m, input_ids, targets, R=2, thinking_token_id=THINK_ID,
        max_positions=8, max_prefix_len=24, pad_id=PAD_ID, eos_id=EOS_ID,
        selective=True, generator=gen)
    assert out is not None, "should sample (plenty of valid positions)"
    loss, mean_delta, n = out
    assert n == 8
    assert torch.is_tensor(loss) and loss.requires_grad
    assert isinstance(mean_delta, float)
    # The per-step gate stash is restored to the sentinel (transparent loss).
    assert m._last_gate is sentinel, "loss must restore _last_gate stash"
