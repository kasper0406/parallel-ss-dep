"""Regression test for latent-space thinking (2026-05-28).

Locks in the validated result: a depth curriculum with final-answer-ONLY
supervision teaches a small DeltaNet to perform iterative latent computation;
latent thinking at R=K vastly beats no-think, and the discrete-token variant
fails (high bandwidth is essential).
"""
import argparse

import pytest
import torch

from experiments import latent_think as lt


def _args(**kw):
    a = argparse.Namespace(
        N=8, K=3, R_train=3, eval_R="0,3", extrapolate_R=0,
        train_mode="latent", deep_supervision=False, k_curriculum=True,
        state_write=False, steps=1500, batch=256, lr=2e-3,
        d_model=96, n_layers=2, n_heads=3, d_head=32,
        log_every=10000, seed=0, device="cuda", save="", smoke=False,
        task="chase", gate_weight=1.0,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_task_shapes_and_chain():
    """Pointer-chase batch returns aligned chain targets (CPU, no model)."""
    ids, ans, chain, vocab = lt.make_pointer_chase_batch(
        4, N=6, K=3, device="cpu",
        generator=torch.Generator().manual_seed(0))
    assert ids.shape == (4, 2 * 6 + 2)
    assert chain.shape == (4, 3)
    assert vocab == 6 + 3
    # answer is the last hop
    assert torch.equal(ans, chain[:, -1])
    # every chain entry is a valid node id
    assert (chain < 6).all() and (chain >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/FLA")
def test_curriculum_makes_latent_thinking_load_bearing():
    _, res = lt.train(_args())
    none = res["none"]
    latent = res["latent_R3"]
    # Latent thinking at R=K solves the chain; no-think is at the floor.
    assert latent > 0.80, f"latent R=K should solve the task, got {latent}"
    assert none < 0.50, f"no-think should be at floor, got {none}"
    assert latent - none > 0.40, "latent thinking must be load-bearing"


def test_fixedpoint_task_has_absorbing_answer():
    """Fixed-point chase: answer is the absorbing node, reached in L hops."""
    ids, ans, Ls, vocab = lt.make_fixedpoint_chase_batch(
        8, N=8, L_max=4, device="cpu",
        generator=torch.Generator().manual_seed(1))
    assert ids.shape == (8, 2 * 8 + 2)
    assert (Ls >= 1).all() and (Ls <= 4).all()
    assert (ans < 8).all() and (ans >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/FLA")
def test_adaptive_halting_learns_when_to_stop():
    a = _args(task="fixedpoint", N=8, K=4, gate_weight=1.0, steps=2500,
              d_model=96, n_heads=3)
    _, res = lt.train_fixedpoint(a)
    assert res["answer_acc"] > 0.85, f"should emit the absorbing node: {res}"
    assert res["halt_exact"] > 0.70, f"gate should halt near L: {res}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/FLA")
def test_discrete_token_feedback_fails():
    """Constant-[THINK]-embedding feedback (token mode) cannot chain —
    proves the high-bandwidth latent CONTENT is what carries the computation."""
    model, _ = lt.train(_args())
    # token mode at R=K should be far below latent at R=K.
    res = lt.evaluate(model, N=8, K=3, thinking_id=9, R_list=[3],
                      device="cuda", n_eval=1024, batch=512)
    assert res["latent_R3"] - res["token_R3"] > 0.40, \
        f"latent must beat token-mode by a wide margin: {res}"
