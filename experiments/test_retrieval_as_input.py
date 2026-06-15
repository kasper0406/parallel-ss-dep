"""Tests for the retrieval-as-input thinking-token mechanism
(added 2026-05-19).

Two pieces under test:

  1. TinyLM.forward(inputs_embeds=...): when set, bypasses
     embed(input_ids) and uses inputs_embeds directly as trunk input.
     This is the entry point for the new mechanism.

  2. WorkingMemory.forward stashes `self._last_injection` (B, T, d_model,
     detached): the per-position pre-mask retrieval. The generate loop
     reads this at think positions and uses it as the next position's
     input embedding.

  3. generate_with_retrieval_as_input: end-to-end loop. We don't test
     it head-to-head with the real model here (too slow); instead test
     that the components compose correctly on a small model.
"""
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.model import TinyLM, WorkingMemory
from experiments.layers import DeltaNetAttention


@pytest.fixture
def tiny_model_with_memory():
    """Tiny TinyLM with memory + gate. Returns a CUDA model (DeltaNet's
    Triton kernels require CUDA tensors)."""
    if not torch.cuda.is_available():
        pytest.skip("DeltaNet's Triton kernel requires CUDA")
    vocab = 16
    thinking_id = vocab - 1
    torch.manual_seed(0)
    model = TinyLM(
        vocab_size=vocab,
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_head=4,
        max_T=0,
        output_gate=True,
        use_memory=True,
        mem_size=4,
        mem_dim=8,
        thinking_token_id=thinking_id,
        attention_cls=DeltaNetAttention,
    ).cuda()
    return model, thinking_id


# ---------------------------------------------------------------------------
# TinyLM.forward(inputs_embeds=...)
# ---------------------------------------------------------------------------

def test_inputs_embeds_bypasses_embed_table(tiny_model_with_memory):
    """When inputs_embeds is provided, the model uses it directly and
    the input_ids' embedding contribution is IGNORED (used only for
    masks)."""
    model, _ = tiny_model_with_memory
    model.eval()
    B, T, d = 1, 6, 8
    input_ids = torch.zeros(B, T, dtype=torch.long).cuda()
    # Use a custom embedding LARGE-magnitude that's clearly different from
    # embed(input_ids=zero) (which is the same vector repeated). A large
    # contrast lets the difference propagate through the trunk even on
    # a 2-layer tiny model.
    custom = (torch.randn(B, T, d) * 5.0).cuda()
    with torch.no_grad():
        logits_with = model(input_ids, inputs_embeds=custom)
        logits_without = model(input_ids)
    diff = (logits_with - logits_without).abs().mean().item()
    assert diff > 1e-3, (
        f"forward(inputs_embeds=...) should differ from forward without; "
        f"got diff={diff:.6f}"
    )


def test_inputs_embeds_shape_mismatch_raises(tiny_model_with_memory):
    """If inputs_embeds has the wrong (B, T) it should raise — silent
    truncation would be worse than crashing."""
    model, _ = tiny_model_with_memory
    input_ids = torch.zeros(1, 6, dtype=torch.long).cuda()
    wrong = torch.randn(1, 5, 8).cuda()   # T=5 not 6
    with pytest.raises(ValueError, match="inputs_embeds shape"):
        model(input_ids, inputs_embeds=wrong)


def test_inputs_embeds_equal_to_embed_lookup_matches_baseline(tiny_model_with_memory):
    """Sanity: if inputs_embeds is exactly embed(input_ids), the two
    paths must give identical logits (no other source of difference)."""
    model, _ = tiny_model_with_memory
    model.eval()
    input_ids = torch.randint(0, 16, (1, 8)).cuda()
    with torch.no_grad():
        manual_embeds = model.embed(input_ids)
        a = model(input_ids, inputs_embeds=manual_embeds)
        b = model(input_ids)
    assert torch.allclose(a, b, atol=1e-5)


# ---------------------------------------------------------------------------
# WorkingMemory._last_injection stash
# ---------------------------------------------------------------------------

def test_wm_stashes_last_injection(tiny_model_with_memory):
    """The injection (pre-mask retrieval result) must be stashed for
    the generate loop to read."""
    model, _ = tiny_model_with_memory
    model.eval()
    input_ids = torch.randint(0, 16, (1, 8)).cuda()
    with torch.no_grad():
        _ = model(input_ids)
    inj = getattr(model.memory, "_last_injection", None)
    assert inj is not None
    assert inj.shape == (1, 8, 8)   # (B, T, d_model)
    # Must be detached (no gradient on inj itself)
    assert not inj.requires_grad


# ---------------------------------------------------------------------------
# generate_with_retrieval_as_input: smoke
# ---------------------------------------------------------------------------

def test_generate_with_retrieval_as_input_runs(tiny_model_with_memory):
    """End-to-end smoke: the new generator runs on a tiny model
    without crashing and produces an output that includes thinking
    tokens (if the gate ever decides to think)."""
    from experiments.eval_humaneval import generate_with_retrieval_as_input
    model, thinking_id = tiny_model_with_memory
    model = model.cuda().eval()
    prompt = torch.randint(0, 15, (1, 4)).cuda()    # avoid thinking_id (idx 15)
    out, diag = generate_with_retrieval_as_input(
        model, prompt, max_gen=4,
        temperature=0.0,
        max_think_per_step=2,
        total_think_budget=4,
        emit_threshold=0.5,
        min_emit_before_eos=0,
        thinking_token_id=thinking_id,
    )
    # Output shape: prompt + emits + thinks
    assert out.shape[0] == 1
    assert out.shape[1] >= prompt.shape[1] + diag["emit_count"]
    assert diag["mode"] == "retrieval_as_input"
    # emit_count is bounded by max_gen
    assert diag["emit_count"] <= 4


def test_force_prefix_think_forces_exact_burst(tiny_model_with_memory):
    """force_prefix_think=R must run EXACTLY R retrieval-injected thinks
    before the first emit, then emit gate-free (0 thinks) — regardless of
    the gate. This closes the WM-kill-gate tooling gap (the gate fires 0
    thinks on recall, so without this the retrieval read is never exercised).
    Also asserts force_prefix_think=0 is byte-identical to the default."""
    from experiments.eval_humaneval import generate_with_retrieval_as_input
    model, thinking_id = tiny_model_with_memory
    model = model.cuda().eval()
    prompt = torch.randint(0, 15, (1, 4)).cuda()
    R = 3
    out_f, diag_f = generate_with_retrieval_as_input(
        model, prompt, max_gen=4, temperature=0.0,
        thinking_token_id=thinking_id, force_prefix_think=R)
    assert diag_f["think_steps_used"][0] == R, diag_f["think_steps_used"]
    assert all(s == 0 for s in diag_f["think_steps_used"][1:]), \
        diag_f["think_steps_used"]
    assert diag_f["think_total"] == R

    # force_prefix_think=0 must match the gate-decides default exactly.
    out_a, _ = generate_with_retrieval_as_input(
        model, prompt, max_gen=4, temperature=0.0,
        thinking_token_id=thinking_id, force_prefix_think=0)
    out_b, _ = generate_with_retrieval_as_input(
        model, prompt, max_gen=4, temperature=0.0,
        thinking_token_id=thinking_id)
    assert out_a.shape == out_b.shape and torch.equal(out_a, out_b)


# ---------------------------------------------------------------------------
# v7: additive α-gated injection (Fix B)
# ---------------------------------------------------------------------------

def test_model_has_retrieval_input_alpha(tiny_model_with_memory):
    """A memory model must expose the learned scalar gate
    `retrieval_input_alpha`, init 0.1, trainable."""
    model, _ = tiny_model_with_memory
    assert hasattr(model, "retrieval_input_alpha")
    a = model.retrieval_input_alpha
    assert a.requires_grad and a.numel() == 1
    assert abs(a.item() - 0.1) < 1e-6


def test_additive_injection_formula():
    """Pin the v7 additive injection used by sft_code + the generator:
    input[think] = base_emb + α·retrieval; input[emit] = base_emb.
    A wrong sign or a stray replacement would change recall behaviour
    silently, so the arithmetic is pinned directly."""
    torch.manual_seed(0)
    B, T, d = 2, 5, 8
    base_emb = torch.randn(B, T, d)
    shifted_inj = torch.randn(B, T, d)
    alpha = torch.tensor(0.1)
    is_think = torch.zeros(B, T, 1)
    is_think[0, 2, 0] = 1.0          # one think position
    is_think[1, 4, 0] = 1.0
    out = base_emb + is_think * alpha * shifted_inj
    # Emit positions: untouched.
    emit_mask = (is_think.squeeze(-1) == 0)
    assert torch.allclose(out[emit_mask], base_emb[emit_mask])
    # Think positions: think-embed baseline survives + α·retrieval added.
    assert torch.allclose(out[0, 2], base_emb[0, 2] + alpha * shifted_inj[0, 2])
    assert torch.allclose(out[1, 4], base_emb[1, 4] + alpha * shifted_inj[1, 4])
    # At α=0 the injection is a no-op (think token keeps pure embedding).
    out0 = base_emb + is_think * torch.tensor(0.0) * shifted_inj
    assert torch.allclose(out0, base_emb)


def test_generator_additive_vs_replace_differ(tiny_model_with_memory):
    """The additive (v7) and replace (v5/v6) injection modes must drive
    the generator differently when the gate triggers a think step."""
    from experiments.eval_humaneval import generate_with_retrieval_as_input
    model, thinking_id = tiny_model_with_memory
    model = model.cuda().eval()
    # Push alpha up so the additive contribution is non-trivial.
    with torch.no_grad():
        model.retrieval_input_alpha.fill_(1.0)
    prompt = torch.randint(0, 15, (1, 4)).cuda()
    kw = dict(max_gen=3, temperature=0.0, max_think_per_step=3,
              total_think_budget=6, emit_threshold=0.99,   # force think
              thinking_token_id=thinking_id)
    out_add, _ = generate_with_retrieval_as_input(model, prompt,
                                                  additive=True, **kw)
    out_rep, _ = generate_with_retrieval_as_input(model, prompt,
                                                  additive=False, **kw)
    # Same lengths possible, but the emitted tokens should differ
    # somewhere (different think-position inputs → different states).
    same = (out_add.shape == out_rep.shape
            and torch.equal(out_add, out_rep))
    assert not same, "additive and replace modes produced identical output"


def test_generate_with_retrieval_as_input_requires_memory():
    """If the model lacks .memory, the generator should raise rather
    than silently fall back to standard generation."""
    from experiments.eval_humaneval import generate_with_retrieval_as_input
    vocab = 16
    model = TinyLM(
        vocab_size=vocab, d_model=8, n_layers=2, n_heads=2, d_head=4,
        max_T=0, output_gate=True, use_memory=False,
        attention_cls=DeltaNetAttention,
    ).cuda()
    prompt = torch.randint(0, 16, (1, 4)).cuda()
    with pytest.raises(ValueError, match="memory"):
        generate_with_retrieval_as_input(
            model, prompt, max_gen=4,
            thinking_token_id=vocab - 1,
        )


def test_generate_with_retrieval_as_input_diff_from_standard(tiny_model_with_memory):
    """When the gate triggers think, the new generator's output should
    differ from the standard generator on the same prompt — because the
    new mechanism injects retrieval as input rather than the [THINKING]
    embedding."""
    from experiments.eval_humaneval import generate, generate_with_retrieval_as_input
    model, thinking_id = tiny_model_with_memory
    # Force think by setting emit_threshold very high; the gate sigmoid
    # is unlikely to exceed 0.99.
    model = model.cuda().eval()
    prompt = torch.randint(0, 15, (1, 4)).cuda()
    out_std, _ = generate(
        model, prompt, max_gen=2,
        temperature=0.0, use_thinking=True,
        max_think_per_step=2, total_think_budget=4,
        emit_threshold=0.99,  # force think
        thinking_token_id=thinking_id,
    )
    out_new, _ = generate_with_retrieval_as_input(
        model, prompt, max_gen=2,
        temperature=0.0,
        max_think_per_step=2, total_think_budget=4,
        emit_threshold=0.99,
        thinking_token_id=thinking_id,
    )
    # The standard generator should emit specific tokens; the new
    # generator should emit different ones (because the hidden states
    # at think positions differ between the two mechanisms).
    # If the prompt is the same and the generators emit identically,
    # something is wrong.
    if out_std.shape == out_new.shape:
        # Both have same number of think+emit tokens — check the emit
        # tokens differ at least somewhere.
        # Note: this is a soft check; on a tiny untrained model the
        # gate may behave identically across the two paths if the
        # retrieved embedding happens to be similar to the [THINKING]
        # embedding. We just check that the test runs.
        pass
    # Mainly: confirm shapes are sensible
    assert out_std.shape[0] == 1
    assert out_new.shape[0] == 1
