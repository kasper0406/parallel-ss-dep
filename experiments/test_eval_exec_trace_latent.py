"""Tests for the exec-trace latent kill-test harness (`eval_exec_trace_latent.py`,
EXEC_TRACE_LATENT_PLAN.md phase N3, 2026-07-04).

CPU-only, no CUDA / FLA / HF required (SoftmaxAttention tiny model + a
dependency-free fake tokenizer, mirroring `test_latent_reasoning_checkpoint.py`'s
pattern). These tests pin the harness's MECHANICS, not any real ckpt's numbers:

  1. `load_rung` parses the exec-trace/state-track jsonl schema and applies the
     length-margin filter correctly.
  2. `grow_latent_thread` produces the right shapes and, with an untrained
     (zero-init proj, alpha=1) LatentFeedbackAdapter, feeds back the RAW hidden
     state unchanged (adapter identity at cold start).
  3. Teacher-forced and greedy-decode arms AGREE for every R on a fresh tiny
     model with single-token answers -- this is the harness's built-in
     self-consistency check (module docstring: causal invariance means the two
     independently-coded paths must produce identical exact-match verdicts).
  4. `structural_floor_check` reports max_abs_delta==0 for two loads of the
     IDENTICAL state_dict and nonzero after any trunk perturbation -- pinning
     the "adapter-only ckpt no-think path is byte-identical to the base" claim
     the harness is meant to verify.
"""
from __future__ import annotations

import json

import torch

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.eval_exec_trace_latent import (
    load_rung,
    grow_latent_thread,
    eval_record_teacher_forced,
    eval_record_greedy,
)

THINK_ID = 5  # != PAD/EOS (0)
EOS_ID = 1
VOCAB = 64
D_MODEL = 32


def _tiny_model(*, seed: int = 0, state_readonly: bool = False) -> TinyLM:
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=2, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=256,
        state_readonly_at_think=state_readonly,
        use_latent_feedback_adapter=True,
    )
    m.thinking_token_id = THINK_ID
    m.eval()
    return m


class _FakeTok:
    """Deterministic, dependency-free tokenizer stand-in (mirrors
    test_latent_reasoning_checkpoint.py's _FakeTok)."""

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (VOCAB - 4)) for c in text]


# ---------------------------------------------------------------------------
# 1. load_rung
# ---------------------------------------------------------------------------

def _write_rung_file(path, n_records=5, K=3):
    with open(path, "w") as f:
        for i in range(n_records):
            f.write(json.dumps({
                "task_id": f"t/{i}", "prompt": f"x{i} = {i}\nprint(x{i})",
                "answer": i % 10, "rung": K,
                "intermediates": [(i + j) % 10 for j in range(K)],
            }) + "\n")


def test_load_rung_parses_schema(tmp_path):
    prefix = str(tmp_path / "toy")
    _write_rung_file(f"{prefix}_n3.jsonl", n_records=4, K=3)
    tok = _FakeTok()
    recs = load_rung(prefix, 3, tok, max_len=512)
    assert len(recs) == 4
    ex = recs[0]
    assert ex["task_id"] == "t/0"
    assert ex["answer"] == 0
    assert len(ex["inter_ids"]) == 3
    assert isinstance(ex["comment_ids"], list) and len(ex["comment_ids"]) > 0
    assert isinstance(ex["sol_ids"], list) and len(ex["sol_ids"]) >= 1


def test_load_rung_missing_file_returns_empty(tmp_path):
    tok = _FakeTok()
    recs = load_rung(str(tmp_path / "nope"), 3, tok, max_len=512)
    assert recs == []


def test_load_rung_applies_length_margin_filter(tmp_path):
    prefix = str(tmp_path / "toy2")
    _write_rung_file(f"{prefix}_n3.jsonl", n_records=3, K=3)
    tok = _FakeTok()
    # A max_len smaller than any record's prompt length must drop everything.
    recs_tiny = load_rung(prefix, 3, tok, max_len=1)
    assert recs_tiny == []
    recs_full = load_rung(prefix, 3, tok, max_len=512)
    assert len(recs_full) == 3


# ---------------------------------------------------------------------------
# 2. grow_latent_thread shapes + cold-start adapter identity.
# ---------------------------------------------------------------------------

def test_grow_latent_thread_shapes():
    m = _tiny_model(seed=0)
    comment_ids = [2, 3, 4, 5, 6, 7]
    P = len(comment_ids)
    R = 4
    cur_ids, cur_emb = grow_latent_thread(m, comment_ids, R, THINK_ID, "cpu")
    assert cur_ids.shape == (1, P + R)
    assert cur_emb.shape == (1, P + R, D_MODEL)
    # The R appended positions all carry the thinking token id.
    assert cur_ids[0, P:].tolist() == [THINK_ID] * R
    # R=0 is a no-op (byte-identical prompt embed, no growth).
    cur_ids0, cur_emb0 = grow_latent_thread(m, comment_ids, 0, THINK_ID, "cpu")
    assert cur_ids0.shape == (1, P)
    assert torch.equal(cur_emb0, m.embed(torch.tensor([comment_ids])))


def test_grow_latent_thread_cold_start_adapter_is_identity():
    """LatentFeedbackAdapter is zero-init-proj / alpha=1 at cold start, so
    z_adapted == z: the fed-back embedding at each think slot must equal the
    RAW hidden state from the previous forward, unchanged by the adapter."""
    m = _tiny_model(seed=1)
    comment_ids = [2, 3, 4, 5]
    R = 2
    cur_ids, cur_emb = grow_latent_thread(m, comment_ids, R, THINK_ID, "cpu")
    # Recompute what the raw (pre-adapter) hidden feedback should have been at
    # step 1 (forward over the prompt alone) and confirm apply_latent_feedback_
    # adapter is exactly identity on it.
    base_ids = torch.tensor([comment_ids])
    h = m(base_ids, skip_lm_head=True)
    if isinstance(h, tuple):
        h = h[0]
    z_raw = h[:, -1:, :]
    z_adapted = m.apply_latent_feedback_adapter(z_raw)
    assert torch.allclose(z_raw, z_adapted, atol=1e-6)
    assert torch.allclose(cur_emb[:, len(comment_ids):len(comment_ids) + 1, :],
                         z_raw, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Teacher-forced vs greedy-decode agreement (single-token answers).
# ---------------------------------------------------------------------------

def test_teacher_forced_and_greedy_agree_across_R():
    """For single-token answers, argmax-teacher-forced and greedy-decode-exact
    must agree at every R (causal invariance: appending the gold answer span
    cannot change the logits at positions <= the last think slot). This is the
    harness's own mechanics self-check (see eval_rung's
    --assert_tf_greedy_agree)."""
    m = _tiny_model(seed=2)
    comment_ids = [2, 3, 4, 5, 6, 7, 8]
    sol_ids = [9]  # single-token answer
    inter_ids = [3, 4, 5, 6]
    ex = {"task_id": "x", "comment_ids": comment_ids, "sol_ids": sol_ids,
         "answer": 9, "inter_ids": inter_ids}
    for R in (0, 1, 2, 4):
        tf = eval_record_teacher_forced(m, ex, R, THINK_ID, EOS_ID, "cpu")
        gr = eval_record_greedy(m, ex, R, THINK_ID, EOS_ID, "cpu")
        assert tf["tf_exact"] == gr["greedy_exact"], (
            R, tf["tf_pred"], gr["greedy_pred"])
        assert len(tf["perhop"]) == R


def test_perhop_targets_beyond_intermediates_are_none():
    """If R exceeds len(intermediates), the extra hop positions have no
    ground truth and must report None (not silently wrong/right)."""
    m = _tiny_model(seed=3)
    comment_ids = [2, 3, 4, 5]
    sol_ids = [9]
    inter_ids = [3, 4]  # only 2 intermediates
    ex = {"task_id": "y", "comment_ids": comment_ids, "sol_ids": sol_ids,
         "answer": 9, "inter_ids": inter_ids}
    tf = eval_record_teacher_forced(m, ex, R=4, thinking_id=THINK_ID,
                                    eos_id=EOS_ID, device="cpu")
    assert len(tf["perhop"]) == 4
    assert tf["perhop"][:2] == [int(v) for v in tf["perhop"][:2]]  # 0/1 ints
    assert tf["perhop"][2] is None and tf["perhop"][3] is None


# ---------------------------------------------------------------------------
# 4. structural_floor_check.
# ---------------------------------------------------------------------------

def test_structural_floor_check_zero_delta_for_identical_weights(tmp_path):
    from experiments.eval_exec_trace_latent import structural_floor_check

    m = _tiny_model(seed=4)
    cfg = dict(arch="softmax", n_layers=2, vocab_size=VOCAB,
              tokenizer="HuggingFaceTB/SmolLM2-135M")
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    payload = {"state_dict": m.state_dict(), "config": cfg, "step": 0}
    torch.save(payload, ckpt_a)
    torch.save(payload, ckpt_b)

    def _fake_load(ckpt_path, device, bf16=True, tf32=True):
        mm = _tiny_model(seed=4)
        mm.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])
        mm.eval()
        return mm, cfg, THINK_ID, _FakeTok(), EOS_ID

    import experiments.eval_exec_trace_latent as mod
    orig = mod.load_eval_model
    mod.load_eval_model = _fake_load
    try:
        ids_lists = [[2, 3, 4, 5], [2, 3, 4, 5, 6, 7]]
        res = structural_floor_check(str(ckpt_a), str(ckpt_b), ids_lists, "cpu")
    finally:
        mod.load_eval_model = orig

    assert res["skipped"] is False
    assert res["max_abs_delta"] == 0.0
    assert res["pass"] is True


def test_structural_floor_check_nonzero_delta_after_perturbation(tmp_path):
    from experiments.eval_exec_trace_latent import structural_floor_check

    m_a = _tiny_model(seed=5)
    m_b = _tiny_model(seed=5)
    with torch.no_grad():
        # Perturb a trunk weight in b -- the no-think path must now diverge.
        list(m_b.blocks[0].parameters())[0].add_(1.0)
    cfg = dict(arch="softmax", n_layers=2, vocab_size=VOCAB,
              tokenizer="HuggingFaceTB/SmolLM2-135M")
    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    torch.save({"state_dict": m_a.state_dict(), "config": cfg, "step": 0}, ckpt_a)
    torch.save({"state_dict": m_b.state_dict(), "config": cfg, "step": 0}, ckpt_b)

    loaded = {}

    def _fake_load(ckpt_path, device, bf16=True, tf32=True):
        seed = 5
        mm = _tiny_model(seed=seed)
        mm.load_state_dict(torch.load(ckpt_path, weights_only=False)["state_dict"])
        mm.eval()
        return mm, cfg, THINK_ID, _FakeTok(), EOS_ID

    import experiments.eval_exec_trace_latent as mod
    orig = mod.load_eval_model
    mod.load_eval_model = _fake_load
    try:
        ids_lists = [[2, 3, 4, 5]]
        res = structural_floor_check(str(ckpt_a), str(ckpt_b), ids_lists, "cpu")
    finally:
        mod.load_eval_model = orig

    assert res["skipped"] is False
    assert res["max_abs_delta"] > 0.0
    assert res["pass"] is False


def test_structural_floor_check_skips_missing_base(tmp_path):
    from experiments.eval_exec_trace_latent import structural_floor_check
    res = structural_floor_check(str(tmp_path / "x.pt"),
                                 str(tmp_path / "does_not_exist.pt"),
                                 [[2, 3]], "cpu")
    assert res["skipped"] is True
