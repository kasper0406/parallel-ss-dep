"""Tests for Stage-B (Coconut text->latent replacement) trace-curriculum mode
in latent_reasoning_cotrain.py + the eval twin eval_exec_trace_latent_trace.py
(EXEC_TRACE_LATENT_PLAN.md "Staged addendum", 2026-07-13).

The mode gradually replaces the first `s` TEXT trace steps of a Stage-A record
with `s` LATENT (hidden-feedback) slots; it rides the existing
`_answer_span_latent_loss(_batched)` UNCHANGED with R=s_eff. These pins cover:

  A. loader byte-format equality (vs the real Stage-A text data), suffix
     variants for s=0 / s=K / middle s, single-token inter contract, length
     filter.
  B. R=0 == an independent plain teacher-forced CE over the solution span.
  C. s=K path (fully-latent trace): solution == final line only, per-hop over
     all K slots, finite loss + adapter gradient.
  D. middle-s convention: the supervised text span starts at absolute position
     P+s (start=P+s-1 in the shifted frame), and per-hop slots don't overlap it.
  E. curriculum: s_max(0)=0, s_max(~55%)=8, consolidation uniform, s_eff=min(s,K).
  F. default-off: trace_mode absent is byte-identical (ordinary path).
  G. eval reuse + R=0-injection reduces to the direct arm's machinery.

CPU sections use SoftmaxAttention (causal, bit-exact so a clean model's
think-slot logits are unaffected by the trailing answer span, and
clean_latent_thread is a no-op). One CUDA-gated end-to-end check of the batched
trace step mirrors test_latent_reasoning_batched.py.

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv/bin/python -m \
    pytest experiments/test_stageB_trace_curriculum.py -v
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch
import torch.nn.functional as F

from experiments.layers import SoftmaxAttention
from experiments.model import TinyLM
from experiments.latent_reasoning_cotrain import (
    LatentReasoningCotrain,
    _answer_span_latent_loss,
    _load_rung_trace,
    _render_trace_text,
    _trace_render_parts,
    _trace_stage_smax,
)

THINK_ID = 5   # != PAD_ID (0)
PAD_ID = 0
EOS_ID = 1
VOCAB = 64
D_MODEL = 32


# --------------------------------------------------------------------------- #
# Fixtures / helpers.
# --------------------------------------------------------------------------- #

class _FakeTok:
    """Each char -> one token, so a single-digit str is ONE token and a
    multi-digit str is multi-token (the single-token-inter contract probe)."""

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (VOCAB - 4)) for c in text]


def _tiny_cpu_model(*, seed: int = 0) -> TinyLM:
    torch.manual_seed(seed)
    m = TinyLM(
        vocab_size=VOCAB, d_model=D_MODEL, n_layers=2, n_heads=2, d_head=16,
        attention_cls=SoftmaxAttention, max_T=256,
        output_gate=True, state_readonly_at_think=True,
        use_latent_feedback_adapter=True,
    )
    m.thinking_token_id = THINK_ID
    m.train()
    return m


def _trace_record(i: int, K: int) -> dict:
    inter = [(i + j) % 10 for j in range(K)]      # single-digit -> single token
    return {"task_id": f"t/{i}", "prompt": f"prog {i}\nx = {i}\nfinal of x?",
            "answer": inter[-1], "intermediates": inter, "rung": K,
            "tracked_var": "x"}


def _write_trace_rung(path, K, n_records=6, start=0):
    with open(path, "w") as f:
        for i in range(start, start + n_records):
            f.write(json.dumps(_trace_record(i, K)) + "\n")


def _grow_thread_cpu(model, comment_ids, R):
    """CPU growing-thread mirror of the loss helper (adapter + hidden index),
    returning (cur_ids, cur_emb) of length P+R — the position-explicit
    reference used by the middle-s / s=K convention pins."""
    base = torch.tensor([comment_ids], dtype=torch.long)
    cur_ids, cur_emb = base, model.embed(base)
    think = torch.full((1, 1), THINK_ID, dtype=torch.long)
    for _ in range(R):
        out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        h = out[1]
        z = model.apply_latent_feedback_adapter(h[:, -1:, :].to(cur_emb.dtype))
        cur_ids = torch.cat([cur_ids, think], dim=1)
        cur_emb = torch.cat([cur_emb, z.to(cur_emb.dtype)], dim=1)
    return cur_ids, cur_emb


# =========================================================================== #
# A. Loader: byte-format equality, suffix variants, contract, length filter.
# =========================================================================== #

_REAL_TEXT = pathlib.Path("data/exec_trace_text_train.jsonl")
_REAL_SCHEMA = sorted(pathlib.Path("data").glob("exec_trace_train_n*.jsonl"))


@pytest.mark.skipif(not (_REAL_TEXT.exists() and _REAL_SCHEMA),
                    reason="real exec-trace data files not present")
def test_render_matches_stageA_text_byte_for_byte():
    """`_render_trace_text` reproduces the flattened Stage-A `text` field
    byte-for-byte (the load-bearing train/eval-format invariant)."""
    schema = {}
    for path in _REAL_SCHEMA:
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            schema[r["prompt"].rstrip()] = r
    checked = matched = 0
    for line in _REAL_TEXT.open():
        line = line.strip()
        if not line:
            continue
        txt = json.loads(line)["text"]
        idx = txt.find("\n# trace:\n")
        if idx < 0:
            continue
        r = schema.get(txt[:idx])
        if r is None:
            continue
        checked += 1
        assert _render_trace_text(r) == txt
        matched += 1
        if checked >= 500:
            break
    assert matched >= 50, f"only matched {matched} records — data/format drift?"


def test_render_parts_exact_format():
    r = _trace_record(3, 2)              # inter [3,4], answer 4, var x
    prompt, step_lines, final_line = _trace_render_parts(r)
    assert prompt == "prog 3\nx = 3\nfinal of x?\n# trace:\n"
    assert step_lines == ["# step 1: x = 3\n", "# step 2: x = 4\n"]
    assert final_line == "# final: 4\n"
    assert _render_trace_text(r) == prompt + "".join(step_lines) + final_line


def test_loader_suffix_variants_s0_sK_middle(tmp_path):
    prefix = str(tmp_path / "tr")
    K = 4
    _write_trace_rung(f"{prefix}_n{K}.jsonl", K, n_records=5)
    tok = _FakeTok()
    recs = _load_rung_trace(prefix, K, tok, max_len=256)
    assert recs
    r0 = _trace_record(0, K)
    _, step_lines, final_line = _trace_render_parts(r0)
    rec = recs[0]
    assert rec["K"] == K
    assert set(rec["sol_ids_by_s"].keys()) == set(range(0, K + 1))
    # s=0 : the full text trace (all step lines + final).
    assert rec["sol_ids_by_s"][0] == tok.encode(
        "".join(step_lines[0:]) + final_line)
    # s=K : final line ALONE (fully-latent trace).
    assert rec["sol_ids_by_s"][K] == tok.encode(final_line)
    # middle s : step lines s.. + final.
    for s in (1, 2, 3):
        assert rec["sol_ids_by_s"][s] == tok.encode(
            "".join(step_lines[s:]) + final_line)
    # suffix length strictly decreases as more steps go latent.
    lens = [len(rec["sol_ids_by_s"][s]) for s in range(0, K + 1)]
    assert all(lens[s] > lens[s + 1] for s in range(0, K))


def test_loader_inter_ids_single_token(tmp_path):
    prefix = str(tmp_path / "tr")
    _write_trace_rung(f"{prefix}_n3.jsonl", 3, n_records=4)
    recs = _load_rung_trace(prefix, 3, _FakeTok(), max_len=256)
    for rec in recs:
        assert len(rec["inter_ids"]) == 3
        assert all(isinstance(t, int) for t in rec["inter_ids"])


def test_loader_errors_below_50pct_single_token(tmp_path):
    prefix = str(tmp_path / "bad")
    # 4/6 have a multi-token (2-digit) intermediate -> only 2/6 = 33% usable.
    with open(f"{prefix}_n2.jsonl", "w") as f:
        for i in range(6):
            inter = [42, 3] if i < 4 else [3, 4]     # 42 -> "42" -> 2 tokens
            f.write(json.dumps({"prompt": f"p{i}", "answer": inter[-1],
                                "intermediates": inter, "rung": 2,
                                "tracked_var": "x"}) + "\n")
    with pytest.raises(ValueError, match="single-token"):
        _load_rung_trace(prefix, 2, _FakeTok(), max_len=256,
                         require_single_token_inter=True)


def test_loader_length_filter(tmp_path):
    prefix = str(tmp_path / "tr")
    _write_trace_rung(f"{prefix}_n4.jsonl", 4, n_records=5)
    tok = _FakeTok()
    full = _load_rung_trace(prefix, 4, tok, max_len=256)
    assert len(full) == 5
    # A max_len below (prompt + full trace + K + 2) drops everything.
    rec = full[0]
    need = len(rec["prompt_ids"]) + len(rec["sol_ids_by_s"][0]) + 4 + 2
    assert _load_rung_trace(prefix, 4, tok, max_len=need - 1) == []
    assert len(_load_rung_trace(prefix, 4, tok, max_len=need)) == 5


# =========================================================================== #
# B. R=0 == independent plain teacher-forced CE over the solution span.
# =========================================================================== #

def _independent_span_ce(model, comment_ids, sol_ids, eos_id):
    """Position-explicit reference: logit at absolute position P-1+i predicts
    target[i] for target = sol + [eos]. No masking machinery reused."""
    full = torch.tensor([list(comment_ids) + list(sol_ids) + [eos_id]],
                        dtype=torch.long)
    out = model(full, inputs_embeds=model.embed(full))
    logits = out[0] if isinstance(out, tuple) else out
    P = len(comment_ids)
    target = list(sol_ids) + [eos_id]
    ces = [F.cross_entropy(logits[:, P - 1 + i, :],
                           torch.tensor([t], dtype=torch.long))
           for i, t in enumerate(target)]
    return torch.stack(ces).mean()


def test_R0_equals_independent_plain_ce():
    m = _tiny_cpu_model(seed=0)
    comment = [7, 8, 9, 10, 11, 12]
    sol = [13, 14, 15]
    with torch.no_grad():
        ref = _independent_span_ce(m, comment, sol, EOS_ID)
        _total, ans, hop = _answer_span_latent_loss(
            m, comment, sol, EOS_ID, 0, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=[], perhop_weight=1.0,
            return_components=True)
    assert torch.allclose(ans, ref, atol=1e-5, rtol=1e-4), (float(ans),
                                                            float(ref))
    assert float(hop) == 0.0            # no latent slots -> no per-hop term


def test_R0_no_latent_path_breaks_grads_flow_to_trunk():
    """R=0 (pure text stage) — latent loop runs zero times; loss finite and
    trunk gradient flows (the adapter is simply never invoked at R=0)."""
    m = _tiny_cpu_model(seed=1)
    comment, sol = [7, 8, 9, 10], [11, 12]
    for p in m.parameters():
        p.grad = None
    loss = _answer_span_latent_loss(
        m, comment, sol, EOS_ID, 0, THINK_ID, "cpu", checkpoint_latent=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.embed.weight.grad is not None
    assert m.embed.weight.grad.abs().sum() > 0


# =========================================================================== #
# C. s=K path: solution == final line, per-hop over all K slots.
# =========================================================================== #

def test_sK_solution_is_final_line_only(tmp_path):
    prefix = str(tmp_path / "tr")
    K = 3
    _write_trace_rung(f"{prefix}_n{K}.jsonl", K, n_records=4)
    tok = _FakeTok()
    recs = _load_rung_trace(prefix, K, tok, max_len=256)
    _, _steps, final_line = _trace_render_parts(_trace_record(0, K))
    assert recs[0]["sol_ids_by_s"][K] == tok.encode(final_line)


def test_sK_perhop_over_all_K_slots_and_adapter_grad():
    m = _tiny_cpu_model(seed=2)
    comment = [7, 8, 9, 10, 11]
    K = 4
    sol = [13, 14]                     # the "# final: N" tokens (opaque here)
    inter = [3, 5, 7, 9]               # one per latent slot
    for p in m.parameters():
        p.grad = None
    total, ans, hop = _answer_span_latent_loss(
        m, comment, sol, EOS_ID, K, THINK_ID, "cpu", checkpoint_latent=False,
        inter_ids=inter, perhop_weight=1.0, return_components=True)
    assert torch.isfinite(total) and float(hop.detach()) > 0.0
    total.backward()
    g = m.latent_feedback_adapter.proj.weight.grad
    assert g is not None and g.abs().sum() > 0
    # per-hop covered all K slots: matches a position-explicit mean over K slots.
    with torch.no_grad():
        cur_ids, cur_emb = _grow_thread_cpu(m, comment, K)
        out = m(cur_ids, inputs_embeds=cur_emb)
        logits = out[0] if isinstance(out, tuple) else out
        P = len(comment)
        ref = torch.stack([
            F.cross_entropy(logits[:, P + j, :],
                            torch.tensor([inter[j]], dtype=torch.long))
            for j in range(K)]).mean()
    assert torch.allclose(hop, ref, atol=1e-5, rtol=1e-4)


# =========================================================================== #
# D. middle-s convention: text span starts at P+s; per-hop slots don't overlap.
# =========================================================================== #

def test_middle_s_text_span_starts_at_P_plus_s():
    m = _tiny_cpu_model(seed=3)
    comment = [7, 8, 9, 10, 11, 12]
    P = len(comment)
    s = 2
    sol = [20, 21, 22]                 # remaining text trace + final (opaque)
    inter = [4, 6]                     # s intermediates for the s latent slots
    with torch.no_grad():
        _total, ans, _hop = _answer_span_latent_loss(
            m, comment, sol, EOS_ID, s, THINK_ID, "cpu",
            checkpoint_latent=False, inter_ids=inter, perhop_weight=1.0,
            return_components=True)
        # Independent reference: grow s slots, then the answer span is decoded
        # from absolute position (P+s-1) predicting the FIRST text token at
        # absolute (P+s). Written out by hand => pins start = P+s-1.
        cur_ids, cur_emb = _grow_thread_cpu(m, comment, s)
        sol_t = torch.tensor([list(sol) + [EOS_ID]], dtype=torch.long)
        full_ids = torch.cat([cur_ids, sol_t], dim=1)
        full_emb = torch.cat([cur_emb, m.embed(sol_t)], dim=1)
        out = m(full_ids, inputs_embeds=full_emb)
        logits = out[0] if isinstance(out, tuple) else out
        target = list(sol) + [EOS_ID]
        ref = torch.stack([
            F.cross_entropy(logits[:, (P + s - 1) + i, :],
                            torch.tensor([t], dtype=torch.long))
            for i, t in enumerate(target)]).mean()
    assert torch.allclose(ans, ref, atol=1e-5, rtol=1e-4), (float(ans),
                                                           float(ref))
    # Non-overlap: per-hop reads the s latent slots at absolute [P, P+s); the
    # text CE supervises predicted absolute positions [P+s, ...). Disjoint.
    perhop_positions = set(range(P, P + s))
    text_positions = set(range(P + s, P + s + len(target)))
    assert perhop_positions.isdisjoint(text_positions)
    assert min(text_positions) == P + s


# =========================================================================== #
# E. Curriculum: s_max ramp, consolidation, s_eff = min(s, K).
# =========================================================================== #

def test_trace_stage_smax_ramp_endpoints():
    T = 1000
    assert _trace_stage_smax(0, T) == 0
    # ramp reaches 8 just below the 55% mark; at/after 55% -> consolidation.
    assert _trace_stage_smax(549, T) == 8
    assert _trace_stage_smax(550, T) is None
    assert _trace_stage_smax(999, T) is None
    # linear midpoint of the ramp (step 275 of ramp_end 550) -> ~4.
    assert _trace_stage_smax(275, T) == 4


def test_trace_stage_smax_monotonic_nondecreasing_in_ramp():
    prev = -1
    for step in range(0, 550):
        v = _trace_stage_smax(step, 1000)
        assert v is not None and 0 <= v <= 8
        assert v >= prev
        prev = v


def _cpu_reasoner(tmp_path, rungs, n_records=8, batch_examples=False,
                  perhop_weight=1.0):
    prefix = str(tmp_path / "tr")
    for K in rungs:
        _write_trace_rung(f"{prefix}_n{K}.jsonl", K, n_records=n_records)
    return LatentReasoningCotrain(
        train_prefix=prefix, rungs=list(rungs), tok=_FakeTok(),
        thinking_id=THINK_ID, eos_id=EOS_ID, device="cpu", max_len=256,
        seed=0, checkpoint_latent=False, batch_examples=batch_examples,
        perhop_weight=perhop_weight, trace_mode=True)


def test_pick_stage_ramp_frontier_and_consolidation(tmp_path):
    r = _cpu_reasoner(tmp_path, [2, 3, 4])
    # Step 0: s_max=0 -> always 0.
    assert all(r._pick_stage(0, 1000) == 0 for _ in range(20))
    # Deep in the ramp (s_max=8): only {8,7,6} appear.
    seen = {r._pick_stage(549, 1000) for _ in range(200)}
    assert seen <= {6, 7, 8} and 8 in seen
    # Consolidation: uniform over 0..8, and the full range is reachable.
    cons = {r._pick_stage(800, 1000) for _ in range(400)}
    assert cons <= set(range(0, 9))
    assert min(cons) <= 1 and max(cons) >= 7


def test_trace_step_s_eff_is_min_s_K(tmp_path):
    # Only rung 2 loaded -> rung is always 2; force a large s -> s_eff = min = 2.
    r = _cpu_reasoner(tmp_path, [2])
    r._pick_stage = lambda step, total: 8       # force s > K
    m = _tiny_cpu_model(seed=4)
    loss, R = r.step(m, step=0, total_steps=100, n_examples=2)
    assert r.last_K == 2
    assert r.last_s_eff == 2                     # min(8, 2)
    assert R == 2
    assert torch.isfinite(loss)


def test_trace_step_s_eff_small_s(tmp_path):
    r = _cpu_reasoner(tmp_path, [4])
    r._pick_stage = lambda step, total: 1
    m = _tiny_cpu_model(seed=5)
    loss, R = r.step(m, step=0, total_steps=100, n_examples=2)
    assert r.last_K == 4 and r.last_s_eff == 1 and R == 1
    loss.backward()
    assert m.latent_feedback_adapter.proj.weight.grad is not None


def test_trace_step_R0_pure_text_stage(tmp_path):
    """s=0 (pure text) trace step: R=0, no latent slots, loss finite +
    gradient flows (the pure-text stage the curriculum starts from)."""
    r = _cpu_reasoner(tmp_path, [3])
    r._pick_stage = lambda step, total: 0
    m = _tiny_cpu_model(seed=6)
    loss, R = r.step(m, step=0, total_steps=100, n_examples=2)
    assert R == 0 and r.last_s_eff == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert m.embed.weight.grad is not None and m.embed.weight.grad.abs().sum() > 0


# =========================================================================== #
# F. default-off: trace_mode absent is the ordinary path (backwards-compat).
# =========================================================================== #

def test_default_off_uses_ordinary_loader_and_flags(tmp_path):
    # Ordinary (non-trace) records: (prompt, answer) only, loaded as tuples.
    prefix = str(tmp_path / "plain")
    with open(f"{prefix}_n2.jsonl", "w") as f:
        for i in range(4):
            f.write(json.dumps({"prompt": f"x{i} = {i}\nprint(x{i})",
                                "answer": i, "intermediates": [i % 10,
                                (i + 1) % 10]}) + "\n")
    r = LatentReasoningCotrain(
        train_prefix=prefix, rungs=[2], tok=_FakeTok(), thinking_id=THINK_ID,
        eos_id=EOS_ID, device="cpu", max_len=64, seed=0, perhop_weight=1.0)
    assert r.trace_mode is False
    # ordinary loader yields 3-tuples (comment, answer, inter), not dicts.
    ex = r.data[2][0]
    assert isinstance(ex, tuple) and len(ex) == 3


def test_trace_mode_data_are_dicts_with_suffix_variants(tmp_path):
    r = _cpu_reasoner(tmp_path, [2, 3])
    assert r.trace_mode is True
    ex = r.data[2][0]
    assert isinstance(ex, dict)
    assert set(ex) == {"prompt_ids", "sol_ids_by_s", "inter_ids", "K"}


# =========================================================================== #
# G. Eval harness: parser reuse + R=0-injection reduces to the direct arm.
# =========================================================================== #

def test_eval_harness_reuses_stageA_parsers():
    import experiments.eval_exec_trace_latent_trace as ev
    # parse_final is re-exported (imported) from eval_exec_trace_text.
    assert ev.parse_final("# step 1: x = 8\n# final: 4\n") == 4
    assert ev.parse_final("no final here") is None
    assert callable(ev.latent_greedy_answer)
    assert callable(ev.compute_verdict)


def test_eval_R0_injection_is_identity_reduces_to_direct():
    """With R=0, the growing-thread injection is a no-op: grow_latent_thread
    returns the prompt unchanged (never even calls the model / adapter), so the
    subsequent greedy decode reduces to the plain no-latent (direct-style)
    machinery. Tiny scale, no full-model run (R=0 skips every forward)."""
    from experiments.eval_exec_trace_latent import grow_latent_thread
    m = _tiny_cpu_model(seed=7)
    prompt = [7, 8, 9, 10, 11]
    cur_ids, cur_emb = grow_latent_thread(m, prompt, 0, THINK_ID, "cpu")
    assert cur_ids.tolist() == [prompt]                     # unchanged ids
    assert torch.equal(cur_ids, torch.tensor([prompt], dtype=torch.long))
    assert torch.allclose(cur_emb, m.embed(torch.tensor([prompt],
                                                        dtype=torch.long)))
    assert cur_ids.shape[1] == len(prompt)                  # no latent slots


def test_eval_verdict_kill_and_success_lines():
    from experiments.eval_exec_trace_latent_trace import compute_verdict
    rungs = [
        {"K": 2, "latent_RK_answer": 0.9, "direct_answer": 0.1,
         "latent_RK_perhop": 0.9},
        {"K": 4, "latent_RK_answer": 0.60, "direct_answer": 0.12,
         "latent_RK_perhop": 0.55},
        {"K": 6, "latent_RK_answer": 0.58, "direct_answer": 0.14,
         "latent_RK_perhop": 0.60},
    ]
    v = compute_verdict(rungs)
    assert v["not_killed"] is True                    # >5pp lift at K>=4
    assert v["mechanism_gate_pass (perhop>=0.50 @K4)"] is True
    assert v["success_bar_pass (>=0.55 @K4-8)"] is True


def test_eval_verdict_kill_when_lift_below_5pp():
    from experiments.eval_exec_trace_latent_trace import compute_verdict
    rungs = [
        {"K": 4, "latent_RK_answer": 0.14, "direct_answer": 0.12,
         "latent_RK_perhop": 0.11},                   # +2pp only -> KILLED
        {"K": 6, "latent_RK_answer": 0.60, "direct_answer": 0.14,
         "latent_RK_perhop": 0.60},
    ]
    v = compute_verdict(rungs)
    assert v["not_killed"] is False
    assert v["mechanism_gate_pass (perhop>=0.50 @K4)"] is False
    assert v["success_bar_pass (>=0.55 @K4-8)"] is False   # K=4 at 0.14


# =========================================================================== #
# H. CUDA-gated: batched trace step end-to-end on a tiny DeltaNet.
# =========================================================================== #

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="DeltaNet Triton kernels require CUDA")
def test_batched_trace_step_runs_and_grads(tmp_path):
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    m = TinyLM(vocab_size=VOCAB, d_model=32, n_layers=3, n_heads=2, d_head=16,
               attention_cls=DeltaNetAttention, max_T=0, output_gate=True,
               use_latent_feedback_adapter=True)
    m.thinking_token_id = THINK_ID
    m.train()
    m = m.cuda()
    prefix = str(tmp_path / "tr")
    for K in (2, 3, 4):
        _write_trace_rung(f"{prefix}_n{K}.jsonl", K, n_records=8)
    r = LatentReasoningCotrain(
        train_prefix=prefix, rungs=[2, 3, 4], tok=_FakeTok(),
        thinking_id=THINK_ID, eos_id=EOS_ID, device="cuda", max_len=256,
        seed=0, checkpoint_latent=False, batch_examples=True,
        perhop_weight=1.0, trace_mode=True)
    # Deep in the ramp so s_eff can be > 0 for the small rungs (s_eff=min(s,K)).
    loss, R = r.step(m, step=549, total_steps=1000, n_examples=4)
    assert torch.isfinite(loss)
    assert 0 <= R <= r.last_K
    loss.backward()
    assert m.latent_feedback_adapter.proj.weight.grad is not None


# =========================================================================== #
# H2. Per-hop read units (regression: 2026-07-13 first Stage-B eval read
# structurally 0.000 because the argmax TOKEN ID was compared to the RAW int
# value — encode_inter_token_ids pins the conversion).
# =========================================================================== #

def test_encode_inter_token_ids_units():
    from experiments.eval_exec_trace_latent_trace import encode_inter_token_ids

    class _FakeTok:
        def encode(self, s, add_special_tokens=False):
            # single digit -> one id offset by 100; multi-char -> two ids
            return [100 + int(s)] if len(s) == 1 else [1, 2]

    ids = encode_inter_token_ids(_FakeTok(), [4, 7, 12])
    assert ids == [104, 107, None]      # raw values NEVER compared directly


def test_latent_perhop_reads_skips_none_entries():
    """None (multi-token) intermediates are skipped, not scored as wrong."""
    import experiments.eval_exec_trace_latent_trace as ev
    m = _tiny_cpu_model(seed=4)
    got = ev.latent_perhop_reads(m, [7, 8, 9, 10], 3, THINK_ID,
                                 [5, None, 6], "cpu")
    assert len(got) == 2                # the None slot contributes no entry


# =========================================================================== #
# I. Depth-weighted sampling (the hop-7+ cliff fix arm, 2026-07-13).
# =========================================================================== #

def _mk_trace_jsonl(tmp_path, rungs=(2, 8)):
    import json as _json
    for K in rungs:
        recs = []
        for i in range(4):
            inter = [(i + j) % 10 for j in range(K)]
            lines = "\n".join(f"x = {v}" for v in inter)
            recs.append({"prompt": f"prog {i}\n{lines}\n", "answer": inter[-1],
                         "intermediates": inter, "rung": K,
                         "tracked_var": "x"})
        p = tmp_path / f"tr_n{K}.jsonl"
        p.write_text("\n".join(_json.dumps(r) for r in recs) + "\n")
    return str(tmp_path / "tr")


class _CharTok:
    """Every value 0..9 is a single token (id 30+v)."""
    def encode(self, s, add_special_tokens=False):
        if len(s) == 1 and s.isdigit():
            return [30 + int(s)]
        return [ord(c) % 29 for c in s]


def _mk_cotrain(tmp_path, **kw):
    prefix = _mk_trace_jsonl(tmp_path)
    return LatentReasoningCotrain(
        train_prefix=prefix, rungs=[2, 8], tok=_CharTok(), thinking_id=THINK_ID,
        eos_id=EOS_ID, device="cpu", max_len=512, perhop_weight=1.0,
        trace_mode=True, **kw)


def test_depth_weighted_stage_distribution(tmp_path):
    """Consolidation P(s) ~ (1+s): s=8 drawn ~9x as often as s=0."""
    c = _mk_cotrain(tmp_path, depth_weighted=True, no_ramp=True, seed=0)
    draws = [c._pick_stage(step=10, total_steps=100) for _ in range(4000)]
    n8 = draws.count(8)
    n0 = draws.count(0)
    # expected 9:1; allow generous sampling noise
    assert n8 > 4 * max(1, n0), (n8, n0)
    assert min(draws) == 0 and max(draws) == 8


def test_depth_weighted_rung_distribution(tmp_path):
    """Rung P(K) ~ K over [2, 8]: K=8 drawn ~4x as often as K=2."""
    c = _mk_cotrain(tmp_path, depth_weighted=True, no_ramp=True, seed=0)
    ks = []
    for _ in range(2000):
        if c.depth_weighted:
            w = torch.tensor([float(r) for r in c.rungs])
            ks.append(c.rungs[int(torch.multinomial(w, 1,
                                                    generator=c.g).item())])
    r = ks.count(8) / max(1, ks.count(2))
    assert 2.5 < r < 6.5, r                     # expected 4.0


def test_no_ramp_skips_ramp_in_trace_mode(tmp_path):
    """no_ramp=True: even at step 0 the stage draw is consolidation (can draw
    the full range), not the ramp frontier s_max=0."""
    c = _mk_cotrain(tmp_path, no_ramp=True, seed=0)
    draws = {c._pick_stage(step=0, total_steps=1000) for _ in range(300)}
    assert max(draws) == 8                      # ramp would pin ALL draws to 0


def test_default_uniform_unchanged(tmp_path):
    """depth_weighted absent: consolidation stays uniform (regression)."""
    c = _mk_cotrain(tmp_path, no_ramp=True, seed=0)
    draws = [c._pick_stage(step=10, total_steps=100) for _ in range(4000)]
    counts = [draws.count(s) for s in range(9)]
    assert min(counts) > 0.5 * (4000 / 9), counts
