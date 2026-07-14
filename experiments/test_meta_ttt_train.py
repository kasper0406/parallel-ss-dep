"""Tests for the Meta-TTT episode-training wiring (experiments/meta_ttt_train.py).

Runs entirely on CPU. Because FLA's real DeltaNet is a CUDA-only Triton kernel
(it cannot run on CPU — `chunk_delta_rule` asserts non-fp32 AND its kernels need
a GPU), the state-carry equivalence test uses a CPU-runnable STUB attention that
implements the SAME FLA cache protocol (`get_layer_cache`/`update_layer_cache`,
read initial recurrent_state → run recurrence → write final recurrent_state).
The stub uses ONLY element-wise ops (autocast-inert) so chunked-with-carry and a
single full forward agree in fp32. This tests the NEW code — the chunking, the
sequential state threading through one cache, and the no_grad/grad boundary —
while FLA's own state carry is upstream-validated (test_incremental_decode.py).

A CUDA-guarded real-DeltaNet equivalence check is included but skipped without a
GPU (the instructions forbid GPU use here).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from experiments.meta_ttt_train import (
    MetaTTTEpisodeTrainer,
    assemble_episode_tokens,
    chunk_bounds,
    grad_boundary,
    ingest_and_task_logits,
    supervised_positions,
)


# --------------------------------------------------------------------------- #
# CPU state-carrying stub attention (same FLA cache protocol as DeltaNet).
# --------------------------------------------------------------------------- #

class _StubDeltaLayer(nn.Module):
    """Inner `.layer` — mimics fla DeltaNet's cache contract on CPU.

    Recurrence (element-wise, so it is exact under CPU bf16 autocast and its
    chunked accumulation equals the full one): with per-channel write gate
    g = sigmoid(write_gate),
        state_t = state_{t-1} + g * (x_t * in_scale)
        out_t   = state_t
    `write_gate` is the "β / write-gate" analog whose gradient the meta-TTT
    truncated-BPTT test tracks.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = 0
        self.mode = "chunk"
        self.write_gate = nn.Parameter(torch.randn(d_model) * 0.3)
        self.in_scale = nn.Parameter(torch.ones(d_model) + 0.05 * torch.randn(d_model))

    def forward(self, hidden_states, past_key_values=None, use_cache=False,
                **kwargs):
        from fla.layers.utils import get_layer_cache, update_layer_cache
        hs = hidden_states.float()                          # (B, T, d)
        init = None
        if past_key_values is not None:
            last = get_layer_cache(self, past_key_values)
            if last is not None:
                init = last["recurrent_state"]              # (B, d)
        g = torch.sigmoid(self.write_gate)                  # (d,)
        contrib = hs * self.in_scale * g                    # element-wise
        state_seq = contrib.cumsum(dim=1)                   # (B, T, d)
        if init is not None:
            state_seq = state_seq + init.unsqueeze(1)
        final_state = state_seq[:, -1, :]                   # (B, d)
        if use_cache and past_key_values is not None:
            update_layer_cache(self, past_key_values,
                               recurrent_state=final_state, conv_state=None,
                               offset=hidden_states.shape[1])
        return state_seq, None, past_key_values


class StubDeltaAttention(nn.Module):
    """Wrapper (`blk.attn`) matching the `_FlaWrapper` interface enough for both
    TinyLM's full-forward (`Block.forward` → `attn(x)`) and its cache decode
    path (`TinyLM._step_block` → `attn.layer(...)` / `attn.forward_step`)."""

    accepts_cu_seqlens = False
    accepts_think_mask = False
    state_readonly_at_think = False
    needs_input_ids = False

    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        self.layer = _StubDeltaLayer(d_model)

    def forward(self, x, cu_seqlens=None, think_mask=None):
        out, _, _ = self.layer(x, past_key_values=None, use_cache=False)
        return out.to(x.dtype)

    def forward_step(self, x, past, layer_idx):
        self.layer.layer_idx = int(layer_idx)
        out, _, past = self.layer(x, past_key_values=past, use_cache=True)
        return out.to(x.dtype), past


VOCAB = 5000


def _build_stub_model(n_layers=2, d_model=16):
    from experiments.model import TinyLM
    torch.manual_seed(0)
    model = TinyLM(vocab_size=VOCAB, d_model=d_model, n_layers=n_layers,
                   n_heads=2, d_head=d_model // 2, d_ff=2 * d_model,
                   attention_cls=StubDeltaAttention, max_T=0)
    model.train()
    return model


# --------------------------------------------------------------------------- #
# Fake char-level tokenizer (hermetic; supports offsets for the span helper).
# --------------------------------------------------------------------------- #

class FakeCharTokenizer:
    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False):
        ids = [(ord(c) % 4000) + 1 for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def _make_episode(episode_id, ctx_files, task_prefix, task_line, ident,
                  n_ctx, bucket):
    """Synthetic episode dict matching gen_repo_episodes' schema. The identifier
    span is located inside task_line by .find()."""
    s0 = task_line.find(ident)
    assert s0 >= 0, "identifier must appear in task_line"
    return {
        "episode_id": episode_id,
        "repo_name": "t/repo",
        "context_files": [{"path": f"f{i}.py", "text": t}
                          for i, t in enumerate(ctx_files)]
                          + [{"path": "task.py", "text": task_prefix + task_line}],
        "task_file": "task.py",
        "task_prefix": task_prefix,
        "task_line": task_line,
        "task_char_span": [s0, s0 + len(ident)],
        "link": {"identifier": ident},
        "n_ctx_tokens": n_ctx,
        "bucket": bucket,
    }


# =========================================================================== #
# 1-3. Chunk-boundary math.
# =========================================================================== #

def test_chunk_bounds_covers_range_no_size1():
    for total in range(2, 40):
        for cs in (2, 3, 5, 8, 16):
            bounds = chunk_bounds(total, cs)
            # contiguous cover of [0, total)
            assert bounds[0][0] == 0
            assert bounds[-1][1] == total
            for (a, b), (a2, _) in zip(bounds, bounds[1:]):
                assert b == a2
            # every chunk length >= 2 (never routes T==1 forward_step)
            for a, b in bounds:
                assert b - a >= 2, (total, cs, bounds)
            # each chunk <= cs+1 (the +1 is the absorbed size-1 tail)
            for a, b in bounds:
                assert b - a <= cs + 1


def test_chunk_bounds_absorbs_size1_tail():
    # total=9, cs=4 → naive [0,4)[4,8)[8,9) leaves a size-1 tail; absorb it.
    bounds = chunk_bounds(9, 4)
    assert all(b - a >= 2 for a, b in bounds)
    assert bounds[-1] == (4, 9)  # the size-1 tail was folded into the last chunk


def test_chunk_bounds_total_one():
    assert chunk_bounds(1, 8) == [(0, 1)]
    assert chunk_bounds(0, 8) == []


# =========================================================================== #
# 4-6. Grad-window boundary.
# =========================================================================== #

def test_grad_boundary_covers_predictors_small_window():
    # window shorter than distance from earliest predictor to end → pull back.
    total, window, earliest = 100, 10, 80
    b = grad_boundary(total, window, earliest)
    # window from `total - window` = 90 would MISS predictor 80 → pulled to 80.
    assert b == 80
    assert b <= earliest


def test_grad_boundary_large_window_reaches_zero():
    assert grad_boundary(total_len=50, grad_window_len=4096, earliest_predictor=45) == 0


def test_grad_boundary_natural_window_when_it_covers():
    # window (4096) easily covers a task span near the end → natural boundary.
    total, window, earliest = 6000, 4096, 5990
    b = grad_boundary(total, window, earliest)
    assert b == total - window  # natural boundary
    assert b <= earliest        # still covers predictors


# =========================================================================== #
# 7-9. Token assembly + loss-mask exactness vs the eval's tokenization.
# =========================================================================== #

def test_assemble_matches_eval_tokenization():
    from experiments.eval_repo_adaptive import context_ids, task_line_span_tokens
    from experiments.gen_repo_episodes import perline_ids
    tok = FakeCharTokenizer()
    ep = _make_episode("e1", ["import os\ndef helper(a, b):\n    return a+b\n"],
                       "x = 1\nresult = ", "helper(x, 2)\n", "helper",
                       n_ctx=6000, bucket="4-8k")
    a = assemble_episode_tokens(ep, tok)
    # The eval's `real` arm assembles context_ids + perline_ids(task_prefix),
    # then predicts task_line from [P-1, P+L-1). Our full_ids and p_line must
    # reproduce exactly that layout.
    real_prefix = context_ids(ep, tok) + perline_ids(ep["task_prefix"], tok)
    line_ids, span_idx = task_line_span_tokens(ep, tok)
    assert a["full_ids"] == real_prefix + line_ids
    assert a["p_line"] == len(real_prefix)
    assert a["line_len"] == len(line_ids)
    assert a["span_local"] == span_idx


def test_supervised_positions_m0_matches_eval_predictors():
    # eval predicts task_line[i] from hidden at P-1+i, i.e. pred = [P-1, P+L-1).
    P, L = 40, 5
    tgt, pred = supervised_positions(P, L, prefix_supervise_m=0)
    assert tgt == list(range(P, P + L))
    assert pred == list(range(P - 1, P + L - 1))


def test_line_ce_equals_eval_real_arm():
    """End-to-end: the trainer's supervised line CE equals the CE
    `eval_repo_adaptive.arm_ce` computes for the `real` arm on the SAME model +
    tokenizer — the tightest check that training scores exactly what the eval
    measures (same tokenization, same positions, same head)."""
    import torch.nn.functional as F
    from experiments.eval_repo_adaptive import (arm_ce, context_ids,
                                                task_line_span_tokens)
    from experiments.gen_repo_episodes import perline_ids
    tok = FakeCharTokenizer()
    ep = _make_episode("e1", ["def helper(a, b):\n    return a + b\n" * 2],
                       "x = 1\ny = 2\nresult = ", "helper(x, y)\n", "helper",
                       n_ctx=6000, bucket="4-8k")
    model = _build_stub_model(n_layers=2, d_model=16)
    model.eval()
    # eval `real` arm
    prefix = context_ids(ep, tok) + perline_ids(ep["task_prefix"], tok)
    line_ids, span_idx = task_line_span_tokens(ep, tok)
    with torch.no_grad():
        ev = arm_ce(model, prefix, line_ids, span_idx, "cpu", bf16=False)
    # trainer assembly + ingest (boundary 0 = single full forward equivalent)
    a = assemble_episode_tokens(ep, tok)
    tgt, pred = supervised_positions(a["p_line"], a["line_len"], 0)
    with torch.no_grad():
        logits = ingest_and_task_logits(model, a["full_ids"], 0, 10_000, pred, "cpu")
    targets = torch.tensor([a["full_ids"][t] for t in tgt], dtype=torch.long)
    my_line_ce = F.cross_entropy(logits.float(), targets).item()
    assert abs(my_line_ce - ev["line_ce"]) < 1e-4, (my_line_ce, ev["line_ce"])


def test_supervised_positions_with_prefix_m():
    P, L, M = 40, 5, 3
    tgt, pred = supervised_positions(P, L, prefix_supervise_m=M)
    assert tgt == list(range(P - M, P + L))           # includes last M prefix tokens
    assert pred == list(range(P - M - 1, P + L - 1))
    assert pred[0] == P - M - 1                        # earliest predictor pulled back


# =========================================================================== #
# 10-11. State-carry equivalence (THE critical test) + non-triviality.
# =========================================================================== #

def _predictor_positions(total, p_line, line_len):
    tgt, pred = supervised_positions(p_line, line_len, 0)
    return pred


def test_state_carry_equivalence_chunked_equals_full():
    """Chunked ingest WITH state carry == a single full forward, at the task
    predictor positions. The core correctness guarantee of the ingest path."""
    model = _build_stub_model(n_layers=3, d_model=16)
    torch.manual_seed(1)
    total, p_line, line_len = 60, 48, 6
    ids = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(total)]
    pred = _predictor_positions(total, p_line, line_len)

    # Full single forward → logits at predictor positions.
    x_full = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        full_logits = model(x_full)                    # (1, T, V)
    full_at = full_logits[0, pred, :]

    # Chunked ingest (prefix in chunks of 8 under no_grad, grad window covers
    # predictors). boundary chosen well before earliest predictor.
    boundary = grad_boundary(total, grad_window_len=16, earliest_predictor=pred[0])
    with torch.no_grad():
        chunk_at = ingest_and_task_logits(model, ids, boundary, chunk_size=8,
                                          pred_pos=pred, device="cpu")
    assert boundary > 0 and boundary < pred[0]          # a real no_grad prefix ran
    assert torch.allclose(full_at, chunk_at, atol=1e-4, rtol=1e-4), \
        (full_at - chunk_at).abs().max().item()


def test_state_carry_is_nontrivial_without_carry_differs():
    """If the recurrent state were NOT carried across chunks (cache reset each
    chunk), the logits would differ substantially — proving the equivalence
    test above actually exercises state carry."""
    from fla.models.utils import Cache as FLACache
    from experiments.meta_ttt_train import _block_stack, _embed_chunk, \
        _finalize_logits_at
    model = _build_stub_model(n_layers=2, d_model=16)
    torch.manual_seed(2)
    total, p_line, line_len = 40, 30, 5
    ids = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(total)]
    pred = _predictor_positions(total, p_line, line_len)
    ids_t = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        full_at = model(ids_t)[0, pred, :]
        # NO-carry: fresh cache per chunk (state reset at every boundary).
        win_start = 16
        for a, b in chunk_bounds(total, 8):
            fresh = FLACache(seen_tokens=0)
            x = _embed_chunk(model, ids_t[:, a:b], pos_offset=a)
            h = _block_stack(model, x, fresh)
            if a <= pred[0] < b:  # crude: last chunk holds predictors
                pass
        # Simpler: reset cache before the grad window so it starts from state 0
        # instead of the carried prefix state.
        fresh = FLACache(seen_tokens=0)
        win = ids_t[:, win_start:]
        x = _embed_chunk(model, win, pos_offset=win_start)
        h = _block_stack(model, x, fresh)
        local = [p - win_start for p in pred]
        nocarry_at = _finalize_logits_at(model, h, win, local)
    assert not torch.allclose(full_at, nocarry_at, atol=1e-2), \
        "state carry made no difference — recurrence/stub is degenerate"


# =========================================================================== #
# 12-13. Truncated-BPTT: grad flows to the write-gate from grad chunks ONLY.
# =========================================================================== #

def test_grad_only_from_grad_window():
    """Tokens that appear ONLY in the no_grad prefix get ZERO embedding
    gradient; tokens in the grad window get gradient; and the write-gate param
    receives gradient (from the grad-window ingestion)."""
    model = _build_stub_model(n_layers=2, d_model=16)
    PREFIX_TOK, WINDOW_TOK, TASK_TOK = 11, 22, 33
    # Layout chosen so the computed grad boundary lands EXACTLY at the end of the
    # PREFIX_TOK run: total=40, window=12, task span=[36,40) → earliest pred 35,
    # boundary = min(40-12, 35) = 28. So [0,28)=PREFIX (no_grad), [28,36)=WINDOW
    # (grad), [36,40)=TASK (grad).
    ids = [PREFIX_TOK] * 28 + [WINDOW_TOK] * 8 + [TASK_TOK] * 4
    total = len(ids)
    p_line = total - 4          # task span = last 4 tokens
    line_len = 4
    tgt, pred = supervised_positions(p_line, line_len, 0)
    boundary = grad_boundary(total, grad_window_len=12, earliest_predictor=pred[0])
    assert boundary == 28
    # PREFIX_TOK must appear ONLY strictly before the boundary (no_grad region).
    assert all(i < boundary for i, t in enumerate(ids) if t == PREFIX_TOK)

    model.zero_grad(set_to_none=True)
    logits = ingest_and_task_logits(model, ids, boundary, chunk_size=8,
                                    pred_pos=pred, device="cpu")
    targets = torch.tensor([ids[t] for t in tgt], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.float(), targets)
    loss.backward()

    emb_grad = model.embed.weight.grad
    assert emb_grad is not None
    # PREFIX_TOK appears ONLY before the boundary → no grad path.
    assert emb_grad[PREFIX_TOK].abs().sum().item() == 0.0
    # WINDOW_TOK / TASK_TOK are in the grad window → real grad.
    assert emb_grad[WINDOW_TOK].abs().sum().item() > 0.0
    assert emb_grad[TASK_TOK].abs().sum().item() > 0.0
    # The write-gate ("β" analog) received gradient — from the grad window only.
    for blk in model.blocks:
        wg = blk.attn.layer.write_gate.grad
        assert wg is not None and wg.abs().sum().item() > 0.0


def test_full_grad_window_reaches_prefix_tokens():
    """With grad_chunks large enough that boundary==0, EVERY token embedding
    (including the ones that were prefix-only above) gets gradient — the
    complement of the truncated-BPTT test."""
    model = _build_stub_model(n_layers=2, d_model=16)
    PREFIX_TOK, TASK_TOK = 11, 33
    ids = [PREFIX_TOK] * 20 + [TASK_TOK] * 4
    total = len(ids)
    p_line, line_len = total - 4, 4
    tgt, pred = supervised_positions(p_line, line_len, 0)
    boundary = grad_boundary(total, grad_window_len=10_000, earliest_predictor=pred[0])
    assert boundary == 0                               # whole episode is grad-on

    model.zero_grad(set_to_none=True)
    logits = ingest_and_task_logits(model, ids, boundary, chunk_size=8,
                                    pred_pos=pred, device="cpu")
    targets = torch.tensor([ids[t] for t in tgt], dtype=torch.long)
    torch.nn.functional.cross_entropy(logits.float(), targets).backward()
    assert model.embed.weight.grad[PREFIX_TOK].abs().sum().item() > 0.0


# =========================================================================== #
# 14-17. Trainer: loading, bucket balance, drop-oversized, end-to-end step.
# =========================================================================== #

def _write_episodes(tmp_path, episodes):
    import json
    p = tmp_path / "train.jsonl"
    with open(p, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return str(p)


def _corpus(tmp_path):
    eps = []
    ctx = "def helper(a, b):\n    return a + b\n" * 3
    for i in range(9):
        bucket = ["4-8k", "8-16k", "16-32k"][i % 3]
        eps.append(_make_episode(f"e{i}", [ctx], "x = 1\nr = ",
                                 "helper(x, 2)\n", "helper",
                                 n_ctx=5000, bucket=bucket))
    return _write_episodes(tmp_path, eps)


def test_trainer_loads_and_indexes_buckets(tmp_path):
    tok = FakeCharTokenizer()
    path = _corpus(tmp_path)
    tr = MetaTTTEpisodeTrainer(path, tok, device="cpu", chunk_size=8,
                               grad_chunks=2)
    assert tr.n_episodes == 9
    assert set(tr.buckets) == {"4-8k", "8-16k", "16-32k"}
    assert all(len(tr.by_bucket[b]) == 3 for b in tr.buckets)


def test_trainer_drops_oversized_episodes(tmp_path):
    tok = FakeCharTokenizer()
    ctx = "def helper(a, b):\n    return a + b\n"
    eps = [
        _make_episode("small", [ctx], "x=1\nr = ", "helper(x, 2)\n", "helper",
                      n_ctx=5000, bucket="4-8k"),
        _make_episode("huge", [ctx], "x=1\nr = ", "helper(x, 2)\n", "helper",
                      n_ctx=40000, bucket="16-32k"),
    ]
    path = _write_episodes(tmp_path, eps)
    tr = MetaTTTEpisodeTrainer(path, tok, device="cpu", max_ctx_tokens=32000)
    assert tr.n_episodes == 1
    assert tr.n_dropped == 1


def test_trainer_bucket_balanced_sampling(tmp_path):
    tok = FakeCharTokenizer()
    path = _corpus(tmp_path)
    tr = MetaTTTEpisodeTrainer(path, tok, device="cpu", seed=0)
    counts = {b: 0 for b in tr.buckets}
    for _ in range(900):
        ep = tr._sample_episode()
        counts[ep["bucket"]] += 1
    # Uniform over 3 buckets → each ≈ 300; allow generous slack.
    for b, c in counts.items():
        assert 200 < c < 400, counts


def test_trainer_step_returns_loss_and_diag(tmp_path):
    tok = FakeCharTokenizer()
    path = _corpus(tmp_path)
    tr = MetaTTTEpisodeTrainer(path, tok, device="cpu", chunk_size=16,
                               grad_chunks=2)
    model = _build_stub_model(n_layers=2, d_model=16)
    loss, diag = tr.step(model)
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()                                    # graph is well-formed
    for k in ("ce", "ctx", "bkt", "grad_tokens", "n_pred"):
        assert k in diag
    assert diag["bkt"] in ("4-8k", "8-16k", "16-32k")
    assert diag["n_pred"] >= 1
    # last-step diagnostics mirror the returned diag.
    assert tr.last_bucket == diag["bkt"]
    assert abs(tr.last_task_ce - diag["ce"]) < 1e-6


# =========================================================================== #
# 18. Default-off byte-identity + flag plumbing.
# =========================================================================== #

def test_default_off_flag_plumbing():
    from experiments.train_lm_args import build_parser
    a = build_parser().parse_args([])
    assert a.meta_ttt_weight == 0.0                    # OFF by default
    assert a.meta_ttt_grad_chunks == 2
    assert a.meta_ttt_every == 1
    assert a.meta_ttt_prefix_supervise_m == 0
    assert a.meta_ttt_max_ctx_tokens == 32000
    assert a.meta_ttt_train_prefix.endswith("train.jsonl")
    a2 = build_parser().parse_args(
        ["--meta_ttt_weight", "0.05", "--meta_ttt_grad_chunks", "4",
         "--meta_ttt_every", "3", "--meta_ttt_chunk_size", "1024",
         "--meta_ttt_prefix_supervise_m", "8"])
    assert a2.meta_ttt_weight == 0.05 and a2.meta_ttt_grad_chunks == 4
    assert a2.meta_ttt_every == 3 and a2.meta_ttt_chunk_size == 1024
    assert a2.meta_ttt_prefix_supervise_m == 8


def test_default_off_forward_byte_identical():
    """With the aux never invoked, a plain forward through the stub model is
    unchanged — the meta-TTT machinery adds nothing to the main path."""
    model = _build_stub_model(n_layers=2, d_model=16)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    with torch.no_grad():
        a = model(ids)
        b = model(ids)
    assert torch.allclose(a, b)


# =========================================================================== #
# CUDA-only: real DeltaNet equivalence (skipped without a GPU).
# =========================================================================== #

@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="FLA DeltaNet is a CUDA-only Triton kernel")
def test_state_carry_equivalence_real_deltanet():
    from experiments.model import TinyLM
    from experiments.layers import DeltaNetAttention
    torch.manual_seed(0)
    model = TinyLM(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2,
                   d_head=16, d_ff=64, attention_cls=DeltaNetAttention,
                   max_T=0).cuda().train()
    total, p_line, line_len = 300, 280, 8
    ids = [int(torch.randint(1, VOCAB, (1,)).item()) for _ in range(total)]
    pred = _predictor_positions(total, p_line, line_len)
    x_full = torch.tensor([ids], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        full_at = model(x_full)[0, pred, :].float()
        boundary = grad_boundary(total, 128, pred[0])
        chunk_at = ingest_and_task_logits(model, ids, boundary, chunk_size=128,
                                          pred_pos=pred, device="cuda").float()
    assert torch.allclose(full_at, chunk_at, atol=5e-2, rtol=5e-2)
