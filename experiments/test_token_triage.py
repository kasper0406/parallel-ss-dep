"""Tests for Rho-1-style selective-token-loss (triage) — `token_triage.py`.

Covers the pure routing function on hand-built distributions (KD / DROP / EASY),
keep_frac quantile math, top-K-miss fallback, per-source accounting, the
default-off byte-identity of the trainer wiring (a plain-CE-equivalent config
reduces to baseline CE), gradient masking (dropped tokens ⇒ zero grad),
composition with grad-accum, and the mode-b ref-CE store round-trip.

Run:
  PYTHONPATH=. .venv/bin/python -m pytest experiments/test_token_triage.py -v
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.token_triage import (  # noqa: E402
    ROUTE_DROP, ROUTE_EASY, ROUTE_IGNORE, ROUTE_KD,
    RefCEStoreReader, RefCEStoreWriter, TriageConfig,
    compute_triage_mask, format_triage_log, ref_ce_from_topk,
    teacher_entropy_from_topk, triage_config_from_args,
)


# ---------------------------------------------------------------------------
# Helpers to build a top-k store from a full teacher distribution.
# ---------------------------------------------------------------------------
def _topk_logprob_store(full_logits, k):
    """(B,T,V) raw teacher logits -> (ids[B,T,k], logprobs[B,T,k]) as the vLLM
    store holds them (log-softmax over the FULL vocab, top-k kept)."""
    logp = torch.log_softmax(full_logits, dim=-1)
    vals, ids = torch.topk(logp, k, dim=-1)
    return ids, vals


# ---------------------------------------------------------------------------
# 1-3. Routing correctness on hand-built distributions.
# ---------------------------------------------------------------------------
def test_route_kd_teacher_confident_student_wrong():
    # One token: teacher very confident on target (low ref CE), student wrong
    # (high student CE) -> high excess -> KD route.
    V, k = 32, 8
    logits = torch.full((1, 1, V), -10.0)
    logits[0, 0, 3] = 10.0                     # teacher confident on id 3
    targets = torch.tensor([[3]])
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.tensor([[6.0]])          # student is wrong/uncertain
    cfg = TriageConfig(keep_frac=1.0)           # everything eligible -> kept
    res = compute_triage_mask(student_ce, (ids, vals), targets, cfg)
    assert res.route[0, 0].item() == ROUTE_KD
    assert res.kd_mask[0, 0].item() == 1.0
    assert res.ce_weights[0, 0].item() == cfg.w_ce_kept
    # ref CE ~ -log softmax(target) ~ small (teacher confident)
    assert res.ref_ce[0, 0].item() < 0.5


def test_route_drop_both_wrong_high_entropy():
    # Teacher spreads mass (high entropy) AND target not in top-k -> DROP.
    V, k = 64, 8
    logits = torch.zeros(1, 1, V)               # uniform teacher -> max entropy
    targets = torch.tensor([[50]])              # not in the (arbitrary) top-8
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.tensor([[5.0]])          # student also wrong
    cfg = TriageConfig(keep_frac=0.6, entropy_cutoff=1.0, vocab_size=V)
    res = compute_triage_mask(student_ce, (ids, vals), targets, cfg)
    assert res.route[0, 0].item() == ROUTE_DROP
    assert res.ce_weights[0, 0].item() == 0.0
    assert res.kd_mask[0, 0].item() == 0.0


def test_route_easy_both_right():
    # Teacher confident AND student confident on target -> low excess -> EASY.
    V, k = 32, 8
    logits = torch.full((1, 1, V), -10.0)
    logits[0, 0, 7] = 10.0
    targets = torch.tensor([[7]])
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.tensor([[0.05]])         # student also nearly right
    # Explicit cutoff above this token's (tiny) excess so it is NOT kept and,
    # with no hard-CE cutoff, routes to EASY.
    cfg = TriageConfig(excess_cutoff=1.0, w_ce_easy=0.1)
    res = compute_triage_mask(student_ce, (ids, vals), targets, cfg)
    assert res.route[0, 0].item() == ROUTE_EASY
    assert abs(res.ce_weights[0, 0].item() - 0.1) < 1e-7
    assert res.kd_mask[0, 0].item() == 0.0


def test_three_way_split_on_a_batch():
    # Build 3 tokens, one per route, and check all three land correctly.
    V, k = 64, 8
    logits = torch.full((1, 3, V), -12.0)
    # token 0: teacher confident on target 3 (KD candidate; student wrong)
    logits[0, 0, 3] = 12.0
    # token 1: teacher confident on target 4 (EASY; student right)
    logits[0, 1, 4] = 12.0
    # token 2: uniform teacher (DROP by entropy), target off-support
    logits[0, 2] = 0.0
    targets = torch.tensor([[3, 4, 60]])
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.tensor([[7.0, 0.05, 6.0]])
    cfg = TriageConfig(keep_frac=0.5, entropy_cutoff=1.0, vocab_size=V)
    res = compute_triage_mask(student_ce, (ids, vals), targets, cfg)
    assert res.route.flatten().tolist() == [ROUTE_KD, ROUTE_EASY, ROUTE_DROP]


# ---------------------------------------------------------------------------
# 4. keep_frac quantile math.
# ---------------------------------------------------------------------------
def test_keep_frac_quantile_fraction():
    torch.manual_seed(0)
    B, T = 4, 64
    V, k = 128, 16
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.rand(B, T) * 8.0
    cfg = TriageConfig(keep_frac=0.25, vocab_size=V)
    res = compute_triage_mask(student_ce, (ids, vals), targets, cfg)
    kept = (res.route == ROUTE_KD).float().mean().item()
    # ~25% kept (quantile rounding on 256 tokens).
    assert abs(kept - 0.25) < 0.03


def test_keep_frac_one_keeps_all_and_zero_keeps_almost_none():
    torch.manual_seed(1)
    B, T, V, k = 2, 32, 64, 8
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.rand(B, T) * 5.0
    r_all = compute_triage_mask(student_ce, (ids, vals), targets,
                                TriageConfig(keep_frac=1.0, vocab_size=V))
    assert (r_all.route == ROUTE_KD).all()
    r_none = compute_triage_mask(student_ce, (ids, vals), targets,
                                 TriageConfig(keep_frac=0.0, vocab_size=V))
    # keep_frac=0 -> cutoff at the max, only the top token(s) kept.
    assert (r_none.route == ROUTE_KD).float().mean().item() < 0.1


def test_explicit_excess_cutoff_overrides_keep_frac():
    student_ce = torch.tensor([[1.0, 5.0, 9.0]])
    ref_ce = torch.tensor([[0.5, 0.5, 0.5]])
    targets = torch.tensor([[0, 1, 2]])
    cfg = TriageConfig(excess_cutoff=4.0)  # excess = [0.5, 4.5, 8.5]
    res = compute_triage_mask(student_ce, ref_ce, targets, cfg)
    assert res.route.flatten().tolist() == [ROUTE_EASY, ROUTE_KD, ROUTE_KD]


# ---------------------------------------------------------------------------
# 5. Top-K-miss fallback.
# ---------------------------------------------------------------------------
def test_topk_miss_residual_fallback_ref_ce():
    # Target NOT in the stored top-k: residual-mass fallback gives a large ref CE.
    V, k = 100, 8
    logits = torch.full((1, 1, V), -20.0)
    logits[0, 0, 0] = 20.0                       # teacher mass on id 0
    targets = torch.tensor([[99]])               # id 99 far off support
    ids, vals = _topk_logprob_store(logits, k)
    assert 99 not in ids[0, 0].tolist()
    cfg = TriageConfig(vocab_size=V, residual_fallback=True)
    ref = ref_ce_from_topk(ids, vals, targets, cfg)
    # tail mass ~ e^-40 split over 92 ids -> huge ref CE.
    assert ref[0, 0].item() > 10.0


def test_topk_hit_ref_ce_is_exact_for_logprob_store():
    # In-top-k target with a log-prob store -> ref CE == exact -log softmax.
    V, k = 40, 8
    torch.manual_seed(3)
    logits = torch.randn(1, 1, V)
    targets = torch.tensor([[int(logits.argmax())]])   # top-1 is in top-k
    ids, vals = _topk_logprob_store(logits, k)
    cfg = TriageConfig(vocab_size=V)
    ref = ref_ce_from_topk(ids, vals, targets, cfg)
    exact = -torch.log_softmax(logits, -1)[0, 0, targets[0, 0]]
    assert torch.allclose(ref[0, 0], exact, atol=1e-4)


def test_raw_logit_store_softmax_over_topk():
    # stored_is_logprob=False: values are raw logits -> softmax over top-k support.
    V, k = 40, 8
    torch.manual_seed(4)
    logits = torch.randn(1, 1, V)
    vals, ids = torch.topk(logits, k, dim=-1)         # RAW top-k logits
    tgt = int(ids[0, 0, 0])                            # the top-1 id
    targets = torch.tensor([[tgt]])
    cfg = TriageConfig(stored_is_logprob=False)
    ref = ref_ce_from_topk(ids, vals, targets, cfg)
    exact_topk = -torch.log_softmax(vals, -1)[0, 0, 0]
    assert torch.allclose(ref[0, 0], exact_topk, atol=1e-5)


def test_topk_padded_duplicate_ids_no_double_count():
    # Regression (review finding #1): the vLLM store pads short rows by
    # duplicating the last id with logprob -30. If the target == that id, a
    # SUM over matching slots would inflate ref CE by ~30·n_pad. The max fix
    # must recover the real log-prob.
    V, k = 40, 8
    ids = torch.zeros(1, 1, k, dtype=torch.long)
    logp = torch.full((1, 1, k), -30.0)
    # 3 real distinct top-k entries, then 5 padded duplicates of id 7 @ -30.
    real_ids = [3, 5, 7]
    real_lp = [-0.2, -1.0, -2.5]
    for c, (i, lp) in enumerate(zip(real_ids, real_lp)):
        ids[0, 0, c] = i
        logp[0, 0, c] = lp
    for c in range(3, k):
        ids[0, 0, c] = 7          # padded duplicate of the last real id (7)
    targets = torch.tensor([[7]])  # target == the duplicated id
    cfg = TriageConfig(stored_is_logprob=True, vocab_size=V)
    ref = ref_ce_from_topk(ids, logp, targets, cfg)
    # Correct ref CE = -(-2.5) = 2.5, NOT 2.5 + 5*30.
    assert abs(ref[0, 0].item() - 2.5) < 1e-4


def test_teacher_entropy_monotone():
    V, k = 64, 16
    peaked = torch.full((1, 1, V), -20.0)
    peaked[0, 0, 0] = 20.0
    flat = torch.zeros(1, 1, V)
    cfg = TriageConfig()
    _, vp = _topk_logprob_store(peaked, k)
    _, vf = _topk_logprob_store(flat, k)
    hp = teacher_entropy_from_topk(vp, cfg)[0, 0].item()
    hf = teacher_entropy_from_topk(vf, cfg)[0, 0].item()
    assert hf > hp
    assert hp < 0.5           # confident -> near-zero entropy


# ---------------------------------------------------------------------------
# 6. Per-source accounting.
# ---------------------------------------------------------------------------
def test_per_source_accounting_sums():
    student_ce = torch.tensor([[1.0, 8.0, 0.1, 9.0]])
    ref_ce = torch.tensor([[0.5, 0.5, 0.05, 0.5]])
    targets = torch.tensor([[0, 1, 2, 3]])
    source_ids = torch.tensor([[10, 10, 20, 20]])
    cfg = TriageConfig(excess_cutoff=4.0)   # excess=[0.5,7.5,0.05,8.5]->E,K,E,K
    res = compute_triage_mask(student_ce, ref_ce, targets, cfg,
                              source_ids=source_ids)
    assert set(res.per_source.keys()) == {10, 20}
    for s, rec in res.per_source.items():
        # route fractions sum to 1 over valid tokens of each source.
        tot = (rec["frac_drop"] + rec["frac_kd"] + rec["frac_easy"])
        assert abs(tot - 1.0) < 1e-6
        assert rec["n_valid"] == 2
    # source 10: [easy, kd]; source 20: [easy, kd]
    assert res.per_source[10]["n_kd"] == 1
    assert res.per_source[20]["n_kd"] == 1


def test_invalid_positions_ignored_everywhere():
    student_ce = torch.tensor([[5.0, 5.0, 5.0]])
    ref_ce = torch.tensor([[0.0, 0.0, 0.0]])
    targets = torch.tensor([[1, -100, 2]])       # middle position ignored
    cfg = TriageConfig(keep_frac=1.0)
    res = compute_triage_mask(student_ce, ref_ce, targets, cfg)
    assert res.route[0, 1].item() == ROUTE_IGNORE
    assert res.ce_weights[0, 1].item() == 0.0
    assert res.kd_mask[0, 1].item() == 0.0
    assert res.stats["n_valid"] == 2


# ---------------------------------------------------------------------------
# 7. Byte-identity: a plain-CE-equivalent config reduces to baseline CE.
# ---------------------------------------------------------------------------
def _weighted_ce(ce_per_tok, weights, valid):
    denom = (weights * valid).sum().clamp(min=1.0)
    return (ce_per_tok * weights * valid).sum() / denom


def test_plain_ce_equivalent_config_matches_baseline_loss():
    torch.manual_seed(5)
    B, T, V, k = 3, 40, 64, 8
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.rand(B, T) * 4.0
    valid = (targets != -100).float()
    baseline = (student_ce * valid).sum() / valid.sum().clamp(min=1.0)
    # keep_frac=1 (all kept, weight 1), easy weight irrelevant, no drops.
    cfg = TriageConfig(keep_frac=1.0, w_ce_kept=1.0, w_ce_easy=1.0, vocab_size=V)
    res = compute_triage_mask(student_ce, (ids, vals), targets, cfg)
    got = _weighted_ce(student_ce, res.ce_weights, valid)
    assert torch.allclose(got, baseline, atol=1e-6)
    assert torch.allclose(res.ce_weights, valid, atol=1e-7)


# ---------------------------------------------------------------------------
# 8. Gradient masking: dropped tokens contribute zero gradient.
# ---------------------------------------------------------------------------
def test_dropped_tokens_zero_gradient():
    torch.manual_seed(6)
    B, T, V = 2, 6, 16
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    ce_per_tok = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                 reduction="none").reshape(B, T)
    # Route via explicit ref/cutoff so we control exactly which tokens drop.
    ref_ce = torch.zeros(B, T)
    # Force alternating student CE so ~half are high-excess (kept), and mark the
    # low-excess ones as DROP via hard_ce_cutoff below.
    # Use a raw excess override: keep only tokens with excess >= 100 (none) so
    # everything is non-kept, then hard_ce_cutoff drops the high-CE ones.
    cfg = TriageConfig(excess_cutoff=1e9, hard_ce_cutoff=0.0)  # all non-kept, all high CE -> DROP
    res = compute_triage_mask(ce_per_tok.detach(), ref_ce, targets, cfg)
    assert (res.route == ROUTE_DROP).all()
    loss = _weighted_ce(ce_per_tok, res.ce_weights,
                        (targets != -100).float())
    # All weights zero -> loss is a zero-denominator-clamped 0; grad must be zero.
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_only_kept_tokens_receive_gradient():
    torch.manual_seed(7)
    B, T, V = 1, 4, 16
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    ce_per_tok = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                                 reduction="none").reshape(B, T)
    ref_ce = torch.zeros(B, T)
    # Excess = ce; pick cutoff between token CEs so exactly the top-1 is kept.
    ce_sorted = ce_per_tok.detach().flatten().sort().values
    cutoff = float((ce_sorted[-1] + ce_sorted[-2]) / 2)
    # w_ce_easy=0 so the non-kept (EASY) tokens carry NO loss -> no gradient;
    # only the single kept token should receive gradient.
    cfg = TriageConfig(excess_cutoff=cutoff, w_ce_easy=0.0)
    res = compute_triage_mask(ce_per_tok.detach(), ref_ce, targets, cfg)
    kept_idx = (res.route.flatten() == ROUTE_KD).nonzero().flatten()
    assert kept_idx.numel() == 1
    loss = _weighted_ce(ce_per_tok, res.ce_weights, (targets != -100).float())
    loss.backward()
    # Only the kept position's logits row has non-zero grad.
    grad_rows = logits.grad.reshape(-1, V).norm(dim=-1)
    nonzero = (grad_rows > 1e-9).nonzero().flatten()
    assert nonzero.tolist() == kept_idx.tolist()


# ---------------------------------------------------------------------------
# 9. Composition with grad-accum: summed micro-losses == full-batch loss when
#    normalised by the shared valid-token denominator (the trainer's contract).
# ---------------------------------------------------------------------------
def test_grad_accum_equivalence():
    torch.manual_seed(8)
    B, T, V = 4, 20, 32
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    student_ce = torch.rand(B, T) * 6.0
    ref_ce = torch.rand(B, T) * 2.0
    cfg = TriageConfig(keep_frac=0.5, w_ce_easy=0.2)
    # Full batch.
    full = compute_triage_mask(student_ce, ref_ce, targets, cfg)
    valid = (targets != -100).float()
    denom = (full.ce_weights * valid).sum()
    full_loss = (student_ce * full.ce_weights * valid).sum() / denom
    # Two microbatches with the SAME excess cutoff (grad-accum shares a cfg;
    # the trainer computes the cutoff per-microbatch, but for equivalence we
    # pin the cutoff so the routing matches the full-batch routing).
    cfg2 = TriageConfig(keep_frac=0.5, w_ce_easy=0.2,
                        excess_cutoff=full.stats["cutoff"])
    num = torch.zeros(())
    den = torch.zeros(())
    for b0, b1 in ((0, 2), (2, 4)):
        r = compute_triage_mask(student_ce[b0:b1], ref_ce[b0:b1],
                                targets[b0:b1], cfg2)
        v = (targets[b0:b1] != -100).float()
        num = num + (student_ce[b0:b1] * r.ce_weights * v).sum()
        den = den + (r.ce_weights * v).sum()
    assert torch.allclose(full_loss, num / den, atol=1e-5)


# ---------------------------------------------------------------------------
# 10. Log formatting + config-from-args.
# ---------------------------------------------------------------------------
def test_format_triage_log():
    stats = {"frac_keep": 0.6, "frac_kd": 0.6, "frac_drop": 0.1, "frac_easy": 0.3}
    s = format_triage_log(stats)
    assert "keep=60.0%" in s and "kd=60.0%" in s
    assert "drop=10.0%" in s and "easy=30.0%" in s


def test_triage_config_from_args_sentinels():
    class A:
        triage_keep_frac = 0.6
        triage_entropy_cutoff = -1.0     # off
        triage_hard_ce_cutoff = 3.0      # on
        triage_easy_weight = 0.05
        triage_store_raw_logits = True
    cfg = triage_config_from_args(A(), vocab_size=1000)
    assert cfg.entropy_cutoff is None
    assert cfg.hard_ce_cutoff == 3.0
    assert cfg.w_ce_easy == 0.05
    assert cfg.stored_is_logprob is False
    assert cfg.vocab_size == 1000


# ---------------------------------------------------------------------------
# 11. Mode-b ref-CE store round-trip (the SmolLM2-cache reader contract).
# ---------------------------------------------------------------------------
def test_ref_ce_store_roundtrip(tmp_path):
    n = 1000
    ref_ce = torch.rand(n) * 10.0
    input_ids = torch.randint(0, 50000, (n,))
    with RefCEStoreWriter(str(tmp_path), ref_model="SmolLM2-360M",
                          tokenizer_name="tok", shard_max_tokens=256) as w:
        # append in irregular blocks; multiple shards flushed.
        for lo in range(0, n, 137):
            hi = min(lo + 137, n)
            w.append(ref_ce[lo:hi], input_ids[lo:hi])
    r = RefCEStoreReader(str(tmp_path))
    assert len(r) == n
    assert r.ref_model == "SmolLM2-360M"
    # sequential cursor reconstructs the exact stream.
    got_ce, got_ids = [], []
    for _ in range(5):
        ce, ids = r.next_block(200)
        got_ce.append(ce.float())
        got_ids.append(ids)
    got_ce = torch.cat(got_ce)
    got_ids = torch.cat(got_ids)
    assert torch.allclose(got_ce, ref_ce, atol=1e-2)   # fp16 store tolerance
    assert torch.equal(got_ids, input_ids)


def test_ref_ce_store_reader_overrun_raises(tmp_path):
    with RefCEStoreWriter(str(tmp_path)) as w:
        w.append(torch.rand(10), torch.arange(10))
    r = RefCEStoreReader(str(tmp_path))
    try:
        r.next_block(11)
        assert False, "expected IndexError"
    except IndexError:
        pass


# ---------------------------------------------------------------------------
# 12. End-to-end mode-a via the flexible `reference` positional (tuple form).
# ---------------------------------------------------------------------------
def test_reference_positional_tuple_and_tensor():
    V, k = 32, 8
    logits = torch.full((1, 1, V), -10.0)
    logits[0, 0, 5] = 10.0
    targets = torch.tensor([[5]])
    ids, vals = _topk_logprob_store(logits, k)
    student_ce = torch.tensor([[4.0]])
    cfg = TriageConfig(keep_frac=1.0, vocab_size=V)
    # tuple form (mode a)
    r_tuple = compute_triage_mask(student_ce, (ids, vals), targets, cfg)
    # tensor form (mode b) using the ref CE the tuple form computed
    r_tensor = compute_triage_mask(student_ce, r_tuple.ref_ce, targets, cfg)
    assert r_tuple.route[0, 0] == r_tensor.route[0, 0] == ROUTE_KD
    assert torch.allclose(r_tuple.excess, r_tensor.excess, atol=1e-5)


def test_reject_both_reference_modes():
    student_ce = torch.tensor([[1.0]])
    targets = torch.tensor([[0]])
    ids = torch.zeros(1, 1, 4, dtype=torch.long)
    vals = torch.zeros(1, 1, 4)
    try:
        compute_triage_mask(student_ce, None, targets, TriageConfig(),
                            teacher_ids=ids, teacher_vals=vals,
                            ref_ce=torch.tensor([[1.0]]))
        assert False, "expected ValueError"
    except ValueError:
        pass


# ===========================================================================
# Trainer-integration tests: the MixedSourceStream source channel, the
# batch-unpack helper, and _nonthink_forward_loss triage wiring (CPU, tiny).
# ===========================================================================
import torch.nn as nn  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from experiments.train_lm import (  # noqa: E402
    _nonthink_forward_loss, _unpack_train_batch,
)


class _TinyLM(nn.Module):
    """Minimal stand-in for TinyLM's non-gate forward: (x, doc_ids=...) -> logits."""
    def __init__(self, vocab, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x, doc_ids=None, **kw):
        return self.head(self.emb(x))


class _FakeStore:
    """In-memory teacher top-k store with a LogitStoreReader-like cursor and a
    next_block call counter (to prove single-read reuse). Built so the target
    `y` is the top-1 teacher id with a chosen log-prob (=> controlled ref CE)."""
    def __init__(self, x, y, k=8, ref_logp=-0.2, vocab=64):
        self.k = k
        self.vocab_size = vocab
        self.tokenizer_name = "fake"
        self.teacher_model = "fake"
        n = x.numel()
        ids = torch.zeros(n, k, dtype=torch.int64)
        logp = torch.full((n, k), -20.0)
        yf = y.reshape(-1).clamp(min=0)          # -100 -> 0 (masked out anyway)
        for j in range(n):
            ids[j, 0] = yf[j]
            logp[j, 0] = ref_logp
            # fill remaining slots with distinct dummy ids
            for c in range(1, k):
                ids[j, c] = (int(yf[j]) + c) % vocab
        self._ids = ids
        self._logp = logp.to(torch.float16)
        self._inids = x.reshape(-1).to(torch.int64)
        self._cursor = 0
        self.calls = 0

    def next_block(self, n):
        self.calls += 1
        a, b = self._cursor, self._cursor + n
        self._cursor = b
        return (self._ids[a:b].clone(), self._logp[a:b].clone(),
                self._inids[a:b].clone())


def _base_args(**over):
    a = dict(output_gate=False, gate_entropy_aux_weight=0.0, aux_brackets=False,
             gist_loss_weight=0.0, distill_weight=0.0, distill_temp=2.0,
             kd_objective="legacy", token_triage=False, triage_kd_weight=-1.0)
    a.update(over)
    return SimpleNamespace(**a)


# --- MixedSourceStream source channel -------------------------------------
def _mk_stream(tmp_path, **over):
    import json
    from experiments.data_mix import MixedSourceStream, SourceConfig

    class _Tok:
        eos_token_id = 0
        bos_token_id = 0
        vocab_size = 64

        def encode(self, text, add_special_tokens=False):
            return [((ord(c) % 60) + 1) for c in text][:40] or [1, 2, 3]

    def _jsonl(tag):
        p = tmp_path / f"{tag}.jsonl"
        with open(p, "w") as f:
            for i in range(200):
                f.write(json.dumps({"text": f"{tag}doc{i} " * 8}) + "\n")
        return str(p)

    srcs = [SourceConfig(name="A", weight=0.5, jsonl_path=_jsonl("A")),
            SourceConfig(name="B", weight=0.5, jsonl_path=_jsonl("B"))]
    return MixedSourceStream(sources=srcs, tokenizer=_Tok(), block_size=16,
                             thinking_token_id=None, think_burst_prob=0.0,
                             base_seed=0, **over)


def test_stream_source_channel_present_and_constant(tmp_path):
    ds = _mk_stream(tmp_path, emit_doc_ids=True, emit_source_ids=True)
    it = iter(ds)
    for _ in range(3):
        out = next(it)
        assert len(out) == 4                     # x, y, doc_ids, source_ids
        x, y, doc_ids, src = out
        assert src.shape == x.shape
        assert int(src.min()) == int(src.max())  # constant across the chunk
        assert int(src.max()) in (0, 1)


def test_stream_source_channel_off_is_byte_identical_arity(tmp_path):
    ds_off = _mk_stream(tmp_path, emit_doc_ids=True, emit_source_ids=False)
    out = next(iter(ds_off))
    assert len(out) == 3                          # x, y, doc_ids (unchanged)


def test_stream_source_ids_implies_doc_ids(tmp_path):
    # Regression (review finding #2): emit_source_ids must force emit_doc_ids so
    # the source channel is always the LAST element WITH doc_ids present (never
    # a 3-tuple where source_ids would be misread as doc_ids by the unpack).
    ds = _mk_stream(tmp_path, emit_doc_ids=False, emit_source_ids=True)
    assert ds.emit_doc_ids is True
    out = next(iter(ds))
    assert len(out) == 4                          # x, y, doc_ids, source_ids
    batch = tuple(t.unsqueeze(0) for t in out)    # add a batch dim
    xb, yb, doc, mem, src = _unpack_train_batch(batch, True, device="cpu")
    # source is recovered as the LAST channel; mem_read_mask stays None.
    assert mem is None and src is not None
    assert int(src.min()) == int(src.max())


# --- _unpack_train_batch --------------------------------------------------
def test_unpack_batch_off_is_identical_to_legacy():
    x = torch.randint(0, 10, (2, 4))
    y = torch.randint(0, 10, (2, 4))
    doc = torch.zeros(2, 4, dtype=torch.long)
    rm = torch.ones(2, 4, dtype=torch.long)
    # 3-tuple (doc only)
    xb, yb, d, m, s = _unpack_train_batch((x, y, doc), False, device="cpu")
    assert d is not None and m is None and s is None
    # 4-tuple (doc + read_mask): legacy took _rest[1] as mem_read_mask
    xb, yb, d, m, s = _unpack_train_batch((x, y, doc, rm), False, device="cpu")
    assert torch.equal(m, rm) and s is None


def test_unpack_batch_source_ids_no_collision():
    x = torch.randint(0, 10, (2, 4))
    y = torch.randint(0, 10, (2, 4))
    doc = torch.zeros(2, 4, dtype=torch.long)
    rm = torch.ones(2, 4, dtype=torch.long)
    src = torch.full((2, 4), 3, dtype=torch.long)
    # doc + source (NO read_mask): source is last, mem_read_mask must stay None.
    xb, yb, d, m, s = _unpack_train_batch((x, y, doc, src), True, device="cpu")
    assert m is None and torch.equal(s, src)
    # doc + read_mask + source: both recovered, no collision.
    xb, yb, d, m, s = _unpack_train_batch((x, y, doc, rm, src), True, device="cpu")
    assert torch.equal(m, rm) and torch.equal(s, src)


# --- _nonthink_forward_loss triage wiring ----------------------------------
def _forward(model, x, y, args, triage_cfg=None, store=None, ref_reader=None,
             source_ids=None):
    return _nonthink_forward_loss(
        model, x, y, args, step=0, bracket_deltas=None, doc_ids=None,
        gist_horizons=None, fwd_model=model, mem_read_mask=None,
        kd_teacher=None, kd_thinking_token_id=None, kd_logit_store=store,
        triage_cfg=triage_cfg, triage_ref_reader=ref_reader,
        source_ids=source_ids)


def test_forward_triage_off_returns_none_stats():
    torch.manual_seed(0)
    V = 64
    model = _TinyLM(V)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    out = _forward(model, x, y, _base_args())
    assert len(out) == 8
    assert out[-1] is None                         # triage_stats


def test_forward_plain_ce_equivalent_matches_baseline():
    # keep_frac=1 + w_ce_kept=1 => all tokens weight 1 => lm_loss == baseline.
    torch.manual_seed(1)
    V = 64
    model = _TinyLM(V)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    base = _forward(model, x, y, _base_args())
    base_lm = base[2]
    cfg = TriageConfig(keep_frac=1.0, w_ce_kept=1.0, w_ce_easy=1.0, vocab_size=V)
    store = _FakeStore(x, y, vocab=V)
    tri = _forward(model, x, y, _base_args(token_triage=True), triage_cfg=cfg,
                   store=store)
    assert torch.allclose(base_lm, tri[2], atol=1e-6)
    assert tri[-1] is not None                     # stats populated
    assert tri[-1]["stats"]["frac_kd"] == 1.0
    assert store.calls == 1                         # single store read (ref only)


def test_forward_triage_reweights_and_drops():
    torch.manual_seed(2)
    V = 64
    model = _TinyLM(V)
    x = torch.randint(0, V, (3, 10))
    y = torch.randint(0, V, (3, 10))
    # Reference is confident (ref_logp=-0.05 -> ref CE ~0.05); many student CEs
    # exceed it -> ~half kept. Drop EASY via w_ce_easy=0.
    store = _FakeStore(x, y, ref_logp=-0.05, vocab=V)
    cfg = TriageConfig(keep_frac=0.5, w_ce_easy=0.0, vocab_size=V)
    out = _forward(model, x, y, _base_args(token_triage=True), triage_cfg=cfg,
                   store=store)
    st = out[-1]["stats"]
    assert 0.0 < st["frac_kd"] < 1.0
    assert st["frac_kd"] + st["frac_easy"] + st["frac_drop"] == 1.0
    assert torch.isfinite(out[2])


def test_forward_kd_restricted_and_single_store_read():
    # With distill_weight>0 AND triage on, the KD term reuses the triage block
    # (store read exactly once) and is restricted to kd_mask.
    torch.manual_seed(3)
    V = 64
    model = _TinyLM(V)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    store = _FakeStore(x, y, vocab=V)
    cfg = TriageConfig(keep_frac=0.5, vocab_size=V)
    out = _forward(model, x, y,
                   _base_args(token_triage=True, distill_weight=1.0),
                   triage_cfg=cfg, store=store)
    kd_loss = out[6]
    assert torch.isfinite(kd_loss)
    assert store.calls == 1                         # NOT double-read


def test_forward_dropped_tokens_zero_gradient_through_forward():
    # A triage config that DROPS every token (excess cutoff huge, hard-CE drop)
    # yields zero gradient into the model params.
    torch.manual_seed(4)
    V = 64
    model = _TinyLM(V)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    store = _FakeStore(x, y, vocab=V)
    cfg = TriageConfig(excess_cutoff=1e9, hard_ce_cutoff=0.0, vocab_size=V)
    out = _forward(model, x, y, _base_args(token_triage=True), triage_cfg=cfg,
                   store=store)
    lm_loss = out[2]
    lm_loss.backward()
    total = sum(p.grad.abs().sum().item() for p in model.parameters()
                if p.grad is not None)
    assert total == 0.0


def test_forward_grad_accum_store_cursor_advances_per_call():
    # Two microbatches -> two next_block calls, cursor advances in lockstep.
    torch.manual_seed(5)
    V = 64
    model = _TinyLM(V)
    x0 = torch.randint(0, V, (2, 8))
    y0 = torch.randint(0, V, (2, 8))
    x1 = torch.randint(0, V, (2, 8))
    y1 = torch.randint(0, V, (2, 8))
    # A single store covering BOTH microbatches, in order.
    x_all = torch.cat([x0, x1], dim=0)
    y_all = torch.cat([y0, y1], dim=0)
    store = _FakeStore(x_all, y_all, vocab=V)
    cfg = TriageConfig(keep_frac=0.5, vocab_size=V)
    _forward(model, x0, y0, _base_args(token_triage=True), triage_cfg=cfg,
             store=store)
    assert store.calls == 1 and store.tell() if hasattr(store, "tell") else True
    _forward(model, x1, y1, _base_args(token_triage=True), triage_cfg=cfg,
             store=store)
    assert store.calls == 2
    assert store._cursor == x_all.numel()
