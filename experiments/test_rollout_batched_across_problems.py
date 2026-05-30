"""CPU-only tests for cross-problem turn-0 rollout batching.

`rollout_turn0_batched_across_problems` runs ONE decode loop over R = B*N rows
(all B problems' turn-0 rollouts) instead of B separate N-row loops. This is
the ~3-4x speedup lever for grader-RL (DeltaNet decode is memory-bandwidth-
bound, so one R-row loop costs ~the same as one N-row loop).

CORRECTNESS is paramount — this generates the RL training data. The key safety
net is an EQUIVALENCE test: the batched-across-problems path must produce the
SAME per-row rollouts (emit ids/positions/log_probs, gate decisions, depths,
row-local full_ids prefix) as running each problem through the existing
`rollout_group_batched` path, given identical inputs + greedy decode.

We use the full-forward decode path (`can_incremental=False`) because the
incremental (prefill/forward_step) path needs Triton/CUDA and cannot run on
CPU. Two stub models:

  * `StubLM` (constant logits) — proves bookkeeping + regrouping equivalence.
  * `HistoryStubLM` (logits depend on the row's OWN prefix) — proves NO
    padding leakage: a short prompt batched next to a long one yields the
    identical rollout to running it alone.

Run ONLY this file (the full suite has CUDA tests that would OOM a co-resident
training run):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python -m pytest \\
        experiments/test_rollout_batched_across_problems.py -v
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments.train_rl_grader import (
    Rollout,
    compute_think_budget_spread,
    rollout_group_batched,
    rollout_turn0_batched_across_problems,
    _merge_prefill_caches,
)


THINK_ID = 6   # != winning emit token (5), != EOS (1), != PAD (0)
EOS_ID = 1


class StubTokenizer:
    eos_token_id = EOS_ID

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


class StubLM(nn.Module):
    """Constant logits + fixed gate. No prefill/forward_step → full-forward
    path. Output is content-INDEPENDENT (good for bookkeeping equivalence)."""

    def __init__(self, vocab_size: int, *, gate_value: float):
        super().__init__()
        self.vocab_size = vocab_size
        self._gate_value = gate_value
        self.dummy = nn.Linear(1, 1)
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        logits = torch.zeros(B, T, self.vocab_size, device=ids.device,
                             dtype=torch.float32)
        logits[..., 5] = 5.0   # token 5 always wins (greedy)
        g = torch.full((B, T), float(self._gate_value), device=ids.device,
                       dtype=torch.float32)
        self._last_gate = g
        self._last_gate_logits = torch.logit(g.clamp(1e-6, 1 - 1e-6))
        return logits


class HistoryStubLM(nn.Module):
    """Content-DEPENDENT logits: the winning token + gate at position p depend
    on a CAUSAL prefix statistic (running sum of token ids mod something). If
    a row's real tokens ever attended to padding from another row, the chosen
    token / gate would change — so batched==alone proves no leakage.

    Deterministic (greedy-friendly): the argmax token at position p is
    `7 + (prefix_sum % 40)` so it never collides with THINK/EOS/PAD. The gate
    is a deterministic function of the prefix so emit/think decisions are
    reproducible. Causal: position p only sees ids[:, :p+1]."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Linear(1, 1)
        self._last_gate = None
        self._last_gate_logits = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        device = ids.device
        # Causal running sum of ids up to and including each position.
        prefix_sum = torch.cumsum(ids, dim=1)                       # (B, T)
        win_tok = 7 + (prefix_sum % 40)                             # (B, T)
        logits = torch.full((B, T, self.vocab_size), -10.0,
                            device=device, dtype=torch.float32)
        logits.scatter_(2, win_tok.unsqueeze(-1), 10.0)
        # Gate: deterministic in (0,1), high so the model mostly emits but
        # occasionally thinks depending on prefix parity.
        g = torch.where((prefix_sum % 7) == 0,
                        torch.full_like(prefix_sum, 0, dtype=torch.float32) + 0.2,
                        torch.full_like(prefix_sum, 0, dtype=torch.float32) + 0.9)
        self._last_gate = g
        self._last_gate_logits = torch.logit(g.clamp(1e-6, 1 - 1e-6))
        return logits


COMMON = dict(
    thinking_token_id=THINK_ID, eos_token_id=EOS_ID,
    max_gen=10, max_think_per_step=4, emit_threshold=0.5, gate_floor=0.0,
    temperature=0.0, min_emit_before_eos=4,
)


def _rollouts_equal(a: Rollout, b: Rollout, *, check_full_prefix=True) -> bool:
    """Compare the load-bearing fields. full_ids may differ in trailing
    lock-step garbage between paths, so compare only the meaningful prefix
    up to the last recorded position."""
    if a.emit_token_ids != b.emit_token_ids:
        return False
    if a.emit_positions != b.emit_positions:
        return False
    if a.depth != b.depth:
        return False
    # log-probs: same logits + same temperature → identical (float compare).
    if len(a.emit_log_probs) != len(b.emit_log_probs):
        return False
    for x, y in zip(a.emit_log_probs, b.emit_log_probs):
        if abs(x - y) > 1e-5:
            return False
    if a.gate_decisions != b.gate_decisions:
        return False
    if a.gate_positions != b.gate_positions:
        return False
    if check_full_prefix:
        last = max(a.emit_positions) if a.emit_positions else a.prompt_len - 1
        if a.full_ids[:last + 1] != b.full_ids[:last + 1]:
            return False
    return True


def _per_problem_reference(model, tok, prompts, n, budgets, **kw):
    """Run each problem through the existing rollout_group_batched."""
    out = []
    for b, p in enumerate(prompts):
        rs = rollout_group_batched(
            model, tok, p, n_rollouts=n, total_think_budget=budgets[b], **kw)
        out.append(rs)
    return out


# ---------------------------------------------------------------------------
# (a) Equivalence: batched-across-problems == per-problem (the safety net).
# ---------------------------------------------------------------------------

def test_equivalence_constant_model_bookkeeping():
    """StubLM (constant logits) + greedy. The flat batched rollouts, regrouped
    per problem, must exactly match the per-problem rollout_group_batched."""
    m = StubLM(64, gate_value=0.9)
    tok = StubTokenizer()
    torch.manual_seed(0)
    prompts = [torch.randint(7, 50, (1, L)) for L in (4, 9, 6)]
    n = 3
    budgets = [compute_think_budget_spread(8, n, 0.0) for _ in prompts]

    ref = _per_problem_reference(m, tok, prompts, n, budgets, **COMMON)
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=n, budgets_per_problem=budgets, **COMMON)

    # Regroup flat -> per problem.
    grouped = [[] for _ in prompts]
    for pi, r in flat:
        grouped[pi].append(r)
    assert [len(g) for g in grouped] == [n] * len(prompts)
    for b in range(len(prompts)):
        for r_batched, r_ref in zip(grouped[b], ref[b]):
            assert _rollouts_equal(r_batched, r_ref), (b,)


def test_equivalence_history_model_no_leakage():
    """HistoryStubLM (content-dependent, causal) + greedy. Equivalence here
    REQUIRES that no row's logits were polluted by another row's padding —
    the model's output literally depends on the row's own prefix."""
    m = HistoryStubLM(64)
    tok = StubTokenizer()
    torch.manual_seed(3)
    prompts = [torch.randint(7, 50, (1, L)) for L in (3, 11, 7, 5)]
    n = 2
    budgets = [compute_think_budget_spread(6, n, 0.5) for _ in prompts]

    ref = _per_problem_reference(m, tok, prompts, n, budgets, **COMMON)
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=n, budgets_per_problem=budgets, **COMMON)

    grouped = [[] for _ in prompts]
    for pi, r in flat:
        grouped[pi].append(r)
    for b in range(len(prompts)):
        for r_batched, r_ref in zip(grouped[b], ref[b]):
            assert _rollouts_equal(r_batched, r_ref), \
                (b, r_batched.emit_token_ids, r_ref.emit_token_ids)


# ---------------------------------------------------------------------------
# (b) Padding leakage: a short prompt batched next to a long one yields the
#     IDENTICAL rollout to running it alone.
# ---------------------------------------------------------------------------

def test_short_prompt_next_to_long_is_unaffected():
    m = HistoryStubLM(64)
    tok = StubTokenizer()
    torch.manual_seed(11)
    short = torch.randint(7, 50, (1, 3))
    long = torch.randint(7, 50, (1, 12))
    n = 2
    budgets_one = [compute_think_budget_spread(6, n, 0.0)]
    budgets_two = [budgets_one[0], budgets_one[0]]

    # The short prompt run ALONE.
    alone = rollout_turn0_batched_across_problems(
        m, tok, [short], n_rollouts=n, budgets_per_problem=budgets_one, **COMMON)
    alone_rollouts = [r for _, r in alone]

    # The short prompt batched alongside a much longer one.
    together = rollout_turn0_batched_across_problems(
        m, tok, [short, long], n_rollouts=n,
        budgets_per_problem=budgets_two, **COMMON)
    short_in_batch = [r for pi, r in together if pi == 0]

    assert len(alone_rollouts) == len(short_in_batch) == n
    for ra, rb in zip(alone_rollouts, short_in_batch):
        assert _rollouts_equal(ra, rb), \
            (ra.emit_token_ids, rb.emit_token_ids)


def test_long_prompt_next_to_short_is_unaffected():
    """Symmetric: the LONG prompt must also be unaffected by being batched
    next to a short one (its frontier advances past the short row's pad)."""
    m = HistoryStubLM(64)
    tok = StubTokenizer()
    torch.manual_seed(13)
    short = torch.randint(7, 50, (1, 4))
    long = torch.randint(7, 50, (1, 13))
    n = 3
    budgets_one = [compute_think_budget_spread(6, n, 0.3)]
    budgets_two = [budgets_one[0], budgets_one[0]]

    alone = rollout_turn0_batched_across_problems(
        m, tok, [long], n_rollouts=n, budgets_per_problem=budgets_one, **COMMON)
    alone_rollouts = [r for _, r in alone]
    together = rollout_turn0_batched_across_problems(
        m, tok, [short, long], n_rollouts=n,
        budgets_per_problem=budgets_two, **COMMON)
    long_in_batch = [r for pi, r in together if pi == 1]
    for ra, rb in zip(alone_rollouts, long_in_batch):
        assert _rollouts_equal(ra, rb), \
            (ra.emit_token_ids, rb.emit_token_ids)


# ---------------------------------------------------------------------------
# (c) Per-problem regrouping correctness.
# ---------------------------------------------------------------------------

def test_regrouping_indices_and_counts():
    m = StubLM(64, gate_value=0.9)
    tok = StubTokenizer()
    torch.manual_seed(5)
    prompts = [torch.randint(7, 50, (1, L)) for L in (5, 8)]
    n = 4
    budgets = [compute_think_budget_spread(8, n, 0.0) for _ in prompts]
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=n, budgets_per_problem=budgets, **COMMON)
    # B*N rollouts, problem indices in [0, B), each appears exactly N times,
    # and they appear contiguously in row order (problem 0 first, then 1).
    assert len(flat) == len(prompts) * n
    pidxs = [pi for pi, _ in flat]
    assert pidxs == [0] * n + [1] * n
    # Each rollout's prompt_len matches its problem's prompt length.
    for pi, r in flat:
        assert r.prompt_len == prompts[pi].shape[1]


def test_per_row_budget_routing():
    """Distinct per-row budgets with an always-think gate: each rollout's depth
    must equal exactly its assigned budget, and the routing must be per
    (problem, row)."""
    m = StubLM(64, gate_value=0.01)   # σ < threshold → always think
    tok = StubTokenizer()
    torch.manual_seed(7)
    prompts = [torch.randint(7, 50, (1, 4)), torch.randint(7, 50, (1, 9))]
    budgets = [[1, 3, 5], [2, 4, 6]]
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=3, budgets_per_problem=budgets,
        thinking_token_id=THINK_ID, eos_token_id=EOS_ID, max_gen=16,
        max_think_per_step=100, emit_threshold=0.5, gate_floor=0.0,
        temperature=0.0, min_emit_before_eos=4)
    grouped = [[] for _ in prompts]
    for pi, r in flat:
        grouped[pi].append(r)
    for b in range(2):
        assert [r.depth for r in grouped[b]] == budgets[b]


# ---------------------------------------------------------------------------
# (d) Stochastic gate path: regrouping + budget routing still hold (decisions
#     are recorded; we don't assert token equivalence under sampling because
#     RNG interleaving differs between batched and per-problem paths).
# ---------------------------------------------------------------------------

def test_deterministic_stochastic_gate_equivalence_no_rng_divergence():
    """The inconclusive earlier A/B couldn't separate a real batching bug from
    RNG-stream divergence (the batched path draws one R-row Bernoulli, the
    per-problem path draws B separate N-row Bernoullis → different RNG
    consumption order). Here we DELIBERATELY remove the RNG variable: with
    stochastic_gate=True but every gate value placed STRICTLY OUTSIDE the
    sample range [low, high], EVERY decision is the deterministic threshold
    (no `torch.bernoulli` draw is recorded), so the two paths must produce
    BIT-IDENTICAL rollouts including gate bookkeeping. A failure here would be
    a genuine batching/cache-merge bug, not RNG noise.

    Content-dependent HistoryStubLM + greedy emit → any padding leakage in the
    full-forward joint tensor would also surface as a token mismatch."""
    m = HistoryStubLM(64)
    tok = StubTokenizer()
    torch.manual_seed(101)
    prompts = [torch.randint(7, 50, (1, L)) for L in (3, 10, 6, 4)]
    n = 3
    budgets = [compute_think_budget_spread(6, n, 0.4) for _ in prompts]
    kw = dict(
        thinking_token_id=THINK_ID, eos_token_id=EOS_ID,
        max_gen=10, max_think_per_step=4, emit_threshold=0.5, gate_floor=0.0,
        temperature=0.0, min_emit_before_eos=4,
        stochastic_gate=True,
        # HistoryStubLM gate values are 0.2 or 0.9; a [0.3, 0.85] window puts
        # BOTH outside the sample range → all decisions are deterministic.
        gate_sample_range_low=0.3, gate_sample_range_high=0.85,
    )
    ref = _per_problem_reference(m, tok, prompts, n, budgets, **kw)
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=n, budgets_per_problem=budgets, **kw)
    grouped = [[] for _ in prompts]
    for pi, r in flat:
        grouped[pi].append(r)
    for b in range(len(prompts)):
        for r_batched, r_ref in zip(grouped[b], ref[b]):
            assert _rollouts_equal(r_batched, r_ref), \
                (b, r_batched.emit_token_ids, r_ref.emit_token_ids,
                 r_batched.gate_n_decisive, r_ref.gate_n_decisive)
            # No Bernoulli draws should have been recorded (all decisive).
            assert r_batched.gate_n_sampled == 0
            assert r_batched.gate_decisions == []


def test_incremental_deterministic_gate_equivalence_no_rng_divergence():
    """Same RNG-free stochastic_gate equivalence, on the INCREMENTAL
    (prefill + merged-cache + forward_step) production decode path. This is
    the path the 3× cross-problem batching speedup runs on — proving the
    merged cache is byte-identical to per-problem caches here is the safety
    net the earlier temp>0 A/B could not provide."""
    m = FakeIncrementalLM(64)
    tok = StubTokenizer()
    torch.manual_seed(202)
    prompts = [torch.randint(7, 50, (1, L)) for L in (4, 9, 5, 7)]
    n = 2
    budgets = [compute_think_budget_spread(6, n, 0.5) for _ in prompts]
    kw = dict(
        thinking_token_id=THINK_ID, eos_token_id=EOS_ID,
        max_gen=10, max_think_per_step=4, emit_threshold=0.5, gate_floor=0.0,
        temperature=0.0, min_emit_before_eos=4,
        stochastic_gate=True,
        gate_sample_range_low=0.3, gate_sample_range_high=0.85,
    )
    ref = _per_problem_reference(m, tok, prompts, n, budgets, **kw)
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=n, budgets_per_problem=budgets, **kw)
    grouped = [[] for _ in prompts]
    for pi, r in flat:
        grouped[pi].append(r)
    for b in range(len(prompts)):
        for r_batched, r_ref in zip(grouped[b], ref[b]):
            assert _rollouts_equal(r_batched, r_ref, check_full_prefix=False), \
                (b, r_batched.emit_token_ids, r_ref.emit_token_ids)
            assert r_batched.gate_n_sampled == 0


def test_stochastic_gate_records_decisions_per_problem():
    m = StubLM(64, gate_value=0.5)   # σ=0.5 → Bernoulli draws recorded
    tok = StubTokenizer()
    torch.manual_seed(9)
    prompts = [torch.randint(7, 50, (1, 5)), torch.randint(7, 50, (1, 7))]
    n = 2
    budgets = [compute_think_budget_spread(8, n, 0.0) for _ in prompts]
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=n, budgets_per_problem=budgets,
        stochastic_gate=True, **COMMON)
    for pi, r in flat:
        assert r.gate_decisions is not None
        assert r.gate_positions is not None
        # Recorded gate positions are row-local indices into full_ids.
        for gp in r.gate_positions:
            assert r.prompt_len <= gp < len(r.full_ids)


# ---------------------------------------------------------------------------
# (e) Cache-merge helper: concat correctness on synthetic cache dicts.
# ---------------------------------------------------------------------------

def test_merge_prefill_caches_concats_recurrent_state_and_seen():
    from fla.models.utils import Cache as FLACache

    def _fake_cache(B, T, d=8, n_layers=2, with_wm=True):
        states = tuple(
            dict(recurrent_state=torch.randn(B, 2, d, d),
                 attn_state=None, conv_state=None, ffn_state=None)
            for _ in range(n_layers))
        fla = FLACache.from_legacy_cache(states, seen_tokens=T)
        wm = None
        if with_wm:
            wm = {
                "gate": torch.rand(B, T),
                "value": torch.randn(B, T, d),
                "pos": torch.arange(T).unsqueeze(0).expand(B, T).contiguous(),
                "tok": torch.randint(0, 50, (B, T)),
            }
        return {"fla_cache": fla, "seen": T, "wm_buf": wm,
                "lagged_sources": None, "think_run_len": None}

    c0 = _fake_cache(2, 4)
    c1 = _fake_cache(2, 9)   # different prompt length
    merged = _merge_prefill_caches([c0, c1], [2, 2], [4, 9])
    # recurrent_state concatenated along batch dim (2 + 2 = 4 rows).
    assert merged["fla_cache"][0]["recurrent_state"].shape[0] == 4
    # seen_per_row reflects each problem's prompt length, tiled to N rows.
    assert merged["seen_per_row"].tolist() == [4, 4, 9, 9]
    # WM buffer left-padded to the common max width (9).
    assert merged["wm_buf"]["gate"].shape == (4, 9)
    # The first problem's rows (prompt_len 4) are LEFT-padded: their first
    # (9-4)=5 pos slots are the +SENTINEL, the real positions sit at the end.
    pos0 = merged["wm_buf"]["pos"][0]
    assert (pos0[:5] >= 10 ** 8).all()        # pad slots = sentinel
    assert pos0[5:].tolist() == [0, 1, 2, 3]  # real buffer positions
    # The long problem's rows are unpadded.
    pos2 = merged["wm_buf"]["pos"][2]
    assert pos2.tolist() == list(range(9))
    # Pad slots carry gate=0 so the causal-masked top-K never weights them.
    assert (merged["wm_buf"]["gate"][0][:5] == 0).all()


# ---------------------------------------------------------------------------
# (f) Incremental decode path (prefill + forward_step). A fake model with a
#     REAL FLA Cache and a content-dependent recurrent state proves the
#     cross-problem batched incremental path == per-problem rollout_group_batched
#     (the production path, which can't run a DeltaNet on CPU).
# ---------------------------------------------------------------------------

class FakeIncrementalLM(nn.Module):
    """A tiny causal model with prefill/forward_step over a REAL FLA Cache.

    The recurrent "state" is the running sum of token ids per row (stored in
    the cache's recurrent_state, 1 layer). Logits + gate are deterministic
    functions of that state, so:
      * the output depends on the row's OWN history (cross-row pollution would
        change it → leakage test);
      * greedy decode is reproducible (token = 7 + state % 40).
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Linear(1, 1)
        self._last_gate = None
        self._last_gate_logits = None

    def _logits_and_gate(self, state):   # state: (B,) running sum
        B = state.shape[0]
        win = (7 + (state % 40)).long()
        logits = torch.full((B, 1, self.vocab_size), -10.0, dtype=torch.float32)
        logits.scatter_(2, win.view(B, 1, 1), 10.0)
        g = torch.where((state % 7) == 0,
                        torch.full_like(state, 0.2, dtype=torch.float32),
                        torch.full_like(state, 0.9, dtype=torch.float32))
        self._last_gate = g.view(B, 1)
        self._last_gate_logits = torch.logit(self._last_gate.clamp(1e-6, 1 - 1e-6))
        return logits

    def prefill(self, ids):
        from fla.models.utils import Cache as FLACache
        B, T = ids.shape
        state = ids.sum(dim=1).float()                 # (B,) running sum
        # Stash state as recurrent_state in a real FLA Cache (shape (B,1,1)).
        fla = FLACache.from_legacy_cache(
            (dict(recurrent_state=state.view(B, 1, 1),
                  attn_state=None, conv_state=None, ffn_state=None),),
            seen_tokens=T)
        # All-position logits so callers reading [:, -1] get the last state.
        logits = self._logits_and_gate(state)          # (B,1,V) at last pos
        # prefill returns (cache, last_logits) where last_logits is (B,T,V);
        # only [:, -1] is used, so a (B,1,V) is broadcast-safe via expand.
        last_logits = logits.expand(B, T, self.vocab_size)
        cache = {"fla_cache": fla, "seen": int(T), "wm_buf": None,
                 "lagged_sources": None, "think_run_len": None}
        return cache, last_logits

    def forward_step(self, input_id, cache):
        if input_id.dim() == 1:
            input_id = input_id.unsqueeze(-1)
        B = input_id.shape[0]
        state = cache["fla_cache"][0]["recurrent_state"].view(B)
        state = state + input_id.view(B).float()
        cache["fla_cache"][0]["recurrent_state"] = state.view(B, 1, 1)
        cache["seen"] = int(cache["seen"]) + 1
        if cache.get("seen_per_row") is not None:
            cache["seen_per_row"] = cache["seen_per_row"] + 1
        logits = self._logits_and_gate(state)          # (B,1,V)
        return logits, cache


def test_incremental_path_equivalence_with_per_problem():
    """The cross-problem batched INCREMENTAL path (per-problem prefill +
    concat cache + joint forward_step) must equal running each problem through
    rollout_group_batched on the SAME incremental model. Content-dependent
    recurrence + greedy → any leakage or mis-merged cache would diverge."""
    m = FakeIncrementalLM(64)
    tok = StubTokenizer()
    torch.manual_seed(21)
    prompts = [torch.randint(7, 50, (1, L)) for L in (3, 8, 5)]
    n = 2
    budgets = [compute_think_budget_spread(6, n, 0.5) for _ in prompts]

    ref = _per_problem_reference(m, tok, prompts, n, budgets, **COMMON)
    flat = rollout_turn0_batched_across_problems(
        m, tok, prompts, n_rollouts=n, budgets_per_problem=budgets, **COMMON)
    grouped = [[] for _ in prompts]
    for pi, r in flat:
        grouped[pi].append(r)
    for b in range(len(prompts)):
        for r_batched, r_ref in zip(grouped[b], ref[b]):
            assert _rollouts_equal(r_batched, r_ref, check_full_prefix=False), \
                (b, r_batched.emit_token_ids, r_ref.emit_token_ids,
                 r_batched.depth, r_ref.depth)


def test_incremental_short_next_to_long_no_leakage():
    m = FakeIncrementalLM(64)
    tok = StubTokenizer()
    torch.manual_seed(23)
    short = torch.randint(7, 50, (1, 3))
    long = torch.randint(7, 50, (1, 11))
    n = 2
    budgets_one = [compute_think_budget_spread(6, n, 0.0)]
    alone = rollout_turn0_batched_across_problems(
        m, tok, [short], n_rollouts=n, budgets_per_problem=budgets_one, **COMMON)
    together = rollout_turn0_batched_across_problems(
        m, tok, [short, long], n_rollouts=n,
        budgets_per_problem=[budgets_one[0], budgets_one[0]], **COMMON)
    short_alone = [r for _, r in alone]
    short_batch = [r for pi, r in together if pi == 0]
    for ra, rb in zip(short_alone, short_batch):
        assert _rollouts_equal(ra, rb, check_full_prefix=False), \
            (ra.emit_token_ids, rb.emit_token_ids)


def test_merge_prefill_caches_no_wm():
    from fla.models.utils import Cache as FLACache

    def _fake_cache(B, T, d=8, n_layers=2):
        states = tuple(
            dict(recurrent_state=torch.randn(B, 2, d, d),
                 attn_state=None, conv_state=None, ffn_state=None)
            for _ in range(n_layers))
        fla = FLACache.from_legacy_cache(states, seen_tokens=T)
        return {"fla_cache": fla, "seen": T, "wm_buf": None,
                "lagged_sources": None, "think_run_len": None}

    merged = _merge_prefill_caches(
        [_fake_cache(3, 5), _fake_cache(3, 5)], [3, 3], [5, 5])
    assert merged["wm_buf"] is None
    assert merged["fla_cache"][1]["recurrent_state"].shape[0] == 6
    assert merged["seen_per_row"].tolist() == [5, 5, 5, 5, 5, 5]
