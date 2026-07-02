"""CPU-only correctness tests for the depth-via-iteration synthetic tasks.

These check that the homogeneous / heterogeneous chase batches encode a
sequence whose embedded program/tables actually compose to the returned answer
(no model / CUDA needed).  The decisive learning results live in the runner
report, not here.
"""
import torch

from experiments.depth_via_iteration import (
    make_hetero_chase_batch, make_multitable_chase_batch, hetero_layout,
    get_baked_ops, task_meta,
)
from experiments.latent_think import make_pointer_chase_batch


def test_homo_pointer_chase_answer_is_fK():
    g = torch.Generator().manual_seed(0)
    N, K = 7, 4
    ids, ans, chain, vocab = make_pointer_chase_batch(8, N, K, "cpu", g)
    assert chain.shape == (8, K)
    assert torch.equal(ans, chain[:, -1])
    assert (ans < N).all()


def test_hetero_baked_program_composes():
    N, K, L = 8, 5, 3
    perms = get_baked_ops(N, L)
    OP_BASE, QUERY, THINK, PAD, vocab = hetero_layout(N, L)
    g = torch.Generator().manual_seed(1)
    ids, ans, chain, prog, vocab = make_hetero_chase_batch(
        6, N, K, L, "cpu", g, fold=False)
    # s is the LAST token; apply baked ops per program
    for b in range(6):
        s = ids[b, -1].item()
        x = s
        for r in range(K):
            x = perms[prog[b, r].item(), x].item()
        assert x == ans[b].item()


def test_hetero_fold_puts_s_first():
    N, K, L = 6, 4, 2
    perms = get_baked_ops(N, L)
    g = torch.Generator().manual_seed(2)
    ids, ans, chain, prog, vocab = make_hetero_chase_batch(
        5, N, K, L, "cpu", g, fold=True)
    for b in range(5):
        s = ids[b, 0].item()          # s FIRST in the fold layout
        x = s
        for r in range(K):
            x = perms[prog[b, r].item(), x].item()
        assert x == ans[b].item()


def test_multitable_recall_program_composes():
    N, K, L = 8, 5, 2
    OP_BASE, QUERY, THINK, PAD, vocab = hetero_layout(N, L)
    g = torch.Generator().manual_seed(3)
    ids, ans, chain, prog, vocab = make_multitable_chase_batch(
        6, N, K, L, "cpu", g)
    # parse the per-example recalled tables back out and re-derive
    for b in range(6):
        seq = ids[b].tolist()
        pos = 0
        f = {}
        for j in range(L):
            assert seq[pos] == OP_BASE + j
            pos += 1
            t = {}
            for _ in range(N):
                t[seq[pos]] = seq[pos + 1]
                pos += 2
            f[j] = t
        assert seq[pos] == QUERY
        pos += 1
        pg = [seq[pos + r] - OP_BASE for r in range(K)]
        pos += K
        s = seq[pos]
        x = s
        for r in range(K):
            x = f[pg[r]][x]
        assert x == ans[b].item()


def test_homo_mt_control_is_same_op_and_length_matched():
    N, K, L = 8, 5, 2
    from experiments.depth_via_iteration import make_multitable_chase_batch
    g = torch.Generator().manual_seed(9)
    ids_h, ans, _c, prog, _v = make_multitable_chase_batch(
        6, N, K, L, "cpu", g, homogeneous=True)
    # every program row is a single repeated op
    for b in range(6):
        assert len(set(prog[b].tolist())) == 1
    # same sequence length as the heterogeneous multi-table task
    ids_ht, *_ = make_multitable_chase_batch(
        6, N, K, L, "cpu", torch.Generator().manual_seed(9), homogeneous=False)
    assert ids_h.shape[1] == ids_ht.shape[1]


def test_task_meta_sizes_max_T_for_eval_K():
    # hetero positional length grows with K -> max_T must cover eval_K_max
    _, _, mt = task_meta("hetero", N=8, K_max=8, L_ops=3)
    assert mt >= 8 + 2 + 1
    _, _, mt_mt = task_meta("hetero_mt", N=8, K_max=8, L_ops=3)
    assert mt_mt >= 3 * (2 * 8 + 1) + 8
