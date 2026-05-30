"""Probe: is "thinking helps here" a learnable per-position signal?

ONE question, on a pretrained checkpoint:

  At a real-text position t, does feeding the trunk K state-readonly
  [THINKING] tokens improve the next-token prediction? And — the key
  number — is that "thinking helps" label LINEARLY DECODABLE from the
  trunk's final hidden state h_t?

If yes (high held-out AUC of a linear probe h_t -> y_t), then the
output gate — which is itself a linear head on the same h_t — *can*
learn to fire think exactly where it helps, i.e. a gate-calibration
BCE loss has a real, learnable target. If no, no post-pretrain
calibration loss can make the gate selective and we should stop trying.

Mechanism (mirrors the planned gate-calibration aux loss):
  1. Baseline forward over a clean real-text prefix -> per-position
     logits + final hidden h + current gate sigma.
     lp0_t = log p(true_{t+1} | prefix_0..t)              [no think]
  2. Per sampled position t, an EXTRA forward over
        [prefix_0..t, THINK_ID * K]
     reads the next-token logp at the LAST think slot:
     lpK_t = log p(true_{t+1} | prefix_0..t, K thinks)    [post-think]
     The think tokens are state-readonly (beta=0 at think positions,
     installed by build_model_from_ckpt when the ckpt has
     state_readonly_at_think) so they READ the recurrent state but
     never corrupt it.
  3. Delta_t = lpK_t - lp0_t ;  y_t = 1{Delta_t > margin}.
  4. Capture h_t (detached) + current gate sigma_t (snapshotted from
     the BASELINE forward, before any extra forward clobbers
     model._last_gate_logits).

Reported:
  - SIGNAL: %positions with Delta>margin, mean/median Delta, histogram.
  - LEARNABILITY (THE KEY NUMBER): logistic-regression linear probe
    h_t -> y_t, held-out AUC on a test split.
  - CURRENT-GATE baseline: AUC of the existing sigma_t against y_t
    (expected low; it's entropy-trained, not calibration-trained).
  - Optional (--fit_steps>0): freeze everything but the gate head,
    train BCE(gate_logit, y_t) a few hundred steps over fresh batches,
    report sigma-vs-y AUC + mean sigma before/after.

This is a DIAGNOSTIC. It allocates GPU memory only if CUDA is visible
to it; pin to the FREE gpu with CUDA_VISIBLE_DEVICES=1.

Usage (the integrator runs this on GPU 1):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      experiments/probe_gate_calibration.py \
      --ckpt checkpoints/pretrain_v8_wide_step5723_tok1500250112.pt \
      --n_positions 4000 --K 4

  # also run the gate-head-only calibration fit:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
      experiments/probe_gate_calibration.py --ckpt <ckpt> \
      --n_positions 4000 --K 4 --fit_steps 400
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F


# pad id used to build the batched extra-forward. MUST differ from the
# thinking token id (pad-as-think silently triggers state_readonly /
# mem_write_only_at_think on padding positions, corrupting the
# after-forward's recurrent state). See CLAUDE.md "Pad-id MUST differ".
PAD_ID = 0


# ---------------------------------------------------------------------------
# Forward helpers — usable against any object exposing the TinyLM contract
# (model.forward(input_ids, return_hidden=, ...) + model._last_gate_logits).
# Kept model-agnostic so the CPU test can pass a tiny fake model.
# ---------------------------------------------------------------------------

def _unwrap_logits_hidden(out):
    """forward(return_hidden=True) returns (logits, h) in eval mode, but in
    training mode TinyLM may append a gist-loss scalar -> (logits, h, loss).
    We always run the model in eval(), but be defensive: take the first two
    tensor outputs as (logits, hidden)."""
    if isinstance(out, torch.Tensor):
        raise ValueError("expected (logits, hidden) tuple from return_hidden=True")
    logits, hidden = out[0], out[1]
    return logits, hidden


@torch.no_grad()
def baseline_forward(model, input_ids: torch.Tensor):
    """One clean forward over real tokens. Returns (logits_f32, hidden_f32,
    gate_logits_f32) — every tensor snapshotted/detached/cast to fp32 so the
    later extra forward (which overwrites model._last_gate_logits) cannot
    corrupt them.

    logits: (B, T, V)  hidden: (B, T, d)  gate_logits: (B, T) or None
    """
    out = model(input_ids, return_hidden=True)
    logits, hidden = _unwrap_logits_hidden(out)
    # Cast to fp32 BEFORE any log_softmax (bf16 autocast otherwise).
    logits = logits.float()
    hidden = hidden.float().detach()
    gate_logits = getattr(model, "_last_gate_logits", None)
    if gate_logits is not None:
        gate_logits = gate_logits.detach().float().clone()
    return logits, hidden, gate_logits


@torch.no_grad()
def post_think_logp(model, prefixes: torch.Tensor, true_next: torch.Tensor,
                    K: int, thinking_token_id: int):
    """Batched post-think next-token logp.

    `prefixes` (N, Lmax) is LEFT-padded with PAD_ID so every row has the same
    length; the *real* prefix for row i ends at its last column (column
    Lmax-1). We append K think tokens to every row and read the logp of the
    true next token at the LAST think slot.

    Left-padding choice: DeltaNet is a causal linear RNN, so leading PAD_ID
    tokens prepend benign state before the real prefix. PAD_ID != think id is
    asserted by the caller, so the appended think tokens are the ONLY
    state-readonly positions. Returns lpK: (N,).
    """
    assert thinking_token_id != PAD_ID, (
        "thinking_token_id must differ from PAD_ID (pad-as-think corrupts "
        "the state-readonly mask)")
    N, Lmax = prefixes.shape
    think_block = torch.full((N, K), int(thinking_token_id),
                             dtype=prefixes.dtype, device=prefixes.device)
    seq = torch.cat([prefixes, think_block], dim=1)          # (N, Lmax+K)
    out = model(seq, return_hidden=False)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    logits = logits.float()
    last_think = Lmax + K - 1                                 # last think slot
    lp = F.log_softmax(logits[:, last_think, :], dim=-1)      # (N, V)
    return lp.gather(1, true_next.view(-1, 1)).squeeze(1)     # (N,)


# ---------------------------------------------------------------------------
# Position sampling: from a clean (no-think) batch, pick positions t whose
# next token (t+1) is a normal real token (not think/pad/eos), build the
# baseline lp0, the post-think lpK, capture h_t + gate sigma_t.
# ---------------------------------------------------------------------------

def _clean_position_mask(input_ids: torch.Tensor, targets: torch.Tensor,
                         thinking_token_id: int, eos_id: int) -> torch.Tensor:
    """(B, T) bool: positions t that are valid probe positions.

    Valid iff:
      - target_t (== input_{t+1}) is not -100 (loss-masked),
      - target_t is a real token: not think, not pad, not eos,
      - input_t is itself real (not think/pad) so h_t is a normal hidden.
    """
    valid = targets != -100
    valid &= targets != int(thinking_token_id)
    valid &= targets != PAD_ID
    valid &= targets != int(eos_id)
    valid &= input_ids != int(thinking_token_id)
    valid &= input_ids != PAD_ID
    return valid


@torch.no_grad()
def collect_one_batch(model, input_ids: torch.Tensor, targets: torch.Tensor,
                      *, K: int, thinking_token_id: int, eos_id: int,
                      max_positions_per_batch: int, margin: float,
                      think_batch: int, device, generator):
    """Run the baseline + extra forwards for ONE (B,T) batch.

    Returns dict of CPU tensors: h (M,d), lp0 (M,), lpK (M,), delta (M,),
    y (M,), gate_sigma (M,) — M = number of sampled positions this batch.
    """
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    B, T = input_ids.shape

    logits, hidden, gate_logits = baseline_forward(model, input_ids)
    logp_all = F.log_softmax(logits, dim=-1)                  # (B, T, V)

    valid = _clean_position_mask(input_ids, targets, thinking_token_id, eos_id)
    # Position t needs a prefix [0..t]; t ranges over [0, T-2] (need t+1 target).
    valid[:, T - 1] = False
    bsel, tsel = torch.nonzero(valid, as_tuple=True)          # (P,), (P,)
    P = bsel.numel()
    if P == 0:
        return None
    if P > max_positions_per_batch:
        perm = torch.randperm(P, generator=generator, device=bsel.device
                              )[:max_positions_per_batch]
        bsel, tsel = bsel[perm], tsel[perm]
        P = bsel.numel()

    true_next = targets[bsel, tsel]                           # (P,) the t+1 token
    lp0 = logp_all[bsel, tsel, :].gather(1, true_next.view(-1, 1)).squeeze(1)
    h_t = hidden[bsel, tsel, :]                               # (P, d)
    if gate_logits is not None:
        sigma_t = torch.sigmoid(gate_logits[bsel, tsel])      # (P,)
    else:
        sigma_t = torch.full((P,), float("nan"), device=device)

    # Build per-position left-padded prefixes [0..t] and run the post-think
    # forward in chunks (think_batch positions at a time) to bound memory.
    Lmax = int(tsel.max().item()) + 1                         # longest prefix
    lpK = torch.empty(P, device=device)
    for s in range(0, P, think_batch):
        e = min(s + think_batch, P)
        rows = []
        for j in range(s, e):
            b = int(bsel[j].item())
            t = int(tsel[j].item())
            pref = input_ids[b, : t + 1]                      # (t+1,)
            padlen = Lmax - pref.numel()
            if padlen > 0:
                pad = torch.full((padlen,), PAD_ID, dtype=pref.dtype,
                                 device=device)
                pref = torch.cat([pad, pref], dim=0)          # LEFT pad
            rows.append(pref)
        chunk_pref = torch.stack(rows, dim=0)                 # (e-s, Lmax)
        chunk_true = true_next[s:e]
        lpK[s:e] = post_think_logp(model, chunk_pref, chunk_true, K,
                                   thinking_token_id)

    delta = lpK - lp0
    y = (delta > margin).float()
    return dict(
        h=h_t.cpu(), lp0=lp0.cpu(), lpK=lpK.cpu(), delta=delta.cpu(),
        y=y.cpu(), gate_sigma=sigma_t.cpu(),
    )


# ---------------------------------------------------------------------------
# AUC + linear probe (no sklearn dependency for the AUC itself so the test
# can assert the math; sklearn used only for the LogisticRegression fit).
# ---------------------------------------------------------------------------

def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """ROC AUC via the Mann-Whitney U statistic (rank-based, ties averaged).

    scores, labels: 1-D tensors; labels in {0,1}. Returns AUC in [0,1], or
    nan if a class is absent."""
    scores = scores.flatten().double()
    labels = labels.flatten().double()
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks (1-based) handling ties.
    order = torch.argsort(scores)
    s_sorted = scores[order]
    ranks = torch.empty_like(s_sorted)
    i = 0
    n = s_sorted.numel()
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        ranks[i:j + 1] = avg_rank
        i = j + 1
    rank_of = torch.empty_like(ranks)
    rank_of[order] = ranks
    sum_ranks_pos = float(rank_of[labels == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def linear_probe_auc(h_train, y_train, h_test, y_test):
    """Fit logistic-regression h -> y on train, return held-out test AUC.

    Standardises features (fit on train). Returns (test_auc, train_auc,
    test_scores)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    Xtr = h_train.numpy().astype("float64")
    Xte = h_test.numpy().astype("float64")
    ytr = y_train.numpy().astype("int64")
    yte = y_test.numpy().astype("int64")
    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    s_te = torch.from_numpy(clf.decision_function(Xte).astype("float64"))
    s_tr = torch.from_numpy(clf.decision_function(Xtr).astype("float64"))
    test_auc = roc_auc(s_te, torch.from_numpy(yte))
    train_auc = roc_auc(s_tr, torch.from_numpy(ytr))
    return test_auc, train_auc, s_te


# ---------------------------------------------------------------------------
# Optional: gate-head-only calibration fit (does the ACTUAL gate head learn
# the y_t target if we train it with BCE while freezing everything else?).
# ---------------------------------------------------------------------------

def gate_head_calibration_fit(model, h_all: torch.Tensor, y_all: torch.Tensor,
                              *, fit_steps: int, device, batch: int = 256,
                              lr: float = 1e-3):
    """Freeze everything except model.gate_head; train BCE(gate_head(h), y)
    on the captured (h, y) pairs. Returns (sigma_before, sigma_after) as the
    gate's mean sigmoid output on the full set before/after.

    Operates DIRECTLY on captured hidden states h (the same h the gate head
    sees in forward), so it isolates "can the linear gate head fit y" without
    re-running the trunk.
    """
    gate_head = model.gate_head
    h_all = h_all.to(device)
    y_all = y_all.to(device)

    @torch.no_grad()
    def mean_sigma():
        return float(torch.sigmoid(gate_head(h_all).squeeze(-1)).mean())

    sigma_before = mean_sigma()
    auc_before = roc_auc(
        torch.sigmoid(gate_head(h_all).squeeze(-1)).detach().cpu(),
        y_all.cpu())

    opt = torch.optim.Adam(gate_head.parameters(), lr=lr)
    N = h_all.shape[0]
    gen = torch.Generator(device=device).manual_seed(0)
    for step in range(fit_steps):
        idx = torch.randint(0, N, (min(batch, N),), generator=gen, device=device)
        logit = gate_head(h_all[idx]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logit, y_all[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    sigma_after = mean_sigma()
    auc_after = roc_auc(
        torch.sigmoid(gate_head(h_all).squeeze(-1)).detach().cpu(),
        y_all.cpu())
    return sigma_before, sigma_after, auc_before, auc_after


# ---------------------------------------------------------------------------
# Data + run driver (only invoked from __main__; the test exercises the
# helpers directly with a tiny model so it never touches HF / datasets).
# ---------------------------------------------------------------------------

def _build_stream(cfg, *, block_size: int, base_seed: int):
    """MixedSourceStream over the v4 mix with think-burst injection OFF
    (think_burst_prob=0.0 -> every position is a clean real token)."""
    from transformers import AutoTokenizer
    from experiments.data_mix import MixedSourceStream, load_sources_from_yaml

    tok = AutoTokenizer.from_pretrained(
        cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    sources = load_sources_from_yaml("configs/pretrain_mix_v4.yaml")
    thinking_token_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    stream = MixedSourceStream(
        sources=sources, tokenizer=tok, block_size=block_size,
        thinking_token_id=thinking_token_id,
        think_burst_prob=0.0,           # CLEAN real tokens only
        base_seed=base_seed,
    )
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else (
        tok.bos_token_id if tok.bos_token_id is not None else 0)
    return stream, tok, thinking_token_id, eos_id


def run_probe(ckpt_path: str, *, n_positions: int, K: int, margin: float,
              block_size: int, batch: int, max_positions_per_batch: int,
              think_batch: int, fit_steps: int, test_frac: float,
              base_seed: int):
    from experiments.eval_bracket_structure import build_model_from_ckpt

    print(f"[probe] loading ckpt: {ckpt_path}")
    model, cfg = build_model_from_ckpt(ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    thinking_token_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    assert thinking_token_id != PAD_ID, (
        f"thinking_token_id ({thinking_token_id}) must differ from PAD_ID "
        f"({PAD_ID})")
    print(f"[probe] device={device}  thinking_token_id={thinking_token_id}  "
          f"state_readonly_at_think={getattr(model, 'state_readonly_at_think', '?')}  "
          f"output_gate={getattr(model, 'output_gate', '?')}")

    stream, tok, _, eos_id = _build_stream(cfg, block_size=block_size,
                                           base_seed=base_seed)
    it = iter(stream)
    gen = torch.Generator(device=device).manual_seed(base_seed)

    chunks = {k: [] for k in ("h", "lp0", "lpK", "delta", "y", "gate_sigma")}
    n_have = 0
    n_batches = 0
    while n_have < n_positions:
        rows_in, rows_tg = [], []
        for _ in range(batch):
            x, y, *_ = next(it)
            rows_in.append(x); rows_tg.append(y)
        input_ids = torch.stack(rows_in, 0)
        targets = torch.stack(rows_tg, 0)
        res = collect_one_batch(
            model, input_ids, targets, K=K,
            thinking_token_id=thinking_token_id, eos_id=eos_id,
            max_positions_per_batch=max_positions_per_batch, margin=margin,
            think_batch=think_batch, device=device, generator=gen)
        n_batches += 1
        if res is None:
            continue
        for k in chunks:
            chunks[k].append(res[k])
        n_have += res["y"].numel()
        print(f"[probe] batch {n_batches}: +{res['y'].numel()} positions "
              f"(total {n_have}/{n_positions})")

    data = {k: torch.cat(v, 0)[:n_positions] for k, v in chunks.items()}
    _report(data, cfg=cfg, K=K, margin=margin, test_frac=test_frac)

    if fit_steps > 0 and getattr(model, "output_gate", False):
        print("\n[probe] === gate-head-only calibration fit ===")
        sb, sa, ab, aa = gate_head_calibration_fit(
            model, data["h"], data["y"], fit_steps=fit_steps, device=device)
        print(f"  gate mean sigma:  before={sb:.4f}  after={sa:.4f}")
        print(f"  gate sigma-vs-y AUC:  before={ab:.4f}  after={aa:.4f}")
    return data


def _report(data, *, cfg, K, margin, test_frac):
    delta = data["delta"]; y = data["y"]; h = data["h"]
    sigma = data["gate_sigma"]
    N = y.numel()
    print("\n" + "=" * 64)
    print(f"GATE-CALIBRATION PROBE  (N={N}, K={K}, margin={margin})")
    print("=" * 64)

    # --- SIGNAL ---
    frac_help = float((delta > margin).float().mean())
    print(f"\n[signal] %positions thinking helps (Delta>{margin}): "
          f"{frac_help*100:.1f}%")
    print(f"[signal] Delta logp  mean={float(delta.mean()):+.4f}  "
          f"median={float(delta.median()):+.4f}  "
          f"std={float(delta.std()):.4f}")
    edges = [-1e9, -1.0, -0.25, -0.05, 0.0, 0.05, 0.25, 1.0, 1e9]
    counts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        counts.append(int(((delta >= lo) & (delta < hi)).sum()))
    labels = ["<-1", "-1..-.25", "-.25..-.05", "-.05..0",
              "0..+.05", "+.05..+.25", "+.25..+1", ">+1"]
    print("[signal] Delta histogram:")
    for lab, c in zip(labels, counts):
        print(f"    {lab:>12s}: {c:5d}  {'#'*int(40*c/max(1,N))}")

    # --- LEARNABILITY (the key number) ---
    n_test = max(1, int(N * test_frac))
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1234))
    te, tr = perm[:n_test], perm[n_test:]
    if float(y[tr].sum()) in (0.0, float(tr.numel())) or \
       float(y[te].sum()) in (0.0, float(te.numel())):
        print("\n[learnability] DEGENERATE: a split has a single class — "
              "cannot fit a probe. (Adjust --margin or sample more.)")
        probe_auc = float("nan")
    else:
        probe_auc, train_auc, _ = linear_probe_auc(h[tr], y[tr], h[te], y[te])
        print(f"\n[learnability] linear-probe (h_t -> y_t) HELD-OUT AUC = "
              f"{probe_auc:.4f}   (train AUC {train_auc:.4f})")
        print("    >> THE KEY NUMBER. >~0.65 => the gate (also linear on h_t)")
        print("       can learn 'thinking helps here'. ~0.5 => not decodable.")

    # --- CURRENT-GATE baseline ---
    if not torch.isnan(sigma).all():
        gate_auc = roc_auc(sigma, y)
        print(f"\n[current-gate] existing sigma_t vs y_t AUC = {gate_auc:.4f}  "
              f"(mean sigma={float(sigma.mean()):.4f})")
        print("    (expected ~0.5 — the gate is entropy-trained, not "
              "calibration-trained)")
    else:
        print("\n[current-gate] model has no output gate — skipped.")
    print("=" * 64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_positions", type=int, default=4000)
    p.add_argument("--K", type=int, default=4,
                   help="number of think tokens appended in the extra forward")
    p.add_argument("--margin", type=float, default=0.0,
                   help="y_t = 1{Delta_logp > margin}")
    p.add_argument("--block_size", type=int, default=512,
                   help="prefix sequence length for the baseline forward")
    p.add_argument("--batch", type=int, default=4,
                   help="rows per baseline forward")
    p.add_argument("--max_positions_per_batch", type=int, default=256,
                   help="cap sampled positions per (B,T) batch")
    p.add_argument("--think_batch", type=int, default=64,
                   help="positions per extra (post-think) forward chunk")
    p.add_argument("--fit_steps", type=int, default=0,
                   help=">0 also runs the gate-head-only BCE calibration fit")
    p.add_argument("--test_frac", type=float, default=0.3)
    p.add_argument("--base_seed", type=int, default=0)
    args = p.parse_args()

    run_probe(args.ckpt, n_positions=args.n_positions, K=args.K,
              margin=args.margin, block_size=args.block_size, batch=args.batch,
              max_positions_per_batch=args.max_positions_per_batch,
              think_batch=args.think_batch, fit_steps=args.fit_steps,
              test_frac=args.test_frac, base_seed=args.base_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
