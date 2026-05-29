"""
Thinking ⊗ Memory unification — S1 validation (2026-05-29, THINKING_MEMORY_PLAN.md).

Tests ALL the architecture bets TOGETHER on a memory-required reasoning task:
  FiLM (depth via cross-layer feedback) + WorkingMemory (dynamic RAG) +
  latent thinking (retrieval-as-input) + state-readonly (β=0, protect bindings).

Task: pointer-chase f^K(start) with a LARGE table (N nodes) presented in-context.
For large N the bounded DeltaNet recurrent state saturates, so the fact table
must be retrieved from WM; computing f^K needs K sequential hops (depth), each of
which must RETRIEVE f(current) from memory. This is the task that only
"retrieve-while-thinking" can solve.

Unified think step (mode="both"):
  - read WM with the current thought  → injection
  - feed (think_embed + α·injection) as the next think-slot input  (retrieval-as-input)
  - β=0 at think (state-readonly): protect the recurrent bindings
  - WM write-at-think is automatic: the think-slot hidden is a write candidate,
    so later think steps can retrieve earlier thoughts (scratchpad)
  - FiLM carries the running computation across think steps (D3)

Three-way ablation (one trained model):
  - both         : retrieve-while-thinking (R=K, WM read fed as input)
  - think_only   : R=K latent thinking but WM read DISABLED (hidden fed back) →
                   thinking over the saturating recurrent state alone
  - single_read  : R=1 (one retrieval, cannot chain)

Usage:
  PYTHONPATH=. .venv/bin/python experiments/latent_mem.py --smoke
  PYTHONPATH=. .venv/bin/python experiments/latent_mem.py --N 40 --K 4 --steps 6000
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM
from experiments.latent_think import make_pointer_chase_batch


def make_distractor_chase_batch(B, N, K, D, device="cuda", generator=None):
    """Pointer-chase with a DISTRACTOR span: the bounded recurrent state decays
    over D filler tokens (forgets the early table), but WM — which write-selects
    the pair positions — retains it. Forces memory at small (WM-storable) N.

    Sequence: [i0,f(i0),...,i(N-1),f(i(N-1)), <D random node distractors>, QUERY, s].
    Returns (ids, answers, chain (B,K), vocab)."""
    QUERY = N
    vocab = N + 3
    g = generator
    perm = torch.rand(B, N, generator=g).argsort(dim=1)
    order = torch.rand(B, N, generator=g).argsort(dim=1)
    tgt = perm.gather(1, order)
    pairs = torch.stack([order, tgt], dim=2).reshape(B, 2 * N)
    distract = torch.randint(0, N, (B, D), generator=g) if D > 0 \
        else torch.zeros(B, 0, dtype=torch.long)
    s = torch.randint(0, N, (B, 1), generator=g)
    x = s.clone()
    chain_cols = []
    for _ in range(K):
        x = perm.gather(1, x)
        chain_cols.append(x.clone())
    chain = torch.cat(chain_cols, dim=1)
    answers = chain[:, -1]
    query_col = torch.full((B, 1), QUERY, dtype=torch.long)
    ids = torch.cat([pairs, distract, query_col, s], dim=1)
    return ids.to(device), answers.to(device), chain.to(device), vocab


def build_model(vocab, N, d_model, n_layers, n_heads, d_head, max_T,
                mem_size, use_film=True, use_mem=True, film_self_k=3,
                device="cuda"):
    thinking_id = N + 1
    kw = {}
    if use_film:
        kw = dict(feedback_mode="film",
                  feedback_pairs=((1, n_layers - 1),),
                  feedback_self_k=int(film_self_k))
    model = TinyLM(
        vocab_size=vocab, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_head=d_head, attention_cls=DeltaNetAttention,
        max_T=max_T,
        use_memory=use_mem,
        mem_size=mem_size,
        thinking_token_id=thinking_id,
        state_readonly_at_think=True,
        activation_checkpointing=False,
        **kw,
    ).to(device)
    return model, thinking_id


def think_forward_mem(model, base_ids, R, thinking_id, mode="both",
                      alpha=0.5, return_steps=False):
    """Unified retrieve-while-thinking forward. Returns final answer logits,
    or (B, R, vocab) per-step logits if return_steps.

    mode: 'both' (retrieval-as-input), 'think_only' (hidden feedback, WM read
    off), 'single_read' (handled by caller via R=1 + 'both')."""
    B, Lb = base_ids.shape
    device = base_ids.device
    think_tok = torch.full((B, 1), int(thinking_id), dtype=torch.long, device=device)
    think_emb = model.embed(think_tok)                    # (B,1,d) baseline
    emb = model.embed(base_ids)                           # (B, Lb, d)
    ids = base_ids

    # mem_read_mask: 'both' reads at think positions (default → None lets the
    # model use input_ids==thinking_id); 'think_only' forces NO read.
    # Optionally bypass the FiLM multipass during think steps (rely on
    # hidden-feedback for think-depth; FiLM gives per-token depth and deploys
    # single-forward anyway). Tests whether the FiLM multipass × cross-forward
    # retrieval chaining is the FiLM×WM breaker (D9).
    bypass = getattr(model, "_think_film_bypass", False)

    def fwd(ids_, emb_):
        if bypass:
            model._film_bypass = True
        if mode == "think_only":
            rm = torch.zeros(ids_.shape, dtype=emb_.dtype, device=device)
            return model(ids_, inputs_embeds=emb_, return_hidden=True,
                         mem_read_mask=rm)
        return model(ids_, inputs_embeds=emb_, return_hidden=True)

    if R == 0:
        logits, _h = fwd(ids, emb)
        out = logits[:, -1, :]
        return out.unsqueeze(1) if return_steps else out

    def make_feed(h):
        """Think-slot input for the next step.
          think_only : hidden feedback only (carries the running thread; no mem).
          both       : retrieval-as-input only (think_embed + α·retrieval) [D3, refuted].
          hybrid     : hidden feedback + α·retrieval (thread + new info) [D7 fix]."""
        last_h = h[:, -1:, :]
        if mode == "think_only":
            return last_h
        inj = model.memory._last_injection_grad[:, -1:, :]
        # Learned α (init small) so hybrid STARTS == pure hidden-feedback
        # (which works) and grows the retrieval term only if it helps — the
        # FiLM-α / retrieval_input_alpha lesson. Fixed large α swamps the
        # hidden thread with untrained-WM noise (D8).
        a = getattr(model, "mem_alpha", None)
        a = a if a is not None else alpha
        if mode == "both":
            return think_emb + a * inj
        # hybrid
        return last_h + a * inj

    logits, h = fwd(ids, emb)
    feed = make_feed(h)
    step_logits = []
    for _ in range(R):
        ids = torch.cat([ids, think_tok], dim=1)
        emb = torch.cat([emb, feed], dim=1)
        logits, h = fwd(ids, emb)
        step_logits.append(logits[:, -1, :])
        feed = make_feed(h)
    if return_steps:
        return torch.stack(step_logits, dim=1)
    return step_logits[-1]


@torch.no_grad()
def evaluate(model, N, K, thinking_id, device, distract=0, n_eval=1024,
             batch=256, seed=4321):
    model.eval()
    res = {}
    has_mem = getattr(model, "use_memory", False)
    configs = [("no_think", "think_only", 0), ("think_only", "think_only", K)]
    if has_mem:
        configs += [("single_read", "hybrid", 1), ("hybrid", "hybrid", K)]
    for name, mode, R in configs:
        correct = total = done = 0
        gg = torch.Generator().manual_seed(seed)
        while done < n_eval:
            b = min(batch, n_eval - done)
            ids, ans, _chain, _ = make_distractor_chase_batch(
                b, N, K, distract, device, gg)
            logits = think_forward_mem(model, ids, R, thinking_id, mode=mode)
            pred = logits[:, :N].argmax(-1)
            correct += (pred == ans).sum().item()
            total += b
            done += b
        res[name] = correct / total
    model.train()
    return res


def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    vocab = args.N + 3
    max_T = 2 * args.N + args.distract + 2 + args.K   # pairs + distract + QUERY + s + thinks
    mem_size = max(args.N + 8, args.mem_size)  # buffer must hold the table
    use_film = not args.no_film
    use_mem = not args.no_mem
    model, thinking_id = build_model(
        vocab, args.N, args.d_model, args.n_layers, args.n_heads, args.d_head,
        max_T, mem_size, use_film=use_film, use_mem=use_mem,
        film_self_k=args.film_self_k, device=device)
    if use_mem:
        # Learned, no-WD retrieval-mix scalar, init small (D8).
        model.mem_alpha = torch.nn.Parameter(
            torch.tensor(float(args.alpha_init), device=device))
    if args.film_bypass_think:
        model._think_film_bypass = True
    train_mode = "hybrid" if use_mem else "think_only"
    print(f"[latent-mem] N={args.N} K={args.K} mem_size={mem_size} "
          f"L={args.n_layers} d={args.d_model} "
          f"film={use_film} mem={use_mem} mode={train_mode}  "
          f"params={model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=args.lr * 0.1)
    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # depth curriculum 1->K with consolidation (validated recipe)
        if step < 0.6 * args.steps:
            frac = step / (0.6 * args.steps)
            K_cur = max(1, min(args.K, int(round(1 + frac * (args.K - 1)))))
        else:
            K_cur = int(torch.randint(1, args.K + 1, (1,), generator=g).item())
        # Distractor-length curriculum: ramp 0->distract so WM learns selective
        # writing (pairs over distractors) gradually (D10).
        if args.distract_curriculum and args.distract > 0:
            D_cur = int(round(args.distract * min(1.0, step / (0.6 * args.steps))))
        else:
            D_cur = args.distract
        ids, ans, chain, _ = make_distractor_chase_batch(
            args.batch, args.N, K_cur, D_cur, device, g)
        # deep supervision: each think step r decodes f^r(s)
        step_logits = think_forward_mem(model, ids, K_cur, thinking_id,
                                        mode=train_mode, return_steps=True)
        tgt = chain[:, :step_logits.shape[1]]
        loss = F.cross_entropy(step_logits.reshape(-1, step_logits.shape[-1]),
                               tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                acc = (step_logits[:, -1, :args.N].argmax(-1) == ans).float().mean().item()
            print(f"  step {step:>5}  loss {loss.item():.4f}  "
                  f"acc {acc:.3f}  K_cur {K_cur}  ({time.time()-t0:.0f}s)")

    res = evaluate(model, args.N, args.K, thinking_id, device, distract=args.distract)
    print("\n=== ABLATION ===")
    for k in ["no_think", "single_read", "think_only", "hybrid"]:
        if k in res:
            print(f"  {k:>12}: {res[k]:.3f}")
    if args.save:
        torch.save({"state_dict": model.state_dict(),
                    "config": {"N": args.N, "K": args.K, "vocab": vocab,
                               "thinking_id": thinking_id, "mem_size": mem_size}},
                   args.save)
        print(f"[saved] {args.save}")
    return model, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=40)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--mem_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_head", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--distract", type=int, default=0,
                    help="distractor-span length (forces memory: recurrent "
                         "forgets over it, WM retains)")
    ap.add_argument("--distract_curriculum", action="store_true",
                    help="ramp distractor length 0->distract over training")
    ap.add_argument("--alpha_init", type=float, default=0.1,
                    help="initial retrieval-mix α (learned, no WD)")
    ap.add_argument("--film_bypass_think", action="store_true",
                    help="bypass FiLM multipass during think steps (test fix)")
    ap.add_argument("--film_self_k", type=int, default=3,
                    help="FiLM self-feed passes (1 = single-pass, no multipass)")
    ap.add_argument("--no_film", action="store_true", help="disable FiLM (isolation)")
    ap.add_argument("--no_mem", action="store_true", help="disable WM (isolation)")
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.N, args.K, args.steps, args.batch = 24, 3, 400, 64
        args.log_every = 50
    train(args)


if __name__ == "__main__":
    main()
