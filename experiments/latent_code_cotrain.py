"""Co-train latent thinking ON CODE so it stops corrupting code-token prediction.

The Δlogp probe (2026-06-06) showed latent thinking on `sft_baked_pure` HURTS
code prediction by -3.7..-6.4 nats (the adapter was co-trained on reasoning
traces -> OOD for code), which is why inference-time thinking is net-negative on
MBPP at every gating threshold. This trainer adds `thinking.latent_cotrain_loss`
(grad CE on the post-R-latent-think prediction of the TRUE next code token) to
the normal no-think LM loss, so the trunk+adapter learn to make a latent step on
code USEFUL (or at least harmless: Δlogp -> 0+). The no-think LM loss is kept so
code ability isn't forgotten. Validation signal: mean_delta_logp climbs from
≈ -3.7 toward 0/positive. Then re-probe + re-eval MBPP no-think vs thinking.
"""
import argparse, json, random, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from experiments.thinking import latent_cotrain_loss, load_latent_model


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def load_code(jsonl, tok, comment, max_len, min_score, limit):
    """Return list of (input_ids, prompt_len) for passing-solution code."""
    rows = []
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            if float(d.get("score", 0)) < min_score or not d.get("extracted_code"):
                continue
            pre = tok.encode(comment + d["problem_prompt"], add_special_tokens=False)
            code = tok.encode(d["extracted_code"], add_special_tokens=False)
            ids = (pre + code)[:max_len]
            if len(ids) < 16:
                continue
            rows.append((ids, len(pre)))
            if len(rows) >= limit:
                break
    return rows


def make_batch(rows, idxs, pad_id, device):
    seqs = [rows[i][0] for i in idxs]
    L = max(len(s) for s in seqs)
    inp = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    tgt = torch.full((len(seqs), L), -100, dtype=torch.long)
    for r, (ids, plen) in enumerate([rows[i] for i in idxs]):
        t = torch.tensor(ids, dtype=torch.long)
        inp[r, :len(ids)] = t
        # next-token targets; mask the prompt span (-100) so LM loss is on CODE.
        tgt[r, :len(ids) - 1] = t[1:]
        tgt[r, :plen - 1] = -100
    return inp.to(device), tgt.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft_baked_pure.pt")
    ap.add_argument("--save", default="checkpoints/latent_code_cotrain.pt")
    ap.add_argument("--jsonl", default="data/sft_phase_c_combined.jsonl")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--R", type=int, default=2)
    ap.add_argument("--cotrain_weight", type=float, default=1.0)
    ap.add_argument("--kl_anchor", type=float, default=0.0,
                    help="KL(current no-think || frozen original no-think) weight; "
                         "anti-forgetting anchor so the aux loss preserves base code")
    ap.add_argument("--max_positions", type=int, default=24)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_score", type=float, default=0.99)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--selective", action="store_true")
    ap.add_argument("--freeze_trunk", action="store_true",
                    help="train ONLY the latent_feedback_adapter -> no-think path "
                         "byte-identical to original (zero forgetting)")
    ap.add_argument("--train_gate", action="store_true",
                    help="RESERVED (currently rejected): no gate-gradient-bearing "
                         "loss is wired here — latent_cotrain_loss's CE never "
                         "touches gate_head, so the gate would silently stay at "
                         "its base values. Use gate_calibration / latent_rl to "
                         "train the gate.")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 5
    if args.train_gate:
        raise SystemExit(
            "--train_gate is rejected: under this trainer the only gradient-"
            "bearing loss (latent_cotrain_loss CE) never touches gate_head, so "
            "the gate would receive zero gradient and silently stay at its base "
            "values while looking 'co-calibrated'. Train the gate with "
            "gate_calibration (SFT) or latent_rl (RL) instead.")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg, tid, tok, eos = load_latent_model(args.base, device, train=True)
    # FREEZE-TRUNK: train ONLY the latent_feedback_adapter. The no-think forward
    # NEVER uses the adapter (it fires only at think positions), so freezing the
    # trunk leaves the no-think path BYTE-IDENTICAL to the original -> base code
    # ability preserved EXACTLY (no forgetting), while the adapter learns to
    # make latent thinking on code non-harmful.
    if args.freeze_trunk:
        ntrain = 0
        for n, p in model.named_parameters():
            keep = "latent_feedback_adapter" in n
            p.requires_grad = keep
            ntrain += p.numel() if keep else 0
        print(f"[freeze_trunk] training only adapter: {ntrain:,} params", flush=True)
    # Frozen reference = the ORIGINAL base, used to KL-anchor the no-think logits
    # so co-training the latent-think aux loss does NOT degrade base code ability
    # (the cw=1.0 narrow-data run forgot ~18% of MBPP). KL pins no-think behaviour
    # while the cotrain loss is free to fix the latent-think path. Pointless (and
    # structurally zero) under --freeze_trunk, hence rejected there.
    ref = None
    if args.kl_anchor > 0:
        if args.freeze_trunk:
            raise SystemExit("--kl_anchor with --freeze_trunk is a no-op that "
                             "costs a full ref model: the no-think path is "
                             "frozen byte-identical, so the KL is exactly 0.")
        ref, _, _, _, _ = load_latent_model(args.base, device, train=False)
        for p in ref.parameters():
            p.requires_grad = False
    comment = "# Complete the following Python function.\n"
    rows = load_code(args.jsonl, tok, comment, args.max_len, args.min_score, args.limit)
    print(f"[code-cotrain] base={args.base} code_examples={len(rows)} R={args.R} "
          f"cw={args.cotrain_weight} selective={args.selective} params={model.num_params():,}",
          flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        idxs = [rng.randrange(len(rows)) for _ in range(args.batch)]
        inp, tgt = make_batch(rows, idxs, pad_id=0, device=device)
        opt.zero_grad()
        is_log_step = (step % args.log_every == 0 or step == 1)
        lm = torch.zeros((), device=device)
        kl = torch.zeros((), device=device)
        if args.freeze_trunk:
            # The no-think path is frozen byte-identical: the LM loss carries no
            # gradient and would roughly DOUBLE step wall-clock if computed every
            # step. Compute it under no_grad on log steps only (sanity readout —
            # it should never move).
            if is_log_step:
                with torch.no_grad():
                    logits = _logits(model(inp))
                    lm = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                        tgt[:, :-1].reshape(-1), ignore_index=-100)
        else:
            # 1) no-think LM loss (preserve code ability)
            logits = _logits(model(inp))
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 tgt[:, :-1].reshape(-1), ignore_index=-100)
            # 1b) KL anchor: keep no-think logits close to the frozen original so
            # the aux loss can't erode base code ability (anti-forgetting).
            if ref is not None:
                with torch.no_grad():
                    rlog = _logits(ref(inp))[:, :-1].float()
                cur = logits[:, :-1].float()
                mask = (tgt[:, :-1] != -100).reshape(-1)
                kl_tok = F.kl_div(F.log_softmax(cur, -1).reshape(-1, cur.shape[-1]),
                                  F.log_softmax(rlog, -1).reshape(-1, rlog.shape[-1]),
                                  reduction="none", log_target=True).sum(-1)
                kl = (kl_tok * mask).sum() / mask.sum().clamp(min=1)
        # 2) latent co-train loss (make thinking useful/harmless on code)
        res = latent_cotrain_loss(model, inp, tgt, R=args.R, thinking_token_id=tid,
                                  max_positions=args.max_positions,
                                  max_prefix_len=args.max_len, pad_id=0, eos_id=eos,
                                  selective=args.selective, generator=gen)
        if res is None:
            ct, dlp, npos = torch.zeros((), device=device), float("nan"), 0
        else:
            ct, dlp, npos = res
        loss = args.cotrain_weight * ct
        if not args.freeze_trunk:
            loss = loss + lm + args.kl_anchor * kl
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
        if is_log_step:
            dlpv = float(dlp) if not isinstance(dlp, torch.Tensor) else float(dlp.detach())
            print(f"step {step:>4}/{args.steps}  lm={float(lm.detach()):.3f}  "
                  f"cotrain={float(ct.detach()):.3f}  kl={float(kl.detach()):.4f}  "
                  f"Δlogp={dlpv:+.3f}  npos={npos}", flush=True)
        if not args.smoke and step % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "step": step, "config": cfg}, args.save)
            print(f"[saved] {args.save} @ {step}", flush=True)
    if args.smoke:
        print("SMOKE OK", flush=True); return
    torch.save({"state_dict": model.state_dict(), "step": args.steps, "config": cfg}, args.save)
    print(f"saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
