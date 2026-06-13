"""Gate-only calibration on code: aim the gate at WHERE latent thinking helps.

Diagnostic finding (probe_gate_placement.py on latent_code_adapteronly.pt):
the gate is ANTI-aimed — corr(P_think, Delta_logp) ≈ -0.10, mean Delta_logp
where the gate fires think ≈ -2.0 (it fires think exactly where thinking
HURTS). Meanwhile thinking genuinely helps at ~10% of positions by ≈+0.3.
So the marginal aggregate win is achieved DESPITE a mis-aimed gate.

This trainer freezes EVERYTHING except gate_head and trains it with the dense
per-position teacher y = 1{Delta_logp(R latent steps) > 0} (BCE,
class-balanced via pos_weight). The trunk + adapter + lm_head are untouched, so
no-think code ability and the latent thought CONTENT are preserved
byte-identically; only the emit/think DECISION changes. Goal: flip the
correlation positive so the gate fires think on the helpful minority and emits
elsewhere — removing the -2.0 left-tail harm and capturing the +0.3 upside.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/gate_calibrate_code.py \
      --base checkpoints/latent_code_adapteronly.pt \
      --save checkpoints/latent_code_gatecal.pt --steps 400
"""
import argparse, json, random, sys
import torch
sys.path.insert(0, ".")
from experiments.gate_calibration import compute_gate_calibration_loss
from experiments.thinking import load_latent_model


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def load_code(jsonl, tok, comment, max_len, min_score, limit):
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
        tgt[r, :len(ids) - 1] = t[1:]
        tgt[r, :plen - 1] = -100        # teacher only on CODE positions
    return inp.to(device), tgt.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/latent_code_adapteronly.pt")
    ap.add_argument("--save", default="checkpoints/latent_code_gatecal.pt")
    ap.add_argument("--jsonl", default="data/sft_phase_c_combined.jsonl")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--R", type=int, default=2, help="latent steps in the teacher "
                    "(match the co-train R the deployed gate will trigger)")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--max_positions", type=int, default=64)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--pos_weight", type=float, default=8.0,
                    help="up-weight the 'think helps' class (~neg/pos≈8 on code) "
                         "so BCE doesn't collapse to never-think")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_score", type=float, default=0.99)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 5
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg, tid, tok, eos = load_latent_model(args.base, device, train=True)
    # Freeze everything except gate_head: no-think code ability + latent thought
    # content preserved byte-identically; only the emit/think decision moves.
    ntrain = 0
    for n, p in model.named_parameters():
        keep = "gate_head" in n
        p.requires_grad = keep
        ntrain += p.numel() if keep else 0
    if ntrain == 0:
        raise SystemExit(f"--base {args.base} has no gate_head to calibrate.")
    print(f"[gatecal] base={args.base} gate_head params={ntrain:,} R={args.R} "
          f"pos_weight={args.pos_weight}", flush=True)

    comment = "# Complete the following Python function.\n"
    rows = load_code(args.jsonl, tok, comment, args.max_len, args.min_score, args.limit)
    print(f"[gatecal] code_examples={len(rows)}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        idxs = [rng.randrange(len(rows)) for _ in range(args.batch)]
        inp, tgt = make_batch(rows, idxs, pad_id=0, device=device)
        opt.zero_grad()
        # Main forward to capture the GRAD-CARRYING gate logits BEFORE the
        # calibration helper's extra forwards clobber model._last_gate_logits.
        _ = _logits(model(inp))
        gate_snap = model._last_gate_logits           # (B,T) grad-carrying
        res = compute_gate_calibration_loss(
            model, inp, tgt, gate_snap, thinking_token_id=tid, latent_R=args.R,
            margin=args.margin, sample_frac=args.sample_frac,
            max_positions=args.max_positions, eos_id=eos,
            pos_weight=args.pos_weight, generator=gen)
        if res is None:
            continue
        res.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"step {step:>4}/{args.steps}  bce={float(res.loss.detach()):.4f}  "
                  f"tgt1={res.target_frac_pos:.3f}  σ={res.mean_sigma:.3f}  "
                  f"Δlogp={res.mean_delta:+.3f}  n={res.n_positions}", flush=True)
        if not args.smoke and step % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "step": step, "config": cfg},
                       args.save)
            print(f"[saved] {args.save} @ {step}", flush=True)
    if args.smoke:
        print("SMOKE OK", flush=True); return
    torch.save({"state_dict": model.state_dict(), "step": args.steps, "config": cfg},
               args.save)
    print(f"saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
