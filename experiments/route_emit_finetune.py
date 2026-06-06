"""Gate-routing fine-tune: teach the gate to EMIT on code (route_emit_all), so it
stops firing the OOD latent op mid-generation (the degenerate-collapse cause).

Freezes EVERYTHING except gate_head, so code competence + output format are
preserved bit-identically; only the emit/think decision changes. Target: the
gate emits (P(emit)=1) at every code position. Result should be: no-think
unchanged (8/164), thinking-ON no longer collapses (gate emits on code → no
mid-gen think → thinking-ON ≈ no-think → invariant restored). The latent op is
untouched, so it remains available for reasoning tasks where the gate learned to
fire.
"""
import argparse, json, random, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft_baked_code.pt")
    ap.add_argument("--data", default="data/sft_phase_c_combined.jsonl")
    ap.add_argument("--save", default="checkpoints/route_emit_code.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--max_pairs", type=int, default=20000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.max_pairs = 10, 200

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    model, cfg = build_model_from_ckpt(args.base)
    model = model.to(device).train()
    model._film_bypass = True
    assert getattr(model, "output_gate", False), "ckpt has no output gate"
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))

    # freeze all but the gate head
    n_tr = 0
    for name, p in model.named_parameters():
        train_it = "gate_head" in name
        p.requires_grad = train_it
        n_tr += p.numel() if train_it else 0
    print(f"[route_emit] base={args.base} gate-only trainable params: {n_tr:,}", flush=True)

    # load code completions (problem+solution), tokenize
    rows = [json.loads(l) for l in open(args.data) if l.strip()][: args.max_pairs * 2]
    comment = "# Complete the following Python function.\n"
    data = []
    for r in rows:
        prob = r.get("problem_prompt") or r.get("problem") or r.get("prompt") or ""
        sol = r.get("extracted_code") or r.get("solution") or r.get("completion") or ""
        if not prob or not sol:
            continue
        ids = tok.encode(comment + prob + "\n" + sol, add_special_tokens=False)[: args.max_len]
        if len(ids) > 16:
            data.append(ids)
        if len(data) >= args.max_pairs:
            break
    print(f"[route_emit] usable code sequences: {len(data)}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(0)
    opt.zero_grad()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        loss_acc = 0.0
        for _ in range(args.accum):
            ids = torch.tensor([rng.choice(data)], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = model(ids)
                gl = model._last_gate_logits  # (1,T) pre-sigmoid P(emit)
                # route_emit: every code position -> EMIT (target 1)
                loss = F.binary_cross_entropy_with_logits(
                    gl.reshape(-1), torch.ones_like(gl.reshape(-1))) / args.accum
            loss.backward()
            loss_acc += loss.item()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad()
        if step % args.log_every == 0 or step <= 3:
            sig = torch.sigmoid(gl).mean().item()
            print(f"step {step:>4}/{args.steps} gate_bce={loss_acc:.4f} "
                  f"mean_sigma(emit)={sig:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if args.smoke:
        print("SMOKE OK", flush=True); return
    torch.save({"state_dict": model.state_dict(), "step": args.steps, "config": cfg}, args.save)
    print(f"saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
