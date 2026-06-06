"""RL on LATENT thinking (no CoT, no discrete think-tokens).

Each rollout: build a fixed-R latent ponder prefix (Coconut feedback — the model's
own out_norm hidden, mapped by the latent-feedback adapter, fed back as the next
input embedding; state-readonly), then SAMPLE pure code. Reward = code_grader tier
(execution-grounded). GRPO group-relative advantage + clipped-PPO on the emitted
tokens + KL to a frozen reference. The PPO re-forward REBUILDS the latent prefix
WITH grad, so the reward gradient flows into the latent thinking itself — i.e. RL
optimizes WHAT the model thinks latently, not a discrete token.

Contrast with train_rl_grader.py, which appends the discrete [THINKING] token id
(no hidden feedback) — that trains discrete-token thinking, which never helps.
"""
import argparse, math, random, re, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from experiments.eval_bracket_structure import build_model_from_ckpt
from transformers import AutoTokenizer
import experiments.code_grader as CG


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def grade_clean(prob, raw_text):
    """Dense reward that doesn't undercount a name-inventing, prose-leading model
    (the two bugs the audit found): strip any leading prose to the first def/import,
    and alias the model's first top-level def to the expected entry_point so the
    mbpp test can resolve it. Then code_grader.grade for the dense tier score."""
    lines = raw_text.split("\n")
    start = 0
    for i, l in enumerate(lines):
        if re.match(r"^(def |import |from |class |@)", l):
            start = i
            break
    code = "\n".join(lines[start:])
    names = re.findall(r"^def (\w+)", code, re.M)
    if names and prob.entry_point not in names:
        code = code + f"\n{prob.entry_point} = {names[0]}\n"
    return CG.grade(prob, code)


def build_prefix(model, comment_ids, R, thinking_id, device):
    """prompt + R latent think slots (inputs_embeds). Grad-aware (respects the
    enclosing no_grad / grad context). Returns (cur_ids, cur_emb)."""
    cur_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_emb = model.embed(cur_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long, device=device)
    for _ in range(R):
        out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        h = out[1]
        z = model.apply_latent_feedback_adapter(h[:, -1:, :]).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    return cur_ids, cur_emb


@torch.no_grad()
def rollout(model, comment_ids, R, thinking_id, eos_id, device, max_gen, temp,
            min_emit_before_eos=10):
    """Latent prefix then sample pure code. Returns (emit_tokens, old_logps)."""
    cur_ids, cur_emb = build_prefix(model, comment_ids, R, thinking_id, device)
    emits, logps = [], []
    for _ in range(max_gen):
        lg = _logits(model(cur_ids, inputs_embeds=cur_emb))[0, -1].float()
        lg[int(thinking_id)] = -float("inf")
        if len(emits) < min_emit_before_eos and eos_id is not None:
            lg[int(eos_id)] = -float("inf")   # avoid halt-after-docstring trap
        probs = F.softmax(lg / max(temp, 1e-6), dim=-1)
        tok = int(torch.multinomial(probs, 1))
        logps.append(float(torch.log(probs[tok] + 1e-12)))
        emits.append(tok)
        if tok == eos_id:
            break
        te = torch.tensor([[tok]], dtype=torch.long, device=device)
        cur_ids = torch.cat([cur_ids, te], dim=1)
        cur_emb = torch.cat([cur_emb, model.embed(te)], dim=1)
    return emits, logps


def seq_logps(model, comment_ids, R, emit_tokens, thinking_id, eos_id, device,
              temp, min_emit):
    """Grad-on re-forward: rebuild latent prefix (grad) + teacher-force emits,
    return per-emit-token log-prob. MUST match the rollout's sampling distribution
    (same temperature + think/eos masking) so the PPO ratio is 1.0 at step 0."""
    cur_ids, cur_emb = build_prefix(model, comment_ids, R, thinking_id, device)
    pr = cur_ids.shape[1]
    sol = torch.tensor([emit_tokens], dtype=torch.long, device=device)
    full_ids = torch.cat([cur_ids, sol], dim=1)
    full_emb = torch.cat([cur_emb, model.embed(sol)], dim=1)
    logits = _logits(model(full_ids, inputs_embeds=full_emb))
    pred = logits[0, pr - 1: pr - 1 + len(emit_tokens), :].float() / max(temp, 1e-6)
    pred[:, int(thinking_id)] = -float("inf")              # rollout masks think
    if min_emit > 0 and eos_id is not None:
        for i in range(min(min_emit, len(emit_tokens))):   # rollout masks early eos
            pred[i, int(eos_id)] = -float("inf")
    lp = F.log_softmax(pred, dim=-1).gather(1, sol[0].unsqueeze(1)).squeeze(1)
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft_baked_pure.pt")
    ap.add_argument("--dataset", default="mbpp_combined")
    ap.add_argument("--save", default="checkpoints/latent_rl.pt")
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--max_gen", type=int, default=200)
    ap.add_argument("--min_emit_before_eos", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 3
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg = build_model_from_ckpt(args.base, force_state_readonly=True,
                                       force_use_latent_feedback_adapter=True)
    model = model.to(device).train()
    if getattr(model, "use_memory", False):
        model.use_memory = False           # latent thread runs WM-off (training parity)
    ref, _ = build_model_from_ckpt(args.base, force_state_readonly=True,
                                   force_use_latent_feedback_adapter=True)
    ref = ref.to(device).eval()
    for p in ref.parameters():
        p.requires_grad = False
    if getattr(ref, "use_memory", False):
        ref.use_memory = False
    thinking_id = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id
    probs_all = CG.LOADERS[args.dataset]()
    print(f"[latent-rl] base={args.base} R={args.R} dataset={args.dataset} "
          f"({len(probs_all)} problems) params={model.num_params():,}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    comment = "# Complete the following Python function.\n"
    for step in range(1, args.steps + 1):
        batch = [rng.choice(probs_all) for _ in range(args.batch)]
        opt.zero_grad()
        step_reward, step_pass, n_roll, loss_val, kl_val = 0.0, 0, 0, 0.0, 0.0
        for prob in batch:
            cids = tok.encode(comment + prob.prompt, add_special_tokens=False)
            rolls = []
            for _ in range(args.n_group):
                emits, old_lp = rollout(model, cids, args.R, thinking_id, eos, device,
                                        args.max_gen, args.temperature,
                                        min_emit_before_eos=args.min_emit_before_eos)
                code = tok.decode([t for t in emits if t != thinking_id],
                                  skip_special_tokens=True)
                res = grade_clean(prob, code)
                rolls.append((emits, old_lp, res.score, res.passed))
                step_reward += res.score; step_pass += int(res.passed)
            rewards = torch.tensor([r[2] for r in rolls], dtype=torch.float32)
            adv = rewards - rewards.mean()
            if rewards.std() > 1e-6:
                adv = adv / (rewards.std() + 1e-6)
            for (emits, old_lp, _, _), a in zip(rolls, adv.tolist()):
                if not emits or abs(a) < 1e-8:
                    continue
                new_lp = seq_logps(model, cids, args.R, emits, thinking_id, eos, device,
                                   args.temperature, args.min_emit_before_eos)
                old_lp_t = torch.tensor(old_lp, device=device)
                ratio = torch.exp(new_lp - old_lp_t)
                surr = torch.minimum(ratio * a,
                                     torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * a)
                with torch.no_grad():
                    ref_lp = seq_logps(ref, cids, args.R, emits, thinking_id, eos, device,
                                       args.temperature, args.min_emit_before_eos)
                kl = (new_lp - ref_lp).mean()
                loss = (-surr.mean() + args.kl_coef * kl) / (args.batch * args.n_group)
                loss.backward()
                loss_val += float(loss); kl_val += float(kl)
                n_roll += 1
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        nb = args.batch * args.n_group
        if step % args.log_every == 0:
            print(f"step {step:>4}/{args.steps}  reward={step_reward/nb:.3f}  "
                  f"pass={step_pass}/{nb}  updated={n_roll}  loss={loss_val:.4f}  "
                  f"kl={kl_val/max(1,n_roll):.4f}", flush=True)
        if not args.smoke and step % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "step": step, "config": cfg}, args.save)
            print(f"[saved] {args.save} @ step {step}", flush=True)
    if args.smoke:
        print("SMOKE OK", flush=True); return
    torch.save({"state_dict": model.state_dict(), "step": args.steps, "config": cfg}, args.save)
    print(f"saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
