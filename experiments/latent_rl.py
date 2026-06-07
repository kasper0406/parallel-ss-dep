"""GATE-DRIVEN RL on LATENT thinking (no CoT, no discrete think-tokens).

The thinking gate dynamically decides — per problem and per position — whether
and how many latent ponder steps to take. There is NO fixed R. At each step:

    g = sigmoid(gate_head(out_norm(h)))  =  P(emit)

is treated as a Bernoulli RL action. If the draw says THINK, we do one
Coconut-style latent feedback step (the model's own out_norm(h), mapped by the
latent-feedback adapter, fed back as the next input embedding; state-readonly so
DeltaNet's recurrent state is never corrupted). If it says EMIT, we sample a
pure-code token. Both the gate decisions AND the emit tokens are policy
variables: GRPO group-relative advantage + clipped-PPO surrogate over the union
of (gate-decision log-probs at non-forced positions) and (emit-token log-probs),
plus a KL to a frozen reference. Reward = code_grader tier (execution-grounded).

This mirrors the canonical gate-driven decode in
`eval_humaneval.generate_latent_think` (interleaved think bursts whose length the
gate chooses), so train and eval use the SAME thinking mechanism.

PPO replay is a SINGLE grad forward over the recorded (ids, inputs_embeds)
sequence: DeltaNet is causal and think positions are state-readonly (β=0), so
the full-sequence forward reproduces every per-position hidden the sequential
rollout produced — hence exact gate + token log-probs (ratio == 1 at step 0)
with O(T) memory. The latent feedback embeddings are taken from the rollout
(detached) in the replay, so grad flows into the TRUNK (which produces the
thought hidden), the GATE (when/how-much to think), and the lm_head — i.e. RL
optimizes WHEN to think and WHAT the trunk thinks; the linear feedback adapter
map is held fixed-from-rollout (it was trained in latent SFT).

Contrast with train_rl_grader.py, which appends the discrete [THINKING] token id
(no hidden feedback) — that trains discrete-token thinking, which never helps.
"""
import argparse, math, random, re, sys
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
    (the two bugs the audit found): strip any leading prose to the first
    def/import, and alias the model's first top-level def to the expected
    entry_point so the mbpp/HumanEval test can resolve it. Then code_grader.grade
    for the dense tier score."""
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


# ---- action/log-prob helpers ------------------------------------------------

def _gate_p_emit(gate_logit, temp: float, floor: float, ceil: float):
    """Clamped P(emit) = clamp(sigmoid(gate_logit / temp), floor, ceil).
    `temp` < 1 SHARPENS the gate (less thinking — start near the no-think
    baseline and let RL ADD thinking where it helps, rather than starting by
    over-thinking). MUST be applied identically in rollout & replay so the PPO
    ratio is 1.0 at step 0. Grad-aware; clamp passes grad in the unclamped
    region and zeros it at saturation. The (1e-6, 1-1e-6) guard keeps
    log(p)/log(1-p) finite even when floor/ceil hit 0/1."""
    return torch.sigmoid(gate_logit / temp).clamp(floor, ceil).clamp(1e-6, 1 - 1e-6)


def _gate_logp(gate_logit, emit: bool, temp: float, floor: float, ceil: float):
    """log P(action) under Bernoulli(_gate_p_emit(...))."""
    p = _gate_p_emit(gate_logit, temp, floor, ceil)
    return torch.log(p) if emit else torch.log1p(-p)


@torch.no_grad()
def rollout(model, comment_ids, thinking_id, eos_id, device, max_gen, temp,
            max_think_per_step, total_think_budget, gate_floor,
            min_emit_before_eos, gate_temp=1.0):
    """Gate-driven latent rollout. The gate stochastically chooses emit-vs-think
    at every step; think -> one latent feedback step; emit -> sample a token.

    Returns a trajectory dict with everything the grad replay needs:
      ids       : (1, L) full token-id sequence (prompt + think_id/emit tokens)
      embeds    : (1, L, d) inputs_embeds used (latents at think positions) DETACHED
      steps     : list of per-action records, each:
          {"emit": bool, "forced": bool, "gate_old_lp": float,
           "tok": int|None, "tok_old_lp": float|None, "pos": int}
        where pos = index (in ids) of the position whose hidden produced the
        decision (i.e. the last position BEFORE the appended action token).
      code      : decoded emit-only string (for grading)
    """
    gate_ceil = 1.0 - gate_floor               # symmetric: think-prob floor too
    cur_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_emb = model.embed(cur_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long, device=device)
    steps = []
    emit_toks = []
    emit_count = 0
    think_total = 0
    thinks_this_step = 0
    while emit_count < max_gen:
        out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        logits = _logits(out)[0, -1].float()
        h = out[1][:, -1:, :]                                   # (1,1,d)
        pos = cur_ids.shape[1] - 1                              # decision position
        gl_t = getattr(model, "_last_gate_logits", None)
        g_eff = (1.0 if gl_t is None
                 else float(_gate_p_emit(gl_t[0, -1], gate_temp,
                                         gate_floor, gate_ceil)))  # clamped P(emit)
        # Forced-emit guard (matches generate_latent_think): cannot think forever.
        forced = (thinks_this_step >= max_think_per_step
                  or think_total >= total_think_budget)
        if forced:
            emit = True
        else:
            emit = (torch.rand(1, device=device).item() < g_eff)
        # gate log-prob recorded against the SAME clamped distribution the replay
        # recomputes (sigmoid(raw_logit) == g, clamped identically) -> ratio 1.0.
        gate_old_lp = math.log(g_eff) if emit else math.log(1.0 - g_eff)

        if not emit:
            # THINK: latent feedback (state-readonly via appended think token).
            latent = model.apply_latent_feedback_adapter(h).to(cur_emb.dtype)
            cur_ids = torch.cat([cur_ids, think_tok], dim=1)
            cur_emb = torch.cat([cur_emb, latent], dim=1)
            steps.append({"emit": False, "forced": forced, "gate_old_lp": gate_old_lp,
                          "tok": None, "tok_old_lp": None, "pos": pos})
            thinks_this_step += 1
            think_total += 1
            continue
        # EMIT: sample a code token from THIS forward's logits.
        lg = logits.clone()
        lg[int(thinking_id)] = -float("inf")
        if eos_id is not None and emit_count < min_emit_before_eos:
            lg[int(eos_id)] = -float("inf")
        probs = F.softmax(lg / max(temp, 1e-6), dim=-1)
        tok = int(torch.multinomial(probs, 1))
        tok_lp = float(torch.log(probs[tok] + 1e-12))
        steps.append({"emit": True, "forced": forced, "gate_old_lp": gate_old_lp,
                      "tok": tok, "tok_old_lp": tok_lp, "pos": pos})
        emit_toks.append(tok)
        emit_count += 1
        thinks_this_step = 0
        if tok == eos_id:
            break
        te = torch.tensor([[tok]], dtype=torch.long, device=device)
        cur_ids = torch.cat([cur_ids, te], dim=1)
        cur_emb = torch.cat([cur_emb, model.embed(te)], dim=1)

    return {"ids": cur_ids.detach(), "embeds": cur_emb.detach(), "steps": steps,
            "emit_toks": emit_toks}


def replay_logps(model, traj, thinking_id, eos_id, temp, min_emit_before_eos,
                 gate_floor, gate_temp=1.0):
    """Single grad forward over the recorded (ids, embeds) sequence. DeltaNet is
    causal + think positions state-readonly, so per-position hiddens match the
    rollout exactly. Returns (gate_new_lp, gate_old_lp, tok_new_lp, tok_old_lp)
    as 1-D tensors (gate terms exclude FORCED positions)."""
    gate_ceil = 1.0 - gate_floor
    ids = traj["ids"]
    emb = traj["embeds"]
    out = model(ids, inputs_embeds=emb, return_hidden=True)
    logits = _logits(out)                                       # (1, L, V)
    gate_logits = model._last_gate_logits[0]                    # (L,) grad-aware
    gate_new, gate_old = [], []
    tok_new, tok_old = [], []
    emit_idx = 0
    for st in traj["steps"]:
        p = st["pos"]
        if not st["forced"]:
            gl = gate_logits[p]
            gate_new.append(
                _gate_logp(gl, st["emit"], gate_temp, gate_floor, gate_ceil).unsqueeze(0))
            gate_old.append(st["gate_old_lp"])
        if st["emit"]:
            lg = logits[0, p].float() / max(temp, 1e-6)
            lg = lg.clone()
            lg[int(thinking_id)] = -float("inf")
            # Replicate the rollout's per-emit-index eos mask EXACTLY so the
            # normalized distribution (and hence the PPO ratio) matches at step 0.
            if eos_id is not None and emit_idx < min_emit_before_eos:
                lg[int(eos_id)] = -float("inf")
            lp_full = F.log_softmax(lg, dim=-1)
            tok_new.append(lp_full[st["tok"]].unsqueeze(0))
            tok_old.append(st["tok_old_lp"])
            emit_idx += 1
    dev = logits.device
    g_new = torch.cat(gate_new) if gate_new else torch.zeros(0, device=dev)
    t_new = torch.cat(tok_new) if tok_new else torch.zeros(0, device=dev)
    g_old = torch.tensor(gate_old, device=dev) if gate_old else torch.zeros(0, device=dev)
    t_old = torch.tensor(tok_old, device=dev) if tok_old else torch.zeros(0, device=dev)
    return g_new, g_old, t_new, t_old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft_baked_pure.pt")
    ap.add_argument("--dataset", default="mbpp_combined")
    ap.add_argument("--save", default="checkpoints/latent_rl.pt")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--gate_lr_mult", type=float, default=10.0,
                    help="gate_head LR multiplier over the base LR (the gate is the "
                         "thing that must move to fix the code-miscalibrated gate)")
    ap.add_argument("--max_gen", type=int, default=200)
    ap.add_argument("--max_think_per_step", type=int, default=8)
    ap.add_argument("--total_think_budget", type=int, default=400)
    ap.add_argument("--gate_floor", type=float, default=0.05,
                    help="clamp P(emit) into [floor, 1-floor] for sampling stability "
                         "and a symmetric think/emit exploration floor")
    ap.add_argument("--gate_temperature", type=float, default=1.0,
                    help="gate sharpening: sigmoid(logit/temp). 1.0 = neutral (the "
                         "base gate as-is). >1 softens toward 0.5; <1 sharpens AWAY "
                         "from 0.5 (this base's gate is negative on code, so <1 "
                         "INCREASES thinking). Kept as a knob; default neutral.")
    ap.add_argument("--min_emit_before_eos", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--ponder_cost", type=float, default=0.004,
                    help="per-latent-think penalty subtracted from the reward before "
                         "the GRPO advantage. Makes the gate suppress thinking that "
                         "doesn't flip correctness; selective thinking survives. 0=off.")
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--kl_coef", type=float, default=0.05)
    ap.add_argument("--gate_entropy_bonus", type=float, default=0.01,
                    help="Bernoulli-entropy reward on the gate to prevent collapse "
                         "to never-think / always-think")
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
    model._gist_loss_enabled = False        # no aux gist loss during RL
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
    print(f"[gate-latent-rl] base={args.base} dataset={args.dataset} "
          f"({len(probs_all)} problems) params={model.num_params():,}", flush=True)

    # Separate, higher-LR group for the gate_head: it is the ~900-param head that
    # must move most to fix the code-miscalibrated gate, while the trunk/lm_head
    # stay at the conservative base LR (KL-anchored) to avoid degrading code.
    gate_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (gate_params if "gate_head" in n else base_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": base_params, "lr": args.lr},
         {"params": gate_params, "lr": args.lr * args.gate_lr_mult}])
    print(f"[gate-latent-rl] gate_head params={sum(p.numel() for p in gate_params)} "
          f"@ lr={args.lr * args.gate_lr_mult:.1e}; base @ lr={args.lr:.1e}", flush=True)
    rng = random.Random(args.seed)
    comment = "# Complete the following Python function.\n"

    def do_rollout(cids):
        return rollout(model, cids, thinking_id, eos, device, args.max_gen,
                       args.temperature, args.max_think_per_step,
                       args.total_think_budget, args.gate_floor,
                       args.min_emit_before_eos, args.gate_temperature)

    for step in range(1, args.steps + 1):
        batch = [rng.choice(probs_all) for _ in range(args.batch)]
        opt.zero_grad()
        step_reward, step_pass, n_roll = 0.0, 0, 0
        loss_val, kl_val, think_steps_tot, think_rate_acc = 0.0, 0.0, 0, 0.0
        for prob in batch:
            cids = tok.encode(comment + prob.prompt, add_special_tokens=False)
            rolls = []
            for _ in range(args.n_group):
                traj = do_rollout(cids)
                code = tok.decode(traj["emit_toks"], skip_special_tokens=True)
                res = grade_clean(prob, code)
                n_think = sum(1 for s in traj["steps"] if not s["emit"])
                n_emit = sum(1 for s in traj["steps"] if s["emit"])
                # Ponder-cost-shaped reward: a small per-think penalty gives the
                # gate a CLEAN gradient to suppress thinking that doesn't pay for
                # itself. Selective thinking survives only where it flips
                # correctness enough to beat the cost (the correctness-only
                # advantage was too noisy to teach this). Raw score is logged.
                shaped = res.score - args.ponder_cost * n_think
                rolls.append((traj, shaped, res.passed))
                step_reward += res.score; step_pass += int(res.passed)
                think_steps_tot += n_think
                think_rate_acc += n_think / max(1, n_think + n_emit)
            rewards = torch.tensor([r[1] for r in rolls], dtype=torch.float32)
            adv = rewards - rewards.mean()
            if rewards.std() > 1e-6:
                adv = adv / (rewards.std() + 1e-6)
            for (traj, _, _), a in zip(rolls, adv.tolist()):
                if not traj["steps"] or abs(a) < 1e-8:
                    continue
                # OLD log-probs via the SAME full-forward path (no_grad) the grad
                # replay uses -> ratio == 1.0 at step 0 exactly (the incremental
                # rollout's chunked-kernel numerics differ from the full forward;
                # evaluating old & new identically removes that discrepancy).
                with torch.no_grad():
                    g_old, _, t_old, _ = replay_logps(
                        model, traj, thinking_id, eos, args.temperature,
                        args.min_emit_before_eos, args.gate_floor, args.gate_temperature)
                g_new, _, t_new, _ = replay_logps(
                    model, traj, thinking_id, eos, args.temperature,
                    args.min_emit_before_eos, args.gate_floor, args.gate_temperature)
                # PPO surrogate over the union of gate + token actions.
                new_lp = torch.cat([g_new, t_new])
                old_lp = torch.cat([g_old, t_old])
                if new_lp.numel() == 0:
                    continue
                ratio = torch.exp(new_lp - old_lp)
                surr = torch.minimum(
                    ratio * a,
                    torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * a)
                # Gate-entropy bonus (maximize) on the gate-decision positions.
                ent = torch.zeros((), device=device)
                if args.gate_entropy_bonus > 0 and g_new.numel() > 0:
                    # H(Bernoulli) from the new gate log-prob of the taken action
                    # is not directly available; use σ from gate_logits via g_new.
                    # g_new = log p(action); recover p, then Bernoulli entropy.
                    p_act = g_new.exp().clamp(1e-6, 1 - 1e-6)
                    ent = -(p_act * p_act.log() + (1 - p_act) * (1 - p_act).log()).mean()
                # KL to frozen ref on the EMIT-token distribution (the deployed
                # output); gate-KL omitted (the gate is the thing we WANT to move).
                with torch.no_grad():
                    _, _, t_ref, _ = replay_logps(
                        ref, traj, thinking_id, eos, args.temperature,
                        args.min_emit_before_eos, args.gate_floor, args.gate_temperature)
                kl = (t_new - t_ref).mean() if t_new.numel() > 0 else torch.zeros((), device=device)
                loss = (-surr.mean() + args.kl_coef * kl
                        - args.gate_entropy_bonus * ent) / (args.batch * args.n_group)
                loss.backward()
                loss_val += float(loss.detach()); kl_val += float(kl.detach()); n_roll += 1
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        nb = args.batch * args.n_group
        if step % args.log_every == 0:
            print(f"step {step:>4}/{args.steps}  reward={step_reward/nb:.3f}  "
                  f"pass={step_pass}/{nb}  think_rate={think_rate_acc/nb:.3f}  "
                  f"think/roll={think_steps_tot/nb:.1f}  updated={n_roll}  "
                  f"loss={loss_val:.4f}  kl={kl_val/max(1,n_roll):.4f}", flush=True)
        if not args.smoke and step % args.save_every == 0:
            torch.save({"state_dict": model.state_dict(), "step": step, "config": cfg}, args.save)
            print(f"[saved] {args.save} @ step {step}", flush=True)
    if args.smoke:
        print("SMOKE OK", flush=True); return
    torch.save({"state_dict": model.state_dict(), "step": args.steps, "config": cfg}, args.save)
    print(f"saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
