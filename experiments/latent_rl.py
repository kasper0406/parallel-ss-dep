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
with O(T) memory. SCOPE OF THE GRADIENT (be precise — fair-claims): the latent
feedback embeddings are recorded DETACHED, and think positions are
state-readonly, so the reward gradient reaches the GATE (when/how-much to
think), the trunk/lm_head at EMIT positions, and think-position trunk compute
only via the gate term + FiLM-lag leakage. It does NOT optimize the thought
CONTENT — the adapter and the hidden→latent production get no task-reward
gradient (that would need re-chaining the latent forwards with grad, the old
O(T²)-memory path). Thought content must come pre-trained (latent SFT /
latent_code_cotrain); this trainer optimizes the POLICY around it.

Contrast with train_rl_grader.py, which appends the discrete [THINKING] token id
(no hidden feedback) — that trains discrete-token thinking, which never helps.
"""
import argparse, random, re, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
from experiments.thinking import load_latent_model
import experiments.code_grader as CG


def _logits(out):
    return out[0] if isinstance(out, tuple) else out


def grade_clean(prob, raw_text):
    """Dense reward that doesn't undercount a name-inventing, prose-leading model
    (the two bugs the audit found): strip any leading prose to the first
    def/import, and alias the model's first top-level def to the expected
    entry_point so the mbpp test can resolve it. Then code_grader.grade for the
    dense tier score. NL-prompt problems only: for prompt_is_code problems
    (HumanEval/LeetCode-style) grade() prepends prob.prompt AND truncates at the
    first top-level stop token, so the strip/alias surgery below would corrupt
    the submission — pass those through untouched."""
    if getattr(prob, "prompt_is_code", False):
        return CG.grade(prob, raw_text)
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


def _masked_emit_logits(row, emit_idx: int, thinking_id, eos_id, temp: float,
                        min_emit_before_eos: int):
    """Think/eos masking + temperature on one position's logits. The SINGLE
    source of truth shared by rollout sampling and PPO replay — the ratio==1
    invariant depends on the two normalized distributions matching exactly, so
    never fork this logic."""
    lg = row.float() / max(temp, 1e-6)
    lg = lg.clone()
    lg[int(thinking_id)] = -float("inf")
    if eos_id is not None and emit_idx < min_emit_before_eos:
        lg[int(eos_id)] = -float("inf")
    return lg


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
          {"emit": bool, "forced": bool, "tok": int|None, "pos": int}
        where pos = index (in ids) of the position whose hidden produced the
        decision (i.e. the last position BEFORE the appended action token).
        (No rollout log-probs are recorded: the PPO "old" is the detached
        replay — see main() — so the incremental rollout's chunked-kernel
        numerics never enter the ratio.)
      emit_toks : emit-only token list (for grading)
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

        if not emit:
            # THINK: latent feedback (state-readonly via appended think token).
            latent = model.apply_latent_feedback_adapter(h).to(cur_emb.dtype)
            cur_ids = torch.cat([cur_ids, think_tok], dim=1)
            cur_emb = torch.cat([cur_emb, latent], dim=1)
            steps.append({"emit": False, "forced": forced, "tok": None, "pos": pos})
            thinks_this_step += 1
            think_total += 1
            continue
        # EMIT: sample a code token from THIS forward's logits.
        lg = _masked_emit_logits(logits, emit_count, thinking_id, eos_id,
                                 temp, min_emit_before_eos)
        probs = F.softmax(lg, dim=-1)
        tok = int(torch.multinomial(probs, 1))
        steps.append({"emit": True, "forced": forced, "tok": tok, "pos": pos})
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
    rollout exactly. Returns (gate_lp, tok_lp) as 1-D tensors under the CURRENT
    model (gate terms exclude FORCED positions — the policy had no choice
    there). The shared _masked_emit_logits keeps the emit distribution
    identical to the rollout's."""
    gate_ceil = 1.0 - gate_floor
    ids = traj["ids"]
    emb = traj["embeds"]
    out = model(ids, inputs_embeds=emb, return_hidden=True)
    logits = _logits(out)                                       # (1, L, V)
    gate_logits = model._last_gate_logits[0]                    # (L,) grad-aware
    gate_lp, tok_lp = [], []
    emit_idx = 0
    for st in traj["steps"]:
        p = st["pos"]
        if not st["forced"]:
            gl = gate_logits[p]
            gate_lp.append(
                _gate_logp(gl, st["emit"], gate_temp, gate_floor, gate_ceil).unsqueeze(0))
        if st["emit"]:
            lg = _masked_emit_logits(logits[0, p], emit_idx, thinking_id,
                                     eos_id, temp, min_emit_before_eos)
            tok_lp.append(F.log_softmax(lg, dim=-1)[st["tok"]].unsqueeze(0))
            emit_idx += 1
    dev = logits.device
    g = torch.cat(gate_lp) if gate_lp else torch.zeros(0, device=dev)
    t = torch.cat(tok_lp) if tok_lp else torch.zeros(0, device=dev)
    return g, t


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
                    help="per-latent-think penalty subtracted (ABSOLUTE, after the "
                         "group z-score of the task score — separate-ponder-norm) "
                         "from the GRPO advantage. Makes the gate suppress thinking "
                         "that doesn't flip correctness; selective thinking "
                         "survives. 0=off.")
    ap.add_argument("--ponder_warmup_steps", type=int, default=50,
                    help="ramp the ponder cost 0→full over N steps (cold-start "
                         "ponder bite was the grader-RL v1 collapse trigger)")
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

    model, cfg, thinking_id, tok, eos = load_latent_model(
        args.base, device, train=True)
    if not hasattr(model, "_last_gate_logits") and not any(
            "gate_head" in n for n, _ in model.named_parameters()):
        raise SystemExit(f"--base {args.base} has no output gate (gate_head): "
                         "gate-driven RL needs one. Train/SFT with the gate on, "
                         "or use a ckpt that has it.")
    ref, _, _, _, _ = load_latent_model(args.base, device, train=False)
    for p in ref.parameters():
        p.requires_grad = False
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
                rolls.append((traj, res.score, n_think))
                step_reward += res.score; step_pass += int(res.passed)
                think_steps_tot += n_think
                think_rate_acc += n_think / max(1, n_think + n_emit)
            # Separate ponder norm (the validated --grpo_separate_ponder_norm
            # semantics from thinking.compute_grpo_advantages): z-score the TASK
            # score only, then subtract the ABSOLUTE ponder cost. Folding the
            # cost into the reward before the group norm has two documented
            # pathologies: (a) high task variance washes the small ponder term
            # into noise; (b) tied task scores make ponder the ONLY variance, so
            # the norm amplifies it to a full-magnitude anti-think gradient.
            # The cost is also warmed up over --ponder_warmup_steps (the
            # grader-RL v1 collapse trigger was a cold-start ponder bite).
            scores = torch.tensor([r[1] for r in rolls], dtype=torch.float32)
            adv = scores - scores.mean()
            if scores.std() > 1e-6:
                adv = adv / (scores.std() + 1e-6)
            pcost = args.ponder_cost * min(1.0, step / max(1, args.ponder_warmup_steps))
            adv = adv - pcost * torch.tensor([float(r[2]) for r in rolls])
            for (traj, _, _), a in zip(rolls, adv.tolist()):
                if not traj["steps"] or abs(a) < 1e-8:
                    continue
                # Grad replay over the recorded sequence. The PPO "old" is the
                # detached replay output: there is exactly ONE update per
                # trajectory (no multi-epoch PPO) and no optimizer step between
                # "old" and "new", so a separate no-grad replay would return the
                # bit-identical values at the cost of one extra full forward.
                # Ratio == 1 by construction; clip_eps only matters if multi-
                # epoch replay is ever added (then restore a real old-policy
                # replay — do NOT use rollout-time log-probs, whose incremental
                # chunked-kernel numerics differ from the full forward).
                g_new, t_new = replay_logps(
                    model, traj, thinking_id, eos, args.temperature,
                    args.min_emit_before_eos, args.gate_floor, args.gate_temperature)
                # PPO surrogate over the union of gate + token actions.
                new_lp = torch.cat([g_new, t_new])
                old_lp = new_lp.detach()
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
                # k3 estimator (exp(δ)-δ-1, δ = t_ref - t_new): the naive
                # (t_new - t_ref).mean() is the k1 estimator, whose pathwise
                # gradient on on-policy samples is ZERO-MEAN (E[∇log π] = 0) —
                # i.e. no restoring force at all, the exact failure that let
                # grader-RL v1 collapse. k3's gradient (1 - exp(δ))·∇t_new pulls
                # t_new toward t_ref from both sides and is non-negative.
                with torch.no_grad():
                    _, t_ref = replay_logps(
                        ref, traj, thinking_id, eos, args.temperature,
                        args.min_emit_before_eos, args.gate_floor, args.gate_temperature)
                if t_new.numel() > 0:
                    delta = t_ref - t_new
                    kl = (delta.exp() - delta - 1.0).mean()
                else:
                    kl = torch.zeros((), device=device)
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
