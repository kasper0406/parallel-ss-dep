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
from experiments.optim_utils import is_film_alpha


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


def load_recall(jsonl, tok, max_len, limit):
    """Return list of (input_ids, prompt_len) for long-context recall tasks.

    The program (`problem_prompt`, ending in `print(s)`) is the prefix; the
    bound value (`answer`) is the supervised continuation. make_batch masks the
    prompt span to -100, so latent_cotrain_loss samples ONLY the answer
    position(s) — the recall trigger where WM retrieval must surface the value
    bound far earlier. This is the Stage A targeting: the loss reward is 'predict
    the recalled value', exactly where WM addressing has to work.

    CRITICAL: the binding (`s = N`) is at the TOP of the program, so we must NOT
    left-truncate the prefix to fit (that would drop the binding and make recall
    structurally impossible). Instead SKIP any example whose full
    program+answer exceeds `max_len` — the latent loss re-windows the prefix to
    max_prefix_len=max_len, so a kept example always has its binding in-window
    AND in the WM buffer the grad-forward builds from that same window. Distances
    above ~max_len are simply not trained (and the kill-gate is at ≥384, well
    inside a 768 window)."""
    from experiments.sft_code import _flatten_to_oneline
    rows = []
    n_skip_long = 0
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            ans = d.get("answer")
            if ans is None or not d.get("problem_prompt"):
                continue
            # MUST match the eval prompt EXACTLY (eval_longctx_recall builds
            # `"# " + _flatten_to_oneline(problem_prompt) + "\n"`). Training on the
            # raw multi-line prompt while the eval flattens to one line shifts the
            # token stream the WM addressing (W_q/W_k) sees — a silent train/eval
            # distribution mismatch (M0 review A2). Build the prefix identically.
            prompt = "# " + _flatten_to_oneline(d["problem_prompt"]) + "\n"
            pre = tok.encode(prompt, add_special_tokens=False)
            ans_ids = tok.encode(str(ans), add_special_tokens=False)
            if not ans_ids:
                continue
            ids = pre + ans_ids
            if len(ids) < 16:
                continue
            if len(ids) > max_len:          # binding would fall outside the window
                n_skip_long += 1
                continue
            rows.append((ids, len(pre)))
            if len(rows) >= limit:
                break
    if n_skip_long:
        print(f"[load_recall] skipped {n_skip_long} examples longer than "
              f"max_len={max_len} (binding-out-of-window); kept {len(rows)}",
              flush=True)
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
    ap.add_argument("--wm_on", action="store_true",
                    help="STAGE A: keep WorkingMemory on, attach a fresh DKV "
                         "addressing head + mem_alpha, train ONLY WM addressing "
                         "+ mem_alpha (trunk + adapter frozen → no-think path "
                         "byte-identical), on long-context recall data. The "
                         "cooperation latent step feeds adapter(z)+mem_alpha·WM. "
                         "Implies the recall dataset + freeze.")
    ap.add_argument("--recall_jsonl", default="data/longctx_recall_train.jsonl",
                    help="recall dataset for --wm_on (program + bound `answer`)")
    ap.add_argument("--unfreeze_trunk", action="store_true",
                    help="FIX-TEST (--wm_on only): also train the TRUNK (not just "
                         "WM+mem_alpha), so the hiddens can become content-"
                         "addressable. The no-think LM loss is added back (no "
                         "longer byte-identical). Trunk gets lr*--trunk_lr_mult; "
                         "WM/mem_alpha get full lr. Tests whether co-adapting the "
                         "trunk sharpens the read onto the binding (the root-cause "
                         "fix), at the cost of base-ability drift.")
    ap.add_argument("--trunk_lr_mult", type=float, default=0.2,
                    help="trunk lr multiplier under --unfreeze_trunk (WM stays 1x)")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.unfreeze_trunk and not args.wm_on:
        raise SystemExit("--unfreeze_trunk is only meaningful with --wm_on")
    if args.smoke:
        args.steps = 5
    if args.wm_on and args.max_len == 256:
        # The binding must stay inside the latent re-window (max_prefix_len=
        # max_len) AND the WM buffer the grad-forward builds from it. 768 holds
        # the kill-gate band (≥384) with margin while keeping the fixed-shape
        # extra-forward (max_positions × max_len) inside 32 GB; longer examples
        # are skipped by load_recall rather than binding-truncated.
        args.max_len = 768
    if args.wm_on and args.max_positions == 24:
        # Recall data supervises ONLY the answer span (~4 tokens/example; the
        # rest of the sequence is masked -100), so the default 24 would exceed
        # the valid-position count of a small batch and latent_cotrain_loss would
        # return None every step (P < max_positions → skip). 8 with batch≥6
        # (~24 valid positions) keeps the fixed-shape contract satisfiable AND
        # the (max_positions × max_len) grad extra-forward inside memory.
        args.max_positions = 8
    if args.train_gate:
        raise SystemExit(
            "--train_gate is rejected: under this trainer the only gradient-"
            "bearing loss (latent_cotrain_loss CE) never touches gate_head, so "
            "the gate would receive zero gradient and silently stay at its base "
            "values while looking 'co-calibrated'. Train the gate with "
            "gate_calibration (SFT) or latent_rl (RL) instead.")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    model, cfg, tid, tok, eos = load_latent_model(
        args.base, device, train=True, wm_on=args.wm_on, dkv=args.wm_on)
    # FREEZE-TRUNK: train ONLY the latent_feedback_adapter. The no-think forward
    # NEVER uses the adapter (it fires only at think positions), so freezing the
    # trunk leaves the no-think path BYTE-IDENTICAL to the original -> base code
    # ability preserved EXACTLY (no forgetting), while the adapter learns to
    # make latent thinking on code non-harmful.
    #
    # STAGE A (--wm_on): freeze EVERYTHING except the WM module (memory.*) and
    # mem_alpha. The no-think forward on think-free recall data masks the WM
    # injection to zero (read_mask = input_ids==think_id is empty), so WM params
    # are structurally inert there → the no-think path is byte-identical and base
    # ability is preserved EXACTLY, while the WM addressing (W_q/W_k/W_proj +
    # logit_scale/gate_bias_beta) and mem_alpha learn to retrieve the bound value
    # through the cooperation channel.
    if args.wm_on and args.unfreeze_trunk:
        # FIX-TEST: train EVERYTHING (trunk + WM + mem_alpha). The no-think path
        # is no longer byte-identical → frozen_base=False adds the LM loss back.
        ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
        has_dkv = any(n.startswith("memory.W_k") for n, _ in model.named_parameters())
        model.activation_checkpointing = True
        print(f"[wm_on+unfreeze] FIX-TEST: training trunk+WM+mem_alpha: "
              f"{ntrain:,} params (DKV={'on' if has_dkv else 'OFF!'}, "
              f"mem_alpha={float(model.mem_alpha.detach()):.3f}, "
              f"trunk_lr_mult={args.trunk_lr_mult}, act_ckpt=on)", flush=True)
    elif args.wm_on:
        ntrain = 0
        for n, p in model.named_parameters():
            keep = n.startswith("memory.") or n == "mem_alpha"
            p.requires_grad = keep
            ntrain += p.numel() if keep else 0
        # The grad latent extra-forward runs the full trunk over
        # (max_positions × max_len) R times; block-level activation checkpointing
        # cuts its stored activations ~n_layers× so the fixed-shape forward fits
        # in 32 GB. Only active when grad is enabled (the no-think readout is
        # no_grad → unaffected).
        model.activation_checkpointing = True
        has_dkv = any(n.startswith("memory.W_k") for n, _ in model.named_parameters())
        print(f"[wm_on] Stage A: training WM addressing + mem_alpha: {ntrain:,} "
              f"params (DKV={'on' if has_dkv else 'OFF!'}, "
              f"mem_alpha={float(model.mem_alpha.detach()):.3f}, "
              f"act_ckpt=on)", flush=True)
    elif args.freeze_trunk:
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
    # The no-think path is frozen byte-identical under --freeze_trunk (only the
    # think-only adapter trains) and plain --wm_on (only WM, inert on think-free
    # data, + mem_alpha). Either way the LM/KL terms are no-ops. NOT frozen under
    # --unfreeze_trunk (the whole point) → the LM loss is added back.
    frozen_base = args.freeze_trunk or (args.wm_on and not args.unfreeze_trunk)
    ref = None
    if args.kl_anchor > 0:
        if frozen_base:
            raise SystemExit("--kl_anchor with --freeze_trunk/--wm_on is a no-op "
                             "that costs a full ref model: the no-think path is "
                             "frozen byte-identical, so the KL is exactly 0.")
        ref, _, _, _, _ = load_latent_model(args.base, device, train=False)
        for p in ref.parameters():
            p.requires_grad = False
    if args.wm_on:
        rows = load_recall(args.recall_jsonl, tok, args.max_len, args.limit)
        print(f"[wm-cotrain] base={args.base} recall_examples={len(rows)} R={args.R} "
              f"data={args.recall_jsonl} selective={args.selective} "
              f"params={model.num_params():,}", flush=True)
    else:
        comment = "# Complete the following Python function.\n"
        rows = load_code(args.jsonl, tok, comment, args.max_len, args.min_score, args.limit)
        print(f"[code-cotrain] base={args.base} code_examples={len(rows)} R={args.R} "
              f"cw={args.cotrain_weight} selective={args.selective} "
              f"params={model.num_params():,}", flush=True)
    if not rows:
        raise SystemExit(f"no training rows loaded (check data path / fields)")

    # mem_alpha (and any FiLM-α-style scalar) gets NO weight decay — the FiLM-α
    # curriculum mandate (WD fights the gradient that grows it only if useful).
    # Under --unfreeze_trunk the TRUNK (non-WM, non-α) gets lr*trunk_lr_mult so
    # the big trunk adapts gently while the small WM addressing learns at full lr.
    def _is_wm(n):
        return n.startswith("memory.") or n == "mem_alpha"
    no_wd, wm_decay, trunk_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_film_alpha(n):
            no_wd.append(p)
        elif _is_wm(n) or not args.unfreeze_trunk:
            wm_decay.append(p)            # WM (and, when frozen, the only group)
        else:
            trunk_decay.append(p)         # trunk, only populated under unfreeze
    groups = [{"params": wm_decay, "weight_decay": 0.01, "lr": args.lr},
              {"params": no_wd, "weight_decay": 0.0, "lr": args.lr}]
    if trunk_decay:
        groups.append({"params": trunk_decay, "weight_decay": 0.01,
                       "lr": args.lr * args.trunk_lr_mult})
    opt = torch.optim.AdamW(groups, lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        idxs = [rng.randrange(len(rows)) for _ in range(args.batch)]
        inp, tgt = make_batch(rows, idxs, pad_id=0, device=device)
        opt.zero_grad()
        is_log_step = (step % args.log_every == 0 or step == 1)
        lm = torch.zeros((), device=device)
        kl = torch.zeros((), device=device)
        if frozen_base:
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
        if not frozen_base:
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
