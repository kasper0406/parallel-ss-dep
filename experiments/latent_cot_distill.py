"""Per-step-supervised latent-CoT distillation for the small DeltaNet code model.

WHAT THIS IS
------------
The existing real-model latent co-train (`latent_code_cotrain.py`) supervises
ONLY the post-R-latent-think prediction of the true next code token
(answer-CE). On synthetic random-access reasoning, final-answer-only
supervision can't learn the per-step *selection* — it drifts to chance —
while PROCESS-SUPERVISION (teach the per-step target, then FREEZE the
mechanism) makes it hold, robust to <=30% label noise.

HYPOTHESIS: adding PER-STEP CoT supervision (latent step r is supervised
toward the teacher's r-th chain-of-thought step) + teach-then-FREEZE will
make latent thinking help MORE than answer-CE-only, without harming the base.

This trainer = the latent_code_cotrain recipe (adapter-only co-train, the
no-think path stays byte-identical → Pareto-safe) PLUS:

  1. CoT-step parser: teacher numbered CoT "1. ... 2. ..." -> ordered steps.
  2. Per-step latent supervision: run R = n_cot latent (Coconut, state-readonly,
     hidden-feedback) steps over the PROMPT, CE each step r toward the FIRST
     TOKEN of teacher CoT step r. Supports --label_noise / --label_coverage.
  3. Answer-CE: `thinking.latent_cotrain_loss` on the (prompt+solution)
     sequence (the validated latent_code_cotrain objective).
  4. Teach-then-FREEZE curriculum (--teach_frac, --freeze_mode): the per-step
     weight weans full->0 over the first teach_frac of steps; freeze mode
     additionally LOCKS the latent-mechanism params (the LatentFeedbackAdapter)
     at teach_frac while the answer path keeps training.
  5. --freeze_trunk (default): adapter-only co-train -> no-think path is
     byte-identical to the base (zero forgetting; the Pareto-safety property).

ARM SWITCH: --perstep_weight 0 == the control == answer-CE-only (latent_code_cotrain).

NOTE on the latent forward: `latent_think.think_forward` feeds the RAW trunk
hidden back (no adapter), so under --freeze_trunk it would have ZERO trainable
params and could not learn. We therefore mirror its structure but route the
fed-back hidden through the trainable LatentFeedbackAdapter (exactly what
`thinking._latent_think_logits_grad` / `latent_code_cotrain` train). Pass
--latent_mode raw to use `think_forward` verbatim (only sensible without
--freeze_trunk).

USAGE
-----
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python \
    experiments/latent_cot_distill.py --smoke

Single-GPU only. fp32 + tf32 (matches latent_code_cotrain).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time

sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from experiments.latent_think import think_forward            # reused (raw mode)
from experiments.optim_utils import _is_latent_feedback_adapter
from experiments.sft_code import _flatten_to_oneline
from experiments.thinking import (
    clean_latent_thread,
    latent_cotrain_loss,
    latent_think_logp,
    load_latent_model,
)

PAD_ID = 0   # latent_code_cotrain convention; pad token differs from thinking id


# ---------------------------------------------------------------------------
# 1. CoT-step parser
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(r"(?m)^[ \t]*(\d+)\.[ \t]+(.+?)[ \t]*$")


def parse_cot_steps(cot_text: str, max_steps: int | None = None) -> list[str]:
    """Extract the ordered reasoning steps from a numbered teacher CoT.

    Takes the FIRST contiguous 1,2,3,... run of numbered lines (Qwen often
    restates the plan as a second "Steps:\\n1. ...\\n2. ..." block — we stop at
    the second "1."). Placeholder steps whose body is exactly "..." are skipped
    (a chunk of the distill corpus has literal "1. ...\\n2. ..." placeholders).
    """
    steps: list[str] = []
    expected = 1
    for m in _STEP_RE.finditer(cot_text):
        num = int(m.group(1))
        body = m.group(2).strip()
        if num == 1 and steps:
            break                                  # second numbered block — stop
        if num == expected and body and body != "...":
            steps.append(body)
            expected += 1
        # else: out-of-order / placeholder line — skip, keep scanning
    if max_steps is not None:
        steps = steps[:max_steps]
    return steps


def _first_token_id(tok, step_text: str) -> int | None:
    """First token id of a CoT step — the cleanest 'what is this step about'
    signal. (Lossy by design; see report caveats.)"""
    ids = tok.encode(step_text, add_special_tokens=False)
    return int(ids[0]) if ids else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _prompt_text(d: dict) -> str:
    return d.get("problem_text") or d.get("problem_prompt") or ""


def _solution_text(d: dict) -> str:
    return (d.get("solution_code") or d.get("extracted_code") or "").strip()


def load_cot_examples(jsonl: str, tok, *, comment: str, max_len: int,
                      max_cot_steps: int, limit: int,
                      min_steps: int = 2) -> list[dict]:
    """Load (prompt, CoT-steps, solution) examples.

    Each row dict:
      prompt_ids        : token ids of `# {one-line problem}\\n`
      full_ids          : prompt_ids + solution_ids (for answer-CE)
      prompt_len        : len(prompt_ids)
      step_first_tokens : list[int] first-token id of each teacher CoT step
      n_cot             : len(step_first_tokens) (clamped to max_cot_steps)
    """
    rows: list[dict] = []
    n_no_steps = n_no_code = 0
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            cot = d.get("cot_text") or ""
            sol = _solution_text(d)
            prompt = _prompt_text(d)
            if not sol or sol == "..." or not prompt:
                n_no_code += 1
                continue
            steps = parse_cot_steps(cot, max_steps=max_cot_steps)
            if len(steps) < min_steps:
                n_no_steps += 1
                continue
            first_toks = [_first_token_id(tok, s) for s in steps]
            first_toks = [t for t in first_toks if t is not None]
            if len(first_toks) < min_steps:
                n_no_steps += 1
                continue
            prompt_line = f"# {_flatten_to_oneline(prompt)}\n" \
                if not (prompt.startswith("# ") and "\n" in prompt.rstrip("\n")) \
                else (prompt if prompt.endswith("\n") else prompt + "\n")
            prompt_ids = tok.encode(comment + prompt_line, add_special_tokens=False)
            sol_ids = tok.encode(sol, add_special_tokens=False)
            full = (prompt_ids + sol_ids)[:max_len]
            if len(full) - len(prompt_ids) < 4:
                n_no_code += 1
                continue
            rows.append({
                "prompt_ids": prompt_ids[:max_len],
                "full_ids": full,
                "prompt_len": len(prompt_ids),
                "step_first_tokens": first_toks,
                "n_cot": len(first_toks),
            })
            if len(rows) >= limit:
                break
    print(f"[load] {jsonl}: kept {len(rows)} (skipped {n_no_steps} no-steps, "
          f"{n_no_code} no-code)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# 2. Per-step latent forward (adapter-routed mirror of think_forward)
# ---------------------------------------------------------------------------
def _perstep_latent_logits(model, prefixes: torch.Tensor, R: int,
                           thinking_id: int, *, latent_mode: str = "adapter"
                           ) -> torch.Tensor:
    """Per-step latent logits (N, R, V) at the appended think slot.

    latent_mode='adapter' (default): mirrors `latent_think.think_forward` but
      feeds the trunk hidden back THROUGH `model.apply_latent_feedback_adapter`,
      so the (trainable) adapter is in the gradient path — required for the
      adapter-only --freeze_trunk co-train. Uses skip_lm_head + lm_head at the
      slot only (memory-frugal: avoids the full (N,L,V) logits per step).
    latent_mode='raw': verbatim `think_forward(return_steps=True)` (no adapter).
    """
    if latent_mode == "raw":
        return think_forward(model, prefixes, R, thinking_id,
                             mode="latent", return_steps=True)
    base_emb = model.embed(prefixes)                              # (N, L, d)
    N = prefixes.shape[0]
    think_col = torch.full((N, 1), int(thinking_id),
                           dtype=prefixes.dtype, device=prefixes.device)
    ids = torch.cat([prefixes, think_col], dim=1)                 # (N, L+1)
    h0 = model(prefixes, skip_lm_head=True)                       # (N, L, d)
    z = h0[:, -1:, :]
    step_logits = []
    for _ in range(max(1, int(R))):
        zi = model.apply_latent_feedback_adapter(z)              # adapter in graph
        ie = torch.cat([base_emb, zi.to(base_emb.dtype)], dim=1)  # (N, L+1, d)
        h = model(ids, inputs_embeds=ie, skip_lm_head=True)      # (N, L+1, d)
        z = h[:, -1:, :]
        step_logits.append(model.lm_head(h[:, -1, :]))           # (N, V)
    return torch.stack(step_logits, dim=1)                        # (N, R, V)


def build_prompt_batch(rows, idxs, max_R, device):
    """Left-pad prompts (so the last position is real for the latent init) and
    build per-step first-token targets (-100 beyond each example's n_cot)."""
    prompts = [rows[i]["prompt_ids"] for i in idxs]
    L = max(len(p) for p in prompts)
    inp = torch.full((len(idxs), L), PAD_ID, dtype=torch.long)
    for r, p in enumerate(prompts):
        inp[r, L - len(p):] = torch.tensor(p, dtype=torch.long)
    tgt = torch.full((len(idxs), max_R), -100, dtype=torch.long)
    for r, i in enumerate(idxs):
        st = rows[i]["step_first_tokens"][:max_R]
        for k, tid in enumerate(st):
            tgt[r, k] = tid
    return inp.to(device), tgt.to(device)


def perstep_ce_from_logits(logits, step_tgt_clean, vocab, *,
                           label_noise=0.0, label_coverage=1.0,
                           generator=None, device="cuda"):
    """Per-step CE + alignment given step logits (B,R,V) and clean per-step
    first-token targets (B,R) (-100 = padding beyond n_cot).

    label_coverage<1 drops a fraction of valid per-(b,r) targets (mask -100,
    process-supervision coverage knob); label_noise>0 flips targets to a random
    token (label-noise robustness knob). Alignment is always measured against
    the CLEAN target (the quality metric, unaffected by the knobs).
    """
    R_eff = logits.shape[1]
    tgt = step_tgt_clean[:, :R_eff].clone()
    valid = tgt != -100
    if valid.sum() == 0:
        return logits.sum() * 0.0, 0.0, 0
    if label_coverage < 1.0:
        drop = (torch.rand(tgt.shape, generator=generator, device=device)
                > label_coverage) & valid
        tgt[drop] = -100
    if label_noise > 0.0:
        flip = (torch.rand(tgt.shape, generator=generator, device=device)
                < label_noise) & (tgt != -100)
        rand = torch.randint(0, int(vocab), tgt.shape,
                             generator=generator, device=device)
        tgt = torch.where(flip, rand, tgt)
    n_kept = int((tgt != -100).sum())
    if n_kept == 0:
        return logits.sum() * 0.0, 0.0, 0
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           tgt.reshape(-1), ignore_index=-100)
    with torch.no_grad():
        pred = logits.argmax(-1)
        align = (((pred == step_tgt_clean[:, :R_eff]) & valid).sum().float()
                 / valid.sum().clamp(min=1))
    return loss, float(align), n_kept


def perstep_loss(model, inp, step_tgt_clean, R, thinking_id, vocab, *,
                 label_noise=0.0, label_coverage=1.0, latent_mode="adapter",
                 generator=None, device="cuda"):
    """Run the per-step latent forward, then per-step CE + alignment."""
    with clean_latent_thread(model, no_activation_ckpt=True):
        logits = _perstep_latent_logits(model, inp, R, thinking_id,
                                        latent_mode=latent_mode)   # (B,R,V)
    return perstep_ce_from_logits(logits, step_tgt_clean, vocab,
                                  label_noise=label_noise,
                                  label_coverage=label_coverage,
                                  generator=generator, device=device)


# ---------------------------------------------------------------------------
# 4. Teach-then-freeze curriculum
# ---------------------------------------------------------------------------
def wean_weight(step: int, steps: int, teach_frac: float,
                freeze_mode: str) -> float:
    """Per-step-loss weight multiplier w(t): 1.0 at t=0, linearly -> 0 at
    teach_frac*steps, 0 after (freeze/wean). Constant 1.0 for freeze_mode=none."""
    if freeze_mode == "none":
        return 1.0
    teach_steps = max(1.0, teach_frac * steps)
    return float(max(0.0, 1.0 - (step - 1) / teach_steps))


def should_freeze(step: int, steps: int, teach_frac: float,
                  freeze_mode: str) -> bool:
    """True once we've reached the teach->lock boundary (freeze mode only)."""
    return freeze_mode == "freeze" and (step - 1) >= teach_frac * steps


# ---------------------------------------------------------------------------
# 5/6. Evals
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_perstep_alignment(model, rows, idxs, max_R, thinking_id, *,
                           latent_mode="adapter", device="cuda") -> float:
    """Held-out per-step alignment: argmax(step_logits[r]) == teacher step-r
    first token, averaged over valid steps."""
    was_training = model.training
    model.eval()
    inp, tgt = build_prompt_batch(rows, idxs, max_R, device)
    with clean_latent_thread(model, no_activation_ckpt=True):
        logits = _perstep_latent_logits(model, inp, max_R, thinking_id,
                                        latent_mode=latent_mode)
    valid = tgt != -100
    pred = logits.argmax(-1)
    align = (((pred == tgt) & valid).sum().float()
             / valid.sum().clamp(min=1)).item()
    if was_training:
        model.train()
    return float(align)


@torch.no_grad()
def eval_answer_ce_fair(model, rows, idxs, max_R, thinking_id, *,
                        max_prefix_len=256, max_pairs=64, latent_mode="adapter",
                        device="cuda") -> dict:
    """Fair answer-CE: code-token prediction with latent ON (R=n_cot) vs the
    MANDATORY no-think control (R=0). Returns CE_off/CE_on/delta_logp.

    Samples code-token (prefix -> true_next) pairs from held-out rows, buckets
    by R, and compares lp0 (plain forward) against lpR (latent_think_logp)."""
    was_training = model.training
    model.eval()
    # Gather pairs: (prefix_ids, true_next, R)
    pairs: list[tuple[list[int], int, int]] = []
    for i in idxs:
        full = rows[i]["full_ids"]
        plen = rows[i]["prompt_len"]
        R = max(1, min(max_R, rows[i]["n_cot"]))
        positions = list(range(plen - 1, len(full) - 1))         # t s.t. t+1 is code
        if not positions:
            continue
        stride = max(1, len(positions) // 3)
        for t in positions[::stride][:3]:
            prefix = full[max(0, t + 1 - max_prefix_len): t + 1]
            pairs.append((prefix, int(full[t + 1]), R))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    if not pairs:
        if was_training:
            model.train()
        return {"ce_off": float("nan"), "ce_on": float("nan"),
                "delta_logp": float("nan"), "n": 0}

    # Bucket by R for batched latent forwards.
    by_R: dict[int, list[tuple[list[int], int]]] = {}
    for prefix, nxt, R in pairs:
        by_R.setdefault(R, []).append((prefix, nxt))

    lp0_all, lpR_all = [], []
    for R, items in by_R.items():
        Lmax = min(max_prefix_len, max(len(p) for p, _ in items))
        N = len(items)
        prefixes = torch.full((N, Lmax), PAD_ID, dtype=torch.long, device=device)
        nxt = torch.empty(N, dtype=torch.long, device=device)
        for r, (p, nt) in enumerate(items):
            p = p[-Lmax:]
            prefixes[r, Lmax - len(p):] = torch.tensor(p, device=device)
            nxt[r] = nt
        # no-think (R=0): plain forward, logp(true_next) at the last position.
        logits = model(prefixes)
        logits = logits[0] if isinstance(logits, tuple) else logits
        lp0 = F.log_softmax(logits[:, -1, :].float(), dim=-1).gather(
            1, nxt.view(-1, 1)).squeeze(1)
        # latent ON (R=n_cot)
        with clean_latent_thread(model, no_activation_ckpt=True):
            lpR = latent_think_logp(model, prefixes, nxt, R=R,
                                    thinking_token_id=thinking_id, pad_id=PAD_ID)
        lp0_all.append(lp0)
        lpR_all.append(lpR)
    lp0 = torch.cat(lp0_all)
    lpR = torch.cat(lpR_all)
    if was_training:
        model.train()
    return {"ce_off": float(-lp0.mean()), "ce_on": float(-lpR.mean()),
            "delta_logp": float((lpR - lp0).mean()), "n": int(lp0.numel())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft_qwen_coder_05b.pt")
    ap.add_argument("--save", default="checkpoints/latent_cot_distill.pt")
    ap.add_argument("--jsonl", default="data/cot_distill_v1.jsonl")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    # per-step supervision
    ap.add_argument("--perstep_weight", type=float, default=1.0,
                    help="0.0 => CONTROL arm (answer-CE-only = latent_code_cotrain)")
    ap.add_argument("--max_cot_steps", type=int, default=8,
                    help="clamp R / number of supervised latent steps")
    ap.add_argument("--label_noise", type=float, default=0.0,
                    help="per-(b,r) prob the per-step target is flipped random")
    ap.add_argument("--label_coverage", type=float, default=1.0,
                    help="per-(b,r) prob the per-step target is KEPT (else dropped)")
    ap.add_argument("--latent_mode", choices=["adapter", "raw"], default="adapter")
    # answer-CE (latent_cotrain_loss)
    ap.add_argument("--answer_weight", type=float, default=1.0)
    ap.add_argument("--answer_R", type=int, default=4)
    ap.add_argument("--max_positions", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--nothink_lm_weight", type=float, default=1.0,
                    help="no-think LM loss (preserve base) — only when NOT freeze_trunk")
    # teach-then-freeze
    ap.add_argument("--teach_frac", type=float, default=0.6)
    ap.add_argument("--freeze_mode", choices=["freeze", "wean", "none"],
                    default="freeze")
    ap.add_argument("--freeze_trunk", action="store_true", default=True,
                    help="adapter-only co-train (no-think byte-identical). Default on.")
    ap.add_argument("--no_freeze_trunk", dest="freeze_trunk", action="store_false")
    # data / bookkeeping
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--heldout", type=int, default=24)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="0 = eval only at init/teach-end/final")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.smoke:
        # Don't clobber explicitly-passed values (so `--smoke --steps 20` works).
        if args.steps == ap.get_default("steps"):
            args.steps = 60
        if args.limit == ap.get_default("limit"):
            args.limit = 300

    device = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.latent_mode == "raw" and args.freeze_trunk:
        raise SystemExit("--latent_mode raw has no trainable params under "
                         "--freeze_trunk (think_forward bypasses the adapter); "
                         "use --no_freeze_trunk or --latent_mode adapter.")

    model, cfg, tid, tok, eos = load_latent_model(
        args.base, device, train=True, wm_on=False)
    vocab = int(cfg["vocab_size"])

    # Trainable split: latent-mechanism params (adapter) vs answer-path params
    # (trunk + tied head; only trainable when NOT freeze_trunk). Everything else
    # frozen. The no-think forward never invokes the adapter, so freezing the
    # trunk leaves the no-think path BYTE-IDENTICAL to the base (Pareto-safe).
    latent_named = [(n, p) for n, p in model.named_parameters()
                    if _is_latent_feedback_adapter(n)]
    answer_named = [(n, p) for n, p in model.named_parameters()
                    if not _is_latent_feedback_adapter(n)]
    for _, p in latent_named:
        p.requires_grad = True
    for _, p in answer_named:
        p.requires_grad = not args.freeze_trunk
    latent_params = [p for _, p in latent_named]
    n_latent = sum(p.numel() for p in latent_params)
    n_answer = sum(p.numel() for _, p in answer_named) if not args.freeze_trunk else 0
    print(f"[trainable] latent(adapter)={n_latent:,}  answer={n_answer:,}  "
          f"freeze_trunk={args.freeze_trunk}  freeze_mode={args.freeze_mode}",
          flush=True)
    # GUARD: adapter-only (freeze_trunk) + freeze_mode=freeze would freeze the
    # adapter AND the trunk at teach_frac -> ZERO trainable params -> the last
    # (1-teach_frac) of steps silently no-op (loss.requires_grad False, backward
    # skipped). For adapter-only the answer-compute must keep learning after the
    # per-step teacher fades -> use wean. (Both code reviews flagged this.)
    if args.freeze_trunk and args.freeze_mode == "freeze":
        raise SystemExit(
            "freeze_trunk + freeze_mode=freeze => after teach_frac the adapter AND "
            "trunk are frozen => 0 trainable params => the last (1-teach_frac) of "
            "steps train nothing. Use --freeze_mode wean for adapter-only "
            "(the adapter keeps learning answer-CE after the per-step teacher weans).")
    if not args.freeze_trunk:
        model.activation_checkpointing = True

    # Optimizer: no WD on the adapter scalar α (FiLM-α mandate).
    no_wd, decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_wd if n.endswith(".alpha") else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_wd, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    comment = "# Complete the following Python function.\n"
    rows = load_cot_examples(args.jsonl, tok, comment=comment, max_len=args.max_len,
                             max_cot_steps=args.max_cot_steps, limit=args.limit)
    if len(rows) < args.heldout + args.batch:
        raise SystemExit(f"too few rows ({len(rows)})")
    n_held = min(args.heldout, max(8, len(rows) // 10))
    held_rows = rows[-n_held:]
    train_rows = rows[:-n_held]
    held_idx = list(range(len(held_rows)))
    import statistics
    print(f"[data] train={len(train_rows)} heldout={len(held_rows)}  "
          f"median n_cot={statistics.median(r['n_cot'] for r in rows)}  "
          f"arm={'CONTROL(answer-only)' if args.perstep_weight==0 else 'TREATMENT'}",
          flush=True)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = random.Random(args.seed)

    def run_evals(tag):
        al = eval_perstep_alignment(model, held_rows, held_idx, args.max_cot_steps,
                                    tid, latent_mode=args.latent_mode, device=device)
        ce = eval_answer_ce_fair(model, held_rows, held_idx, args.max_cot_steps,
                                 tid, max_prefix_len=args.max_len, device=device)
        print(f"[eval:{tag}] perstep_align={al:.3f}  "
              f"answer CE_off(no-think)={ce['ce_off']:.3f}  "
              f"CE_on(latent R=n_cot)={ce['ce_on']:.3f}  "
              f"Δlogp(on-off)={ce['delta_logp']:+.3f}  n={ce['n']}", flush=True)
        return al, ce

    align0, _ = run_evals("init")
    teach_end_done = False
    t0 = time.time()
    frozen = False

    for step in range(1, args.steps + 1):
        if should_freeze(step, args.steps, args.teach_frac, args.freeze_mode) \
                and not frozen:
            for p in latent_params:
                p.requires_grad = False
            frozen = True
            print(f"[freeze] step {step}: latent-mechanism (adapter) LOCKED "
                  f"({n_latent:,} params); answer path "
                  f"{'still trains' if n_answer else 'also frozen'}", flush=True)
            if not teach_end_done:
                run_evals("teach-end")
                teach_end_done = True

        idxs = [rng.randrange(len(train_rows)) for _ in range(args.batch)]
        opt.zero_grad(set_to_none=True)

        # --- per-step supervision (treatment only) ---
        w = wean_weight(step, args.steps, args.teach_frac, args.freeze_mode)
        ps_loss = torch.zeros((), device=device)
        ps_align, ps_n = 0.0, 0
        if args.perstep_weight > 0 and w > 0:
            R = max(1, min(args.max_cot_steps,
                           max(train_rows[i]["n_cot"] for i in idxs)))
            inp, step_tgt = build_prompt_batch(train_rows, idxs, R, device)
            ps_loss, ps_align, ps_n = perstep_loss(
                model, inp, step_tgt, R, tid, vocab,
                label_noise=args.label_noise, label_coverage=args.label_coverage,
                latent_mode=args.latent_mode, generator=gen, device=device)

        # --- answer-CE (validated latent_code_cotrain objective) ---
        full = [train_rows[i]["full_ids"] for i in idxs]
        plens = [train_rows[i]["prompt_len"] for i in idxs]
        L = max(len(s) for s in full)
        a_inp = torch.full((len(full), L), PAD_ID, dtype=torch.long)
        a_tgt = torch.full((len(full), L), -100, dtype=torch.long)
        for r, (ids, pl) in enumerate(zip(full, plens)):
            t = torch.tensor(ids, dtype=torch.long)
            a_inp[r, :len(ids)] = t
            a_tgt[r, :len(ids) - 1] = t[1:]
            a_tgt[r, :pl - 1] = -100
        a_inp, a_tgt = a_inp.to(device), a_tgt.to(device)
        res = latent_cotrain_loss(model, a_inp, a_tgt, R=args.answer_R,
                                  thinking_token_id=tid,
                                  max_positions=args.max_positions,
                                  max_prefix_len=args.max_len, pad_id=PAD_ID,
                                  eos_id=eos, generator=gen)
        if res is None:
            a_loss, a_dlp, a_n = torch.zeros((), device=device), float("nan"), 0
        else:
            a_loss, a_dlp, a_n = res

        # --- optional no-think LM loss (preserve base; only when unfrozen) ---
        lm_loss = torch.zeros((), device=device)
        if not args.freeze_trunk and args.nothink_lm_weight > 0:
            out = model(a_inp)
            lg = out[0] if isinstance(out, tuple) else out
            lm_loss = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                      a_tgt[:, :-1].reshape(-1), ignore_index=-100)

        loss = args.answer_weight * a_loss \
            + args.perstep_weight * w * ps_loss \
            + (args.nothink_lm_weight * lm_loss if not args.freeze_trunk else 0.0)

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step {step:>4}/{args.steps}  "
                  f"answer_ce={float(a_loss):.3f} (Δlogp={a_dlp:+.2f}, n={a_n})  "
                  f"perstep_ce={float(ps_loss):.3f} (align={ps_align:.3f}, "
                  f"w={w:.2f}, n={ps_n})  "
                  f"lm={float(lm_loss):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            run_evals(f"step{step}")

    if not teach_end_done and args.freeze_mode == "freeze":
        run_evals("teach-end")
    alignF, ceF = run_evals("final")
    print(f"\n[SUMMARY] arm={'CONTROL' if args.perstep_weight==0 else 'TREATMENT'}  "
          f"perstep_align {align0:.3f} -> {alignF:.3f}  "
          f"answer Δlogp(on-off)={ceF['delta_logp']:+.3f}  "
          f"(CE no-think {ceF['ce_off']:.3f} vs latent {ceF['ce_on']:.3f})",
          flush=True)

    if not args.smoke:
        torch.save({"state_dict": model.state_dict(), "step": args.steps,
                    "config": cfg}, args.save)
        print(f"[saved] {args.save}", flush=True)
    else:
        print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
