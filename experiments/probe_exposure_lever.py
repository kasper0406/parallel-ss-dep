"""Validate the under-exposure lever ON THE REAL MODEL: does CLEAN, REPEATED
exposure fix failures that ~22 malformed exposures couldn't?

The capacity probe (probe_pkm_capacity.py) showed facts need ~100 gradient
exposures to lock in; the real model got ~22 (some malformed) for its failures.
If continue-training on CLEAN gold solutions of failing problems (with ~80
exposures each) flips them to pass, the root cause (under-exposure + dirty data,
NOT capacity) is confirmed on the real model and the real lever is: build a
cleaned, tail-up-sampled corpus and retrain.

Honest scope: training on a problem's gold then solving it is MEMORIZATION —
this confirms LEARNABILITY (the model CAN learn these given clean exposure, so
it's not capacity), not generalization to unseen problems. A disjoint CONTROL
set (not trained) checks for catastrophic forgetting.

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
      experiments/probe_exposure_lever.py checkpoints/sft_baked_pure.pt
"""
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from transformers import AutoTokenizer
from experiments.eval_bracket_structure import build_model_from_ckpt
from experiments.eval_humaneval import generate
import experiments.code_grader as CG

DEVICE = "cuda"


def clean_grade(prob, code):
    lines = code.split("\n"); st = 0
    for k, l in enumerate(lines):
        if re.match(r"^(def |import |from |class |@)", l):
            st = k; break
    body = "\n".join(lines[st:])
    names = re.findall(r"^def (\w+)", body, re.M)
    if names and prob.entry_point not in names:
        body = body + f"\n{prob.entry_point} = {names[0]}\n"
    return CG.grade(prob, body).passed


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft_baked_pure.pt"
    exposures = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
    mode = sys.argv[4] if len(sys.argv) > 4 else "full"   # full | pkm
    torch.backends.cuda.matmul.allow_tf32 = True
    m, cfg = build_model_from_ckpt(ckpt, force_state_readonly=True)
    m = m.to(DEVICE)
    tid = int(cfg.get("thinking_token_id", cfg["vocab_size"] - 1))
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    eos = tok.eos_token_id
    comment = "# Complete the following Python function.\n"
    probs = CG.LOADERS["mbpp_combined"]()

    def solved(prob):
        cids = tok.encode(comment + prob.prompt, add_special_tokens=False)
        with torch.no_grad():
            g, _ = generate(m, torch.tensor([cids], device=DEVICE), max_gen=200,
                            temperature=0.0, eos_token_id=eos, min_emit_before_eos=10)
        code = tok.decode([t for t in g[0, len(cids):].tolist() if t != tid],
                          skip_special_tokens=True)
        return clean_grade(prob, code)

    # find failing problems, split into TRAIN (we'll expose) + CONTROL (forgetting)
    m.eval()
    fails = []
    for prob in probs:
        if not prob.gold_solution:
            continue
        if not solved(prob):
            fails.append(prob)
        if len(fails) >= 80:
            break
    train_probs, ctrl_probs = fails[:40], fails[40:80]
    # also a set of problems it SOLVES (to check those aren't broken by training)
    solves = [p for p in probs[:300] if p.gold_solution and solved(p)][:30]
    print(f"[exposure-lever] ckpt={ckpt} exposures/ex={exposures} lr={lr}", flush=True)
    print(f"  train(fails)={len(train_probs)} control(fails)={len(ctrl_probs)} "
          f"solved-control={len(solves)}", flush=True)

    # build training batch tensors (comment+prompt -> gold code; mask prompt span)
    rows = []
    for p in train_probs:
        pre = tok.encode(comment + p.prompt, add_special_tokens=False)
        code = tok.encode("\n" + p.gold_solution.replace("\r\n", "\n"),
                          add_special_tokens=False)
        ids = (pre + code)[:320]
        if len(ids) > len(pre) + 2:
            rows.append((ids, len(pre)))

    def make_batch(batch_rows):
        L = max(len(r[0]) for r in batch_rows)
        inp = torch.zeros(len(batch_rows), L, dtype=torch.long)
        tgt = torch.full((len(batch_rows), L), -100, dtype=torch.long)
        for r, (ids, plen) in enumerate(batch_rows):
            t = torch.tensor(ids)
            inp[r, :len(ids)] = t
            tgt[r, :len(ids) - 1] = t[1:]
            tgt[r, :plen - 1] = -100
        return inp.to(DEVICE), tgt.to(DEVICE)

    def evalset(ps):
        return sum(int(solved(p)) for p in ps)

    tr0, ct0, sv0 = evalset(train_probs), evalset(ctrl_probs), evalset(solves)
    print(f"  BEFORE: train_pass={tr0}/{len(train_probs)} "
          f"control_pass={ct0}/{len(ctrl_probs)} solved_ctrl={sv0}/{len(solves)}",
          flush=True)

    # continue-train on clean gold of TRAIN problems.
    # mode=full: all params (interferes -> forgetting). mode=pkm: only the PKM
    # value/query params (addressable memory -> should add facts WITHOUT
    # overwriting the trunk -> no catastrophic forgetting). lr_mult for pkm: the
    # value-table gradient is heavily dampened, so use a higher LR (matches the
    # pkm_value_lr_mult lesson).
    m.train()
    m._gist_loss_enabled = False
    if mode in ("pkm", "pkmval"):
        # pkm    : all pkm_layer params (addressing + values) — shared addressing
        #          shifts globally -> still forgets.
        # pkmval : ONLY value tables, addressing FROZEN. New facts route to their
        #          pretrained slots; forgetting only via value-slot OVERLAP.
        #          If this forgets far less -> interference-free memory is
        #          achievable (frozen-address + value writes); the remaining
        #          forgetting is slot overlap (-> need sparser/dedicated slots).
        sel = "pkm_layer" if mode == "pkm" else "pkm_layer.values"
        ntr = 0
        for n, p in m.named_parameters():
            keep = sel in n
            p.requires_grad = keep
            ntr += p.numel() if keep else 0
        if ntr == 0:
            raise SystemExit(f"--mode {mode} but ckpt has no {sel} params")
        lr = lr * 50.0    # value-table gradient dampening compensation
        print(f"  [mode={mode}] training only {sel}: {ntr:,} params @ lr={lr:.1e}",
              flush=True)
    train_params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=lr)
    import random
    rng = random.Random(0)
    bs = 8
    steps = max(1, exposures * len(rows) // bs)
    for s in range(steps):
        bb = [rows[rng.randrange(len(rows))] for _ in range(bs)]
        inp, tgt = make_batch(bb)
        out = m(inp)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               tgt[:, :-1].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if (s + 1) % 20 == 0:
            print(f"    step {s+1}/{steps} loss={float(loss.detach()):.3f}", flush=True)

    m.eval()
    tr1, ct1, sv1 = evalset(train_probs), evalset(ctrl_probs), evalset(solves)
    print(f"\n=== Exposure-lever result ===")
    print(f"  TRAIN (clean exposed)  : {tr0} -> {tr1} / {len(train_probs)}  "
          f"(+{tr1-tr0})  <- did clean repeated exposure FIX them?")
    print(f"  CONTROL (failing, not exposed): {ct0} -> {ct1} / {len(ctrl_probs)}  "
          f"({ct1-ct0:+d})  <- spillover/none expected")
    print(f"  SOLVED-CONTROL (forgetting): {sv0} -> {sv1} / {len(solves)}  "
          f"({sv1-sv0:+d})  <- should stay ~flat")
    print(f"\nVERDICT: TRAIN jumps (e.g. {tr0}->high) with solved-control intact "
          f"=> the model CAN learn these given CLEAN repeated exposure; the "
          f"failures were under-exposure+dirty-data, NOT capacity. Real lever: "
          f"cleaned + tail-up-sampled corpus retrain. If TRAIN barely moves => "
          f"deeper capacity/generalization limit.")


if __name__ == "__main__":
    main()
