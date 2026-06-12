"""Depth-matched latent-reasoning co-train for real pretrain (2026-06-05).

The validated fix for "latent thinking is inert in real pretrain". The shipped
`thinking.latent_cotrain_loss` supervises R latent steps to predict a RANDOM
natural-text next token; that signal is broken (natural next-token rarely needs
sequential depth) so the trunk learns "thinking isn't needed" (probe: mean
Δlogp -0.69, only 28% of positions benefit, gate AUC 0.43). Instead, draw from a
DEPTH-BOUND, NON-PARALLELIZABLE reasoning corpus (pointer-chase f^K) and supervise
the ANSWER with R = the problem's depth + a depth curriculum — the recipe
validated standalone in `latent_arith_real.py` (fair lift +0.40-0.65 over a
fully-trained no-think baseline; autonomous gate allocates exactly n hops,
0.94-1.00). This co-trains it at low weight ALONGSIDE the general pretrain mix so
the trunk keeps general/code ability (post-pretrain bolt-ons go inert / forget).

Clean latent thread: WorkingMemory is toggled OFF and FiLM bypassed ONLY during
the latent forwards (WM injects at think positions and contaminates the fed-back
hidden — the documented blocker; the validated run used --no_memory + _film_bypass).
The main general-data forward is untouched.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from experiments.thinking import clean_latent_thread


def _answer_span_latent_loss(model, comment_ids, sol_ids, eos_id, R,
                             thinking_id, device, gate_weight=0.0):
    """Answer-span CE after an R-step latent ponder burst (one example).

    Self-contained twin of latent_sft.latent_sft_loss, but robust to the
    pretrain forward's return arity: with the gist loss active, training-mode
    `model(..., return_hidden=True)` returns (logits, hidden, gist) — a 3-tuple.
    The convention is hidden-at-index-1 (gist appended last), so we INDEX rather
    than tuple-unpack (the latent_sft version assumed exactly 2 → crashed here).

    ``gate_weight`` > 0 ALSO trains the output gate to invoke + halt thinking on
    its own (the latent_arith_real.autonomous_halt_loss recipe): P(emit) is
    supervised at the R+1 decision positions P-1..P+R-1 → THINK (0) for the
    first R, EMIT (1) at the last. Without this the bake installs the reasoning
    CAPABILITY but the gate never fires it (avg_steps≈0.77 vs target n).
    """
    base_ids = torch.tensor([comment_ids], dtype=torch.long, device=device)
    cur_ids = base_ids
    cur_emb = model.embed(base_ids)
    think_tok = torch.full((1, 1), int(thinking_id), dtype=torch.long,
                           device=device)
    P = len(comment_ids)
    for _ in range(int(R)):
        out = model(cur_ids, inputs_embeds=cur_emb, return_hidden=True)
        h = out[1]                                   # hidden = index 1 always
        z = h[:, -1:, :].to(cur_emb.dtype)
        z = model.apply_latent_feedback_adapter(z).to(cur_emb.dtype)
        cur_ids = torch.cat([cur_ids, think_tok], dim=1)
        cur_emb = torch.cat([cur_emb, z], dim=1)
    sol_t = torch.tensor([list(sol_ids) + [int(eos_id)]], dtype=torch.long,
                         device=device)
    full_ids = torch.cat([cur_ids, sol_t], dim=1)
    full_emb = torch.cat([cur_emb, model.embed(sol_t)], dim=1)
    out = model(full_ids, inputs_embeds=full_emb)
    logits = out[0] if isinstance(out, tuple) else out
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:].clone()
    start = P + int(R) - 1                            # supervise sol[0]..eos only
    shift_labels[:, :start] = -100
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), ignore_index=-100)
    if gate_weight > 0.0 and getattr(model, "_last_gate_logits", None) is not None:
        gate_logits = model._last_gate_logits             # (1, T) pre-sigmoid emit
        dec = list(range(P - 1, P + int(R)))              # R+1 decision positions
        gl = gate_logits[0, dec]
        tgt = torch.zeros(len(dec), device=device, dtype=gl.dtype)
        tgt[-1] = 1.0                                      # EMIT at the last
        gate_loss = F.binary_cross_entropy_with_logits(gl, tgt)
        loss = loss + float(gate_weight) * gate_loss
    return loss


def _load_rung(prefix: str, n: int, tok, max_len: int) -> list[tuple]:
    """Pointer-chase records → (comment_ids, answer_ids). Mirrors
    latent_arith_real._load_rung (answer rendered as `def solve(): return <ans>`)."""
    out = []
    path = f"{prefix}_n{n}.jsonl"
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        pfx = r["prompt"] + "\ndef solve():\n    return "
        c = tok.encode(pfx, add_special_tokens=False)
        s = tok.encode(str(r["answer"]), add_special_tokens=False)
        if len(c) + len(s) + n + 2 <= max_len:
            out.append((c, s))
    return out


class LatentReasoningCotrain:
    """Holds the depth-bound reasoning corpus and emits one answer-span latent
    loss per call, with a depth curriculum (ramp 1->max over 60% of steps, then
    uniform consolidation). Does NOT call backward — the caller adds the returned
    loss to the total and backprops with the rest of the step."""

    def __init__(self, train_prefix: str, rungs, tok, thinking_id: int,
                 eos_id: int, device, max_len: int = 256, no_ramp: bool = False,
                 gate_weight: float = 0.0, seed: int = 0):
        self.device = device
        self.thinking_id = int(thinking_id)
        self.eos_id = int(eos_id)
        self.no_ramp = bool(no_ramp)
        self.gate_weight = float(gate_weight)
        self.data: dict[int, list[tuple]] = {}
        for n in rungs:
            recs = _load_rung(train_prefix, int(n), tok, max_len)
            if recs:
                self.data[int(n)] = recs
        self.rungs = sorted(self.data.keys())
        if not self.rungs:
            raise ValueError(
                f"no latent-reasoning data at {train_prefix}_n*.jsonl for "
                f"rungs {list(rungs)}")
        self.g = torch.Generator().manual_seed(int(seed))
        self._ptr = {n: 0 for n in self.rungs}
        self._perm = {n: torch.randperm(len(self.data[n]), generator=self.g).tolist()
                      for n in self.rungs}

    def _next(self, n: int):
        if self._ptr[n] >= len(self._perm[n]):
            self._perm[n] = torch.randperm(len(self.data[n]),
                                           generator=self.g).tolist()
            self._ptr[n] = 0
        ex = self.data[n][self._perm[n][self._ptr[n]]]
        self._ptr[n] += 1
        return ex

    def _pick_rung(self, step: int, total_steps: int) -> int:
        if self.no_ramp:
            choices = self.rungs
        else:
            ramp_end = max(1.0, 0.6 * total_steps)
            if step < ramp_end:
                frac = step / ramp_end
                frontier = max(1, int(round(1 + frac * (len(self.rungs) - 1))))
                choices = self.rungs[:frontier]
            else:
                choices = self.rungs
        return choices[int(torch.randint(0, len(choices), (1,),
                                         generator=self.g).item())]

    def step(self, model, step: int, total_steps: int, n_examples: int):
        """Return (loss, rung): mean answer-span CE over n_examples at depth=rung
        (R=rung latent steps). Toggles a clean latent thread for the duration."""
        rung = self._pick_rung(step, total_steps)
        # clean_latent_thread: WM off (no contamination), FiLM bypass (validated
        # + faster at these tiny seqs), activation checkpointing off (the
        # checkpoint backward RECOMPUTES the FLA kernel at the latent path's
        # short/odd lengths → intermittent Blackwell "unspecified launch
        # failure"; full rationale in the contextmanager's docstring).
        with clean_latent_thread(model, film_bypass=True, no_activation_ckpt=True):
            total = None
            for _ in range(max(1, int(n_examples))):
                c, s = self._next(rung)
                l = _answer_span_latent_loss(model, c, s, self.eos_id, rung,
                                             self.thinking_id, self.device,
                                             gate_weight=self.gate_weight)
                total = l if total is None else total + l
            loss = total / max(1, int(n_examples))
        return loss, rung
