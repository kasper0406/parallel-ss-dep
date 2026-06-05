"""
HumanEval pass@1 / pass@k evaluation.

Phase 2 of post-PPL evals. Tests if architectural differences hide
behind PPL parity show up as actual code-generation differences.

Caveat: 135M / 5K-step models will likely score very low on HumanEval
in absolute terms. The point is relative comparison: does film @30L
beat DN @30L on actual code generation, even if PPL is tied?

Usage:
  python experiments/eval_humaneval.py \\
      --ckpt /path/to/model.pt \\
      --n_samples 1 --temperature 0.0 --max_gen 256
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pathlib
import re
import signal
import sys
from contextlib import contextmanager

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from experiments.eval_bracket_structure import build_model_from_ckpt


# Stop sequences typical of function-end at column 0.
_STOP_SEQUENCES = ["\nclass ", "\ndef ", "\nif __name__", "\n#", "\nprint("]


@contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _run_test_in_subprocess(code: str, test: str, entry_point: str,
                             timeout_s: int = 5) -> bool:
    """Execute code+test in a subprocess; return True if check passes."""
    def target(code, test, entry_point, q):
        try:
            ns = {}
            with time_limit(timeout_s):
                exec(code, ns)
                exec(test, ns)
                ns["check"](ns[entry_point])
            q.put(True)
        except Exception:
            q.put(False)

    q = mp.Queue()
    p = mp.Process(target=target, args=(code, test, entry_point, q))
    p.start()
    p.join(timeout=timeout_s + 2)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    try:
        return q.get_nowait()
    except Exception:
        return False


def _truncate_at_stop(text: str) -> str:
    """Trim generated text at first natural function boundary."""
    earliest = len(text)
    for stop in _STOP_SEQUENCES:
        idx = text.find(stop)
        if idx >= 0 and idx < earliest:
            earliest = idx
    return text[:earliest]


@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, max_gen: int = 256,
             temperature: float = 0.0, eos_token_id: int | None = None,
             use_thinking: bool = False,
             thinking_token_id: int | None = None,
             max_think_per_step: int = 8,
             total_think_budget: int | None = None,
             emit_threshold: float = 0.5,
             min_emit_before_eos: int = 0,
             gate_floor: float = 0.0,
             use_incremental: bool = True,
             ) -> tuple[torch.Tensor, dict]:
    """Token-by-token generation.

    When `use_thinking=True`, after each forward pass we consult the output
    gate at the final position. The convention (matching `thinking.py`
    rollout) is: g = σ(gate_head(h)), where g = P(Emit). If g < emit_threshold
    we append `thinking_token_id` and run forward again, up to
    `max_think_per_step` consecutive thinks per emit step; we always force an
    emit when the budget is exhausted. `total_think_budget` (default 2×max_gen)
    caps the lifetime sum of think tokens.

    Returns (out_ids_including_thinks, diagnostics_dict). The caller is
    responsible for stripping `thinking_token_id` from `out` before grading.
    """
    if use_thinking:
        assert thinking_token_id is not None, \
            "use_thinking=True requires thinking_token_id"
        if total_think_budget is None:
            total_think_budget = 2 * max_gen
    # Note: do NOT set `model._film_bypass = True`. The v2 model is
    # trained WITH FiLM and degenerates badly at temperature sampling
    # without it. See train_rl_grader.rollout_group_batched for the
    # full diagnosis (2026-05-23 root-cause).
    # STATE-PASSING INCREMENTAL DECODE (2026-05-23). Replaces the
    # per-step full-forward `model(out)` with `prefill(prompt)` + one
    # `forward_step(next_tok)` per generated token. Constant per-token
    # cost (~6.5 ms) vs the previous T-linear cost (~30 ms/tok at
    # T=512). Falls back to the full-forward path when
    # `use_incremental=False` (sanity flag) or when the loaded model
    # lacks `forward_step` (older ckpts not affected, since the method
    # was added to TinyLM and ALL ckpts use TinyLM).
    can_incremental = (use_incremental
                       and hasattr(model, "forward_step")
                       and hasattr(model, "prefill"))
    if can_incremental:
        cache, last_logits = model.prefill(prompt_ids)
        # `last_logits` at [:, -1] predicts the FIRST emitted/thought
        # token; cache the "current next-logits" for the inner loop.
        pending_logits = last_logits[:, -1:, :]
        cache_pending_valid = True
    else:
        cache = None
        pending_logits = None
        cache_pending_valid = False

    out = prompt_ids.clone()
    emit_count = 0
    think_total = 0
    think_steps_used = []      # length = emit_count; thinks before each emit
    gate_emit_values = []      # gate values at each emit step
    while emit_count < max_gen:
        # Inner think loop: try to coax the gate above emit_threshold.
        thinks_this_step = 0
        while True:
            if can_incremental:
                # The first time through the loop after either a prior
                # forward_step or prefill, `pending_logits` already
                # holds the next-token distribution conditioned on the
                # tokens currently in the cache. No extra forward
                # needed — but we still need _last_gate to have been
                # populated for the gate decision (it was set inside
                # forward_step / prefill).
                if not cache_pending_valid:
                    # Shouldn't reach here in steady state.
                    raise RuntimeError("incremental: pending_logits invalid")
                next_logits = pending_logits[:, -1, :]
            else:
                logits = model(out)
                next_logits = logits[:, -1, :]
            if not use_thinking:
                gate_val = 1.0  # force emit
                break
            # σ(gate_head(h)) at last position; >threshold ⇒ emit.
            # Mirror training's clamp: during pretrain the loss used
            # g.clamp(min=gate_floor_min), so the gate is indifferent to raw
            # values below the floor. Apply the same clamp at inference
            # to keep the train/inference gate distributions aligned.
            gate_t = getattr(model, "_last_gate", None)
            if gate_t is None:
                gate_val = 1.0
            else:
                gate_val = float(gate_t[0, -1].item())
                if gate_floor > 0.0:
                    gate_val = max(gate_val, gate_floor)
            force_emit = (
                thinks_this_step >= max_think_per_step
                or think_total >= total_think_budget
            )
            if gate_val >= emit_threshold or force_emit:
                break
            # Else: append a think token and loop.
            think_tok = torch.full((out.shape[0], 1), int(thinking_token_id),
                                    dtype=out.dtype, device=out.device)
            out = torch.cat([out, think_tok], dim=1)
            if can_incremental:
                pending_logits, cache = model.forward_step(think_tok, cache)
                cache_pending_valid = True
            thinks_this_step += 1
            think_total += 1
        # Emit step: mask thinking_token_id from sampled output.
        if use_thinking and thinking_token_id is not None:
            next_logits = next_logits.clone()
            next_logits[..., int(thinking_token_id)] = -float("inf")
        # Suppress EOS for the first `min_emit_before_eos` emitted tokens.
        # Works around the well-known pretrain "halt-after-docstring" trap
        # where small documents in the training data place EOS right after a
        # closing `"""`, biasing the model to stop at HumanEval's
        # prompt-final docstring.
        if (eos_token_id is not None
                and min_emit_before_eos > 0
                and emit_count < min_emit_before_eos):
            if not (use_thinking and thinking_token_id is not None):
                next_logits = next_logits.clone()
            next_logits[..., int(eos_token_id)] = -float("inf")
        if temperature == 0.0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_tok], dim=1)
        # Advance the incremental cache by the emitted token. If we're
        # about to break (max_gen reached or EOS), we still advance for
        # symmetry — cheap, and it keeps `pending_logits` consistent if
        # the caller resumes generation (unlikely but harmless).
        if can_incremental:
            pending_logits, cache = model.forward_step(next_tok, cache)
            cache_pending_valid = True
        emit_count += 1
        if use_thinking:
            think_steps_used.append(thinks_this_step)
            gate_emit_values.append(gate_val)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    diag = {
        "emit_count": emit_count,
        "think_total": think_total,
        "think_steps_used": think_steps_used,
        "gate_emit_values": gate_emit_values,
        "think_rate": (think_total / max(1, think_total + emit_count))
                       if use_thinking else 0.0,
        "decode_path": "incremental" if can_incremental else "full_forward",
    }
    return out, diag


@torch.no_grad()
def generate_with_retrieval_as_input(
    model, prompt_ids: torch.Tensor, max_gen: int = 256,
    temperature: float = 0.0, eos_token_id: int | None = None,
    thinking_token_id: int | None = None,
    max_think_per_step: int = 8,
    total_think_budget: int | None = None,
    emit_threshold: float = 0.5,
    min_emit_before_eos: int = 0,
    gate_floor: float = 0.0,
    additive: bool = True,
    use_incremental: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Retrieval-as-input thinking-token generation (2026-05-19).

    DIFFERENCE FROM `generate`:
      In the standard `generate`, when the gate decides to think we
      append a discrete [THINKING] token. Its input embedding is the
      SAME on every think step (since [THINKING] is one vocab entry),
      so successive think positions are highly correlated (median
      pairwise cos 0.146 vs 0.060 at emit — see diag_think_position_
      diversity.py).

      Here, when the gate decides to think we INSTEAD inject the
      WorkingMemory retrieval result at that position as the input
      embedding for the next forward — bypassing the [THINKING] token's
      embedding lookup. The retrieval depends on the current hidden
      state, so each think step gets a unique input signal and the
      think-position hidden states become diverse.

      Concretely:
        forward(input_ids → embed table) → h, gate, WM injection stash
        if think:
          next_input_emb = model.memory._last_injection[0, -1, :]
                           (the WM read at the last position)
          input_ids = cat(input_ids, [THINKING_ID])
          inputs_embeds = cat(prior_embeds, next_input_emb)
                          ← passed to next forward instead of input_ids embed
        else:
          emit as usual

    Requires the loaded model to have memory + the new _last_injection
    stash (added 2026-05-19).
    """
    if total_think_budget is None:
        total_think_budget = 2 * max_gen
    if thinking_token_id is None:
        raise ValueError("generate_with_retrieval_as_input requires "
                         "thinking_token_id")
    if not hasattr(model, "memory"):
        raise ValueError("model lacks .memory — this generator needs "
                         "WorkingMemory to provide the retrieval.")
    # Note: do NOT set `model._film_bypass = True`. See the comment in
    # `generate()` above — the v2 model degenerates at temperature
    # sampling without FiLM. Diagnosed 2026-05-23.
    device = prompt_ids.device
    out = prompt_ids.clone()
    # inputs_embeds is the per-position model input; starts as the
    # embedding lookup of the prompt and grows with each gen step.
    # In the INCREMENTAL path we only track the LAST position's
    # `next_input_emb`; in the FULL-FORWARD fallback we grow the full
    # tensor as before.
    inputs_embeds = model.embed(out).clone()    # (B, prompt_len, d)
    # v7 additive α-gated injection: at a think step the next input is
    # think_embed + α·retrieval (α = model.retrieval_input_alpha),
    # instead of v5/v6's destructive `next input = retrieval`. `additive`
    # is set from cfg['retrieval_input_additive'] by the caller — True
    # for v7 ckpts, False for v5/v6.
    _alpha_p = getattr(model, "retrieval_input_alpha", None)
    alpha = float(_alpha_p.detach()) if _alpha_p is not None else 0.1

    # STATE-PASSING INCREMENTAL DECODE (2026-05-23). For the retrieval-
    # as-input generator the per-step input is `inputs_embeds[:, -1:]`,
    # so we maintain it as the single-position tensor we'll pass to
    # forward_step. The prefill runs once over the whole prompt.
    can_incremental = (use_incremental
                       and hasattr(model, "forward_step")
                       and hasattr(model, "prefill"))
    if can_incremental:
        cache, last_logits = model.prefill(out, inputs_embeds=inputs_embeds)
        pending_logits = last_logits[:, -1:, :]
    else:
        cache = None
        pending_logits = None

    emit_count = 0
    think_total = 0
    think_steps_used: list[int] = []
    gate_emit_values: list[float] = []
    while emit_count < max_gen:
        thinks_this_step = 0
        # Inner think loop: keep retrieving until gate flips to emit or
        # we hit a cap.
        while True:
            if can_incremental:
                next_logits = pending_logits[:, -1, :]
            else:
                logits = model(out, inputs_embeds=inputs_embeds)
                next_logits = logits[:, -1, :]
            gate_t = getattr(model, "_last_gate", None)
            gate_val = (1.0 if gate_t is None
                        else float(gate_t[0, -1].item()))
            if gate_floor > 0.0:
                gate_val = max(gate_val, gate_floor)
            force_emit = (
                thinks_this_step >= max_think_per_step
                or think_total >= total_think_budget
            )
            if gate_val >= emit_threshold or force_emit:
                break
            # THINK STEP: replace the next input embedding with the WM
            # read at the current last position. _last_injection has
            # shape (B, T_current, d); we want the read AT the last
            # position (it's the query computed from h_t which depended
            # on the full context).
            inj = getattr(model.memory, "_last_injection", None)
            if inj is None:
                raise RuntimeError(
                    "memory._last_injection missing — model.memory.forward "
                    "must stash injection (added 2026-05-19).")
            retrieved = inj[:, -1:, :].to(inputs_embeds.dtype)   # (B, 1, d)
            think_tok = torch.full((out.shape[0], 1), int(thinking_token_id),
                                    dtype=out.dtype, device=device)
            out = torch.cat([out, think_tok], dim=1)
            if additive:
                # v7: think_embed + α·retrieval — the think token keeps
                # its own signal; the retrieval is a gated addition.
                think_emb = model.embed(think_tok).to(inputs_embeds.dtype)
                inj_input = think_emb + alpha * retrieved
            else:
                # v5/v6: destructive replacement (kept for those ckpts).
                inj_input = retrieved
            if can_incremental:
                pending_logits, cache = model.forward_step(
                    think_tok, cache, inputs_embeds=inj_input,
                    # Force the WM read at this position (regardless of
                    # whether the appended token is the thinking token —
                    # we want injection stashed so the NEXT iteration
                    # can read `_last_injection`).
                    mem_read_mask=torch.ones_like(think_tok, dtype=inj_input.dtype),
                )
            else:
                inputs_embeds = torch.cat([inputs_embeds, inj_input], dim=1)
            thinks_this_step += 1
            think_total += 1
        # EMIT STEP: same as standard generate.
        if thinking_token_id is not None:
            next_logits = next_logits.clone()
            next_logits[..., int(thinking_token_id)] = -float("inf")
        if (eos_token_id is not None
                and min_emit_before_eos > 0
                and emit_count < min_emit_before_eos):
            next_logits[..., int(eos_token_id)] = -float("inf")
        if temperature == 0.0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_tok], dim=1)
        # For an emit step we use the regular embedding-table lookup.
        emit_emb = model.embed(next_tok).to(inputs_embeds.dtype)
        if can_incremental:
            # Advance cache with the emit token. inputs_embeds is the
            # regular embedding-table lookup of the emitted token.
            pending_logits, cache = model.forward_step(
                next_tok, cache, inputs_embeds=emit_emb,
                # Same as above: force the WM read so _last_injection
                # is populated for the next think iteration to consume.
                mem_read_mask=torch.ones_like(next_tok, dtype=emit_emb.dtype),
            )
        else:
            inputs_embeds = torch.cat([inputs_embeds, emit_emb], dim=1)
        emit_count += 1
        think_steps_used.append(thinks_this_step)
        gate_emit_values.append(gate_val)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    diag = {
        "emit_count": emit_count,
        "think_total": think_total,
        "think_steps_used": think_steps_used,
        "gate_emit_values": gate_emit_values,
        "think_rate": (think_total / max(1, think_total + emit_count)),
        "mode": "retrieval_as_input",
        "decode_path": "incremental" if can_incremental else "full_forward",
    }
    return out, diag


@torch.no_grad()
def generate_latent_think(
    model, prompt_ids: torch.Tensor, max_gen: int = 256,
    temperature: float = 0.0, eos_token_id: int | None = None,
    thinking_token_id: int | None = None,
    max_think_per_step: int = 8,
    total_think_budget: int | None = None,
    emit_threshold: float = 0.5,
    min_emit_before_eos: int = 0,
    gate_floor: float = 0.0,
    force_prefix_think: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Latent-ponder generation (2026-05-28, `THINKING_LATENT_2026_05_28.md`).

    `force_prefix_think` (>0): run exactly that many latent steps BEFORE the
    first emit, then emit the rest with no further thinking. This matches the
    `latent_sft.py` "think-before-solution" training layout; with it set the
    per-step gate is ignored.

    At a think step the next input embedding is the model's OWN hidden state
    at the last position (Coconut-style continuous feedback), and think
    positions run STATE-READONLY (DeltaNet β=0) so the recurrent state — the
    long-range bindings — is never corrupted. The emit after a think burst is
    conditioned on the final refined latent (we emit from the last think
    slot's logits, exactly as the validated synthetic/arith decode).

    Unlike the discrete-[THINKING] `generate` (one homogeneous embedding) or
    `generate_with_retrieval_as_input` (a WM lookup), this feeds back a full
    d_model continuous vector — maximum thinking bandwidth.

    Requires the model loaded with `state_readonly_at_think=True`
    (build_model_from_ckpt(force_state_readonly=True)). Full-forward path only
    (correctness-first; the prefix is re-read each step).
    """
    if total_think_budget is None:
        total_think_budget = 2 * max_gen
    if thinking_token_id is None:
        raise ValueError("generate_latent_think requires thinking_token_id")
    # state_readonly_at_think is the validated default, but this is a FULL-forward
    # path (the whole sequence is re-read every step), so it is equally correct for
    # a STATE-WRITABLE model (think tokens write to S) — the eval forward then
    # matches a state-writable training forward exactly. Only warn, don't block
    # (2026-06-04: the CoT-analog state-writable experiment needs this eval).
    if not getattr(model, "state_readonly_at_think", False):
        print("  [generate_latent_think] note: state_readonly_at_think=False "
              "(state-writable) — full-forward eval matches state-writable training.")
    device = prompt_ids.device
    out = prompt_ids.clone()
    inputs_embeds = model.embed(out).clone()      # (B, prompt_len, d)
    emit_count = 0
    think_total = 0
    think_steps_used: list[int] = []
    gate_emit_values: list[float] = []
    while emit_count < max_gen:
        thinks_this_step = 0
        while True:
            logits, h = model(out, inputs_embeds=inputs_embeds,
                              return_hidden=True)
            next_logits = logits[:, -1, :]
            gate_t = getattr(model, "_last_gate", None)
            gate_val = (1.0 if gate_t is None
                        else float(gate_t[0, -1].item()))
            if gate_floor > 0.0:
                gate_val = max(gate_val, gate_floor)
            if force_prefix_think > 0:
                # Forced burst before the first emit only; then no thinking.
                want_think = (emit_count == 0
                              and thinks_this_step < force_prefix_think)
                if not want_think:
                    break
            else:
                force_emit = (thinks_this_step >= max_think_per_step
                              or think_total >= total_think_budget)
                if gate_val >= emit_threshold or force_emit:
                    break
            # THINK STEP: feed the model's own hidden back as the next input
            # embedding. The appended think token makes β=0 fire at this
            # position (state-readonly), so the recurrence is not written.
            latent = h[:, -1:, :].to(inputs_embeds.dtype)     # (B, 1, d)
            # Learned input adapter: map the fed-back out_norm hidden into the
            # input-embedding manifold (identity when the model has none /
            # untrained — byte-identical to the prior path then). Mirrors the
            # measurement/grad primitives in thinking.py so inference matches
            # training.
            latent = model.apply_latent_feedback_adapter(latent).to(
                inputs_embeds.dtype)
            # Unified hybrid (auto-detected via model.mem_alpha): augment the
            # hidden-feedback thread with a learned-α WM retrieval so the model
            # pulls in new info as it thinks (THINKING_MEMORY_PLAN D8/D11).
            _ma = getattr(model, "mem_alpha", None)
            if _ma is not None and hasattr(model, "memory"):
                inj = getattr(model.memory, "_last_injection", None)
                if inj is not None:
                    latent = latent + _ma.to(inputs_embeds.dtype) * \
                        inj[:, -1:, :].to(inputs_embeds.dtype)
            think_tok = torch.full((out.shape[0], 1), int(thinking_token_id),
                                   dtype=out.dtype, device=device)
            out = torch.cat([out, think_tok], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, latent], dim=1)
            thinks_this_step += 1
            think_total += 1
        # EMIT STEP — conditioned on the final refined latent.
        next_logits = next_logits.clone()
        next_logits[..., int(thinking_token_id)] = -float("inf")
        if (eos_token_id is not None and min_emit_before_eos > 0
                and emit_count < min_emit_before_eos):
            next_logits[..., int(eos_token_id)] = -float("inf")
        if temperature == 0.0:
            next_tok = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_tok], dim=1)
        inputs_embeds = torch.cat(
            [inputs_embeds, model.embed(next_tok).to(inputs_embeds.dtype)], dim=1)
        emit_count += 1
        think_steps_used.append(thinks_this_step)
        gate_emit_values.append(gate_val)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    diag = {
        "emit_count": emit_count,
        "think_total": think_total,
        "think_steps_used": think_steps_used,
        "gate_emit_values": gate_emit_values,
        "think_rate": (think_total / max(1, think_total + emit_count)),
        "mode": "latent_think",
        "decode_path": "full_forward",
    }
    return out, diag


def _clone_fla_cache(fla_cache):
    """Deep-clone an FLA Cache so a what-if forward_step on the clone does
    not mutate the original. Each per-layer state slot holds a
    `recurrent_state` (Tensor or tuple of Tensors), `attn_state` (tuple),
    `conv_state`, `ffn_state` — we clone tensors and copy the dict spine.
    """
    from fla.models.utils import Cache as FLACache  # noqa: F401  (type hint)

    def _clone_val(v):
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.clone()
        if isinstance(v, (tuple, list)):
            return type(v)(_clone_val(x) for x in v)
        return v  # opaque state object — fall through

    cls = type(fla_cache)
    try:
        new = cls(seen_tokens=fla_cache._seen_tokens)
    except TypeError:
        new = cls()
        new._seen_tokens = fla_cache._seen_tokens
    # FLALayer holds .state (a dict) and ._seen_tokens.
    new.layers = []
    layer_cls = type(fla_cache.layers[0]) if fla_cache.layers else None
    for layer in fla_cache.layers:
        new_layer = layer_cls.__new__(layer_cls)
        new_layer.__init__()
        if layer.state is not None:
            new_layer.state = {
                k: _clone_val(v) for k, v in layer.state.items()
            }
        new_layer._seen_tokens = layer._seen_tokens
        if hasattr(layer, "device"):
            new_layer.device = layer.device
        new.layers.append(new_layer)
    return new


def _clone_cache(cache: dict) -> dict:
    """Deep-clone the full TinyLM incremental cache returned by
    `prefill`/`forward_step`. Used by `generate_soft_mixture` so the
    think-branch forward never touches the emit-branch state.
    """
    new = {
        "fla_cache": _clone_fla_cache(cache["fla_cache"]),
        "seen": int(cache["seen"]),
    }
    lag = cache.get("lagged_sources")
    if lag is None:
        new["lagged_sources"] = None
    else:
        new["lagged_sources"] = {k: v.clone() for k, v in lag.items()}
    wm = cache.get("wm_buf")
    if wm is None:
        new["wm_buf"] = None
    else:
        new["wm_buf"] = {k: v.clone() for k, v in wm.items()}
    tr = cache.get("think_run_len")
    new["think_run_len"] = tr.clone() if tr is not None else None
    return new


@torch.no_grad()
def generate_soft_mixture(
    model, prompt_ids: torch.Tensor, max_gen: int = 256,
    temperature: float = 0.0, eos_token_id: int | None = None,
    thinking_token_id: int | None = None,
    emit_threshold: float = 0.5,    # unused in soft mode — kept for API parity
    min_emit_before_eos: int = 0,
    gate_floor: float = 0.0,
    additive: bool = True,
    use_incremental: bool = True,
    force_sigma: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Soft-mixture decode (Phase C of THINKING_PLAN v5, 2026-05-26).

    Instead of hard-thresholding the output gate:
        if σ(gate) ≥ τ: emit, else think
    we run BOTH branches per emit step and mix:
        logits_emit  = next-token logits from current state (no think)
        logits_think = next-token logits after inserting ONE think token
                       (retrieval-as-input substitution for the think input,
                       matching `generate_with_retrieval_as_input`)
        final_logits = σ · logits_emit + (1-σ) · logits_think
        sample from final_logits
    Costs 2× per step but never makes a hard wrong decision. Useful as a
    probe of "is the gate's continuous σ output more informative than its
    thresholded version?"

    STATE COHERENCE:
      The emit-branch state is canonical. After sampling from the mixture
      we advance the cache via the emit-branch `forward_step(sampled_tok)`.
      The think-branch state is computed on a deep-cloned cache and
      discarded — its purpose was purely to produce `logits_think` for
      the mixture. This means the model never actually "saw" the think
      token in the canonical state, but the sampled output reflects the
      think branch's beliefs through the σ-weighted average.

    `force_sigma`: if set, override the gate's σ with this constant value
      (used by tests to verify σ=1 → emit-only and σ=0 → think-only).
    """
    if thinking_token_id is None:
        raise ValueError("generate_soft_mixture requires thinking_token_id")
    if not hasattr(model, "memory"):
        raise ValueError("model lacks .memory — this generator needs "
                         "WorkingMemory to provide the retrieval.")
    if not (hasattr(model, "prefill") and hasattr(model, "forward_step")):
        raise ValueError("generate_soft_mixture requires incremental "
                         "decode (prefill / forward_step).")

    device = prompt_ids.device
    out = prompt_ids.clone()
    inputs_embeds = model.embed(out).clone()
    _alpha_p = getattr(model, "retrieval_input_alpha", None)
    alpha = float(_alpha_p.detach()) if _alpha_p is not None else 0.1

    # Emit-branch prefill (canonical state). `pending_logits` predicts
    # the FIRST token; `pending_gate` is σ(gate) at that same position.
    cache, last_logits = model.prefill(out, inputs_embeds=inputs_embeds)
    pending_logits = last_logits[:, -1:, :]
    pending_gate_t = getattr(model, "_last_gate", None)
    if pending_gate_t is None:
        pending_gate = torch.ones((out.shape[0],), device=device)
    else:
        pending_gate = pending_gate_t[:, -1].clone()  # (B,)
    # The injection used by the think branch is the WM read at the LAST
    # processed position — model.memory._last_injection survives across
    # forward_step calls but is OVERWRITTEN, so we snapshot here.
    pending_injection = getattr(model.memory, "_last_injection", None)
    if pending_injection is not None:
        pending_injection = pending_injection[:, -1:, :].clone()

    emit_count = 0
    think_logits_used = 0
    sigma_values: list[float] = []
    mixture_kl: list[float] = []   # KL(emit || think) per step as a diag

    while emit_count < max_gen:
        # 1. Gate σ at this emit position (from the emit-branch state).
        if force_sigma is not None:
            sigma = torch.full_like(pending_gate, float(force_sigma))
        else:
            sigma = pending_gate.clone()
            if gate_floor > 0.0:
                sigma = sigma.clamp(min=float(gate_floor))
        # σ = P(emit), so mixing weight on emit-logits is σ, on think-logits is (1-σ).
        sigma_values.append(float(sigma[0].item()))

        # 2. EMIT-branch logits: already in pending_logits. Strip THINK id.
        emit_logits = pending_logits[:, -1, :].clone()      # (B, V)
        emit_logits[..., int(thinking_token_id)] = -float("inf")

        # 3. THINK-branch logits: clone the cache, append one think token
        #    (retrieval-as-input), get next-token logits.
        if pending_injection is None:
            # No WM injection available — fall back to think-token-embed only.
            think_emb = model.embed(torch.full(
                (out.shape[0], 1), int(thinking_token_id),
                dtype=out.dtype, device=device))
            inj_input = think_emb
        else:
            think_tok = torch.full((out.shape[0], 1), int(thinking_token_id),
                                    dtype=out.dtype, device=device)
            if additive:
                think_emb = model.embed(think_tok).to(inputs_embeds.dtype)
                inj_input = think_emb + alpha * pending_injection.to(inputs_embeds.dtype)
            else:
                inj_input = pending_injection.to(inputs_embeds.dtype)
        think_cache = _clone_cache(cache)
        think_tok = torch.full((out.shape[0], 1), int(thinking_token_id),
                                dtype=out.dtype, device=device)
        think_logits, _ = model.forward_step(
            think_tok, think_cache, inputs_embeds=inj_input,
            mem_read_mask=torch.ones_like(think_tok, dtype=inj_input.dtype),
        )
        think_logits = think_logits[:, -1, :].clone()        # (B, V)
        think_logits[..., int(thinking_token_id)] = -float("inf")
        think_logits_used += 1

        # 4. Mix in PROBABILITY space (mixing logits directly would be
        #    well-defined but harder to interpret as P(token) — the user-
        #    facing description says "mix by σ", we read that as mixing
        #    the two predictive distributions).
        emit_probs  = torch.softmax(emit_logits.float(),  dim=-1)
        think_probs = torch.softmax(think_logits.float(), dim=-1)
        mix_w = sigma.view(-1, 1).to(emit_probs.dtype)
        mixed_probs = mix_w * emit_probs + (1.0 - mix_w) * think_probs
        # Numerical safety + diag.
        mixed_probs = mixed_probs.clamp_min(1e-30)
        mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
        if emit_count < 10:
            kl = (emit_probs * (emit_probs.clamp_min(1e-30).log()
                  - think_probs.clamp_min(1e-30).log())).sum(dim=-1)
            mixture_kl.append(float(kl[0].item()))

        # 5. Suppress EOS for the first N emitted tokens (halt-after-
        #    docstring fix, parity with the hard generator).
        if (eos_token_id is not None
                and min_emit_before_eos > 0
                and emit_count < min_emit_before_eos):
            mixed_probs[..., int(eos_token_id)] = 0.0
            mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)

        # 6. Sample from the mixed distribution.
        if temperature == 0.0:
            next_tok = mixed_probs.argmax(dim=-1, keepdim=True)
        else:
            # T scaling on the MIXED distribution: exponentiate-by-1/T.
            # For T=1 this is just sampling from mixed_probs.
            if temperature == 1.0:
                next_tok = torch.multinomial(mixed_probs, num_samples=1)
            else:
                inv = 1.0 / max(temperature, 1e-6)
                scaled = mixed_probs.clamp_min(1e-30).pow(inv)
                scaled = scaled / scaled.sum(dim=-1, keepdim=True)
                next_tok = torch.multinomial(scaled, num_samples=1)

        out = torch.cat([out, next_tok], dim=1)

        # 7. Advance the EMIT-branch (canonical) cache with the sampled
        #    token. Use the standard embedding lookup (this token is an
        #    emitted output, not a think).
        emit_emb = model.embed(next_tok).to(inputs_embeds.dtype)
        pending_logits, cache = model.forward_step(
            next_tok, cache, inputs_embeds=emit_emb,
            mem_read_mask=torch.ones_like(next_tok, dtype=emit_emb.dtype),
        )
        # Refresh gate σ and WM injection for the next iteration.
        pending_gate_t = getattr(model, "_last_gate", None)
        if pending_gate_t is None:
            pending_gate = torch.ones((out.shape[0],), device=device)
        else:
            pending_gate = pending_gate_t[:, -1].clone()
        pending_injection = getattr(model.memory, "_last_injection", None)
        if pending_injection is not None:
            pending_injection = pending_injection[:, -1:, :].clone()

        emit_count += 1
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break

    diag = {
        "emit_count": emit_count,
        "think_total": think_logits_used,
        "think_steps_used": [1] * emit_count,   # 1 think branch per emit
        "gate_emit_values": sigma_values,
        "think_rate": 0.5,           # by construction the soft mixer is
                                      # 50/50 in COMPUTE; "think_rate" as
                                      # a behavioral knob is undefined.
        "mode": "soft_mixture",
        "decode_path": "incremental",
        "sigma_mean": (sum(sigma_values) / max(1, len(sigma_values))),
        "mixture_kl_first10": mixture_kl,
    }
    return out, diag


_FENCE_PY = __import__("re").compile(
    r"```python\s*\n(.*?)\n?```", __import__("re").DOTALL)
_FENCE_ANY = __import__("re").compile(
    r"```[a-zA-Z]*\s*\n(.*?)\n?```", __import__("re").DOTALL)


def _extract_code_block(text: str) -> str | None:
    """Pull a python fenced block out of `text`; returns None if no
    fence is present. Matches the helper in distill_solutions.py so
    Qwen-distilled student output is parsed the same way as the
    training data was emitted."""
    m = _FENCE_PY.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_ANY.search(text)
    if m:
        return m.group(1).strip()
    return None


def evaluate(ckpt_path: str, n_samples: int = 1, temperature: float = 0.0,
             max_gen: int = 256, max_problems: int | None = None,
             use_thinking: bool = False,
             max_think_per_step: int = 8,
             total_think_budget: int | None = None,
             emit_threshold: float = 0.5,
             min_emit_before_eos: int = 0,
             gate_floor: float = 0.0,
             prompt_style: str = "humaneval",
             extract_code_block: bool = False,
             generator: str = "standard",
             gate_mode: str = "hard",
             force_prefix_think: int = 0):
    """
    prompt_style:
      "humaneval"   — use the original HumanEval prompt verbatim (default).
      "sft_comment" — prepend "# Complete the following Python function.\\n"
                      to match the # <problem>\\n<solution> format that
                      sft_code.py builds for training. Required when the
                      ckpt was SFT'd on distilled JSONL (Qwen-style
                      problem→completion).

    extract_code_block:
      If True, look for a ```python ... ``` fence in the model output and
      use the EXTRACTED CODE as the full submission for grading (rather
      than prompt + raw model text). Required for Qwen-distilled ckpts
      that emit CoT prose around their code block.
    """
    print(f"Loading checkpoint: {ckpt_path}")
    # The latent-think generator needs DeltaNet β=0 at think positions.
    model, cfg = build_model_from_ckpt(
        ckpt_path,
        force_state_readonly=True if generator == "latent_think" else None)
    print(f"  feedback={cfg.get('feedback_mode')}  n_layers={cfg['n_layers']}")

    # Resolve thinking_token_id. We need it whenever use_thinking=True OR the
    # model has a working-memory module (the inference loop should know which
    # token to strip even if thinking is disabled, in case the model emits it).
    has_gate = hasattr(model, "gate_head")
    has_memory = getattr(model, "use_memory", False)
    thinking_token_id = getattr(model, "thinking_token_id", None)
    if thinking_token_id is None:
        thinking_token_id = cfg.get("thinking_token_id")
    if use_thinking:
        if not has_gate:
            raise RuntimeError(
                "use_thinking=True but model has no output gate; this ckpt "
                "wasn't trained with --output_gate.")
        if thinking_token_id is None:
            raise RuntimeError(
                "use_thinking=True but ckpt has no thinking_token_id in cfg "
                "and model.thinking_token_id is None.")
        print(f"  THINKING ON: max_think_per_step={max_think_per_step} "
              f"emit_threshold={emit_threshold} "
              f"total_think_budget={total_think_budget or 2*max_gen} "
              f"thinking_token_id={thinking_token_id} memory={has_memory}")
    else:
        print(f"  THINKING OFF  (gate={has_gate} memory={has_memory} "
              f"thinking_token_id={thinking_token_id})")

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
    ds = load_dataset("openai_humaneval", split="test")

    n_passed = 0
    n_total = 0
    failures: list[dict] = []
    agg_think_total = 0
    agg_emit_total = 0
    agg_gate_values: list[float] = []
    n_problems_with_any_think = 0
    for i, problem in enumerate(ds):
        if max_problems is not None and i >= max_problems:
            break
        raw_prompt = problem["prompt"]
        if prompt_style == "sft_comment":
            prompt = ("# Complete the following Python function.\n"
                      + raw_prompt)
        else:
            prompt = raw_prompt
        # Tokenise; check fits in model's max_T.
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        # max_T==0 means "no positional embedding limit" — the model has no
        # built-in T cap. Use a generous cap of 2048 to keep generation
        # bounded; don't truncate when the prompt is short.
        eff_max_T = cfg["max_T"] if cfg["max_T"] > 0 else 2048
        # Generation may produce up to max_gen emits + total_think_budget
        # think tokens, so reserve room for both when budget-truncating.
        thinks_reserve = (total_think_budget if total_think_budget is not None
                          else 2 * max_gen) if use_thinking else 0
        room_needed = max_gen + thinks_reserve
        if len(prompt_ids) + room_needed > eff_max_T:
            prompt_ids = prompt_ids[-(eff_max_T - room_needed):]
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device="cuda").unsqueeze(0)

        any_passed = False
        last_diag: dict = {}
        for _ in range(n_samples):
            if gate_mode == "soft":
                # Phase C: σ-weighted mixture of emit and think branches.
                # Requires WM (retrieval-as-input think branch) and
                # incremental decode.
                gen, diag = generate_soft_mixture(
                    model, prompt_t, max_gen=max_gen,
                    temperature=temperature, eos_token_id=tok.eos_token_id,
                    thinking_token_id=thinking_token_id,
                    emit_threshold=emit_threshold,
                    min_emit_before_eos=min_emit_before_eos,
                    gate_floor=gate_floor,
                    additive=cfg.get("retrieval_input_additive", True),
                )
            elif generator == "latent_think":
                # Latent-ponder thinking: state-readonly, hidden fed back as
                # the think-slot input embedding (THINKING_LATENT_2026_05_28).
                # BUG FIX (2026-06-05): the latent thread MUST run WorkingMemory
                # OFF here — every training path (latent_reasoning_cotrain,
                # latent_sft, the thinking.py measurement twins via wm_off) feeds
                # back the CLEAN out_norm(h); leaving WM on at eval feeds back
                # out_norm(h)+α·W_proj(WM_read), an OOD signal the latent adapter
                # was built to avoid → corrupted feedback → run-on output. Toggle
                # WM off for the duration, restore after.
                _saved_um = getattr(model, "use_memory", False)
                model.use_memory = False
                try:
                    gen, diag = generate_latent_think(
                        model, prompt_t, max_gen=max_gen,
                        temperature=temperature, eos_token_id=tok.eos_token_id,
                        thinking_token_id=thinking_token_id,
                        max_think_per_step=max_think_per_step,
                        total_think_budget=total_think_budget,
                        emit_threshold=emit_threshold,
                        min_emit_before_eos=min_emit_before_eos,
                        gate_floor=gate_floor,
                        force_prefix_think=force_prefix_think,
                    )
                finally:
                    model.use_memory = _saved_um
            elif generator == "retrieval_as_input":
                # v5+ models trained with --retrieval_as_input_thinking
                # must be evaluated with the matching generator (the
                # think-position input is the WM retrieval, not the
                # [THINKING] embedding).
                gen, diag = generate_with_retrieval_as_input(
                    model, prompt_t, max_gen=max_gen,
                    temperature=temperature, eos_token_id=tok.eos_token_id,
                    thinking_token_id=thinking_token_id,
                    max_think_per_step=max_think_per_step,
                    total_think_budget=total_think_budget,
                    emit_threshold=emit_threshold,
                    min_emit_before_eos=min_emit_before_eos,
                    gate_floor=gate_floor,
                    additive=cfg.get("retrieval_input_additive", False),
                )
            else:
                gen, diag = generate(
                    model, prompt_t, max_gen=max_gen,
                    temperature=temperature, eos_token_id=tok.eos_token_id,
                    use_thinking=use_thinking,
                    thinking_token_id=thinking_token_id,
                    max_think_per_step=max_think_per_step,
                    total_think_budget=total_think_budget,
                    emit_threshold=emit_threshold,
                    min_emit_before_eos=min_emit_before_eos,
                    gate_floor=gate_floor,
                )
            last_diag = diag
            gen_only_full = gen[0, len(prompt_ids):].tolist()
            # Strip think tokens before decoding (they're internal control
            # symbols, not part of the visible output).
            if thinking_token_id is not None:
                gen_only = [t for t in gen_only_full
                            if t != int(thinking_token_id)]
            else:
                gen_only = gen_only_full
            gen_text = tok.decode(gen_only, skip_special_tokens=True)
            if extract_code_block:
                # Qwen-distilled student emits CoT + ```python ... ```
                # — pull the code block out and use IT as the full
                # submission (replaces prompt+gen concat).
                code = _extract_code_block(gen_text)
                if code is not None:
                    full_code = code
                else:
                    # No fence found — fall back to raw text (will likely
                    # syntax_error, but keeps the loop honest).
                    full_code = gen_text
            else:
                gen_text = _truncate_at_stop(gen_text)
                full_code = raw_prompt + gen_text
            if _run_test_in_subprocess(full_code, problem["test"],
                                        problem["entry_point"]):
                any_passed = True
                break
        n_total += 1
        if any_passed:
            n_passed += 1
        else:
            failures.append({
                "task_id": problem["task_id"],
                "gen_preview": gen_text[:120],
            })
        if use_thinking and last_diag:
            agg_think_total += last_diag["think_total"]
            agg_emit_total += last_diag["emit_count"]
            agg_gate_values.extend(last_diag["gate_emit_values"])
            if last_diag["think_total"] > 0:
                n_problems_with_any_think += 1

        if (i + 1) % 20 == 0:
            msg = f"  {i + 1} problems: pass@{n_samples}={n_passed/n_total:.3f}"
            if use_thinking:
                tr = agg_think_total / max(1, agg_think_total + agg_emit_total)
                msg += (f"  think_rate={tr:.3f}  "
                        f"problems_using_think={n_problems_with_any_think}/{n_total}")
            print(msg)

    rate = n_passed / max(1, n_total)
    print(f"\npass@{n_samples} = {rate:.3f}  ({n_passed}/{n_total})")
    result = {"pass_rate": rate, "n_passed": n_passed, "n_total": n_total,
              "n_samples_per_problem": n_samples, "temperature": temperature,
              "failures_first_5": failures[:5]}
    if use_thinking:
        import statistics as _stats
        mean_gate = (_stats.fmean(agg_gate_values) if agg_gate_values else 0.0)
        result.update({
            "use_thinking": True,
            "think_total": agg_think_total,
            "emit_total": agg_emit_total,
            "think_rate": agg_think_total / max(1, agg_think_total + agg_emit_total),
            "problems_using_think": n_problems_with_any_think,
            "mean_gate_at_emit": mean_gate,
        })
        print(f"  think_total={agg_think_total}  emit_total={agg_emit_total}  "
              f"think_rate={result['think_rate']:.3f}  "
              f"mean_gate_at_emit={mean_gate:.3f}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, action="append")
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_gen", type=int, default=256)
    p.add_argument("--max_problems", type=int, default=None)
    p.add_argument("--use_thinking", action="store_true",
                   help="At inference time, consult the output gate and "
                        "append thinking tokens when σ(gate) < emit_threshold.")
    p.add_argument("--max_think_per_step", type=int, default=8,
                   help="Max consecutive think tokens between two emits.")
    p.add_argument("--total_think_budget", type=int, default=None,
                   help="Lifetime cap on think tokens per problem. "
                        "Default = 2 × max_gen.")
    p.add_argument("--emit_threshold", type=float, default=0.5,
                   help="σ(gate) threshold above which we emit (else think).")
    p.add_argument("--min_emit_before_eos", type=int, default=0,
                   help="Suppress eos_token_id for the first N emitted tokens. "
                        "Mitigates the pretrain halt-after-docstring artifact "
                        "where the model learned to predict EOS at "
                        "small-document boundaries (very common in mixed-corpus "
                        "code data). Recommended: 30 for HumanEval.")
    p.add_argument("--gate_floor", type=float, default=0.0,
                   help="Clamp σ(gate) from below at inference, mirroring "
                        "the training-time loss clamp g.clamp(min=gate_floor_min). "
                        "Without this, the gate distribution at deploy can land "
                        "systematically below emit_threshold because training "
                        "made the model indifferent to gate values < floor. "
                        "Use the same value as --gate_floor_min was during "
                        "pretrain (e.g. 0.5).")
    p.add_argument("--prompt_style", type=str, default="humaneval",
                   choices=["humaneval", "sft_comment"],
                   help="'humaneval' (default): original prompt verbatim. "
                        "'sft_comment': prepend '# Complete the following "
                        "Python function.\\n' to match the # <problem>\\n"
                        "<solution> format the sft_code.py distilled trainer "
                        "uses. Required for Qwen-distilled ckpts.")
    p.add_argument("--extract_code_block", action="store_true",
                   help="Parse a ```python ... ``` fence out of the model "
                        "output and use ONLY that as the submission. "
                        "Required for Qwen-distilled ckpts that emit CoT + "
                        "code block (otherwise the CoT prose would be exec'd "
                        "as Python and syntax-error every time).")
    p.add_argument("--gate_mode", type=str, default="hard",
                   choices=["hard", "soft"],
                   help="'hard' (default): existing hard-threshold "
                        "behaviour (gate ≥ emit_threshold → emit). "
                        "'soft': Phase C soft-mixture decode — run both "
                        "branches per step and mix logits by σ. 2× per-step "
                        "compute, but the threshold can never make a wrong "
                        "decision. Requires a model with WorkingMemory "
                        "and incremental decode.")
    p.add_argument("--force_prefix_think", type=int, default=0,
                   help="latent_think generator: force this many latent steps "
                        "before the first emit, then no more thinking (matches "
                        "latent_sft think-before-solution training).")
    p.add_argument("--generator", type=str, default="standard",
                   choices=["standard", "retrieval_as_input", "latent_think"],
                   help="'standard' (default): NO thinking — plain greedy code "
                        "(the no-think baseline; works for any ckpt). "
                        "'latent_think': the VALIDATED thinking mechanism "
                        "(Coconut-style hidden feedback, state-readonly). Use "
                        "this for any thinking eval. "
                        "'retrieval_as_input': DEPRECATED legacy WM-injection "
                        "thinking (v5-v7 ckpts) — gated behind "
                        "--allow_legacy_thinking. Mixing it with latent-trained "
                        "ckpts caused a toy-vs-HumanEval mechanism mismatch.")
    p.add_argument("--allow_legacy_thinking", action="store_true",
                   help="Opt-in to the DEPRECATED thinking mechanisms "
                        "(--generator retrieval_as_input, or --generator "
                        "standard together with --use_thinking = discrete-token "
                        "thinking). Off by default so they can't be selected by "
                        "accident — the validated mechanism is "
                        "--generator latent_think. Only set this to reproduce a "
                        "historical v5-v7 ckpt eval.")
    args = p.parse_args()

    # Guard the deprecated thinking mechanisms behind an explicit opt-in. The
    # toy ('thinking helps') used latent_think; a HumanEval run accidentally used
    # retrieval_as_input ('thinking hurts') — apples-to-oranges. Don't expose the
    # old modes silently.
    _legacy_think = (
        args.generator == "retrieval_as_input"
        or (args.generator == "standard" and args.use_thinking)
    )
    if _legacy_think and not args.allow_legacy_thinking:
        p.error(
            "Refusing a DEPRECATED thinking mechanism: "
            f"--generator {args.generator}"
            + (" with --use_thinking (discrete-token thinking)"
               if args.generator == "standard" else "")
            + ". The validated mechanism is `--generator latent_think`. "
            "If you really need the legacy mode (e.g. a v5-v7 ckpt SFT'd with "
            "--retrieval_as_input_thinking), pass --allow_legacy_thinking.")

    all_results = {}
    for ckpt in args.ckpt:
        print(f"\n{'=' * 70}\nEvaluating: {ckpt}\n{'=' * 70}")
        all_results[ckpt] = evaluate(
            ckpt, n_samples=args.n_samples, temperature=args.temperature,
            max_gen=args.max_gen, max_problems=args.max_problems,
            use_thinking=args.use_thinking,
            max_think_per_step=args.max_think_per_step,
            total_think_budget=args.total_think_budget,
            emit_threshold=args.emit_threshold,
            min_emit_before_eos=args.min_emit_before_eos,
            gate_floor=args.gate_floor,
            prompt_style=args.prompt_style,
            extract_code_block=args.extract_code_block,
            generator=args.generator,
            gate_mode=args.gate_mode,
            force_prefix_think=args.force_prefix_think,
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ckpt':<60} {'pass@k':>10} {'(passed/total)':>16}")
    for ckpt, r in all_results.items():
        name = pathlib.Path(ckpt).stem
        print(f"{name:<60} {r['pass_rate']*100:>9.1f}% "
              f"{r['n_passed']:>5}/{r['n_total']:<5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
