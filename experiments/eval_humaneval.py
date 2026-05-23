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
    # SPEEDUP: at decode time we feed the whole prefix each step, so the
    # FiLM K=3 self-feed has nothing to iterate over (K-pass is for
    # gradient quality during training). Force the single-pass FiLM
    # path for the whole decode loop — measured ~2x faster on the
    # generate path with identical outputs at the K-warmup deploy
    # convention. Restore on exit so we don't leak state to training.
    _orig_film_bypass = getattr(model, "_film_bypass", False)
    if hasattr(model, "_film_bypass"):
        model._film_bypass = True
    out = prompt_ids.clone()
    emit_count = 0
    think_total = 0
    think_steps_used = []      # length = emit_count; thinks before each emit
    gate_emit_values = []      # gate values at each emit step
    while emit_count < max_gen:
        # Inner think loop: try to coax the gate above emit_threshold.
        thinks_this_step = 0
        while True:
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
        emit_count += 1
        if use_thinking:
            think_steps_used.append(thinks_this_step)
            gate_emit_values.append(gate_val)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    if hasattr(model, "_film_bypass"):
        model._film_bypass = _orig_film_bypass
    diag = {
        "emit_count": emit_count,
        "think_total": think_total,
        "think_steps_used": think_steps_used,
        "gate_emit_values": gate_emit_values,
        "think_rate": (think_total / max(1, think_total + emit_count))
                       if use_thinking else 0.0,
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
    # SPEEDUP: force single-pass FiLM at decode (see generate() above).
    _orig_film_bypass = getattr(model, "_film_bypass", False)
    if hasattr(model, "_film_bypass"):
        model._film_bypass = True

    device = prompt_ids.device
    out = prompt_ids.clone()
    # inputs_embeds is the per-position model input; starts as the
    # embedding lookup of the prompt and grows with each gen step.
    inputs_embeds = model.embed(out).clone()    # (B, prompt_len, d)
    # v7 additive α-gated injection: at a think step the next input is
    # think_embed + α·retrieval (α = model.retrieval_input_alpha),
    # instead of v5/v6's destructive `next input = retrieval`. `additive`
    # is set from cfg['retrieval_input_additive'] by the caller — True
    # for v7 ckpts, False for v5/v6.
    _alpha_p = getattr(model, "retrieval_input_alpha", None)
    alpha = float(_alpha_p.detach()) if _alpha_p is not None else 0.1

    emit_count = 0
    think_total = 0
    think_steps_used: list[int] = []
    gate_emit_values: list[float] = []
    while emit_count < max_gen:
        thinks_this_step = 0
        # Inner think loop: keep retrieving until gate flips to emit or
        # we hit a cap.
        while True:
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
        inputs_embeds = torch.cat([inputs_embeds, emit_emb], dim=1)
        emit_count += 1
        think_steps_used.append(thinks_this_step)
        gate_emit_values.append(gate_val)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            break
    if hasattr(model, "_film_bypass"):
        model._film_bypass = _orig_film_bypass
    diag = {
        "emit_count": emit_count,
        "think_total": think_total,
        "think_steps_used": think_steps_used,
        "gate_emit_values": gate_emit_values,
        "think_rate": (think_total / max(1, think_total + emit_count)),
        "mode": "retrieval_as_input",
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
             generator: str = "standard"):
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
    model, cfg = build_model_from_ckpt(ckpt_path)
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
            if generator == "retrieval_as_input":
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
    p.add_argument("--generator", type=str, default="standard",
                   choices=["standard", "retrieval_as_input"],
                   help="'standard' (default): append [THINKING] tokens. "
                        "'retrieval_as_input': at think positions inject "
                        "the WM retrieval as the next input embedding. "
                        "REQUIRED for ckpts SFT'd with "
                        "--retrieval_as_input_thinking (v5+) — evaluating "
                        "those with the standard generator is a "
                        "train/inference mismatch.")
    args = p.parse_args()

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
