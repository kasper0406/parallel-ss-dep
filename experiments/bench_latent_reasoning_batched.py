"""Standalone GPU bench: sequential vs batched latent-reasoning aux, at the
REAL N1 32L config (2026-07-04 fix).

Measures peak memory + wall-clock for one `step()`-equivalent call at the
WORST CASE the running N1 launcher can hit (R=8, n=4,
--latent_reasoning_max_len 512 prompts) — both the OLD sequential-B=1 path
and the NEW batched path — so the "peak stays comfortably under the
combined training envelope" and "step-time saving" claims in the mission are
backed by a number, not a guess.

Architecture matches the live N1 process exactly (checked against its
/proc/<pid>/cmdline, not a launcher script — none exists on disk for this
run): --arch deltanet --d_model 960 --n_layers 32 --n_heads 15 --d_head 64
--d_ff 2560 --tie_embeddings --keep_base_vocab 49152 --feedback none
--use_latent_feedback_adapter (no --output_gate, no --state_readonly_at_think
in the live cmdline — this script matches that, i.e. the gate_weight loss
term is a no-op here exactly as it silently is in the live N1 run, and the
doc_ids isolation is exercised WITHOUT relying on state_readonly_at_think,
which is the point).

This is a standalone measurement of the aux's OWN memory footprint (model +
activations + its own gradients) — not the full combined training process
(data mix, optimizer state, the main pretrain forward). The mission's
reference numbers (~7.5GB used on GPU1 now; the 2026-07-02 OOM fix measured
8.55GB combined; don't regress past ~12GB combined) are from the full
process; this script's job is to check the DELTA this fix adds/removes is
sane, and that peak stays well inside that combined budget on its own.

Run (GPU0 only — GPU1 is the live N1 training job, never touch it):
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        experiments/bench_latent_reasoning_batched.py
"""
from __future__ import annotations

import time

import torch

from experiments.layers import DeltaNetAttention
from experiments.model import TinyLM
from experiments.latent_reasoning_cotrain import (
    _answer_span_latent_loss,
    _answer_span_latent_loss_batched,
)

VOCAB = 49152
THINK_ID = VOCAB - 1
EOS_ID = 0     # arbitrary distinct id for this bench; content doesn't matter
R = 8
N_EXAMPLES = 4
MAX_LEN = 512


def _n1_model() -> TinyLM:
    torch.manual_seed(0)
    m = TinyLM(
        vocab_size=VOCAB, d_model=960, n_layers=32, n_heads=15, d_head=64,
        d_ff=2560, attention_cls=DeltaNetAttention, max_T=0,
        tie_embeddings=True, feedback_mode="none",
        use_latent_feedback_adapter=True,
        output_gate=False, state_readonly_at_think=False,
    )
    m.thinking_token_id = THINK_ID
    m.train()
    return m.cuda()


def _worst_case_examples():
    """n=4 examples near --latent_reasoning_max_len 512, comment lengths
    DIFFER slightly (realistic padding waste) but all near the cap — the
    worst case for both memory (all rows long) and padding correctness
    (rows differ)."""
    g = torch.Generator().manual_seed(0)
    out = []
    comment_lens = [495, 490, 494, 485]   # near max_len - R - slen - 2
    for plen in comment_lens:
        c = torch.randint(2, VOCAB - 2, (plen,), generator=g).tolist()
        s = torch.randint(2, VOCAB - 2, (5,), generator=g).tolist()
        assert plen + len(s) + R + 2 <= MAX_LEN
        out.append((c, s))
    return out


def _typical_rung8_examples():
    """n=4 examples with lengths drawn from the REAL rung-8 corpus's actual
    distribution (mean ~436, not the synthetic near-512 worst case) — what a
    typical late-training step (deep rung, real data) actually looks like,
    complementing the pure worst-case number above."""
    g = torch.Generator().manual_seed(1)
    comment_lens = [436, 410, 460, 400]   # ~ the real n8 corpus mean/spread
    out = []
    for plen in comment_lens:
        c = torch.randint(2, VOCAB - 2, (plen,), generator=g).tolist()
        s = torch.randint(2, VOCAB - 2, (5,), generator=g).tolist()
        out.append((c, s))
    return out


def _measure(label: str, fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = fn()
    loss.backward()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"{label:32s} loss={float(loss.detach()):.4f}  time={dt:6.3f}s  "
          f"peak_alloc={peak_alloc:6.3f} GiB  peak_reserved={peak_reserved:6.3f} GiB")
    return dt, peak_alloc, peak_reserved


def _bench_pair(label: str, examples):
    print(f"--- {label} ---")
    m_seq = _n1_model()

    def _seq():
        losses = []
        for c, s in examples:
            losses.append(_answer_span_latent_loss(
                m_seq, c, s, EOS_ID, R, THINK_ID, "cuda",
                checkpoint_latent=True))
        return torch.stack(losses).mean()

    dt_seq, mem_seq, res_seq = _measure("sequential (old, n=4 B=1 calls)", _seq)
    del m_seq
    torch.cuda.empty_cache()

    m_batch = _n1_model()

    def _batch():
        return _answer_span_latent_loss_batched(
            m_batch, examples, EOS_ID, R, THINK_ID, "cuda",
            checkpoint_latent=True)

    dt_batch, mem_batch, res_batch = _measure("batched (new, one n=4 call)", _batch)
    del m_batch
    torch.cuda.empty_cache()

    print(f"Wall-clock:  sequential={dt_seq:.3f}s  batched={dt_batch:.3f}s  "
          f"speedup={dt_seq / max(dt_batch, 1e-9):.2f}x")
    print(f"Peak alloc:  sequential={mem_seq:.3f} GiB  batched={mem_batch:.3f} GiB  "
          f"delta={mem_batch - mem_seq:+.3f} GiB")
    print(f"Peak reserved: sequential={res_seq:.3f} GiB  batched={res_batch:.3f} GiB")
    print()


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(dev)} "
          f"(CUDA_VISIBLE_DEVICES should pin this to GPU0)")
    print()
    _bench_pair("ABSOLUTE WORST CASE: R=8, n=4, all prompts near max_len=512",
               _worst_case_examples())
    _bench_pair("TYPICAL rung-8 step: R=8, n=4, real corpus length spread "
               "(~400-460 tok, mean~436)",
               _typical_rung8_examples())


if __name__ == "__main__":
    main()
