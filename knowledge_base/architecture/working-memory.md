# Working Memory (WM)

## Summary
A **bounded (K≤1024) write-gated buffer of past hidden states**, read via soft attention at think / `mem_read_mask` positions. Read cost O(T·K·d), no O(T²) attention. Its one clean validated win is **saturated MQAR recall** (+11.1 pp, and +35.5 pp mean in a 5-seed converged A/B) — exactly the regime where the DeltaNet recurrent state can't hold all the items. Whether it is load-bearing on *real* tasks is the project's longest investigation; see the full story in [[working-memory-recall-saga]]. Source: `experiments/model.py::WorkingMemory` (line 974), `project_working_memory_win.md`.

## The validated win (saturated MQAR)
WM helps **only when the DeltaNet state is saturated** (small state AND more pairs than capacity):
- 5-seed converged A/B (d=64, L=2, T=512, K=128): **WM-off 0.500 ± 0.028 (rock stable) vs WM-on 0.855 ± 0.181** (+35.5 pp; 4/5 seeds win big, 5th ties). Reliably helps-or-ties, never loses.
- On a 4×-bigger un-saturated model DeltaNet hits 0.999 at K≤256 → WM is redundant/harmful there. **If you test WM on an unsaturated model you correctly measure no win.**
- The variance is an **addressing-learning failure, not capacity**: the bad seed has 100 % value-coverage in the buffer but diffuse reads. A bigger buffer would NOT help; better addressing would. (`probe_wm_utilization.py`)

## Required design pieces
- **Read-α gate** (`mem_read_alpha_init=1.0`, learnable): WM was the only module with no output gate; the un-gated injection collapsed training at scale. Use init 1.0 (NOT 0.0 — at α=0 the WM weights get zero gradient and stay at random init). α self-adjusts down once trained.
- **Decoupled-KV (DKV) cosine addressing** (`--mem_decoupled_kv`): a dedicated match-key `W_k` separate from the content value `W_v`, L2-normalized cosine with a learnable temperature, so a slot is retrieved by *what it means* not by its payload.
- **α-floor curriculum** (`--mem_read_alpha_floor_start`) holds the read strong during warmup so addressing locks in.
- Thinking-token embedding initialised to the embedding-mean (else random noise enters the recurrence at each think step).

## The hard part: addressing
DKV cosine addressing **fails on near-identical keys** — see [[key-separability]]. The validated fix (2026-06-15) is to **key on the variable-name INPUT-EMBEDDING window** (`--mem_key_from_embedding`, `wm_namekey_probe.py`: top1=1.00 vs chance) + a **copy/pointer readout** (`--use_copy_head`, `CopyReadout` at model.py:1285; `wm_multitok_readout.py`: 100 % exact multi-token recall). This is the v14 plumbing. But on **real code** the recurrence already solves value-recall (~89 %), so WM's only real headroom is **agentic / long-context / saturating multi-key** recall — see [[working-memory-recall-saga]].

## Where WM belongs
- Recall when the recurrent state can't hold all items. NOT on sparse-read tasks (induction: −28 pp) — see [[evals-and-probes]] / read-density threshold.
- Memory should be ON during RL or tasks that pass `mem_read_mask`; in plain SFT/pretrain without think tokens its read gradient is structurally near-zero (route-around). See [[route-around-principle]].

## Related
[[working-memory-recall-saga]] · [[key-separability]] · [[product-key-memory]] · [[route-around-principle]] · #architecture
