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
DKV cosine addressing **fails on near-identical keys** — see [[key-separability]]. The validated fix (2026-06-15) is to **key on the variable-name INPUT-EMBEDDING** (`--mem_key_from_embedding`, `wm_namekey_probe.py`: top1=1.00 vs chance) + a **copy/pointer readout** (`--use_copy_head`, `CopyReadout` at model.py:1285; `wm_multitok_readout.py`: 100 % exact multi-token recall). The deployed v12 WM keys on a **discrete lexical hash** of the identifier span (one-hot, zero cross-talk) — validated leak-free, but EXACT-SPELLING (`cache_size`≠`CACHE_SIZE`).

**KEY-DESIGN UPDATE (2026-06-16, [[key-separability]] refinement, `wm_vqkey_probe.py`):** a 3-arm head-to-head (hash / VQ / soft-continuous) shows **discreteness was the mistake, not softmax**. Given a **separable name-SPAN key** (not the contaminated fixed `mem_key_window`!), a **continuous SOFT attention read** wins BOTH separability (1.00 @N=128) AND surface-variant robustness (case 0.65–0.81, camel 1.00), while the hash is spelling-locked and VQ loses on both. ⟹ the best addressing = **learned continuous name-span key + soft read**, not a discrete code; the hash is the training-free exact-match fallback. CAVEAT: probe-scale, encoder trained directly on recall; **real-model transfer is the open test** (the historical `mem_key_window` soft key failed *because* it pooled name+arrow+value — name-span-ONLY is the fix). On **real code** the recurrence already solves value-recall (~89 %), so WM's headroom is **agentic / long-context / saturating multi-key / surface-variant** recall — see [[working-memory-recall-saga]].

## Where WM belongs
- Recall when the recurrent state can't hold all items. NOT on sparse-read tasks (induction: −28 pp) — see [[evals-and-probes]] / read-density threshold.
- Memory should be ON during RL or tasks that pass `mem_read_mask`; in plain SFT/pretrain without think tokens its read gradient is structurally near-zero (route-around). See [[route-around-principle]].

## Related
[[working-memory-recall-saga]] · [[key-separability]] · [[product-key-memory]] · [[route-around-principle]] · #architecture
