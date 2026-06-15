# Verdict: the Working Memory recall saga

## Summary
The longest investigation in the project. WM has **one clean win** (saturated MQAR, +11.1 pp) and a long history of being **inert on real recall**. The full causal chain was traced: (1) gradient-disconnect — WM never got a recall gradient; (2) key-separability — cosine addressing can't separate near-identical keys; (3) the mechanism *was* fixed and validated to **100 % synthetic recall** with name-key addressing + copy readout; but (4) on **real code** the recurrence already solves value-recall (~89 %), so WM's only genuine headroom is **agentic / long-context / saturating multi-key** recall — which v14 is testing. Source: `project_wm_recall_probe_broken_and_routed_around.md`, `project_wm_addressing_root_cause.md`, `project_recall_discrete_key_direction.md`, `project_recall_*`.

## The arc (in order)
1. **MQAR win** (the existence proof): saturated MQAR T=512/K=128, WM-off 0.500 → WM-on 0.855 (+35.5 pp mean, 5-seed). WM is load-bearing **only** when the recurrent state is saturated. See [[working-memory]].
2. **"recall=0" was a probe artifact** ([[broken-probe-lessons]]): the in-run probe used the SFT `# {flatten}` format, OOD for a pretrain ckpt → 0.000; training-matched format → 0.438. Recall actually works (~44 % at 1 B).
3. **Single-binding recall has zero headroom**: the 287 M base recalls 100 % no-think at every distance — a delta-rule state trivially holds ONE binding. Can't show a WM lift where there's no headroom.
4. **Gradient-disconnect (the mechanistic root cause)**: WM injects only at think positions; think targets are −100; state-readonly zeroes the write; the injection lands on the final hidden (lm_head only), never back into the trunk. The recall target is an *emit* position → its gradient flows through the **recurrence**, never through WM. So the recurrence learned recall; WM got no recall gradient and `read_alpha` decayed 1.0 → 0.081. The read became **recency**, not content-addressable.
5. **Pretrain stream had no answer supervision**: the recall stream fed `problem_prompt` only → the bound value was never a recall target → no content-recall gradient anywhere (v10 AND v11). Fix = `text_field: [problem_prompt, qwen_completion]`.
6. **Addressing is the wall, and the trunk must be addressable**: freeze-trunk addressing FT → mass-on-binding stays ~0.01; joint-trunk co-train → ALSO fails because the keys are non-separable (see [[key-separability]]). The decisive negative: even explicit attention-placement supervision won't converge.
7. **THE FIX (validated 100 % synthetic)**: address on **token-identity** — key the read on the variable-name INPUT-EMBEDDING window (`wm_namekey_probe.py`: top1=1.00 vs chance) + a **copy/pointer readout** over the addressed span (`wm_multitok_readout.py`: EXACT 1.00 / 1.00 / 1.00 at N=48/64/96 vs base ≈0). Larger-state control: 2×/4× d_head and DeltaProduct K=2/3 all score ~0 at N≥48 → **WM fills headroom the trunk provably can't reach, more cheaply**.
8. **Reality check — real code has little WM headroom**: on realistic `code_recall` data the **recurrence already solves value-recall ~89 %** (separable identifiers like `CACHE_SIZE` are naturally cosine-separable, unlike synthetic `vN=MMMM`). v14's first live probe: `recall_off=1.000` → zero WM headroom on the code probe. **WM's only remaining opportunity = AGENTIC recall** (baseline ~0.42, real capacity wall) + synthetic saturating multibind.

## Current verdict
- WM is a **validated mechanism** but its payoff regime (capacity-exceeding, non-memorizable, multi-key recall) is **rare in code at 287 M**. For the code headline, recurrence + [[product-key-memory]] suffice and WM adds ~nothing.
- **v14** (running, GPU0) is the test: WM-recall continuation with embedding-key addressing + copy head + answer-span `mem_read_mask`, measured on **agentic** recall (not the code probe, which has no headroom). See [[open-questions-and-roadmap]].
- This whole saga is the cleanest demonstration of [[route-around-principle]] and [[objective-function-alignment]].

## Related
[[working-memory]] · [[key-separability]] · [[route-around-principle]] · [[objective-function-alignment]] · [[recall-investigation-arc]] · #verdict #recall
