# The recall investigation arc

## Summary
A chronological view of the project's longest single investigation: *is [[working-memory]] load-bearing for recall on real tasks?* It runs from the MQAR win, through a string of artifacts and root-cause discoveries, to a fully-validated synthetic fix, to the sober conclusion that real code recall is solved by the recurrence so WM's only real headroom is agentic/long-context. The verdict lives in [[working-memory-recall-saga]]; this note is the timeline. Sources: the `project_recall_*` and `project_wm_*` memory files.

## Timeline
1. **(2026-05-12) MQAR win** — bounded WM rescues saturated DeltaNet recall (+11.1 pp; later +35.5 pp 5-seed converged). The existence proof. But only at the saturated regime.
2. **(2026-06-04) read-α gate added** — WM was the only module without an output gate; un-gated injection collapsed training at scale. `mem_read_alpha_init=1.0`.
3. **(2026-06-13) single-binding recall has zero headroom** — base recalls 100 % no-think at every distance; can't show a WM lift. Multibind recall (assign N vars, print one) HAS headroom.
4. **(2026-06-13) addressing is the wall** — on multibind, WRITE 100 % but READ ~1.7 % on the queried binding (diffuse / recency). Frozen-trunk addressing FT can't fix it; co-trained v10 read is peaky-but-query-independent.
5. **(2026-06-13) capstone: the route-around** — joint trunk+WM co-train made it WORSE (trunk memorized the in-window program, routed around WM). → [[route-around-principle]].
6. **(2026-06-13/14) root causes at the source** — v10's recall stream was single-binding (no saturation); v10/v11's stream fed `problem_prompt` only (no answer supervision → no recall gradient anywhere).
7. **(2026-06-14) "recall = 0" was a probe artifact** — OOD format; training-matched format read 0.438. Recall actually works ~44 %. But in the correct format the gate **never thinks** → WM (think-only) never exercised → route-around at its cleanest. See [[broken-probe-lessons]].
8. **(2026-06-15) the key-separability root cause** — cosine-on-hidden can't separate near-identical `vN=MMMM` keys; freeze-trunk AND joint-trunk AND explicit attention-supervision all fail. MQAR won because its keys were distinct random. → [[key-separability]].
9. **(2026-06-15) THE FIX, validated 100 % synthetic** — address on the variable-name **input-embedding window** (top1=1.00) + **copy/pointer readout** (EXACT 1.00 at N=48/64/96). Larger-state control (2×/4× d_head, DeltaProduct) all ~0 at N≥48 → WM fills headroom the trunk can't reach, more cheaply.
10. **(2026-06-15) reality check** — on REAL code recall the recurrence already scores ~89 % (real identifiers are separable). WM's only real opportunity is **agentic** (baseline ~0.42, real wall) + synthetic saturating multibind. v14 launched to test agentic WM; its first code-recall probe confirmed `recall_off=1.000` (no code headroom).

## What it taught the project
- The whole arc is the canonical demonstration of [[route-around-principle]] and [[objective-function-alignment]].
- It also produced several [[broken-probe-lessons]] (format-OOD, headroom-less deltas) and a [[fair-baselines]] catch (larger-state control before quoting the WM win).

## Related
[[working-memory-recall-saga]] · [[working-memory]] · [[key-separability]] · [[route-around-principle]] · #history #recall
