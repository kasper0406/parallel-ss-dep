# WHY PKM / WM / latent thinking give only a marginal benefit on MBPP code

Intuition (correct): a small model SHOULD gain hugely from parametric memory
(PKM), context recall (WM), and extra sequential compute (latent thinking).
Reality: on MBPP/HumanEval the combined benefit is marginal (+1/+3). This
documents the experiments that found WHY — it is NOT that the mechanisms are
broken; it is that MBPP at 287M is bottlenecked by something none of them
address.

## The experiments (all on `latent_code_adapteronly.pt`, the co-trained base)

### 1. Per-token thinking ceiling is low and depth-independent
`probe_gate_placement.py` / `probe_retrieval_channels.py`: a latent burst helps
only ~10-13% of code positions, by ~+0.3 logp, FLAT across R=1/2/4/8. (Arithmetic
chains, by contrast, scale strongly with R.) → code-token prediction is mostly
recall/pattern; iterating the trunk adds little.

### 2. No addressable retrieval content for code (WM ruled out, cheaply)
`probe_oracle_retrieval.py`: ORACLE retrieval (best-of-16 buffer slots, peeking
at the label) scored frac-helpful 0.249 — but a matched-norm RANDOM-injection
control scored 0.298 (HIGHER). oracle−random ≈ 0. The "oracle lift" was pure
max-of-K selection noise. The WM buffer has no addressable code signal → no
addressing scheme (trained DKV included) can help. PKM is the only positive
channel (+0.052 frac), modest and already exploited by the base.

### 3. Failures are runnable but fundamentally wrong (not near-misses, not broken)
`probe_failure_bottleneck.py` (120 MBPP, no-think greedy): 27 pass, 81 partial,
7 exec_error, 5 syntax_error. 87% of failures RUN, so it is not a code-formation
gap. But the partial pass-fraction is 0.06 — those runnable functions pass ~6%
of tests → plausible-but-fundamentally-wrong logic, not off-by-one.

Concrete examples: `lucid_number` returns `n` (should return a list of ludic
numbers); `longest_chain` calls itself with no base case (infinite recursion);
`degrees_to_radians` uses `180/π` instead of `π/180`. Also a non-trivial rate of
BAD TESTS (`first_repeated_char` is correct but the test asserts `== "None"` the
string) — so the benchmark even undercounts the model.

### 4. THE CRUX — failures are KNOWLEDGE-bound, not SEARCH-bound
`probe_knowledge_vs_search.py` (120 MBPP): for each FAILED problem, compare under
the model the mean per-token logp of the GOLD solution vs the model's own wrong
greedy output:

| | failures |
|---|---|
| SEARCH-bound (logp_gold > logp_own — knowledge latent, decoding misses it) | **0 (0%)** |
| KNOWLEDGE-bound (logp_gold < logp_own — model prefers its wrong answer) | **93 (100%)** |

mean logp_gold = −2.26, mean logp_own = −0.73 → the model rates its OWN WRONG
answer ~1.53 nats/token (~4.6×) MORE likely than the correct solution, on EVERY
failure. (Style confound rebutted by §3: the model emits a different *algorithm*,
e.g. returns an int where a list is required — not a restyled gold.)

## Conclusion

**MBPP failures at 287M are a base-KNOWLEDGE gap, not a thinking/memory/search
gap.** The model doesn't know the algorithm and confidently confabulates a wrong
one. No inference-time mechanism can supply knowledge the weights don't contain:
- latent thinking refines a distribution centered on the wrong answer;
- WM retrieves from a context that doesn't contain the algorithm;
- PKM would need the algorithm in its 262k slots, but the 287M model trained on
  limited data never learned it.

This is a scale × data (Chinchilla) problem. It also explains every prior result:
the mechanisms ARE load-bearing where the bottleneck matches their function
(WM +11pp on MQAR recall; latent +0.65-0.80 on arithmetic chains; PKM −5 on
HumanEval where parametric knowledge IS the lever), and marginal where it does
not (MBPP general code).

## Implications (where the benefit actually lives)

1. **To SHOW the mechanisms' huge benefit, use a matched-bottleneck task.** The
   "small super-coder with PKM/WM" wins on tasks bottlenecked by memory/recall:
   long-context code (recall a symbol defined 1000s of tokens back), repo/agentic
   settings, long dependency chains. MBPP is the wrong probe — it's short and
   knowledge-bound. Build/borrow a long-context-code recall eval and show WM is
   load-bearing there.
2. **To raise the MBPP headline, inject knowledge, not thinking**: more/better
   pretrain + distillation (put the algorithms in the weights / PKM), or a
   larger model. The cheapest knowledge-injection lever is distillation from a
   strong teacher on the failing problem families.
3. **Stop spending inference-mechanism effort on MBPP** — the addressable surface
   is ~0 there (0/93 search-bound). Reserve thinking/WM for matched tasks.

Probes (reusable): `probe_gate_placement`, `probe_retrieval_channels`,
`probe_oracle_retrieval`, `probe_failure_bottleneck`, `probe_knowledge_vs_search`,
`probe_thinking_passk`.
