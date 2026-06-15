# WM Addressing Redesign — "agentic-semantics" designer proposal

NOTE: written to disk because the StructuredOutput tool would not accept
parameters in this session (repeated empty-arg validation failures). The
parent orchestrator reads only the StructuredOutput call; this file is a
fallback so the design is not lost.

## name
Write-time key/value separation with content-addressable semantic keys
("Slot Cards": separate match-key vs. carried-content, plus a learned
recency/role tag) — a.k.a. Decoupled Key-Value Working Memory.

## core_idea
Today WorkingMemory addresses BY VALUE: one projection `W_v(h)` is BOTH the
match key (dot-producted with the query) AND the retrieved content. Split this
into two heads computed at WRITE time per buffered position: a normalized
match-KEY `k = L2norm(W_k(h))` and a separate carried-VALUE `v = W_v(h)`.
Reads score `q . k` over unit-norm keys (cosine, temperature-scaled), softmax,
then retrieve `v` (not `k`). This is the standard attention K/V split that the
current WM collapsed into one projection. On top of it, attach a small set of
learned write-time "role/recency" tag embeddings added to the key so the query
can compositionally address by (content-similarity AND role AND recency),
which is exactly what coding/agentic recall needs.

## why_reliable
Attacks both diagnosed root causes directly.
(1) Magnitude/degeneracy: L2-normalizing the KEY removes the small-norm ->
flat-softmax -> weak-gradient failure. A flat softmax now means the query and
keys genuinely point the same direction, not that one vector happened to have
tiny norm; the score range is bounded [-1,1]*temp so the softmax always has
usable gradient. Because the key no longer also has to carry content, it is
free to specialize purely for discriminability — content lives in v.
(2) Chicken-and-egg credit assignment: keep the existing read_alpha-floor
curriculum AND add a cheap auxiliary "addressing" loss during the bootstrap
window: an InfoNCE/contrastive term over the read scores at read positions
that pulls the query toward the key of the position the model actually needed
(supervised by the task's read_mask target when available, else self-supervised
by the highest-attended slot that improved the LM logit — a STOP-grad teacher).
This gives the read path strong, dense gradient even while DeltaNet's state
shortcut still suffices, so addressing locks in instead of waiting for the
read to become useful by luck. The contrastive loss is on SCORES only and is
removed after warmup, so it cannot distort the converged solution — it only
breaks the bootstrap deadlock.

## why_generalizes
Coding/agentic recall is SEMANTIC, not symbol-identity: retrieve a def from a
call site, recall a var's type from its name, pull a tool's earlier output
given the current sub-goal, recall a decision/error from 40 steps ago. All of
these are "find the stored item whose KEY is semantically close to a query I
construct now, then use its CONTENT." Decoupling key from value is the minimal
structural requirement: the match surface (what makes two things "the same
binding") must be optimizable independently of the payload (the def body, the
type, the tool output). The role/recency tag supports compositional/multi-key
queries (e.g. "the error, recent" vs "the def, anywhere") which single-hop
embedding-identity addressing cannot express. Unit-norm cosine keys are exactly
what makes retrieval robust across paraphrase/semantic-neighbor queries rather
than exact-token matches.

## concrete_changes
- experiments/model.py WorkingMemory.__init__: add `self.W_k = nn.Linear(d_model,
  d_mem, bias=False)` alongside existing W_v/W_q/W_proj. Add a small learned tag
  table `self.role_tags = nn.Embedding(n_roles, d_mem)` (n_roles ~ 4-8) and a
  write-time role classifier `self.W_role = nn.Linear(d_model, n_roles)` (soft
  assignment); add `self.key_temp = nn.Parameter(tensor(log(1/sqrt(d_mem))))`.
- WorkingMemory.forward write side: compute `k = F.normalize(self.W_k(h), dim=-1)`
  at every position; buffer `buf_k` via the same top-idx gather used for buf_v;
  add soft role tag: `buf_k = F.normalize(buf_k + softmax(W_role)@role_tags)`.
- Read side: `q = F.normalize(self.W_q(h), dim=-1)`; `scores = einsum('btd,bkd->btk',
  q, buf_k) * exp(self.key_temp)` (REPLACE the q.buf_v dot-product). Keep the
  existing `+ log(buf_g)` gate-bias, causal mask, doc mask, all-masked NaN guard.
  `read = einsum('btk,bkd->btd', attn, buf_v)` (retrieve VALUE, not key).
  `injection = self.W_proj(read)`; keep read_alpha + floor curriculum unchanged.
- Bootstrap aux loss (training only, warmup window): expose `scores` pre-softmax;
  add `compute_wm_addressing_loss(scores, read_mask, target_src_pos)` =
  cross-entropy of the read softmax onto the correct source-slot index when the
  task provides one (MQAR: the value position of the queried pair, derivable as
  in probe_wm_utilization.py), else a STOP-grad self-teacher (slot whose removal
  most raised LM CE). Wire a weight `--wm_addressing_aux_weight` that anneals to 0
  over `--wm_addressing_aux_warmup_steps`, mirroring read_alpha_floor.
- forward_step decode path (model.py ~2900-2972): mirror the same K/V split so
  incremental decode matches the full-forward read.
- Back-compat: old ckpts lack W_k/role_tags/key_temp; load with strict=False and
  zero-init W_k -> at init keys are random-but-normalized (fine); OR add a cfg
  flag `mem_decoupled_kv` (default True for new runs, False loads old behavior).

## validation
Synthetic probe (fast, decisive): extend probe_wm_utilization.py to A/B the
current address-by-value WM vs the decoupled-KV WM on saturated MQAR
(d=64,L=2,T=512,K=128). Success criteria, averaged over the SAME 5 seeds that
currently give 0.51..0.999: (a) seed variance collapses (min recall up from
0.51 toward ~0.85+), (b) val_hit / either_hit rate rises and top-attn-mass
goes from ~0.16 toward ~0.9 on the previously-failing seeds, (c) read_alpha no
longer drifts down on failing seeds. Add a SEMANTIC variant of MQAR where the
query is a noised/paraphrased key (key + small random perturbation, or a
synonym table) so address-by-identity fails but content-similarity succeeds —
the decoupled cosine key should win there by construction. Path to real model:
enable on a short SFT/pretrain continuation of pretrain_phase_c.pt with
use_memory on, measure (1) eval_longctx_recall.py held-out recall per distance
bucket (the proper WM probe per CLAUDE.md) and (2) the WM-off ablation delta
(mean-vector ablation per the 2026-05-20 note) — load-bearing means a real drop.

## risks
- Extra params/compute: W_k + role table + a second normalize per position. Cost
  is O(B*T*d_mem) — same order as existing W_v; negligible vs trunk. Verify no
  OOM at b=14.
- The self-supervised teacher (when no read_mask target exists, i.e. natural
  pretrain) could be noisy and push addressing toward whatever the trunk already
  liked — must STOP-grad the teacher and cap aux weight; if it distorts VAL,
  fall back to aux-only-on-tasks-with-read-targets (MQAR / synthetic memory).
- Cosine keys remove magnitude as a usable signal; if the model was implicitly
  using key-norm to encode write-confidence, that moves entirely to the existing
  log(buf_g) gate bias — confirm the gate still carries it (it should).
- Role/recency tags add a discrete-ish routing decision that could collapse to
  one role early (PKM diversity lesson). Mitigate with a tiny entropy bonus on
  the role softmax during warmup, reuse the PKM diversity-loss pattern.
- forward_step parity bug risk (the recurring "decode path bypasses the new
  mechanism" failure in this repo) — add a full-forward==decode equality test.

## novelty
Standard technique at its core: decoupled K/V is exactly what real attention /
KV-cache memories do; the current WM is the unusual one for collapsing them.
Cosine/normalized keys with a learned temperature is also standard (CLIP, modern
retrievers). The repo-specific novel combination is: (1) applying this K/V split
inside a BOUNDED linear-RNN working-memory read (no KV cache) where it has not
been tried here, (2) the write-time role/recency tag for compositional agentic
queries, and (3) the bootstrap addressing-aux loss specifically engineered to
break the documented chicken-and-egg credit-assignment deadlock that makes WM
seed-variant. Prior art: attention K/V (Vaswani 2017), product-key memory
(Lample 2019, already in repo as PKMLayer — but PKM is a STATIC learned table;
this is DYNAMIC content-from-context), Memorizing Transformers / kNN-LM
(content keys over a cache). The contribution is the reliability-engineering for
this exact bounded-state regime, not the K/V idea itself.
