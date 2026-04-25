"""
Induction heads — the canonical recall test (Olsson et al. 2022).

Spec:
  - Sequence of random tokens, with one (trigger, target) pair planted twice.
  - The model must learn: "if I see trigger token A, predict the token
    that followed A's previous occurrence."

Why this is the right minimum-viable recall test:
  - MQAR (Zoology) is too hard for plain linear-attention without
    feature maps; many architectures fail to learn it at small scale.
  - Induction heads is the simplest possible "look up past content"
    pattern. Solvable by any architecture with KV memory.
  - Olsson et al.: "Induction heads form during training... they're the
    main source of in-context learning ability in small transformers."

Concrete construction (one example):
  - Sequence length T, vocabulary [0, vocab_size).
  - Pick a "trigger" token A and a "target" token B (distinct, both ≥ 1).
  - Token 0 is reserved as the "predict here" cue.
  - Pattern at random position p₁ ∈ [1, T/2): [A, B] (the prior occurrence)
  - Pattern at fixed position T-2: [A, ?] (the prediction site)
  - Other positions: random tokens ≥ 1.
  - Label[T-1] = B; loss only at position T-1.
"""
from __future__ import annotations

import torch


def make_batch(B: int, T: int, vocab_size: int = 32,
               device: torch.device | str = "cuda",
               generator: torch.Generator | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (input_ids, labels, loss_mask) for induction heads.

    Returns:
        input_ids:  (B, T) int64
        labels:     (B, T) int64 — target value at the prediction position
        loss_mask:  (B, T) bool   — True only at the prediction position
    """
    assert vocab_size >= 4, "need vocab ≥ 4 (trigger, target, plus distractors)"
    assert T >= 6, "need T ≥ 6"

    # Pick trigger and target per example, distinct, both ≥ 2 (1 reserved as
    # filler, 0 reserved as cue).
    triggers = torch.randint(2, vocab_size, (B,), device=device, generator=generator)
    targets = torch.randint(2, vocab_size, (B,), device=device, generator=generator)
    coll = (targets == triggers)
    while coll.any():
        new = torch.randint(2, vocab_size, (B,), device=device, generator=generator)
        targets = torch.where(coll, new, targets)
        coll = (targets == triggers)

    # Random base sequence — but resample any token that collides with the
    # per-example trigger so the trigger appears ONLY at our planted positions.
    inp = torch.randint(1, vocab_size, (B, T), device=device, generator=generator,
                        dtype=torch.int64)
    # `triggers` is (B,); broadcast comparison gives (B, T).
    while True:
        clash = (inp == triggers[:, None])
        if not clash.any():
            break
        new_vals = torch.randint(1, vocab_size, inp.shape, device=device,
                                 generator=generator, dtype=torch.int64)
        inp = torch.where(clash, new_vals, inp)

    # Plant the prior occurrence at a random position in [1, T-3).
    p1 = torch.randint(1, T - 3, (B,), device=device, generator=generator)
    arange_b = torch.arange(B, device=device)
    inp[arange_b, p1] = triggers
    inp[arange_b, p1 + 1] = targets

    # Plant the prediction site at T-2 (trigger) and T-1 (cue = 0).
    inp[:, T - 2] = triggers
    inp[:, T - 1] = 0                 # cue token; model output here is the prediction

    labels = torch.zeros(B, T, dtype=torch.int64, device=device)
    labels[:, T - 1] = targets

    loss_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    loss_mask[:, T - 1] = True

    return inp, labels, loss_mask
