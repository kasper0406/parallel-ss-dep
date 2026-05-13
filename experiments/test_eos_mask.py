"""Tests for the EOS-mask-in-targets flag in MixedSourceStream.

Run: PYTHONPATH=. .venv/bin/python -m pytest experiments/test_eos_mask.py -v
"""
from __future__ import annotations

import torch

from experiments.data_mix import MixedSourceStream, SourceConfig


class _FakeTok:
    """Minimal tokenizer stub: encode = whitespace split mapping chars→ids."""
    eos_token_id = 99
    bos_token_id = None

    def __init__(self):
        self._vocab = {}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for ch in text:
            if ch not in self._vocab:
                self._vocab[ch] = len(self._vocab) + 1  # avoid 0
            ids.append(self._vocab[ch])
        return ids


def _short_source() -> list[SourceConfig]:
    """One synthetic source returning the same 8-char doc forever — small
    enough that each tokenised sample fits in one chunk multiple times,
    guaranteeing EOS appears inside the chunks."""
    return [SourceConfig(
        name="fake", dataset_id="fake", split="train", weight=1.0,
        text_field="text",
    )]


def _patched_stream(monkey_iter, mask_eos: bool, n_yields: int = 5):
    """Build a MixedSourceStream and replace its per-source iterator with
    a synthetic generator yielding `{'text': 'abcde'}` forever."""
    src = _short_source()
    tok = _FakeTok()
    stream = MixedSourceStream(
        sources=src, tokenizer=tok, block_size=16,
        thinking_token_id=None,
        think_burst_prob=0.0,
        base_seed=0,
        mask_eos_in_targets=mask_eos,
    )

    # Monkey-patch `_open_stream` to return our infinite generator.
    import experiments.data_mix as dm
    orig = dm._open_stream

    def _fake_open(src, seed=0):
        def gen():
            while True:
                yield {"text": "abcde"}
        return gen()
    dm._open_stream = _fake_open
    try:
        out = []
        it = iter(stream)
        for _ in range(n_yields):
            out.append(next(it))
        return out
    finally:
        dm._open_stream = orig


def test_eos_unmasked_default():
    """Without --mask_eos_in_targets, EOS tokens appear in targets."""
    samples = _patched_stream(monkey_iter=None, mask_eos=False, n_yields=3)
    saw_eos = False
    for inputs, targets in samples:
        if (targets == 99).any():
            saw_eos = True
            break
    assert saw_eos, "Expected EOS in targets when mask is OFF"


def test_eos_masked():
    """With --mask_eos_in_targets, EOS positions in targets become -100."""
    samples = _patched_stream(monkey_iter=None, mask_eos=True, n_yields=3)
    eos_count = 0
    masked_count = 0
    for inputs, targets in samples:
        # EOS-as-input still appears (only TARGETS are masked).
        eos_count += int((inputs == 99).sum().item())
        # No EOS should appear in TARGETS — they should be -100 instead.
        assert (targets != 99).all(), f"EOS leaked into targets despite mask"
        masked_count += int((targets == -100).sum().item())
    # Sanity: we did see some EOS in inputs (so the docs had boundaries),
    # AND at least one position got -100 (so the mask actually fired).
    assert eos_count > 0, "Test setup failed: no EOS in any input chunk"
    assert masked_count > 0, "Mask never fired — targets had no -100"
