"""Quick quality check on the teacher-generated distillation shard.

Reports:
  - Shape and counts.
  - Fraction of teacher-finite slots (vs prompt-mask -inf).
  - Mean rank-0 logprob on teacher-finite positions (mode confidence).
  - EOS fraction (a measure of how well-packed the data is).
  - A decoded sample chunk for sanity-check.

Run:
  /home/knielsen/ml/parallel-ss-dep/.venv/bin/python \\
    experiments/distill_pilot_data_check.py \\
    --shards /home/knielsen/ml/parallel-ss-dep-distill/data/distill_pilot_1M
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=str, required=True)
    p.add_argument("--decode_sample", action="store_true",
                   help="Decode a sample chunk via the Qwen tokenizer.")
    args = p.parse_args()

    sd = pathlib.Path(args.shards)
    shards = sorted(sd.glob("shard_*.npz"))
    print(f"Found {len(shards)} shards in {sd}")

    # Load just the first 2 for a quality check (don't need everything).
    z = np.load(shards[0])
    ti, tk, lp = z["token_ids"], z["top_k_ids"], z["top_k_logprobs"]
    n_chunks_first = ti.shape[0]
    total_chunks = sum(np.load(s)["token_ids"].shape[0] for s in shards)
    print(f"  total chunks: {total_chunks}, first shard has {n_chunks_first}")
    print(f"  per-chunk T = {ti.shape[1]}, top-K = {tk.shape[2]}")

    # Quality on first shard.
    finite = np.isfinite(lp[..., 0])
    print(f"  finite teacher slots (rank-0): {finite.mean()*100:.2f}%")
    rank0 = lp[..., 0][finite].astype(np.float32)
    print(f"  rank-0 logprob: mean={rank0.mean():.3f}, "
          f"median={np.median(rank0):.3f}, "
          f"5p={np.percentile(rank0, 5):.3f}, "
          f"95p={np.percentile(rank0, 95):.3f}")

    # Manifest (only present after the run finishes).
    manifest_path = sd / "manifest.npz"
    teacher_model = "QuantTrio/Qwen3.6-35B-A3B-AWQ"
    if manifest_path.exists():
        m = np.load(manifest_path, allow_pickle=True)
        teacher_model = str(m["model"])
        print(f"  manifest: model={teacher_model}, dataset={m['dataset']}, "
              f"vocab_size={m['vocab_size']}, T={m['T']}, top_k={m['top_k']}")
    else:
        print(f"  (manifest not yet written — run still in progress)")

    # eos / boundary frac.
    eos_id = 248046  # Qwen3.6 <|im_end|>
    eos_frac = (ti == eos_id).mean() * 100
    print(f"  eos token frac: {eos_frac:.2f}% (lower = better packed)")

    if args.decode_sample:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(teacher_model,
                                            trust_remote_code=True)
        print("\n--- sample chunk 0 (first 400 chars decoded) ---")
        text = tok.decode(ti[0].tolist())
        print(text[:400])
        print("...")


if __name__ == "__main__":
    main()
