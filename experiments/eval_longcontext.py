"""
Long-context PPL eval — Phase 3.

Tests whether film helps at long T where depth alone can't capture
global structure. Our models were trained at T=512 with max_T=0
(no positional embedding), so they can in-principle handle longer T at
inference; only the linear-RNN cell determines positional behavior.

Eval at T=512 (training T, sanity), T=1024, T=2048 on Python code val.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from experiments.eval_bracket_structure import build_model_from_ckpt


@torch.no_grad()
def eval_ppl(model, tokenizer, T: int, n_chunks: int = 32, batch: int = 4,
             dataset: str = "codeparrot/codeparrot-clean",
             text_field: str = "content") -> float:
    """Stream Python code, score chunks of length T, average loss."""
    from datasets import load_dataset
    val_stream = load_dataset(dataset, split="train", streaming=True
                              ).shuffle(seed=42).skip(20_000)
    buf: list[int] = []
    eos = tokenizer.eos_token_id or tokenizer.bos_token_id or 0

    losses: list[float] = []
    for example in val_stream:
        text = example[text_field]
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= batch * T + 1:
            chunk = buf[: batch * T + 1]
            buf = buf[batch * T:]
            inputs = torch.tensor(chunk[:-1], dtype=torch.long
                                  ).view(batch, T).to("cuda")
            targets = torch.tensor(chunk[1:], dtype=torch.long
                                   ).view(batch, T).to("cuda")
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            )
            losses.append(loss.item())
            if len(losses) >= n_chunks:
                break
        if len(losses) >= n_chunks:
            break

    mean_loss = sum(losses) / len(losses)
    return float(torch.tensor(mean_loss).exp())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, action="append")
    p.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--n_chunks", type=int, default=32)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--dataset", type=str, default="codeparrot/codeparrot-clean")
    p.add_argument("--text_field", type=str, default="content")
    args = p.parse_args()

    from transformers import AutoTokenizer

    all_results = {}
    for ckpt in args.ckpt:
        print(f"\n{'=' * 70}\nEvaluating: {ckpt}\n{'=' * 70}")
        model, cfg = build_model_from_ckpt(ckpt)
        print(f"  feedback={cfg.get('feedback_mode')}  n_layers={cfg['n_layers']}  d_model={cfg['d_model']}")
        tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer", "HuggingFaceTB/SmolLM2-135M"))
        results = {}
        for T in args.T:
            ppl = eval_ppl(model, tok, T, n_chunks=args.n_chunks,
                           batch=args.batch, dataset=args.dataset,
                           text_field=args.text_field)
            results[T] = ppl
            print(f"  T={T}: PPL = {ppl:.2f}")
        all_results[ckpt] = results
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    Ts = args.T
    header = f"{'ckpt':<48}" + "".join(f"  T={T:<6}" for T in Ts)
    print(header)
    for ckpt, results in all_results.items():
        name = pathlib.Path(ckpt).stem
        row = f"{name:<48}" + "".join(f"  {results[T]:<8.2f}" for T in Ts)
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
