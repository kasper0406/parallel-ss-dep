# Shallow-wide trunk

## Summary
An iso-param swap of the deep 30L×576d trunk for a **shallow-wide 10L×896d trunk + 5 dense reverse-FiLM pairs + K=3 self-feed**. Hypothesis: "the brain is shallow with heavy feedback loops" — fewer layers means shorter gradient paths, dense fan-in across depth, and a wider hidden absorbs the freed parameter budget. **Result: matches the 30L baseline on VAL CE at iso-token and trains ~18 % faster in wall-clock.** This is the production trunk shape from v6 onward. Source: `CLAUDE.md`, `README.md`.

## Config
```
--n_layers 10 --d_model 896 --n_heads 14 --d_head 64
--feedback film --feedback_pairs "0,5;1,6;2,7;3,8;4,9" --feedback_self_k 3
```
Every early layer (0–4) reads from a late layer (5–9): dense fan-in across depth. See [[film-feedback]].

## Evidence
- At iso-token, 10L×896d matches the 30L baseline on overall VAL CE.
- ~18 % faster wall-clock (v7.1-film 13.3 h vs v4 16.3 h at the same step budget).
- Validated v6/v7/v7.1 (2026-05-17–18). See [[pretrain-run-history]].
- A negative control: **LayerDrop / Stochastic Depth made it worse** — it redistributed depth utilization as advertised but cost ~0.02–0.10 CE on every source. Depth-concentration was never the bottleneck. Don't enable `--layer_drop_max`.

## Related
[[deltanet-backbone]] · [[film-feedback]] · [[pretrain-run-history]] · #architecture
