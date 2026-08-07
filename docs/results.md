# Results highlights

Compact view of thesis leaderboards. Full narrative: [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

## Main claim (Llama-3.2-3B, Bedrock Haiku GA)

Quality-stratified LoRA + dense merge (`Ours_tier_*`) vs uniform LoRA (`B3`):

| Dataset | Metric | B3 (approx) | Ours (main) |
|---------|--------|-------------|-------------|
| RepLiQA | GA (merge) | ~3.64 | **~3.78** (`Ours_tier_merge`) |
| Quoref | GA (ctx) | lower | **~3.74** (`Ours_tier_ctx`) |
| SQuAD 2.0 | GA (ctx) | lower | **~2.32** (`Ours_tier_ctx`) |

Exact numbers and CIs: `examples/results/leaderboards/*.json` and `RESULTS_SUMMARY.md`.

## Synthetic QA coverage (RepLiQA split-1 A/B)

Document coverage of train-pool QA (excluding `drop` tiers for FT):

| Generator | Lexical coverage | Embedding coverage |
|-----------|------------------|--------------------|
| Fine-tuned generator | ~84.5% | ~90.7% |
| Haiku generator | ~97.8% | ~99.0% |

Source JSONs under `examples/results/coverage/`.

## Low-tier examples

`examples/results/low_tier_slide_examples.{jsonl,txt}` — illustrative low / drop synthetic pairs for slides.

## General ability (base Instruct vs RepLiQA FT merge)

Frozen 100 questions (math / knowledge / commonsense / instruction) at 1B / 3B / 8B:

`examples/results/general_ability/*_base_vs_ft.jsonl`
