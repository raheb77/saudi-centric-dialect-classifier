# Targeted Confusion Subset Comparison

This comparison uses the same `62` targeted confusion subset rows for every model.
The subset is defined in [llm_sonnet_hard_subset_definition.md](./llm_sonnet_hard_subset_definition.md).
All numbers below are subset-only and should not be compared directly to full-dev reports.

## Overall Metrics

| Mode | Accuracy | Present-Label Macro F1 | Stored 4-Label Macro F1 |
| --- | --- | --- |
| Classical | 0.0161 | 0.0270 | 0.0135 |
| Gemini Zero-Shot | 0.8710 | 0.8848 | 0.4424 |
| Gemini Few-Shot | 0.8387 | 0.8672 | 0.4336 |
| Sonnet Zero-Shot | 0.8387 | 0.8611 | 0.4306 |
| Sonnet Few-Shot | 0.8548 | 0.8882 | 0.4441 |

## Interpretation

- What this subset measures:
  model recovery on a repository-specific baseline-failure region, not generic task difficulty.
- What it does not measure:
  full-dev performance, balanced 4-label generalization, or the overall ranking of models on the entire benchmark.
- Why it is useful:
  it isolates the Saudi/Egyptian-to-broader-region confusions already flagged in [ERROR_ANALYSIS.md](../../ERROR_ANALYSIS.md) and compares all models on exactly the same rows.

## Comparison Readout

- The classical baseline is near-floor on this subset because the subset was constructed from its known failure directions.
- Gemini zero-shot remains the strongest subset baseline by present-label macro F1 (`0.8848`) among the non-Sonnet runs.
- Sonnet few-shot slightly exceeds Gemini few-shot on the subset (`0.8882` vs `0.8672` present-label macro F1), while Sonnet zero-shot trails Gemini zero-shot (`0.8611` vs `0.8848`).
- These are like-for-like subset comparisons only. They should not be read as Sonnet-vs-Gemini full-dev conclusions.
