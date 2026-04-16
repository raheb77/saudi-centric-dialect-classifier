# Claude Sonnet Targeted Confusion Subset Definition

This subset is built from `data/processed/dev_core.csv` only.
It is not a generic difficulty sample from the dev set.
It is a repository-specific targeted confusion subset that isolates the Saudi/Egyptian absorption errors already tracked in [ERROR_ANALYSIS.md](../../ERROR_ANALYSIS.md).

## Selection Rule

Keep a `dev_core` row if its `source_id` matches at least one of the following rules:

1. In [classical_baseline_dev_predictions.csv](./classical_baseline_dev_predictions.csv), the pair `("true_label", "predicted_label")` is one of the tracked confusion directions below.
2. In [llm_gemini_flash_lite_dev_predictions.csv](./llm_gemini_flash_lite_dev_predictions.csv), the pair `("true_label", "zero_shot_predicted_label")` is one of the tracked confusion directions below.
3. In [llm_gemini_flash_lite_dev_predictions.csv](./llm_gemini_flash_lite_dev_predictions.csv), the pair `("true_label", "few_shot_predicted_label")` is one of the tracked confusion directions below.

Tracked confusion directions:

- `Saudi -> Levantine`
- `Saudi -> Maghrebi`
- `Egyptian -> Maghrebi`
- `Egyptian -> Levantine`

The final subset is the union of matching rows, deduplicated by `source_id` and kept in original `dev_core` order.

## Counts

- Full `dev_core` rows: `999`
- Classical matches: `61`
- Gemini zero-shot matches: `2`
- Gemini few-shot matches: `4`
- Final unique subset rows: `62`

## True-Label Composition

| True Label | Rows |
| --- | --- |
| Saudi | 26 |
| Egyptian | 36 |
| Levantine | 0 |
| Maghrebi | 0 |

## Selector Overlap

| Selection Source(s) | Rows |
| --- | --- |
| classical | 58 |
| classical, gemini_few | 1 |
| classical, gemini_few, gemini_zero | 2 |
| gemini_few | 1 |

## Interpretation

- What this subset measures:
  recovery on the specific Saudi/Egyptian confusion region where earlier baselines absorbed single-country labels into broader `Levantine` or `Maghrebi` buckets.
- What it does not measure:
  overall `dev_core` difficulty, broad 4-way task coverage, or generic robustness on all label types.
- Why it is still useful:
  it gives a like-for-like comparison on the exact failure region highlighted in [ERROR_ANALYSIS.md](../../ERROR_ANALYSIS.md), without conflating that region with the rest of the benchmark.

## Metric Caveat

Because all tracked confusion directions start from `Saudi` or `Egyptian`, the targeted confusion subset contains only those two true labels.
`Levantine` and `Maghrebi` therefore have true-label support `0` in this subset.
Any 4-label macro F1 reported on this subset is retained for bookkeeping only and is not directly comparable to full-dev 4-label macro F1.
