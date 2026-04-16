# Claude Sonnet Hard Subset Definition

This subset is built from `data/processed/dev_core.csv` only.

## Selection Rule

Keep a dev-set row if at least one existing baseline prediction places it in one of these tracked confusion directions:

- `Saudi -> Levantine`
- `Saudi -> Maghrebi`
- `Egyptian -> Maghrebi`
- `Egyptian -> Levantine`

Baselines checked for inclusion:

- classical baseline (`predicted_label`)
- Gemini Flash-Lite zero-shot (`zero_shot_predicted_label`)
- Gemini Flash-Lite few-shot (`few_shot_predicted_label`)

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

## Implication

Because all tracked confusion directions start from `Saudi` or `Egyptian`, the resulting hard subset contains only those two true labels. Metrics therefore stay comparable across models on the same rows, but 4-class macro F1 will be depressed by the zero-support `Levantine` and `Maghrebi` true classes in this subset.
