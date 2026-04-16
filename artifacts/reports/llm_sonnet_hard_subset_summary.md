# Claude Sonnet Targeted Confusion Subset Summary

This report evaluates Claude Sonnet on the targeted confusion subset only, then compares Sonnet against the classical baseline and Gemini Flash-Lite on the same subset rows.

## Setup

- Provider: `anthropic_messages`
- Model: `claude-sonnet-4-20250514`
- Train path: `data/processed/train_core.csv`
- Dev path: `data/processed/dev_core.csv`
- Targeted confusion subset rows: `62`
- Text column: `processed_text`
- Target column: `macro_label`
- Few-shot examples per class: `2`

## Metric Caveat

- This subset contains only `Saudi` and `Egyptian` true labels.
- `Levantine` and `Maghrebi` have true-label support `0` here and should not be interpreted as true zero-performance on the task.
- The stored 4-label macro F1 is retained for traceability, but it is penalized by absent classes and is not directly comparable to full-dev macro F1.
- The more informative summary metric for this subset is present-label macro F1 over the labels that actually appear: `Saudi` and `Egyptian`.

## Overall Comparison On The Targeted Confusion Subset

| Mode | Accuracy | Present-Label Macro F1 | Stored 4-Label Macro F1 |
| --- | --- | --- |
| Classical | 0.0161 | 0.0270 | 0.0135 |
| Gemini Zero-Shot | 0.8710 | 0.8848 | 0.4424 |
| Gemini Few-Shot | 0.8387 | 0.8672 | 0.4336 |
| Sonnet Zero-Shot | 0.8387 | 0.8611 | 0.4306 |
| Sonnet Few-Shot | 0.8548 | 0.8882 | 0.4441 |

## Sonnet Latency And Cost

| Mode | Requests | Total ms | Avg request ms | Avg row ms | Estimated Cost (USD) |
| --- | --- | --- | --- | --- | --- |
| Zero-Shot | 4 | 13754.5 | 3438.6 | 221.8 | 0.0240 |
| Few-Shot | 4 | 12775.4 | 3193.8 | 206.1 | 0.0266 |

## Sonnet Present-Label Metrics

| Label | Zero P | Zero R | Zero F1 | Few P | Few R | Few F1 | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Saudi | 0.9091 | 0.7692 | 0.8333 | 0.9545 | 0.8077 | 0.8750 | 26 |
| Egyptian | 0.8889 | 0.8889 | 0.8889 | 0.9143 | 0.8889 | 0.9014 | 36 |

## Support Note

- The few-shot setting uses the existing deterministic `8`-example support strategy from `train_core` only.
- Support-source IDs retained from the stored run:
  `subtask1_train_10492`, `subtask1_train_445`, `subtask1_train_9718`, `subtask1_train_4298`, `subtask1_train_5745`, `subtask1_train_10147`, `subtask1_train_13697`, `subtask1_train_3155`.

## Interpretation

- What this subset does measure:
  whether Sonnet recovers rows in the specific Saudi/Egyptian confusion region where earlier baselines mapped them into `Levantine` or `Maghrebi`.
- What it does not measure:
  overall full-dev performance, generic task difficulty, or balanced 4-way coverage.
- Why it is useful:
  it isolates the failure mode highlighted in [ERROR_ANALYSIS.md](../../ERROR_ANALYSIS.md) and makes the Sonnet, Gemini, and classical comparisons like-for-like on exactly those rows.
