# Final Model Comparison

| Model | Setting | Accuracy | Macro F1 | Evaluation Split |
| --- | --- | --- | --- | --- |
| Classical baseline | TF-IDF + Logistic Regression | `0.8869` | `0.8483` | original dev (999 rows) |
| Gemini Flash-Lite | zero-shot | `0.8679` | `0.8330` | original dev (999 rows) |
| Gemini Flash-Lite | few-shot | `0.8749` | `0.8414` | original dev (999 rows) |
| Claude Sonnet | zero-shot | `0.8268` | `0.7908` | original dev (999 rows) |
| Claude Sonnet | few-shot | `0.8408` | `0.8042` | original dev (999 rows) |
| MARBERT | mean +/- std (n=3 seeds) | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` | cleaned dev (998 rows) |

## Recommendation

- MARBERT is strong enough to stand as the v1 encoder result if its multi-seed stability remains in the highly stable or acceptable range.
- A second encoder is optional rather than required; it is justified only if you want architectural diversity or a stronger comparison story.

MARBERT was re-evaluated on 998 rows after removing one dev row due to train/dev processed-text overlap. Classical, Gemini, and Sonnet were evaluated on the original 999-row dev split. The post-dedup MARBERT metrics are reported explicitly and should be interpreted as the benchmark-safe encoder result.
