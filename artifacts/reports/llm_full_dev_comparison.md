# Full-Dev LLM Comparison

This comparison uses the same full `dev_core` split for the classical baseline, Gemini Flash-Lite, and Claude Sonnet.

## Overall Metrics

| Mode | Accuracy | Macro F1 |
| --- | --- | --- |
| Classical | 0.8869 | 0.8483 |
| Gemini Zero-Shot | 0.8679 | 0.8330 |
| Gemini Few-Shot | 0.8749 | 0.8414 |
| Sonnet Zero-Shot | 0.8268 | 0.7908 |
| Sonnet Few-Shot | 0.8408 | 0.8042 |

## Tracked Confusion Directions

| True Label | Predicted Label | Classical | Gemini Zero | Gemini Few | Sonnet Zero | Sonnet Few |
| --- | --- | --- | --- | --- | --- | --- |
| Saudi | Levantine | 15 | 1 | 1 | 9 | 10 |
| Saudi | Maghrebi | 11 | 0 | 0 | 1 | 3 |
| Egyptian | Maghrebi | 22 | 1 | 1 | 2 | 2 |
| Egyptian | Levantine | 13 | 0 | 2 | 1 | 1 |

## Interpretation

- This is the like-for-like full-dev comparison across all currently completed benchmark-safe baselines.
- The four tracked confusion directions remain the main diagnostic for whether a stronger model reduces Saudi and Egyptian absorption into broader regional labels.
- Use this report to decide whether the next phase should focus on encoder training rather than more prompt-only iteration.
