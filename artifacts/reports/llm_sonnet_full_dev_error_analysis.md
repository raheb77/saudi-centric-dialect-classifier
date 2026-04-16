# LLM Baseline Error Analysis

This report compares the zero-shot and few-shot LLM baselines against the existing classical TF-IDF baseline and Gemini Flash-Lite full-dev run on `dev_core`.

## Requested Confusion Directions

### Saudi -> Levantine

- Classical count: `15`
- Gemini zero-shot count: `1`
- Gemini few-shot count: `1`
- Zero-shot count: `9`
- Few-shot count: `10`

### Saudi -> Maghrebi

- Classical count: `11`
- Gemini zero-shot count: `0`
- Gemini few-shot count: `0`
- Zero-shot count: `1`
- Few-shot count: `3`

### Egyptian -> Maghrebi

- Classical count: `22`
- Gemini zero-shot count: `1`
- Gemini few-shot count: `1`
- Zero-shot count: `2`
- Few-shot count: `2`

### Egyptian -> Levantine

- Classical count: `13`
- Gemini zero-shot count: `0`
- Gemini few-shot count: `2`
- Zero-shot count: `1`
- Few-shot count: `1`

| True Label | Predicted Label | Classical | Gemini Zero | Gemini Few | Zero-Shot | Few-Shot |
| --- | --- | --- | --- | --- | --- | --- |
| Saudi | Levantine | 15 | 1 | 1 | 9 | 10 |
| Saudi | Maghrebi | 11 | 0 | 0 | 1 | 3 |
| Egyptian | Maghrebi | 22 | 1 | 1 | 2 | 2 |
| Egyptian | Levantine | 13 | 0 | 2 | 1 | 1 |

## Overall Comparison

| Mode | Accuracy | Macro F1 |
| --- | --- | --- |
| Classical | 0.8869 | 0.8483 |
| Gemini Zero-Shot | 0.8679 | 0.8330 |
| Gemini Few-Shot | 0.8749 | 0.8414 |
| Zero-Shot | 0.8268 | 0.7908 |
| Few-Shot | 0.8408 | 0.8042 |

## Interpretation

- The LLM baseline is evaluated against the same benchmark-safe `train_core` / `dev_core` setup as the classical baseline.
- The requested confusion directions show whether prompting reduces the TF-IDF tendency to absorb Saudi and Egyptian texts into broader regional classes.
- Gemini Flash-Lite provides a like-for-like full-dev LLM comparison using the same benchmark-safe core split.
- Zero-shot performance reflects the model's raw label understanding; few-shot performance adds only a small, train-derived support set and remains within the benchmark-safe core pool.
- Remaining errors should be read together with `ERROR_ANALYSIS.md`: weak local signal, shared colloquial vocabulary, quasi-MSA writing, and topic-driven cues are still expected to matter even for an LLM baseline.
