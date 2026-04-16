# LLM Baseline Error Analysis

This report compares the zero-shot and few-shot LLM baselines against the existing classical TF-IDF baseline on `dev_core`.

## Requested Confusion Directions

### Saudi -> Levantine

- Classical count: `15`
- Zero-shot count: `1`
- Few-shot count: `1`

### Saudi -> Maghrebi

- Classical count: `11`
- Zero-shot count: `0`
- Few-shot count: `0`

### Egyptian -> Maghrebi

- Classical count: `22`
- Zero-shot count: `1`
- Few-shot count: `1`

### Egyptian -> Levantine

- Classical count: `13`
- Zero-shot count: `0`
- Few-shot count: `2`

| True Label | Predicted Label | Classical | Zero-Shot | Few-Shot |
| --- | --- | --- | --- | --- |
| Saudi | Levantine | 15 | 1 | 1 |
| Saudi | Maghrebi | 11 | 0 | 0 |
| Egyptian | Maghrebi | 22 | 1 | 1 |
| Egyptian | Levantine | 13 | 0 | 2 |

## Overall Comparison

| Mode | Accuracy | Macro F1 |
| --- | --- | --- |
| Classical | 0.8869 | 0.8483 |
| Zero-Shot | 0.8679 | 0.8330 |
| Few-Shot | 0.8749 | 0.8414 |

## Interpretation

- The LLM baseline is evaluated against the same benchmark-safe `train_core` / `dev_core` setup as the classical baseline.
- The requested confusion directions show whether prompting reduces the TF-IDF tendency to absorb Saudi and Egyptian texts into broader regional classes.
- Zero-shot performance reflects the model's raw label understanding; few-shot performance adds only a small, train-derived support set and remains within the benchmark-safe core pool.
- Remaining errors should be read together with `ERROR_ANALYSIS.md`: weak local signal, shared colloquial vocabulary, quasi-MSA writing, and topic-driven cues are still expected to matter even for an LLM baseline.
