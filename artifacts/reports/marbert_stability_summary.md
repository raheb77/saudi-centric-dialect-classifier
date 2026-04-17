# MARBERT Stability Summary

This summary aggregates the cleaned post-dedup MARBERT runs on the 998-row `dev_core` split.

## Seed Runs

| Seed | Dev rows | Accuracy | Macro F1 | Saudi F1 | Egyptian F1 | Levantine F1 | Maghrebi F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `42` | `998` | `0.9669` | `0.9595` | `0.9604` | `0.9340` | `0.9735` | `0.9702` |
| `123` | `998` | `0.9719` | `0.9683` | `0.9802` | `0.9442` | `0.9751` | `0.9736` |
| `7` | `998` | `0.9649` | `0.9561` | `0.9406` | `0.9424` | `0.9675` | `0.9738` |

## Mean ± Std

| Metric | Mean ± Std |
| --- | --- |
| Accuracy | `0.9679 +/- 0.0036` |
| Macro F1 | `0.9613 +/- 0.0063` |
| Saudi F1 | `0.9604 +/- 0.0198` |
| Egyptian F1 | `0.9402 +/- 0.0054` |
| Levantine F1 | `0.9720 +/- 0.0040` |
| Maghrebi F1 | `0.9726 +/- 0.0020` |

## Interpretation

- Macro F1 stability: `acceptable stability`
- Thresholds used: `< 0.005` highly stable, `0.005-0.015` acceptable stability, `> 0.015` unstable.
- The MARBERT multi-seed pass is intended to establish encoder stability on the cleaned benchmark-safe split.

## Baseline Context

| Model | Setting | Accuracy | Macro F1 | Run Type |
| --- | --- | --- | --- | --- |
| Classical baseline | TF-IDF + Logistic Regression | `0.8869` | `0.8483` | single run |
| Gemini Flash-Lite | zero-shot | `0.8679` | `0.8330` | single run |
| Gemini Flash-Lite | few-shot | `0.8749` | `0.8414` | single run |
| Claude Sonnet | zero-shot | `0.8268` | `0.7908` | single run |
| Claude Sonnet | few-shot | `0.8408` | `0.8042` | single run |
| MARBERT | mean +/- std (n=3 seeds) | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` | multi-seed |

- Classical, Gemini, and Sonnet were not multi-seed rerun here. Their single-run values are shown for task context, not as a claim of LLM instability.
