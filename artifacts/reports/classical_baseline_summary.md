# Classical Baseline Summary

This baseline trains a Logistic Regression classifier on combined word and char TF-IDF features using `train_core` and evaluates on `dev_core` only.

## Setup

- Train path: `data/processed/train_core.csv`
- Dev path: `data/processed/dev_core.csv`
- Text column: `processed_text`
- Target column: `macro_label`
- Labels: `Saudi, Egyptian, Levantine, Maghrebi`
- Word n-grams: `(1, 2)`
- Char analyzer: `char_wb`
- Char n-grams: `(3, 5)`
- Train rows: `10000`
- Dev rows: `999`

## Overall Metrics

- Accuracy: `0.8869`
- Macro F1: `0.8483`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Saudi | 0.9250 | 0.7400 | 0.8222 | 100 |
| Egyptian | 0.9028 | 0.6500 | 0.7558 | 100 |
| Levantine | 0.8889 | 0.9223 | 0.9053 | 399 |
| Maghrebi | 0.8753 | 0.9475 | 0.9100 | 400 |

## Confusion Matrix

| True \ Pred | Saudi | Egyptian | Levantine | Maghrebi |
| --- | --- | --- | --- | --- |
| Saudi | 74 | 0 | 15 | 11 |
| Egyptian | 0 | 65 | 13 | 22 |
| Levantine | 5 | 5 | 368 | 21 |
| Maghrebi | 1 | 2 | 18 | 379 |
