# MARBERT Summary

This run fine-tunes `UBC-NLP/MARBERT` on `train_core` and evaluates on `dev_core` only.

## Setup

- Model: `UBC-NLP/MARBERT`
- Train path: `data/processed/train_core.csv`
- Dev path: `data/processed/dev_core.csv`
- Text column: `processed_text`
- Target column: `macro_label`
- Labels: `Saudi, Egyptian, Levantine, Maghrebi`
- Max length: `128`
- Train batch size: `32`
- Eval batch size: `64`
- Learning rate: `2e-05`
- Weight decay: `0.01`
- Epochs requested: `5`
- Warmup ratio: `0.1`
- Optimizer: `adamw_torch`
- Scheduler: `linear_with_warmup`
- Loss: `cross_entropy`
- Class weights: `balanced`
- Seed: `123`
- Device: `cuda`
- Parameter dtype: `torch.float32`
- Best checkpoint: `artifacts/checkpoints/marbert_seed_123`
- Best epoch: `5`
- Train rows: `10000`
- Dev rows: `998`

## Dev Metrics

- Accuracy: `0.9719`
- Macro F1: `0.9683`
- Eval loss: `0.1953`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Saudi | 0.9706 | 0.9900 | 0.9802 | 100 |
| Egyptian | 0.9490 | 0.9394 | 0.9442 | 99 |
| Levantine | 0.9702 | 0.9799 | 0.9751 | 399 |
| Maghrebi | 0.9797 | 0.9675 | 0.9736 | 400 |

## Training History

| Epoch | Train Loss | Eval Loss | Eval Accuracy | Eval Macro F1 |
| --- | --- | --- | --- | --- |
| 1 | 0.4865 | 0.2040 | 0.9499 | 0.9324 |
| 2 | 0.1230 | 0.1665 | 0.9639 | 0.9583 |
| 3 | 0.0423 | 0.2788 | 0.9619 | 0.9534 |
| 4 | 0.0179 | 0.1816 | 0.9669 | 0.9609 |
| 5 | 0.0076 | 0.1953 | 0.9719 | 0.9683 |

## Comparison Note

| Baseline | Accuracy | Macro F1 |
| --- | --- | --- |
| Classical | 0.8869 | 0.8483 |
| Gemini Zero-Shot | 0.8679 | 0.8330 |
| Gemini Few-Shot | 0.8749 | 0.8414 |
| Sonnet Zero-Shot | 0.8268 | 0.7908 |
| Sonnet Few-Shot | 0.8408 | 0.8042 |
| MARBERT | 0.9719 | 0.9683 |

- Use this table as a benchmark-safe comparison against the current classical, Gemini full-dev, and Sonnet full-dev runs already present in the repository.
