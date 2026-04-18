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
- Seed: `42`
- Device: `cuda`
- Parameter dtype: `torch.float32`
- Best checkpoint: `artifacts/checkpoints/marbert_seed_42`
- Best epoch: `5`
- Train rows: `10000`
- Dev rows: `998`

## Dev Metrics

- Accuracy: `0.9669`
- Macro F1: `0.9595`
- Eval loss: `0.2202`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Saudi | 0.9510 | 0.9700 | 0.9604 | 100 |
| Egyptian | 0.9388 | 0.9293 | 0.9340 | 99 |
| Levantine | 0.9821 | 0.9649 | 0.9735 | 399 |
| Maghrebi | 0.9631 | 0.9775 | 0.9702 | 400 |

## Training History

| Epoch | Train Loss | Eval Loss | Eval Accuracy | Eval Macro F1 |
| --- | --- | --- | --- | --- |
| 1 | 0.4721 | 0.2794 | 0.9489 | 0.9318 |
| 2 | 0.1365 | 0.3685 | 0.9459 | 0.9347 |
| 3 | 0.0572 | 0.3449 | 0.9569 | 0.9463 |
| 4 | 0.0209 | 0.1827 | 0.9639 | 0.9548 |
| 5 | 0.0057 | 0.2202 | 0.9669 | 0.9595 |

## Comparison Note

| Baseline | Accuracy | Macro F1 |
| --- | --- | --- |
| Classical | 0.8868 | 0.8476 |
| Gemini Zero-Shot | 0.8679 | 0.8330 |
| Gemini Few-Shot | 0.8749 | 0.8414 |
| Sonnet Zero-Shot | 0.8268 | 0.7908 |
| Sonnet Few-Shot | 0.8408 | 0.8042 |
| MARBERT | 0.9669 | 0.9595 |

- Use this table as a benchmark-safe comparison against the current classical baseline and as task-context reference against the current Gemini full-dev and Sonnet full-dev historical runs already present in the repository.
