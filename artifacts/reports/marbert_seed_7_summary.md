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
- Seed: `7`
- Device: `cuda`
- Parameter dtype: `torch.float32`
- Best checkpoint: `artifacts/checkpoints/marbert_seed_7`
- Best epoch: `1`
- Train rows: `10000`
- Dev rows: `998`

## Dev Metrics

- Accuracy: `0.9649`
- Macro F1: `0.9561`
- Eval loss: `0.1783`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Saudi | 0.9314 | 0.9500 | 0.9406 | 100 |
| Egyptian | 0.9783 | 0.9091 | 0.9424 | 99 |
| Levantine | 0.9651 | 0.9699 | 0.9675 | 399 |
| Maghrebi | 0.9702 | 0.9775 | 0.9738 | 400 |

## Training History

| Epoch | Train Loss | Eval Loss | Eval Accuracy | Eval Macro F1 |
| --- | --- | --- | --- | --- |
| 1 | 0.4639 | 0.1783 | 0.9649 | 0.9561 |
| 2 | 0.1191 | 0.1600 | 0.9559 | 0.9474 |
| 3 | 0.0559 | 0.1745 | 0.9609 | 0.9539 |

## Comparison Note

| Baseline | Accuracy | Macro F1 |
| --- | --- | --- |
| Classical | 0.8868 | 0.8476 |
| Gemini Zero-Shot | 0.8679 | 0.8330 |
| Gemini Few-Shot | 0.8749 | 0.8414 |
| Sonnet Zero-Shot | 0.8268 | 0.7908 |
| Sonnet Few-Shot | 0.8408 | 0.8042 |
| MARBERT | 0.9649 | 0.9561 |

- Use this table as a benchmark-safe comparison against the current classical baseline and as task-context reference against the current Gemini full-dev and Sonnet full-dev historical runs already present in the repository.
