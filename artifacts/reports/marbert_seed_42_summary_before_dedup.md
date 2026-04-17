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
- Device: `cpu`
- Parameter dtype: `torch.float32`
- Best checkpoint: `artifacts/checkpoints/marbert_base`
- Best epoch: `5`
- Train rows: `10000`
- Dev rows: `999`

## Dev Metrics

- Accuracy: `0.9720`
- Macro F1: `0.9658`
- Eval loss: `0.2017`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Saudi | 0.9608 | 0.9800 | 0.9703 | 100 |
| Egyptian | 0.9314 | 0.9500 | 0.9406 | 100 |
| Levantine | 0.9774 | 0.9749 | 0.9762 | 399 |
| Maghrebi | 0.9798 | 0.9725 | 0.9762 | 400 |

## Training History

| Epoch | Train Loss | Eval Loss | Eval Accuracy | Eval Macro F1 |
| --- | --- | --- | --- | --- |
| 1 | 0.4687 | 0.1706 | 0.9620 | 0.9548 |
| 2 | 0.1303 | 0.1960 | 0.9650 | 0.9563 |
| 3 | 0.0547 | 0.1603 | 0.9680 | 0.9626 |
| 4 | 0.0229 | 0.2031 | 0.9670 | 0.9592 |
| 5 | 0.0102 | 0.2017 | 0.9720 | 0.9658 |

## Comparison Note

| Baseline | Accuracy | Macro F1 |
| --- | --- | --- |
| Classical | 0.8869 | 0.8483 |
| Gemini Zero-Shot | 0.8679 | 0.8330 |
| Gemini Few-Shot | 0.8749 | 0.8414 |
| Sonnet Zero-Shot | 0.8268 | 0.7908 |
| Sonnet Few-Shot | 0.8408 | 0.8042 |
| MARBERT | 0.9720 | 0.9658 |

- Use this table as a benchmark-safe comparison against the current classical, Gemini full-dev, and Sonnet full-dev runs already present in the repository.
