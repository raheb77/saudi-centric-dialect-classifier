# MARBERT Stability Summary

Multi-seed stability was not completed. The final leakage audit found `1` exact `processed_text` overlap between `train_core` and `dev_core`, so seeds `123` and `7` were intentionally not run.

## Available Seed

| Seed | Accuracy | Macro F1 | Saudi F1 | Egyptian F1 | Levantine F1 | Maghrebi F1 | Artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `42` | `0.9720` | `0.9658` | `0.9703` | `0.9406` | `0.9762` | `0.9762` | `marbert_seed_42_*` |

## Mean ± Std Across Seeds

| Metric | Mean ± Std |
| --- | --- |
| Accuracy | `not computed` |
| Macro F1 | `not computed` |
| Saudi F1 | `not computed` |
| Egyptian F1 | `not computed` |
| Levantine F1 | `not computed` |
| Maghrebi F1 | `not computed` |

## Why Aggregation Was Not Computed

- Run policy for this stage requires stopping before multi-seed if any exact overlap is found in `source_id`, `original_text`, or `processed_text`.
- The leakage audit passed on `source_id` and `original_text`, but failed on `processed_text`.
- With only one valid seed artifact available, a reported standard deviation would be misleading.

## Prepared but Not Run

- `configs/marbert_seed_123.yaml`
- `configs/marbert_seed_7.yaml`
- `artifacts/reports/marbert_seed_123_train.log`
- `artifacts/reports/marbert_seed_7_train.log`
