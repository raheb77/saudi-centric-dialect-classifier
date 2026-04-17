# MARBERT Overlap Investigation

## Scope

- Train source_id: `subtask1_train_15327`
- Dev source_id: `subtask1_dev_1521`
- Files inspected:
  - `data/interim/train_core.csv`
  - `data/interim/dev_core.csv`
  - `data/processed/train_core.csv`
  - `data/processed/dev_core.csv`

## Finding

This case is a **preprocessing-induced collision**, not a true raw duplicate.

## Evidence

- The two rows have different `source_id` values and come from different benchmark splits.
- In the interim data, the underlying text fields are **not identical**.
- The dev-side interim row includes an extra trailing URL placeholder that is absent from the train-side interim row.
- After preprocessing, URL removal and placeholder normalization collapse both rows onto the same exact `processed_text`.
- The labels are consistent across the pair: both rows are `Egyptian`.

## Classification Decision

- Rule applied: if interim shows different source text but processed shows identical `processed_text`, classify as a preprocessing-induced collision.
- Result: **preprocessing-induced collision**

## Benchmark-Safe Resolution

- Clean the evaluation side only.
- Remove `subtask1_dev_1521` from `data/processed/dev_core.csv`.
- Keep the train-side row `subtask1_train_15327` as provenance only in this task.
- Do not modify raw files or interim files.

## Rationale

The encoder consumes `processed_text`, so this collision is benchmark-relevant even though the raw/interim strings are not exact duplicates. Cleaning the dev-side row is the conservative fix because it restores train/dev separation at the actual model-input level without mutating raw data or changing the train split.
