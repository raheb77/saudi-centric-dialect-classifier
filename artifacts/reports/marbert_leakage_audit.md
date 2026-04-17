# MARBERT Leakage Audit

This audit checks exact train/dev overlap on the processed benchmark-safe core split used by the MARBERT encoder pipeline.

## Scope

- Train rows: `10000`
- Dev rows: `998`
- Hard checks 1-3 use exact string equality on `source_id`, `original_text`, and `processed_text`.
- Soft checks 4-5 report duplicate `processed_text` groups within each split.
- Soft check 6 reports dev rows that have at least one train near-duplicate by token Jaccard similarity greater than `0.9`.

## Hard Checks

| Check | Count | Status |
| --- | --- | --- |
| 1. `source_id` overlap train vs dev | `0` | pass |
| 2. exact `original_text` overlap train vs dev | `0` | pass |
| 3. exact `processed_text` overlap train vs dev | `0` | pass |

## Soft Checks

| Check | Count | Notes |
| --- | --- | --- |
| 4. duplicate `processed_text` groups within train | `16` | additional rows beyond first: `17` |
| 5. duplicate `processed_text` groups within dev | `0` | additional rows beyond first: `0` |
| 6. dev rows with a train near-duplicate by token Jaccard > `0.9` | `1` | threshold: `0.9` |

### Representative Duplicate Groups In train

- `subtask1_train_5120`, `subtask1_train_11764`, `subtask1_train_16924`
- `subtask1_train_10727`, `subtask1_train_12439`
- `subtask1_train_11510`, `subtask1_train_15504`
- `subtask1_train_12428`, `subtask1_train_17155`
- `subtask1_train_12497`, `subtask1_train_15871`

### Representative Near-Duplicate Pairs

| Train source_id | Dev source_id | Token Jaccard |
| --- | --- | --- |
| `subtask1_train_9967` | `subtask1_dev_856` | `0.9200` |

## Decision

- Hard checks 1-3 are stop conditions for benchmark-safe reruns.
- Soft checks 4-6 are diagnostic flags only and do not block reruns by themselves.
- Overall audit status: `pass`
