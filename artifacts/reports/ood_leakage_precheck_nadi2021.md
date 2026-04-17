# OOD Leakage Pre-check: NADI 2021 DA dev

This audit checks whether the standalone split can be treated as an OOD evaluation source relative to the current benchmark-safe core data already present in the project.

## Scope

- Candidate path: `data/raw/nadi2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_dev_labeled.tsv`
- Rows before filtering: `5000`
- Rows after dropping out-of-scope labels: `3328`
- Out-of-scope rows dropped: `1672`

## Raw Label Inventory

| Raw label | Count |
| --- | ---: |
| `Algeria` | 430 |
| `Bahrain` | 52 |
| `Djibouti` | 27 |
| `Egypt` | 1041 |
| `Iraq` | 664 |
| `Jordan` | 104 |
| `Kuwait` | 105 |
| `Lebanon` | 157 |
| `Libya` | 314 |
| `Mauritania` | 53 |
| `Morocco` | 207 |
| `Oman` | 355 |
| `Palestine` | 104 |
| `Qatar` | 52 |
| `Saudi_Arabia` | 520 |
| `Somalia` | 49 |
| `Sudan` | 53 |
| `Syria` | 278 |
| `Tunisia` | 173 |
| `United_Arab_Emirates` | 157 |
| `Yemen` | 105 |

## Mapping Outcome

- Successfully mapped raw labels: `Algeria`, `Egypt`, `Jordan`, `Lebanon`, `Libya`, `Morocco`, `Palestine`, `Saudi_Arabia`, `Syria`, `Tunisia`
- Dropped out-of-scope raw labels: `Bahrain`, `Djibouti`, `Iraq`, `Kuwait`, `Mauritania`, `Oman`, `Qatar`, `Somalia`, `Sudan`, `United_Arab_Emirates`, `Yemen`
- Levantine source-label confirmation: all present labels mapped to `Levantine` = `True`; present labels: `Jordan` (104), `Lebanon` (157), `Palestine` (104), `Syria` (278)

## Benchmark Reference Note

- `train_core` interim rows: `10000`; processed rows: `10000`
- `dev_core` interim rows: `999`; processed rows: `998`
- `source_id` and `original_text` overlap checks use the union of the current interim and processed benchmark references.
- `processed_text` overlap checks use the current processed benchmark files as stored.
- Current processed `dev_core` is missing source_ids present in interim: `subtask1_dev_1521`

## Mapped Label Distribution vs Current dev_core

| Label | Candidate count | Candidate % | Current dev_core count | Current dev_core % | Count delta | Percentage-point delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Saudi` | 520 | 15.62% | 100 | 10.01% | 420 | +5.61 pp |
| `Egyptian` | 1041 | 31.28% | 100 | 10.01% | 941 | +21.27 pp |
| `Levantine` | 643 | 19.32% | 399 | 39.94% | 244 | -20.62 pp |
| `Maghrebi` | 1124 | 33.77% | 400 | 40.04% | 724 | -6.27 pp |

## Exact Leakage Checks

| Check | Candidate rows with exact overlap |
| --- | ---: |
| train_core `source_id` | `0` (0.00%) |
| train_core `original_text` | `0` (0.00%) |
| train_core `processed_text` | `0` (0.00%) |
| dev_core `source_id` | `0` (0.00%) |
| dev_core `original_text` | `0` (0.00%) |
| dev_core `processed_text` | `0` (0.00%) |

## Duplicate and Near-Duplicate Diagnostics

- Duplicate `processed_text` groups within the candidate split: `2`
- Additional rows beyond first occurrence inside duplicate groups: `2`
- Train-core near-duplicate rows at token Jaccard > `0.9`: `0` (0.00%)

### Representative Duplicate Groups

- `DEV_1949`, `DEV_3807`
- `DEV_25`, `DEV_4179`

## Decision

- Combined exact-overlap rows across all six checks: `0` (0.00%)
- Classification: `acceptable as OOD evaluation source`
- Rationale: No candidate rows hit any exact overlap check against the current benchmark-safe train_core/dev_core references.
