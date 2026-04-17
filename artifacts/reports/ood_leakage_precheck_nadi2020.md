# OOD Leakage Pre-check: NADI 2020 dev

This audit checks whether the standalone split can be treated as an OOD evaluation source relative to the current benchmark-safe core data already present in the project.

## Scope

- Candidate path: `data/raw/nadi2020/NADI_release/dev_labeled.tsv`
- Rows before filtering: `4957`
- Rows after dropping out-of-scope labels: `3267`
- Out-of-scope rows dropped: `1690`

## Raw Label Inventory

| Raw label | Count |
| --- | ---: |
| `Algeria` | 359 |
| `Bahrain` | 8 |
| `Djibouti` | 10 |
| `Egypt` | 1070 |
| `Iraq` | 636 |
| `Jordan` | 104 |
| `Kuwait` | 70 |
| `Lebanon` | 110 |
| `Libya` | 265 |
| `Mauritania` | 40 |
| `Morocco` | 249 |
| `Oman` | 249 |
| `Palestine` | 102 |
| `Qatar` | 104 |
| `Saudi_Arabia` | 579 |
| `Somalia` | 51 |
| `Sudan` | 51 |
| `Syria` | 265 |
| `Tunisia` | 164 |
| `United_Arab_Emirates` | 265 |
| `Yemen` | 206 |

## Mapping Outcome

- Successfully mapped raw labels: `Algeria`, `Egypt`, `Jordan`, `Lebanon`, `Libya`, `Morocco`, `Palestine`, `Saudi_Arabia`, `Syria`, `Tunisia`
- Dropped out-of-scope raw labels: `Bahrain`, `Djibouti`, `Iraq`, `Kuwait`, `Mauritania`, `Oman`, `Qatar`, `Somalia`, `Sudan`, `United_Arab_Emirates`, `Yemen`
- Levantine source-label confirmation: all present labels mapped to `Levantine` = `True`; present labels: `Jordan` (104), `Lebanon` (110), `Palestine` (102), `Syria` (265)

## Benchmark Reference Note

- `train_core` interim rows: `10000`; processed rows: `10000`
- `dev_core` interim rows: `999`; processed rows: `998`
- `source_id` and `original_text` overlap checks use the union of the current interim and processed benchmark references.
- `processed_text` overlap checks use the current processed benchmark files as stored.
- Current processed `dev_core` is missing source_ids present in interim: `subtask1_dev_1521`

## Mapped Label Distribution vs Current dev_core

| Label | Candidate count | Candidate % | Current dev_core count | Current dev_core % | Count delta | Percentage-point delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Saudi` | 579 | 17.72% | 100 | 10.01% | 479 | +7.71 pp |
| `Egyptian` | 1070 | 32.75% | 100 | 10.01% | 970 | +22.74 pp |
| `Levantine` | 581 | 17.78% | 399 | 39.94% | 182 | -22.16 pp |
| `Maghrebi` | 1037 | 31.74% | 400 | 40.04% | 637 | -8.30 pp |

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

- Duplicate `processed_text` groups within the candidate split: `12`
- Additional rows beyond first occurrence inside duplicate groups: `14`
- Train-core near-duplicate rows at token Jaccard > `0.9`: `0` (0.00%)

### Representative Duplicate Groups

- `Dev_955`, `Dev_990`, `Dev_1792`, `Dev_4944`
- `Dev_1146`, `Dev_2907`
- `Dev_1523`, `Dev_4318`
- `Dev_177`, `Dev_4224`
- `Dev_1886`, `Dev_3311`
- `Dev_1924`, `Dev_2369`
- `Dev_2109`, `Dev_3256`
- `Dev_2964`, `Dev_4073`
- `Dev_3203`, `Dev_4343`
- `Dev_3566`, `Dev_3692`

## Decision

- Combined exact-overlap rows across all six checks: `0` (0.00%)
- Classification: `acceptable as OOD evaluation source`
- Rationale: No candidate rows hit any exact overlap check against the current benchmark-safe train_core/dev_core references.
