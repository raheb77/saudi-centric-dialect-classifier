# Interim Curation Report

This report describes leakage-aware interim dataset generation from the local NADI 2023 Subtask 1 benchmark anchor and bundled supporting sources only.

## train_core

- Output path: `data/interim/train_core.csv`
- Rows kept: `10000`

## train_core Kept by Source

| Key | Count |
| --- | ---: |
| `NADI2023_Subtask1_TRAIN` | 10000 |

## train_core Kept by Macro Label

| Key | Count |
| --- | ---: |
| `Levantine` | 4000 |
| `Maghrebi` | 4000 |
| `Egyptian` | 1000 |
| `Saudi` | 1000 |

## train_core Dropped by Reason

| Key | Count |
| --- | ---: |
| `out_of_scope_country` | 8000 |

## dev_core

- Output path: `data/interim/dev_core.csv`
- Rows kept: `999`

## dev_core Kept by Source

| Key | Count |
| --- | ---: |
| `NADI2023_Subtask1_DEV` | 999 |

## dev_core Kept by Macro Label

| Key | Count |
| --- | ---: |
| `Maghrebi` | 400 |
| `Levantine` | 399 |
| `Egyptian` | 100 |
| `Saudi` | 100 |

## dev_core Dropped by Reason

| Key | Count |
| --- | ---: |
| `out_of_scope_country` | 795 |
| `benchmark_exact_overlap_with_train` | 6 |

## train_aug_candidates

- Output path: `data/interim/train_aug_candidates.csv`
- Rows kept: `27629`

## train_aug_candidates Kept by Source

| Key | Count |
| --- | ---: |
| `NADI2021_TWT` | 14016 |
| `NADI2020_TWT` | 13613 |

## train_aug_candidates Kept by Macro Label

| Key | Count |
| --- | ---: |
| `Maghrebi` | 9153 |
| `Egyptian` | 8723 |
| `Levantine` | 5326 |
| `Saudi` | 4427 |

## train_aug_candidates Dropped by Reason

| Key | Count |
| --- | ---: |
| `out_of_scope_country` | 12972 |
| `conflicting_supporting_label` | 166 |
| `overlap_with_train_core` | 1 |

## Overall Kept by Source

| Key | Count |
| --- | ---: |
| `NADI2021_TWT` | 14016 |
| `NADI2020_TWT` | 13613 |
| `NADI2023_Subtask1_TRAIN` | 10000 |
| `NADI2023_Subtask1_DEV` | 999 |

## Overall Kept by Macro Label

| Key | Count |
| --- | ---: |
| `Maghrebi` | 13553 |
| `Egyptian` | 9823 |
| `Levantine` | 9725 |
| `Saudi` | 5527 |

## Overall Dropped by Reason

| Key | Count |
| --- | ---: |
| `out_of_scope_country` | 21767 |
| `conflicting_supporting_label` | 166 |
| `benchmark_exact_overlap_with_train` | 6 |
| `overlap_with_train_core` | 1 |

## Overlap Removals

- Benchmark train/dev exact overlap text hashes: `6`
- Dev rows removed for benchmark overlap: `6`
- Aug candidate text hashes already present in train_core: `1`
- Aug candidate rows removed for train_core overlap: `1`
- Aug candidate text hashes already present in dev_core: `0`
- Aug candidate rows removed for dev_core overlap: `0`

### Benchmark Overlap Examples

| Text hash | Source dataset | Source ID | Row |
| --- | --- | --- | ---: |
| `928c479414b43312f954565b73429b78eb1e84db` | `NADI2023_Subtask1_DEV` | `subtask1_dev_162` | 163 |
| `b13dc3def48541a44315add213bd8e34ba35a728` | `NADI2023_Subtask1_DEV` | `subtask1_dev_229` | 230 |
| `7269cf7aa18a4b5ac0e4cd65fe6e300d9926eed0` | `NADI2023_Subtask1_DEV` | `subtask1_dev_635` | 636 |
| `122874ea182df90ee81879b0b3732d6ba5eb3dc0` | `NADI2023_Subtask1_DEV` | `subtask1_dev_1688` | 1689 |
| `5bbbe3a3499669d885eef52da6d9e06932e8420f` | `NADI2023_Subtask1_DEV` | `subtask1_dev_1699` | 1700 |
| `1f25e02318f388515a4c423b22a3ebcd4934bdeb` | `NADI2023_Subtask1_DEV` | `subtask1_dev_1704` | 1705 |

### Aug-vs-Train Overlap Examples

| Text hash | Source dataset | Source ID | Row | Label |
| --- | --- | --- | ---: | --- |
| `e9b11bcfb5006715b2cd24a1dde53ad964201863` | `NADI2021_TWT` | `TRAIN_14641` | 14237 | `Egypt` |

### Aug-vs-Dev Overlap Examples

No aug-vs-dev overlap examples recorded.

## Conflict Removals

- Supporting conflict text hashes: `64`
- Supporting rows removed for conflict: `166`
- Conservative rule: any same exact text with conflicting labels anywhere in the canonical supporting pool is dropped from augmentation candidates.
- Label normalization used for conflict accounting:
  - `UAE` -> `UAE`
  - `United_Arab_Emirates` -> `UAE`

### Conflict Examples

| Text hash | Normalized raw labels | Occurrences |
| --- | --- | ---: |
| `04f9b951f0eb0b86c9e1c32e18d70b3e121de3ac` | `Egypt`, `Oman` | 2 |
| `0ea0974025c41c3d54179002bb64ca3cd2a133e8` | `Kuwait`, `Saudi_Arabia`, `UAE` | 3 |
| `150996d08436f11e3b23d5904022657abb5bccb0` | `Iraq`, `Saudi_Arabia` | 2 |
| `2b916b682d38b641f166f61c796a18f18b79935e` | `Egypt`, `Syria` | 2 |
| `3683066c108466ae683b0c52933d38ef58b6f96d` | `Iraq`, `Morocco`, `Qatar` | 4 |
| `58cd5e44cdd174459418160e1ddb046a61adb3d6` | `Algeria`, `Egypt` | 2 |
| `5972ad51ae211ab5feb3547ed285476bebc63ac6` | `Iraq`, `Saudi_Arabia`, `Yemen` | 3 |
| `67493e25cc7cbd26f2ec6e49d54cb311059511ec` | `Algeria`, `Iraq`, `Yemen` | 4 |
| `6d142f5a52c91fb03dda2bc4c850c35dfa3e2c7c` | `Libya`, `Morocco` | 4 |
| `754f1805ee7193788a707f1cc67056415c627d6d` | `Iraq`, `Libya` | 2 |
| `7f4e9bdf19c81172e7a2bdd4d96e2691bae8e0cc` | `Algeria`, `Palestine` | 2 |
| `85bdb9d3f435786bd767c8cf7a619f058499abdb` | `Lebanon`, `Syria` | 3 |
| `8bdc16ace7f760e2814601db0f9d8f0aafc69000` | `Tunisia`, `UAE` | 2 |
| `9cd193931d4af4f10633b376225a57823672e0b0` | `Morocco`, `Saudi_Arabia`, `UAE` | 3 |
| `abc6510a74986d773e73de335775e555daa65681` | `Algeria`, `Egypt`, `Libya`, `Morocco`, `Oman` | 6 |
| `b44ba7fa05179c68562be10a50c5658ed950ab37` | `Algeria`, `Morocco` | 5 |
| `b879a41ed72a13e69b51aee8c27d975986853a58` | `Iraq`, `Jordan` | 2 |
| `d400e3bd2b11c41a8fb835db199d5fd20ffc72d6` | `Libya`, `Morocco`, `Saudi_Arabia` | 3 |
| `e33239927d111dbdad9866cfa81a100c970e9caf` | `Egypt`, `Oman` | 2 |
| `ee5e51b217f08fecc24681c67a043a7b4bfcc3d0` | `Egypt`, `Jordan` | 2 |
