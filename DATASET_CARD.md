# Dataset Card

## Summary

This repository does not publish a new raw corpus. It documents and evaluates a Saudi-centered four-label dialect classification task built from local NADI files already present under `data/raw/`.

The current project state includes:

- validated local raw-source inventories
- leakage-aware interim curation
- processed benchmark files
- in-domain baseline and encoder evaluations
- OOD leakage pre-checks and OOD evaluation
- robustness reporting

The final reported models in this repository use the benchmark anchor training split `train_core`. The benchmark-aligned supporting pool `train_aug_candidates` is curated and documented, but it is not merged into the final reported baseline or MARBERT runs.

## V1 Label Mapping

| v1 label | Raw labels mapped into v1 |
| --- | --- |
| `Saudi` | `Saudi_Arabia` |
| `Egyptian` | `Egypt` |
| `Levantine` | `Jordan`, `Lebanon`, `Palestine`, `Syria` |
| `Maghrebi` | `Algeria`, `Libya`, `Morocco`, `Tunisia` |

All other raw source labels are out of scope for v1 and are dropped rather than forced into the four-label taxonomy.

## Source Hierarchy

- Primary benchmark source: NADI 2023 Subtask 1 train/dev
- Canonical benchmark-aligned supporting sources: bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` inside the NADI 2023 package
- Standalone local NADI 2020 and NADI 2021 DA releases: audited OOD sources and provenance assets
- Reference / future OOD source: `MADAR-2018.tsv`

## Completed Data Pipeline

The current repository state covers the full data pipeline needed for the checked-in reports:

1. Raw-file validation and schema inspection
2. Leakage-aware interim curation
3. Preprocessing into `original_text` and `processed_text`
4. Benchmark leakage audit for the encoder-safe dev split
5. OOD leakage pre-check for standalone NADI 2020 and NADI 2021 DA dev splits

## Benchmark-Core Splits

### Interim Benchmark Files

- `data/interim/train_core.csv`: `10000` rows from `NADI2023_Subtask1_TRAIN.tsv`
- `data/interim/dev_core.csv`: `999` rows from `NADI2023_Subtask1_DEV.tsv` after removing `6` exact benchmark train/dev overlaps
- `data/interim/train_aug_candidates.csv`: `27629` rows from bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` after dropping out-of-scope rows, benchmark overlaps, and same-text conflicting-label cases

### Processed Benchmark Files

- `data/processed/train_core.csv`: processed counterpart of `train_core`
- `data/processed/dev_core.csv`: current cleaned processed dev reference used by MARBERT and robustness reporting
- Cleaned processed dev size: `998`

Current reporting split note:

- The corrected classical baseline rerun, MARBERT, the MARBERT stability summary, and robustness evaluation use the cleaned benchmark-safe `998`-row processed dev view.
- Gemini and Sonnet full-dev reports remain historical prompt-only runs on the original `999`-row `dev_core` view.
- The final comparison documents keep this distinction explicit instead of collapsing the two views into one unlabeled number.

## OOD Sources Used

The repository audited and evaluated two standalone OOD candidate splits after applying the same four-label mapping and preprocessing policy used in-domain.

| OOD Source | Raw Rows | Kept Rows | Exact Overlap vs train/dev | Near-Duplicate Risk vs train_core | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| NADI 2020 dev | `4957` | `3267` | `0` | `0` | acceptable as OOD evaluation source |
| NADI 2021 DA dev | `5000` | `3328` | `0` | `0` | acceptable as OOD evaluation source |

Mapped label counts:

| OOD Source | Saudi | Egyptian | Levantine | Maghrebi |
| --- | ---: | ---: | ---: | ---: |
| NADI 2020 dev | `579` | `1070` | `581` | `1037` |
| NADI 2021 DA dev | `520` | `1041` | `643` | `1124` |

## Preprocessing

The preprocessing stage operates on interim CSVs and writes parallel outputs under `data/processed/` without modifying `data/raw/`.

- removes URLs
- normalizes mentions to `<USER>`
- preserves hashtag text while stripping `#`
- removes Arabic diacritics
- normalizes Alef variants and final `ى` to `ي`
- strips tatweel / kashida
- reduces repeated elongation to at most `2`
- preserves emojis and English tokens
- preserves the source placeholder `NUM`
- keeps both `original_text` and `processed_text` for traceability

## Local Raw Sources

Row counts below refer to non-header rows in the inspected local files.

### Primary Benchmark Source

- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2023_Subtask1_TRAIN.tsv`
  - `18000` rows
  - schema: `#1_id`, `#2_content`, `#3_label`
- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2023_Subtask1_DEV.tsv`
  - `1800` rows
  - same schema as train

Observed country labels in the local NADI 2023 Subtask 1 train/dev files:
`Algeria`, `Bahrain`, `Egypt`, `Iraq`, `Jordan`, `Kuwait`, `Lebanon`, `Libya`, `Morocco`, `Oman`, `Palestine`, `Qatar`, `Saudi_Arabia`, `Sudan`, `Syria`, `Tunisia`, `UAE`, `Yemen`

### Canonical Benchmark-Aligned Supporting Sources

- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2020-TWT.tsv`
  - `20370` rows
  - schema: `#1 tweet_ID`, `#2 tweet_content`, `#3 country_label`, `#4 province_label`
- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2021-TWT.tsv`
  - `20398` rows
  - schema: `#1_tweetid`, `#2_tweet`, `#3_country_label`, `#4_province_label`

These bundled files are the canonical supporting sources because they align to the NADI 2023 label space and are the only supporting files used in the interim-candidate curation step.

### Standalone Provenance and OOD Evaluation Releases

- `data/raw/nadi2020/NADI_release/dev_labeled.tsv`
  - `4957` rows
- `data/raw/nadi2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_dev_labeled.tsv`
  - `5000` rows

The standalone NADI 2020 and NADI 2021 DA releases contain additional country labels outside the NADI 2023 benchmark space, including `Djibouti`, `Mauritania`, and `Somalia`. In the current repository state they are used as audited OOD evaluation sources, not as automatic training data.

### Reference / Future OOD Source

- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/MADAR-2018.tsv`
  - `40000` rows
  - schema: `sentID.BTEC`, `split`, `lang`, `sent`, `city`, `country`
  - not part of the current reported training or OOD evaluation mix

## Packaging Boundary and Local-Only Artifacts

The repository contains local text-bearing artifacts that support experimentation but should not be treated as public packaging targets.

Remain local-only:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- prediction CSVs that contain `original_text` or `processed_text`
- checkpoints and HF cache directories

Reason:

- the local NADI license notes and release restrictions do not support republishing raw tweet text as a public packaged artifact
- interim and processed CSVs retain tweet text or transformed tweet text for local reproducibility only

Packaging-safe outputs are the aggregate artifacts:

- code, configs, tests, and top-level documentation
- metrics JSON files
- confusion matrices
- markdown summaries and audit reports

## Benchmark Safety Policy

- `NADI2023_Subtask1_TRAIN.tsv` and `NADI2023_Subtask1_DEV.tsv` are the benchmark anchor
- exact train/dev overlap is removed before benchmark-safe encoder reporting
- canonical supporting sources are filtered against benchmark overlap and same-text conflicting-label cases
- standalone NADI 2020 and NADI 2021 DA dev splits require leakage pre-checks before OOD use
- current OOD pre-checks passed with zero exact overlaps on `source_id`, `original_text`, and `processed_text`

## Known Limitations

- The four v1 labels are project groupings layered on top of country-level raw labels.
- The benchmark anchor is tweet-domain text, while MADAR is sentence-domain text.
- The repository currently contains both a `999`-row original dev reporting view and a cleaned `998`-row encoder-safe dev view, so report interpretation must keep the split label explicit.
- Local text-bearing artifacts are available for internal reproducibility but are not intended as public release assets.
