# Dataset Card

## Summary
This project does not introduce a new raw corpus. It documents a Saudi-centered four-label classification task that will later be curated from local files already present under `data/raw/`.

The source hierarchy is:

- Primary benchmark source: NADI 2023 Subtask 1
- Canonical benchmark-aligned supporting sources for v1: bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` inside the NADI 2023 Subtask 1 package
- Standalone local NADI 2020 and NADI 2021 DA releases: provenance, inspection, and possible auxiliary evaluation only
- Reference / future OOD source: MADAR-2018

All counts below come from the inspected local files or the bundled local release notes. No final curated v1 dataset size is claimed yet, because filtering and label mapping have not been implemented.

## Local Raw Sources
Row counts below refer to non-header rows in the local copies.

### Primary Benchmark Source
- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2023_Subtask1_TRAIN.tsv`
  - 18,000 rows
  - Columns: `#1_id`, `#2_content`, `#3_label`
  - 18 observed country labels
- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2023_Subtask1_DEV.tsv`
  - 1,800 rows
  - Same schema as train
  - Same 18 observed country labels

Observed country labels in the local NADI 2023 Subtask 1 train/dev files:
`Algeria`, `Bahrain`, `Egypt`, `Iraq`, `Jordan`, `Kuwait`, `Lebanon`, `Libya`, `Morocco`, `Oman`, `Palestine`, `Qatar`, `Saudi_Arabia`, `Sudan`, `Syria`, `Tunisia`, `UAE`, `Yemen`

### Canonical Benchmark-Aligned Supporting Sources
- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2020-TWT.tsv`
  - 20,370 rows
  - Columns: `#1 tweet_ID`, `#2 tweet_content`, `#3 country_label`, `#4 province_label`
  - 18 observed country labels aligned with the NADI 2023 Subtask 1 label space
- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/NADI2021-TWT.tsv`
  - 20,398 rows
  - Columns: `#1_tweetid`, `#2_tweet`, `#3_country_label`, `#4_province_label`
  - 18 observed country labels aligned with the NADI 2023 Subtask 1 label space

These two bundled files are the canonical supporting sources for v1 because they are already packaged with NADI 2023 Subtask 1 and align to its 18-country benchmark label space.

### Standalone Provenance and Auxiliary Evaluation Releases
- `data/raw/nadi2020/NADI_release/train_labeled.tsv`
  - 21,000 rows
  - Country and province labels
- `data/raw/nadi2020/NADI_release/dev_labeled.tsv`
  - 4,957 rows
  - Country and province labels
- `data/raw/nadi2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_train_labeled.tsv`
  - 21,000 rows
  - Country and province labels
- `data/raw/nadi2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_dev_labeled.tsv`
  - 5,000 rows
  - Country and province labels

The standalone NADI 2020 and NADI 2021 DA releases cover 21 country labels in the inspected local files. They include countries not present in the NADI 2023 Subtask 1 benchmark label space, such as `Djibouti`, `Mauritania`, and `Somalia`. They are documented here for provenance, inspection, and possible auxiliary evaluation only, and are not automatically merged into the initial v1 training pool to avoid accidental duplication with the benchmark-aligned copies bundled inside NADI 2023.

### Reference / Future OOD Source
- `data/raw/nadi2023/NADI2023_Release_Train/Subtask1/MADAR-2018.tsv`
  - 40,000 rows
  - Columns: `sentID.BTEC`, `split`, `lang`, `sent`, `city`, `country`
  - 15 observed countries with city metadata
  - Not part of the initial v1 training mixture

Observed countries in the local MADAR file:
`Algeria`, `Egypt`, `Iraq`, `Jordan`, `Lebanon`, `Libya`, `Morocco`, `Oman`, `Palestine`, `Qatar`, `Saudi_Arabia`, `Sudan`, `Syria`, `Tunisia`, `Yemen`

Observed Saudi city labels in the local MADAR file:
`Jeddah`, `Riyadh`

## Observed Schema Differences Across Sources
- NADI 2023 Subtask 1 uses a simpler three-column schema and country labels only.
- NADI 2020 and NADI 2021 DA include both country and province labels.
- MADAR-2018 uses city and country metadata and is sentence-based rather than Twitter-native.
- Raw country naming is not fully normalized across releases. For example, NADI 2023 uses `UAE`, while standalone NADI 2020 and NADI 2021 DA use `United_Arab_Emirates`.

## Intended V1 Label Derivation
The final v1 labels will be derived later by mapping raw provenance labels into the project label set:

| v1 label | Raw labels mapped into v1 |
| --- | --- |
| `Saudi` | `Saudi_Arabia` |
| `Egyptian` | `Egypt` |
| `Levantine` | `Jordan`, `Lebanon`, `Palestine`, `Syria` |
| `Maghrebi` | `Algeria`, `Libya`, `Morocco`, `Tunisia` |

Examples from other countries remain available in the raw packages but are out of scope for the v1 four-way task and should be dropped during curation.

## Selection and Exclusion Policy
Keep only examples that satisfy all of the following:

- Arabic script
- Short-text / sentence-level length
- Dialectal Arabic with enough signal for a confident four-way label
- A raw source label that maps into one of the four v1 labels

Drop examples that are:

- Arabizi or non-Arabic-script dominant
- MSA-heavy
- Mixed dialect
- Unclear or weakly informative
- Outside the v1 raw-label mapping

## Files Present but Not Used as V1 Text Data
The local raw directory also contains material that should be documented but not treated as v1 training text:

- unlabeled tweet-ID files
- unlabeled test inputs
- scorer scripts
- release readmes and license files
- MT task files from NADI 2023 Subtasks 2 and 3
- example gold and submission files
- package artifacts such as `.DS_Store` and `.ipynb_checkpoints`

## Known Limitations
- The benchmark anchor is Twitter-domain text, while MADAR is translated travel-domain text. That domain mismatch is why MADAR is documented as a future OOD source rather than a primary benchmark or part of the initial v1 training mixture.
- The four v1 labels are grouped project labels layered on top of country- or city-level raw sources.
- No final train/dev/test curation logic has been executed yet, so this card defines source policy rather than a finished processed dataset.
