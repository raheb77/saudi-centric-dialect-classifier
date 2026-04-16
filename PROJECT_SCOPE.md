# Project Scope

## Objective
Build a hiring-grade Arabic dialect classification artifact centered on Saudi identification while comparing three model families later:

- a TF-IDF baseline
- one LLM baseline
- one fine-tuned Arabic encoder

This repository is currently in the documentation and data-definition phase. No model code, preprocessing code, or training code is implemented yet.

## Target Applications
- Saudi-market customer-support message routing and intent benchmarking
- Saudi-focused evaluation of Arabic assistants, chatbots, and search UX
- Dialect-aware benchmarking for products expected to serve users in Saudi Arabia

## V1 Task Definition
- Task type: supervised short-text dialect classification
- Input: one short Arabic-script tweet or sentence
- Output: exactly one label from `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`
- Unit of classification: sentence-level / short-text only
- Raw-data constraint: use only local files already present under `data/raw/`

The v1 label set is project-defined. The local raw sources mostly provide country or city labels, so the four final labels will be created later by filtering and mapping those local labels into the v1 taxonomy.

## V1 Label Mapping
| v1 label | Raw source labels kept in v1 |
| --- | --- |
| `Saudi` | `Saudi_Arabia` |
| `Egyptian` | `Egypt` |
| `Levantine` | `Jordan`, `Lebanon`, `Palestine`, `Syria` |
| `Maghrebi` | `Algeria`, `Libya`, `Morocco`, `Tunisia` |

All other raw source labels are out of scope for v1 and should be dropped rather than forced into one of the four labels. This includes non-Saudi Gulf-country data and other country labels such as `Bahrain`, `Kuwait`, `Oman`, `Qatar`, `UAE` / `United_Arab_Emirates`, `Yemen`, `Iraq`, `Sudan`, `Djibouti`, `Somalia`, and `Mauritania`.

## Why Saudi Is Separate from Gulf
The local benchmark sources are country-level, not coarse regional, and they include `Saudi_Arabia` as a distinct label. They also include other Gulf-country labels such as `Bahrain`, `Kuwait`, `Oman`, `Qatar`, and `UAE` / `United_Arab_Emirates`. Using a single `Gulf` label would collapse multiple distinct country varieties that the source datasets already keep separate.

This project is Saudi-centered, so v1 keeps `Saudi` as its own class and does not replace it with a broader `Gulf` bucket. Non-Saudi Gulf examples are dropped in v1 rather than merged into `Saudi`, which preserves a clearer decision boundary and keeps the task aligned with the country-level structure of NADI 2023 Subtask 1.

## In Scope
- Arabic script only
- Short-text / sentence-level classification
- Dialectal Arabic with enough signal to support a four-way label
- Documentation, data description, annotation rules, and data inventory
- Later model comparison against the four v1 labels

## Out of Scope for V1
- Arabizi
- Speech or audio dialect identification
- Long documents or paragraph-level classification
- Automatic downloading of new datasets
- In-place modification of `data/raw/`
- Uncertain, mixed-dialect, or MSA-heavy samples
- Any model implementation in this documentation phase
- NADI 2023 Subtasks 2 and 3, which are MT tasks rather than dialect classification

## Compute Requirements
- Documentation, validation, and the TF-IDF baseline should fit on a standard CPU laptop or workstation.
- The planned encoder baseline should target a single modern GPU rather than a multi-GPU training setup.
- Large-scale distributed training is not a v1 requirement.

## Source Hierarchy
- Primary benchmark source: NADI 2023 Subtask 1 local train/dev files
- Canonical benchmark-aligned supporting sources for v1: `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` bundled inside the NADI 2023 Subtask 1 package
- Standalone local NADI 2020 and NADI 2021 DA releases: documented for provenance, inspection, and possible auxiliary evaluation only; not automatically merged into the initial v1 training pool to avoid accidental duplication
- Reference / future OOD source: local `MADAR-2018.tsv` bundled inside the NADI 2023 Subtask 1 package; not part of the initial v1 training mixture

The bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` files are the canonical supporting sources for v1 because they are already aligned to the NADI 2023 Subtask 1 benchmark label space. The standalone NADI 2020 and NADI 2021 local releases are kept documented in the repo for provenance, inspection, and possible auxiliary evaluation, but they are not automatically merged into the initial v1 training pool.

`MADAR-2018.tsv` is intentionally not the primary v1 benchmark source because it is a translated sentence resource with city and country metadata, not the main Twitter benchmark that anchors the project. It is documented as a later out-of-domain reference set and is not part of the initial v1 training mixture.

## Benchmark Safety Policy
- `NADI2023_Subtask1_TRAIN.tsv` and `NADI2023_Subtask1_DEV.tsv` are the benchmark anchor.
- `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` are the only canonical supporting sources for v1 augmentation planning.
- Standalone NADI 2020 and NADI 2021 DA releases remain provenance / auxiliary evaluation assets and are excluded from canonical leakage accounting.
- Exact text overlap between NADI 2023 train and dev must be removed from dev before benchmark-style evaluation.
- Same exact text with conflicting labels across the canonical supporting sources must be dropped from augmentation candidates.
- For leakage accounting, `UAE` and `United_Arab_Emirates` are treated as the same canonical raw label.
