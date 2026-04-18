# Model Card: MARBERT v1 Encoder Result

## Model Summary

- Model family: Arabic transformer encoder
- Base checkpoint: `UBC-NLP/MARBERT`
- Task: four-way short-text dialect classification
- Output labels: `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`
- Final v1 encoder result: fine-tuned MARBERT evaluated on the cleaned benchmark-safe `998`-row processed `dev_core` split

This file describes the final encoder result used in the repository's comparison story. MARBERT is the strongest overall model in the current project state, but its clean-metric advantage does not eliminate OOD degradation or robustness sensitivity.

## Task Definition

- Input: one short Arabic-script tweet or sentence
- Output: exactly one label from `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`
- Scope exclusions: no Arabizi, no long-form text, no uncertain / mixed-dialect / MSA-heavy samples

## Training Data and Benchmark Notes

- Training split: `data/processed/train_core.csv`
- Training rows: `10000`
- Encoder evaluation split: cleaned benchmark-safe `data/processed/dev_core.csv`
- Encoder dev rows: `998`
- Label order: `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`

Important benchmark note:

- The repository contains two relevant dev views in its current reports.
- The corrected classical baseline rerun, MARBERT, the MARBERT stability pass, and the robustness evaluation use the cleaned benchmark-safe `998`-row processed dev view.
- Gemini and Sonnet full-dev comparisons remain historical prompt-only runs on the original `999`-row `dev_core` view.
- Repository-level comparison documents keep this split distinction explicit and do not treat it as hidden bookkeeping.

Supporting-source note:

- The repo also contains a curated `train_aug_candidates` pool from bundled NADI 2020 and NADI 2021 files.
- The reported MARBERT runs in this repository use `train_core` as the supervised training split; they do not merge `train_aug_candidates` into the final reported encoder result.

## Training Configuration

The current MARBERT runs use the checked-in YAML configs under `configs/`.

- Max length: `128`
- Train batch size: `32`
- Eval batch size: `64`
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Epochs requested: `5`
- Warmup ratio: `0.1`
- Optimizer: `adamw_torch`
- Scheduler: `linear_with_warmup`
- Loss: `cross_entropy`
- Class weights: `balanced`
- Early stopping metric: `eval_macro_f1`

The benchmark-safe single-run reference is seed `42`. Stability was checked across seeds `42`, `123`, and `7`.

## In-Domain Results

### Cleaned Benchmark-Safe Dev Result

| Run | Split | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| MARBERT seed `42` | cleaned dev (`998` rows) | `0.9669` | `0.9595` |
| MARBERT mean +/- std (`42/123/7`) | cleaned dev (`998` rows) | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` |

Per-class F1 for the seed `42` reference run:

| Label | F1 | Support |
| --- | ---: | ---: |
| `Saudi` | `0.9604` | `100` |
| `Egyptian` | `0.9340` | `99` |
| `Levantine` | `0.9735` | `399` |
| `Maghrebi` | `0.9702` | `400` |

Stability interpretation from the checked-in report:

- Macro F1 stability: `acceptable stability`
- Accuracy mean +/- std: `0.9679 +/- 0.0036`
- Macro F1 mean +/- std: `0.9613 +/- 0.0063`

## OOD Notes

### OOD Source Eligibility

Before any OOD claim, the standalone NADI 2020 dev and NADI 2021 DA dev splits were audited against the current benchmark-safe train/dev references.

| Split | In-Scope Rows | Exact Overlap vs train/dev | Near-Duplicate Risk vs train_core | Classification |
| --- | ---: | ---: | ---: | --- |
| NADI 2020 dev | `3267` | `0` | `0` | acceptable as OOD evaluation source |
| NADI 2021 DA dev | `3328` | `0` | `0` | acceptable as OOD evaluation source |

### OOD Evaluation

| Dataset | Accuracy | Macro F1 | Delta Accuracy vs In-Domain | Delta Macro F1 vs In-Domain |
| --- | ---: | ---: | ---: | ---: |
| NADI 2020 dev | `0.6122` | `0.5938` | `-0.3548` | `-0.3657` |
| NADI 2021 DA dev | `0.6656` | `0.6443` | `-0.3014` | `-0.3153` |

Interpretation:

- MARBERT remains stronger than the classical baseline on both audited OOD splits.
- The OOD drop is still material relative to the in-domain cleaned dev result.
- Levantine is the weakest OOD class in both checked-in MARBERT OOD summaries.

## Robustness Notes

Robustness was measured on the cleaned `998`-row dev split under four deterministic perturbation families.

| Perturbation Family | Clean Macro F1 | Perturbed Macro F1 | Macro F1 Drop |
| --- | ---: | ---: | ---: |
| `typo_noise` | `0.9595` | `0.8836` | `0.0760` |
| `elongation` | `0.9595` | `0.9513` | `0.0082` |
| `placeholder_noise` | `0.9595` | `0.9504` | `0.0092` |
| `punctuation_spacing` | `0.9595` | `0.9274` | `0.0321` |

Project-level robustness interpretation:

- MARBERT remains the higher-performing model family on clean text.
- The classical baseline is more robust under all four tested perturbation families when judged by relative degradation.
- The repository therefore treats clean accuracy and perturbation robustness as separate evaluation axes.

## Intended Use

- research and portfolio evaluation of Saudi-centered dialect classification
- benchmark-safe comparison of a classical baseline, prompt-based LLM baselines, and a fine-tuned Arabic encoder
- reproducible local experimentation on inspected local NADI sources

## Out-of-Scope Use

- speaker nationality, ethnicity, or identity inference
- legal, medical, hiring, moderation, or other high-stakes decisions
- Arabizi, speech, long documents, or mixed-dialect / MSA-heavy inputs
- public redistribution of raw or transformed tweet-text corpora

## Limitations and Risks

- The label space is asymmetric: `Saudi` and `Egyptian` are single-country labels, while `Levantine` and `Maghrebi` are grouped regional labels.
- The project operates on short text only, so some items remain weak-signal or genuinely ambiguous even for stronger models.
- Reported repository comparisons span two dev views: historical `999`-row prompt-only full-dev reports and cleaned `998`-row benchmark-safe classical/encoder reports.
- OOD degradation is substantial despite MARBERT's strong in-domain result.
- Robustness under deterministic perturbation is weaker than the classical baseline on the tested families.
- The source material is tweet-domain text, so domain transfer beyond this setting should not be assumed.

## Release and Packaging Note

This model card summarizes aggregate findings only. Raw source data, interim/processed text CSVs, checkpoints, and prediction files that expose text remain local-only artifacts and are not the intended public packaging targets for the final repository.
