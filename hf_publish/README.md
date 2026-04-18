---
title: Saudi-Centric Dialect Classifier Demo
emoji: 🗣️
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
short_description: Arabic dialect summary, benchmark-safe, no data/weights
pinned: false
---

# Saudi-Centric Dialect Classifier: Safe Publishing Package

This release is a lightweight Hugging Face Space companion for the repository. It presents the project and benchmark-safe aggregate results without redistributing datasets, text-bearing artifacts, or model weights.

This package is summary-only. It does not run model inference and does not publish datasets, text-bearing artifacts, or model weights.

## What This Release Includes

- benchmark-safe aggregate metrics and narrative summaries
- a lightweight Gradio summary app in `app.py`
- minimal runtime dependencies in `requirements.txt`
- a short upload checklist in `PUBLISHING_PLAN.md`

## What This Release Excludes

- datasets or raw source packages
- raw tweet text
- interim or processed text files
- text-bearing prediction files
- checkpoints or model weights
- cache directories, train logs, and other large run artifacts

## Explicit Publishing Notice

Raw data, processed text data, text-bearing prediction files, and model weights are not published in this release. This Space is limited to benchmark-safe reporting and documentation-oriented summary content.

## Task Summary

- Task: four-way short-text Arabic dialect classification
- Labels: `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`
- Scope: Arabic script only, short text only, no Arabizi in v1
- Data policy: local-source workflow only in the main repository; no dataset redistribution in this package

## Key Benchmark-Safe Results

Important split note:

- The corrected classical baseline rerun and MARBERT were reported on the cleaned benchmark-safe `998`-row dev view.
- Gemini Flash-Lite and Claude Sonnet remain historical prompt-only runs on the original `999`-row dev view.
- That distinction is part of the published record and should stay explicit.

### In-Domain Comparison

| Model / Setting | Eval View | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Classical TF-IDF + Logistic Regression | cleaned benchmark-safe dev (`998` rows) | `0.8868` | `0.8476` |
| Gemini Flash-Lite few-shot | original dev (`999` rows) | `0.8749` | `0.8414` |
| Claude Sonnet few-shot | original dev (`999` rows) | `0.8408` | `0.8042` |
| MARBERT seed `42` | cleaned dev (`998` rows) | `0.9669` | `0.9595` |
| MARBERT mean +/- std over seeds `42/123/7` | cleaned dev (`998` rows) | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` |

### OOD Summary

Both standalone OOD sources passed the repository's leakage pre-check with `0` exact overlaps and `0` near-duplicate risk against the benchmark-safe train/dev references.

| Dataset | Model | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| NADI 2020 dev | Classical | `0.4763` | `0.4467` |
| NADI 2020 dev | MARBERT | `0.6122` | `0.5938` |
| NADI 2021 DA dev | Classical | `0.5153` | `0.4937` |
| NADI 2021 DA dev | MARBERT | `0.6656` | `0.6443` |

### Robustness Summary

- MARBERT is the strongest model on clean in-domain and current OOD metrics.
- The classical baseline was more robust on all `4/4` tested perturbation families: `typo_noise`, `elongation`, `placeholder_noise`, and `punctuation_spacing`.

## Safe Source-Repo References

These repository-relative references point to existing docs and aggregate reports in the source project. They are safe to cite, but they are not bundled into this release.

- Top-level docs: `README.md`, `PROJECT_SCOPE.md`, `DATASET_CARD.md`, `ANNOTATION_GUIDELINES.md`, `MODEL_CARD.md`, `ERROR_ANALYSIS.md`, `LICENSE_NOTES.md`
- Aggregate reports: `artifacts/reports/final_model_comparison.md`, `artifacts/reports/ood_evaluation_summary.md`, `artifacts/reports/robustness_summary.md`, `artifacts/reports/final_packaging_audit.md`

## Publishing Boundary

This release intentionally ships no datasets, no text-bearing artifacts, and no checkpoints. It is limited to benchmark-safe aggregate reporting and a static summary app.
