# Saudi-Centric Dialect Classifier

This repository contains a completed Arabic dialect classification artifact centered on Saudi identification. The v1 task is four-way short-text classification over `Saudi`, `Egyptian`, `Levantine`, and `Maghrebi`, using only local source files already present under `data/raw/`.

The project now includes benchmark-aware data validation, leakage-aware curation, a classical TF-IDF baseline, prompt-based Gemini and Sonnet baselines, a fine-tuned MARBERT encoder, OOD leakage pre-checks, OOD evaluation, robustness evaluation, and a final packaging audit.

## Task

- Input: one short Arabic-script tweet or sentence
- Output labels: `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`
- Scope exclusions: no Arabizi, no long-form text, no uncertain / mixed / MSA-heavy samples
- Data policy: use only local files already present under `data/raw/`

## Experimental Arc

1. Data validation:
   local raw-source schemas, row counts, and label inventories were validated and documented.
2. Leakage-aware curation:
   `train_core`, `dev_core`, and `train_aug_candidates` were generated from the benchmark anchor and canonical supporting sources.
3. Preprocessing:
   URLs, mentions, diacritics, Alef variants, tatweel, and elongation were normalized into parallel processed CSVs.
4. Classical baseline:
   TF-IDF + Logistic Regression established the lexical baseline, then was rerun on the cleaned benchmark-safe `998`-row `dev_core` view used by the final classical-vs-MARBERT comparison.
5. LLM baselines:
   Gemini Flash-Lite and Claude Sonnet were evaluated in zero-shot and few-shot settings on the original `999`-row `dev_core` view and are retained as historical prompt-only references.
6. Encoder baseline:
   `UBC-NLP/MARBERT` was fine-tuned and reported on the cleaned 998-row benchmark-safe dev split, then checked across three seeds.
7. OOD checks and OOD evaluation:
   standalone NADI 2020 and NADI 2021 DA dev splits passed exact-overlap pre-checks and were evaluated as OOD sources.
8. Robustness evaluation:
   deterministic perturbation families were applied to the cleaned dev split to compare sensitivity rather than only clean accuracy.

## Final In-Domain Results

| Model / Setting | Evaluation Split | Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Classical baseline | cleaned benchmark-safe dev (`998` rows) | `0.8868` | `0.8476` | TF-IDF + Logistic Regression |
| Gemini Flash-Lite zero-shot | original dev (`999` rows) | `0.8679` | `0.8330` | prompt-only |
| Gemini Flash-Lite few-shot | original dev (`999` rows) | `0.8749` | `0.8414` | best prompt-only LLM run |
| Claude Sonnet zero-shot | original dev (`999` rows) | `0.8268` | `0.7908` | prompt-only |
| Claude Sonnet few-shot | original dev (`999` rows) | `0.8408` | `0.8042` | prompt-only |
| MARBERT seed `42` | cleaned benchmark-safe dev (`998` rows) | `0.9669` | `0.9595` | encoder reference result |
| MARBERT mean +/- std (`42/123/7`) | cleaned benchmark-safe dev (`998` rows) | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` | acceptable stability |

Comparison note:

- The corrected classical baseline rerun and MARBERT use the cleaned benchmark-safe `998`-row processed dev view.
- Gemini and Sonnet remain historical prompt-only runs on the original `999`-row `dev_core` view.
- Treat the split distinction as part of the result, not as an omitted footnote.

## OOD and Robustness Summary

### OOD Leakage Pre-check

| Split | In-scope Rows | Exact Overlap vs train/dev | Near-Duplicate Risk vs train_core | Status |
| --- | ---: | ---: | ---: | --- |
| NADI 2020 dev | `3267` | `0` | `0` | acceptable as OOD evaluation source |
| NADI 2021 DA dev | `3328` | `0` | `0` | acceptable as OOD evaluation source |

### OOD Evaluation

| Dataset | Model | Accuracy | Macro F1 | Delta Accuracy vs In-Domain | Delta Macro F1 vs In-Domain |
| --- | --- | ---: | ---: | ---: | ---: |
| NADI 2020 dev | Classical | `0.4763` | `0.4467` | `-0.4105` | `-0.4009` |
| NADI 2020 dev | MARBERT | `0.6122` | `0.5938` | `-0.3548` | `-0.3657` |
| NADI 2021 DA dev | Classical | `0.5153` | `0.4937` | `-0.3714` | `-0.3539` |
| NADI 2021 DA dev | MARBERT | `0.6656` | `0.6443` | `-0.3014` | `-0.3153` |

Evidence in this repository supports an audited distribution-shift explanation more strongly than a leakage explanation: both OOD sources passed the exact-overlap pre-check, the same four-label mapping produces a much more Saudi/Egyptian-heavy and less Levantine-heavy label mix than the cleaned `dev_core`, and Levantine is the weakest MARBERT OOD class in both checked-in OOD summaries. Read together with the in-domain error analysis, topic/domain mismatch and weak-locality text are the most plausible interpretations, while orthographic or noise differences remain a secondary possibility rather than a proven primary cause. The current evidence therefore supports a careful claim of materially weaker domain transfer, not a stronger claim that one isolated factor fully explains the drop.

### Robustness on the Cleaned `998`-Row Dev Split

| Perturbation Family | Classical Macro F1 Drop | MARBERT Macro F1 Drop | More Robust Under This Test |
| --- | ---: | ---: | --- |
| `typo_noise` | `0.0502` | `0.0760` | Classical |
| `elongation` | `0.0004` | `0.0082` | Classical |
| `placeholder_noise` | `-0.0016` | `0.0092` | Classical |
| `punctuation_spacing` | `0.0201` | `0.0321` | Classical |

MARBERT is the strongest overall model on clean in-domain and OOD metrics, but the classical baseline is more robust across all four deterministic perturbation families tested in Phase 9 Part 3.

## What This Project Shows

- Benchmark-safe reporting changes interpretation: the corrected classical baseline and encoder results are tied to the cleaned `998`-row dev split, while Gemini/Sonnet remain historical `999`-row prompt-only references.
- A strong lexical baseline remains competitive: the corrected classical rerun still provides a credible non-neural benchmark next to MARBERT.
- MARBERT is the strongest overall model in the repository on clean in-domain and current OOD evaluations.
- Robustness and clean accuracy are different properties: MARBERT wins on clean metrics, while the classical baseline is less sensitive to the tested deterministic perturbations.

## Repository Layout

- [src](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/src): data, baseline, encoder, and audit code
- [tests](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/tests): unit tests for the core pipeline
- [configs](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/configs): YAML configs for classical, LLM, and MARBERT runs
- [artifacts/reports](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports): aggregate metrics, summaries, audits, and comparison reports

## Practical Setup and Reproducibility

Environment:

- Python `3.12`
- No locked environment file is currently checked in
- Core Python packages used by the codebase: `pyyaml`, `scikit-learn`, `numpy`, `torch`, `transformers`, `pytest`

Suggested setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install pyyaml scikit-learn numpy torch transformers pytest
```

Local data requirement:

- Keep the NADI source packages local under `data/raw/`
- The repo does not download datasets automatically

Core pipeline commands currently exposed in the repo:

```bash
python3 src/validate_data.py
python3 src/generate_interim_data.py
python3 src/preprocess_interim_data.py
python3 src/run_classical_baseline.py --config configs/baseline.yaml
python3 src/run_llm_baseline.py --config configs/llm_gemini_flash_lite.yaml
python3 src/run_llm_baseline.py --config configs/llm_sonnet_full_dev.yaml
python3 src/run_encoder_baseline.py --config configs/marbert_seed_42.yaml
python3 src/run_encoder_baseline.py --config configs/marbert_seed_123.yaml
python3 src/run_encoder_baseline.py --config configs/marbert_seed_7.yaml
python3 src/run_marbert_stability_summary.py --seeds 42 123 7
python3 src/run_marbert_leakage_audit.py
python3 src/run_ood_leakage_precheck.py
pytest -q
```

API-backed runs:

- Gemini baseline requires `GEMINI_API_KEY`
- Sonnet baseline requires `ANTHROPIC_API_KEY`

Report note:

- The repository includes checked-in aggregate reports for OOD evaluation and robustness.
- Not every post-hoc analysis phase is currently exposed through a dedicated standalone runner script in `src/`.

## Packaging Boundary

Safe aggregate artifacts to package in Git:

- code, configs, tests, and top-level docs
- aggregate reports under `artifacts/reports/` that do not expose raw tweet text
- leakage audits, OOD leakage pre-checks, metrics JSON files, confusion matrices, and summary markdown files

Local-only artifacts that should not be framed as public packaging targets:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- prediction CSVs that contain `original_text` or `processed_text`
- checkpoints, HF cache directories, and temporary run logs

This boundary follows the repository's own licensing notes: raw tweet text and redistributed transformed tweet-text artifacts should remain local-only unless a separate license review says otherwise.

## Key Documents

- [PROJECT_SCOPE.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/PROJECT_SCOPE.md)
- [DATASET_CARD.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/DATASET_CARD.md)
- [MODEL_CARD.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/MODEL_CARD.md)
- [ERROR_ANALYSIS.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/ERROR_ANALYSIS.md)
- [artifacts/reports/final_model_comparison.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports/final_model_comparison.md)
- [artifacts/reports/ood_evaluation_summary.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports/ood_evaluation_summary.md)
- [artifacts/reports/robustness_summary.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports/robustness_summary.md)
- [artifacts/reports/final_packaging_audit.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports/final_packaging_audit.md)
