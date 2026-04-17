# Final Cleanup Audit

## Scope

This audit records the Phase 11 Part 3 repository cleanup against the packaging policy documented in:

- `README.md`
- `MODEL_CARD.md`
- `DATASET_CARD.md`
- `artifacts/reports/final_packaging_audit.md`

The cleanup is Git-surface cleanup only. Local copies of data and generated artifacts remain on disk unless otherwise noted.

## Packaging Policy Applied

Keep tracked:

- source code under `src/`
- tests under `tests/`
- configs under `configs/`
- top-level project docs and manifests
- packaging-safe aggregate reports and audits under `artifacts/reports/`

Exclude from Git tracking:

- local-only text-bearing data artifacts
- checkpoints and model cache directories
- train logs
- prediction CSVs containing tweet text
- duplicated validation copies
- intermediate `*_before_dedup.*` report variants

## Cleanup Actions Taken

### `.gitignore` Simplified

The ignore rules were rewritten to make the repository boundary explicit:

- local source and derived text artifacts
- local model artifacts and transient run outputs
- duplicated validation and `before_dedup` report variants
- Python / notebook cache files
- local environment and OS clutter

### Files Untracked From Git

The following tracked files were removed from Git tracking while keeping the local copies on disk:

- `data/interim/train_core.csv`
- `data/interim/dev_core.csv`
- `data/interim/train_aug_candidates.csv`
- `artifacts/reports/marbert_seed_123_train.log`
- `artifacts/reports/marbert_seed_42_train.log`
- `artifacts/reports/marbert_seed_7_train.log`
- `artifacts/reports/marbert_seed_42_classification_report_before_dedup.json`
- `artifacts/reports/marbert_seed_42_confusion_matrix_before_dedup.csv`
- `artifacts/reports/marbert_seed_42_error_analysis_before_dedup.md`
- `artifacts/reports/marbert_seed_42_metrics_before_dedup.json`
- `artifacts/reports/marbert_seed_42_summary_before_dedup.md`
- `artifacts/reports/validation_latest/data_validation_files.csv`
- `artifacts/reports/validation_latest/data_validation_report.json`
- `artifacts/reports/validation_latest/data_validation_summary.md`

## Canonical Tracked Surface After Cleanup

Tracked report classes retained:

- validation reports: canonical top-level copies only
- interim curation report
- classical baseline aggregate reports
- Gemini and Sonnet aggregate reports
- MARBERT aggregate reports for the final benchmark-safe runs
- MARBERT leakage audit and stability summary
- OOD leakage pre-check reports
- OOD evaluation summary and per-dataset aggregate summaries
- robustness summary
- final comparison and packaging audits

Local-only artifacts still present but excluded from Git:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `artifacts/checkpoints/`
- `artifacts/hf_cache/`
- prediction CSVs under `artifacts/reports/`
- train logs and `before_dedup` report variants

## Consistency Check

The cleanup now matches the documented packaging boundary:

- README and cards describe data-bearing files as local-only
- top-level docs remain tracked
- aggregate report summaries remain tracked
- duplicate and intermediate report variants are no longer part of the intended public surface

## Final Status Summary

Current repository changes after cleanup consist of:

- documentation updates from Phase 11 Part 2
- `.gitignore` simplification
- Git untracking of local-only and intermediate artifacts
- packaging-safe audit reports

No raw files were modified. No model results were changed.

## Ready State

From a packaging-hygiene standpoint, the repository is ready for the final commit once the current diff is reviewed as a single documentation-and-cleanup change set.
