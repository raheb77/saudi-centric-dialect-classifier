# Planned Model Card

## Status
No model has been implemented, trained, or evaluated yet. This file defines the intended model-comparison setup for the repository.

## Planned Comparison
The project will eventually compare three model families on the same four-way task:

1. TF-IDF baseline
2. One LLM baseline
3. One fine-tuned Arabic encoder

## Task
- Input: one short Arabic-script tweet or sentence
- Output labels: `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`
- Task type: single-label classification
- Scope exclusions: no Arabizi, no long-form text, no mixed-dialect or MSA-heavy examples

## Data Plan
- Primary benchmark source: local NADI 2023 Subtask 1 train/dev files
- Canonical benchmark-aligned supporting sources for v1: bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` inside the NADI 2023 Subtask 1 package
- Standalone local NADI 2020 and NADI 2021 DA releases: provenance, inspection, and possible auxiliary evaluation only; not automatically merged into the initial v1 training pool
- Reference / future OOD source: local `MADAR-2018.tsv`, not part of the initial v1 training mixture

The final modeling dataset will be a filtered and mapped subset of the primary benchmark source plus its canonical benchmark-aligned supporting files. The standalone NADI 2020 and NADI 2021 releases remain documented local assets rather than automatic members of the initial v1 training pool. No processed dataset has been created yet.

## Intended Evaluation
- Primary metric: macro F1, matching the official NADI dialect-ID evaluation emphasis
- Secondary metrics: accuracy, precision, recall, and error analysis by v1 label
- Primary benchmark split anchor: NADI 2023 Subtask 1 train/dev

## Intended Use
- Research and portfolio evaluation of Saudi-centered Arabic dialect classification
- Comparison of simple and stronger baselines under one consistent label set
- Reproducible local experimentation using only the inspected raw sources already present in the repo

## Out-of-Scope Use
- Production moderation or identity inference
- Any task requiring speaker nationality or ethnicity prediction
- Classification of Arabizi, speech, or long documents
- Use of this future model family comparison as a legal or high-stakes decision system

## Risks and Limitations
- The v1 labels are project groupings layered on top of country- and city-labeled raw sources.
- Saudi is kept separate from Gulf on purpose, which improves project focus but narrows the label space.
- Twitter-domain data and MADAR translated sentences represent different domains.
- Filtering out mixed and MSA-heavy samples may improve label purity while reducing coverage.

## Fields to Fill After Implementation
Populate the sections below only after model code exists:

- preprocessing details
- training configuration
- exact train/dev/test splits
- hyperparameters
- results table
- confusion patterns
- failure cases
- bias and robustness analysis
