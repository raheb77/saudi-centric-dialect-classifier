# AGENTS.md

## Project
Saudi-centered Arabic dialect classification project.

## Goal
Build a hiring-grade ML artifact that compares:
1. TF-IDF baseline
2. one LLM baseline
3. one fine-tuned Arabic encoder

## v1 labels
- Saudi
- Egyptian
- Levantine
- Maghrebi

## Scope
- Arabic script only
- no Arabizi in v1
- short-text / sentence-level classification
- uncertain, mixed, and MSA-heavy samples should be dropped

## Data policy
Use only local files already present under data/raw/.
Do not download new datasets automatically.
Do not modify raw data in place.

## Engineering rules
- Use Python 3.12
- Prefer src/ layout
- Keep code simple and typed
- No notebook-only core logic
- No giant framework abstractions
- No fake metrics or fabricated dataset statistics
- Keep outputs reproducible
- Write small, reviewable changes

## Modeling policy
- Start with data validation and documentation
- Then preprocessing
- Then TF-IDF baseline
- Then one LLM baseline
- Then one encoder model
- Do not add LoRA, quantization, or synthetic data in v1

## Deliverables
- DATASET_CARD.md
- ANNOTATION_GUIDELINES.md
- validation utilities
- preprocessing pipeline
- baseline experiment
- comparison-ready README
