# Saudi-Centric Dialect Classifier

This repository defines a hiring-grade Arabic dialect classification project centered on Saudi identification. The v1 task is a four-way short-text classification problem over `Saudi`, `Egyptian`, `Levantine`, and `Maghrebi`.

Current state: documentation and data-definition only. No validation utilities, preprocessing pipeline, or model code have been implemented yet.

## V1 Scope
- Arabic script only
- No Arabizi in v1
- Short-text / sentence-level classification only
- Drop uncertain, mixed-dialect, and MSA-heavy samples
- Use only local files already present under `data/raw/`

Saudi is intentionally separate from Gulf in v1. The local benchmark sources are country-level and include `Saudi_Arabia` as a distinct label alongside other Gulf-country labels, so collapsing them into `Gulf` would weaken the Saudi-centered goal of the project.

## Target Applications
- Saudi-market customer-support message routing and intent benchmarking
- Saudi-focused evaluation of Arabic assistants, chatbots, and search UX
- Dialect-aware benchmarking for products intended for users in Saudi Arabia

## Source Hierarchy
- Primary benchmark source: NADI 2023 Subtask 1
- Canonical benchmark-aligned supporting sources for v1: bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` inside the NADI 2023 Subtask 1 package
- Standalone NADI 2020 and NADI 2021 DA local releases: provenance, inspection, and possible auxiliary evaluation only
- Reference / future OOD source: `MADAR-2018.tsv`, not part of the initial v1 training mixture

The repo contains only local raw-source documentation at this stage. NADI 2023 Subtask 1 is the main benchmark anchor. The bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` files are the canonical v1 supporting sources because they are benchmark-aligned with that package. The standalone NADI 2020 and NADI 2021 DA releases are documented for provenance, inspection, and possible auxiliary evaluation only, and are not automatically merged into the initial v1 training pool to avoid accidental duplication. `MADAR-2018.tsv` is documented as a future out-of-domain reference rather than part of the initial training mixture.

## Documentation
- `PROJECT_SCOPE.md`: task definition, label policy, and Saudi-vs-Gulf rationale
- `DATASET_CARD.md`: local source inventory for the benchmark-relevant corpora
- `ANNOTATION_GUIDELINES.md`: keep/drop rules and v1 label mapping
- `LICENSE_NOTES.md`: observed local license constraints and repo handling rules
- `DATA_MANIFEST.csv`: full inventory of every file currently present under `data/raw/`
- `MODEL_CARD.md`: planned model-comparison card with no results yet

## Planned Implementation Sequence
1. Validation utilities
2. Preprocessing pipeline
3. TF-IDF baseline
4. One LLM baseline
5. One fine-tuned Arabic encoder

## Not Yet Implemented
- no model code
- no preprocessing code
- no curated processed dataset
- no benchmark results
- no experiment tracking outputs

The current deliverable is an internally consistent documentation pack that defines the task before implementation begins.
