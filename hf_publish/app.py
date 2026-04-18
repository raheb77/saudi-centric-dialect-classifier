from __future__ import annotations

from typing import Final

import gradio as gr


TITLE: Final[str] = "Saudi-Centric Dialect Classifier"

INTRO_MD: Final[str] = """# Saudi-Centric Dialect Classifier Summary

This Space-style package is summary-only. It presents benchmark-safe aggregate results from the source repository without shipping datasets, raw text, processed text, prediction files, checkpoints, or model weights.

This app does not accept user inputs or run model inference.

> Notice: no raw data, processed text, text-bearing prediction files, or model weights are included in this package.
"""

OVERVIEW_MD: Final[str] = """## Project Summary

- Task: four-way short-text Arabic dialect classification
- Labels: `Saudi`, `Egyptian`, `Levantine`, `Maghrebi`
- Scope: Arabic script only, no Arabizi in v1, uncertain or MSA-heavy items dropped
- Main repository workflow: validation, preprocessing, TF-IDF baseline, prompt-only LLM baselines, and a fine-tuned MARBERT encoder

## Included Here

- project summary and benchmark-safe aggregate metrics
- safe file references back to the source repository
- a lightweight Gradio interface for review

## Not Included Here

- any dataset files
- raw or processed text
- prediction CSVs
- checkpoints or model weights
- training logs, caches, or other large artifacts

## Illustrative Task Examples

These examples are illustrative only. They describe the task scope and are not runnable inputs in this Space.

- `لهجتي سعودية وأتكلم عن السوق المحلي كل يوم`
- `هذا التعبير أقرب إلى المصرية في سياق محادثة قصيرة`
- `المثال هنا توضيحي فقط، ولا يتم تشغيل أي نموذج داخل هذه المساحة`
"""

RESULTS_MD: Final[str] = """## In-Domain Metrics

Important split note:

- The corrected classical baseline rerun and MARBERT use the cleaned benchmark-safe `998`-row dev view.
- Gemini Flash-Lite and Claude Sonnet remain historical prompt-only runs on the original `999`-row dev view.

| Model / Setting | Eval View | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Classical TF-IDF + Logistic Regression | cleaned benchmark-safe dev (`998` rows) | `0.8868` | `0.8476` |
| Gemini Flash-Lite few-shot | original dev (`999` rows) | `0.8749` | `0.8414` |
| Claude Sonnet few-shot | original dev (`999` rows) | `0.8408` | `0.8042` |
| MARBERT seed `42` | cleaned dev (`998` rows) | `0.9669` | `0.9595` |
| MARBERT mean +/- std over seeds `42/123/7` | cleaned dev (`998` rows) | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` |

## OOD Metrics

OOD leakage pre-check status:

- NADI 2020 dev: `0` exact overlaps, `0` near-duplicate risk
- NADI 2021 DA dev: `0` exact overlaps, `0` near-duplicate risk

| Dataset | Model | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| NADI 2020 dev | Classical | `0.4763` | `0.4467` |
| NADI 2020 dev | MARBERT | `0.6122` | `0.5938` |
| NADI 2021 DA dev | Classical | `0.5153` | `0.4937` |
| NADI 2021 DA dev | MARBERT | `0.6656` | `0.6443` |

## Robustness Snapshot

- Clean in-domain leader: `MARBERT`
- More robust under tested perturbations: `Classical TF-IDF + Logistic Regression`
- Classical was more robust on `4/4` tested perturbation families.
"""

REFERENCES_MD: Final[str] = """## Safe References

These repository-relative file references point to documentation and aggregate reports in the source project. They are safe to cite and do not imply that those files are bundled into this package.

- `README.md`
- `PROJECT_SCOPE.md`
- `DATASET_CARD.md`
- `ANNOTATION_GUIDELINES.md`
- `MODEL_CARD.md`
- `ERROR_ANALYSIS.md`
- `LICENSE_NOTES.md`
- `artifacts/reports/final_model_comparison.md`
- `artifacts/reports/ood_evaluation_summary.md`
- `artifacts/reports/robustness_summary.md`
- `artifacts/reports/final_packaging_audit.md`

## Packaging Notice

This app is a reporting surface only. It does not expose inference endpoints because no checkpoints or weights are distributed here.
"""


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(INTRO_MD)
    with gr.Tabs():
        with gr.Tab("Overview"):
            gr.Markdown(OVERVIEW_MD)
        with gr.Tab("Results"):
            gr.Markdown(RESULTS_MD)
        with gr.Tab("References"):
            gr.Markdown(REFERENCES_MD)


if __name__ == "__main__":
    demo.launch()
