# Final Model Comparison

This file is the authoritative repository-level comparison summary. It separates the historical `999`-row prompt-only full-dev reports from the cleaned `998`-row benchmark-safe comparison reports, then summarizes OOD and robustness findings.

It does not reproduce raw tweet text and should be read together with the model card, dataset card, OOD summaries, and robustness summary.

## Evaluation Framing

- Historical prompt-only full-dev view: `999` rows
  - used by the Gemini Flash-Lite and Claude Sonnet full-dev runs
- Cleaned benchmark-safe comparison view: `998` rows
  - used by the corrected classical baseline rerun, MARBERT, MARBERT stability, and robustness evaluation
- OOD sources:
  - standalone NADI 2020 dev and NADI 2021 DA dev
  - both passed the exact-overlap OOD leakage pre-check before evaluation

The split distinction remains part of the experimental record. The repository keeps the corrected classical-vs-MARBERT comparison explicit and treats Gemini/Sonnet as historical task-context runs rather than collapsing everything into one unlabeled score table.

## In-Domain Results on the Cleaned `998`-Row Benchmark-Safe View

| Model | Setting | Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Classical baseline | TF-IDF + Logistic Regression | `0.8868` | `0.8476` | cleaned benchmark-safe rerun |
| MARBERT | seed `42` | `0.9669` | `0.9595` | cleaned benchmark-safe reference result |
| MARBERT | mean +/- std over seeds `42/123/7` | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` | acceptable stability |

Takeaways:

- Classical and MARBERT are now compared on the same cleaned benchmark-safe split.
- MARBERT remains the strongest overall model in the repository on clean in-domain metrics.
- The multi-seed pass supports the encoder result as stable enough for the v1 comparison story.

## Historical Prompt-Only Context on the Original `999`-Row Full-Dev View

| Model | Setting | Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Gemini Flash-Lite | zero-shot | `0.8679` | `0.8330` | prompt-only |
| Gemini Flash-Lite | few-shot | `0.8749` | `0.8414` | best prompt-only LLM run |
| Claude Sonnet | zero-shot | `0.8268` | `0.7908` | prompt-only |
| Claude Sonnet | few-shot | `0.8408` | `0.8042` | prompt-only |

Takeaway:

- Gemini few-shot remains the strongest prompt-only run, but these LLM scores are still historical `999`-row full-dev references rather than apples-to-apples comparisons against the cleaned classical/MARBERT view.

## OOD Leakage Pre-check

| Split | In-Scope Rows | Exact Overlap vs train/dev | Near-Duplicate Risk vs train_core | Classification |
| --- | ---: | ---: | ---: | --- |
| NADI 2020 dev | `3267` | `0` | `0` | acceptable as OOD evaluation source |
| NADI 2021 DA dev | `3328` | `0` | `0` | acceptable as OOD evaluation source |

The OOD evaluation below is therefore grounded in audited sources rather than assumed source independence.

## OOD Evaluation

| Dataset | Model | Accuracy | Macro F1 | In-Domain Reference Accuracy | In-Domain Reference Macro F1 | Delta Accuracy | Delta Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NADI 2020 dev | Classical | `0.4763` | `0.4467` | `0.8868` | `0.8476` | `-0.4105` | `-0.4009` |
| NADI 2020 dev | MARBERT | `0.6122` | `0.5938` | `0.9669` | `0.9595` | `-0.3548` | `-0.3657` |
| NADI 2021 DA dev | Classical | `0.5153` | `0.4937` | `0.8868` | `0.8476` | `-0.3714` | `-0.3539` |
| NADI 2021 DA dev | MARBERT | `0.6656` | `0.6443` | `0.9669` | `0.9595` | `-0.3014` | `-0.3153` |

Takeaways:

- MARBERT remains stronger than the classical baseline on both audited OOD splits.
- Classical OOD deltas now reference the corrected cleaned `998`-row in-domain result.
- OOD degradation is still material for both model families, so the current v1 story should not be framed as solved domain transfer.

Evidence in this repository supports an audited distribution-shift explanation more strongly than a leakage explanation: both OOD sources passed the exact-overlap pre-check, the same four-label mapping produces a much more Saudi/Egyptian-heavy and less Levantine-heavy label mix than the cleaned `dev_core`, and Levantine is the weakest MARBERT OOD class in both checked-in OOD summaries. Read together with the in-domain error analysis, topic/domain mismatch and weak-locality text are the most plausible interpretations, while orthographic or noise differences remain a secondary possibility rather than a proven primary cause. The current evidence therefore supports a careful claim of materially weaker domain transfer, not a stronger claim that one isolated factor fully explains the drop.

## Robustness Evaluation on the Cleaned `998`-Row Dev View

| Perturbation Family | Classical Macro F1 Drop | MARBERT Macro F1 Drop | More Robust Under This Test |
| --- | ---: | ---: | --- |
| `typo_noise` | `0.0502` | `0.0760` | Classical |
| `elongation` | `0.0004` | `0.0082` | Classical |
| `placeholder_noise` | `-0.0016` | `0.0092` | Classical |
| `punctuation_spacing` | `0.0201` | `0.0321` | Classical |

Reference clean metrics on the cleaned dev view:

- Classical: accuracy `0.8868`, macro F1 `0.8476`
- MARBERT: accuracy `0.9669`, macro F1 `0.9595`

Takeaways:

- MARBERT remains the higher-performing model on clean text.
- The classical baseline is more robust under all four deterministic perturbation families tested here.
- Robustness therefore remains a separate axis from clean in-domain accuracy.

## Decision Summary

- Strongest overall model: `MARBERT`
- Strongest prompt-only LLM configuration: `Gemini Flash-Lite few-shot`
- Strongest non-neural baseline: `Classical TF-IDF + Logistic Regression`
- Best-supported in-domain encoder claim: MARBERT on the cleaned benchmark-safe `998`-row dev split with three-seed stability summary
- Main remaining risk: material OOD degradation despite strong clean in-domain performance
- Main secondary risk: weaker perturbation robustness than the classical baseline

## Packaging Note

This comparison summary is packaging-safe because it reports aggregate metrics only. Text-bearing local artifacts such as raw data, interim/processed CSVs, and prediction CSVs remain local-only and are not the intended public release surface for the final repository.
