# Final Model Comparison

This file is the authoritative repository-level comparison summary. It separates the original `999`-row full-dev reports from the cleaned `998`-row benchmark-safe encoder reports, then summarizes OOD and robustness findings.

It does not reproduce raw tweet text and should be read together with the model card, dataset card, OOD summaries, and robustness summary.

## Evaluation Framing

- Original full-dev view: `999` rows
  - used by the classical baseline, Gemini Flash-Lite, and Claude Sonnet full-dev runs
- Cleaned benchmark-safe encoder view: `998` rows
  - used by MARBERT, MARBERT stability, and robustness evaluation
- OOD sources:
  - standalone NADI 2020 dev and NADI 2021 DA dev
  - both passed the exact-overlap OOD leakage pre-check before evaluation

The split distinction is part of the experimental record. The repository does not collapse the `999`-row and `998`-row results into a single unlabeled in-domain score table.

## In-Domain Results on the Original `999`-Row Dev View

| Model | Setting | Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Classical baseline | TF-IDF + Logistic Regression | `0.8869` | `0.8483` | strongest non-neural baseline |
| Gemini Flash-Lite | zero-shot | `0.8679` | `0.8330` | prompt-only |
| Gemini Flash-Lite | few-shot | `0.8749` | `0.8414` | best prompt-only LLM run |
| Claude Sonnet | zero-shot | `0.8268` | `0.7908` | prompt-only |
| Claude Sonnet | few-shot | `0.8408` | `0.8042` | prompt-only |

Takeaway:

- The classical baseline outperformed both prompt-only LLM baselines on the original full-dev comparison.

## Cleaned Benchmark-Safe Encoder Result on the `998`-Row Dev View

| Model | Setting | Accuracy | Macro F1 | Notes |
| --- | --- | ---: | ---: | --- |
| MARBERT | seed `42` | `0.9669` | `0.9595` | cleaned benchmark-safe reference result |
| MARBERT | mean +/- std over seeds `42/123/7` | `0.9679 +/- 0.0036` | `0.9613 +/- 0.0063` | acceptable stability |

Takeaways:

- MARBERT is the strongest overall model in the repository on clean in-domain metrics.
- The multi-seed pass supports the encoder result as stable enough for the v1 comparison story.

## OOD Leakage Pre-check

| Split | In-Scope Rows | Exact Overlap vs train/dev | Near-Duplicate Risk vs train_core | Classification |
| --- | ---: | ---: | ---: | --- |
| NADI 2020 dev | `3267` | `0` | `0` | acceptable as OOD evaluation source |
| NADI 2021 DA dev | `3328` | `0` | `0` | acceptable as OOD evaluation source |

The OOD evaluation below is therefore grounded in audited sources rather than assumed source independence.

## OOD Evaluation

| Dataset | Model | Accuracy | Macro F1 | In-Domain Reference Accuracy | In-Domain Reference Macro F1 | Delta Accuracy | Delta Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NADI 2020 dev | Classical | `0.4763` | `0.4467` | `0.8869` | `0.8483` | `-0.4106` | `-0.4016` |
| NADI 2020 dev | MARBERT | `0.6122` | `0.5938` | `0.9669` | `0.9595` | `-0.3548` | `-0.3657` |
| NADI 2021 DA dev | Classical | `0.5153` | `0.4937` | `0.8869` | `0.8483` | `-0.3716` | `-0.3547` |
| NADI 2021 DA dev | MARBERT | `0.6656` | `0.6443` | `0.9669` | `0.9595` | `-0.3014` | `-0.3153` |

Takeaways:

- MARBERT remains stronger than the classical baseline on both audited OOD splits.
- OOD degradation is still material for both model families, so the current v1 story should not be framed as solved domain transfer.

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
