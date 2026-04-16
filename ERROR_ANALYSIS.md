# Error Analysis

This document summarizes the current repository-level error picture from the classical TF-IDF + Logistic Regression baseline evaluated on `data/processed/dev_core.csv`.

It is intended as a stable comparison reference for later baselines, especially:

- a future LLM baseline
- a future fine-tuned Arabic encoder

The current baseline reaches `0.8869` accuracy and `0.8483` macro F1 on `dev_core`, but its errors are not evenly distributed across labels. The main pattern is asymmetry: the single-country labels `Saudi` and `Egyptian` are more often absorbed into the broader grouped labels `Levantine` and `Maghrebi` than the reverse.

## Confusion Summary

The table below focuses on the four confusion directions that matter most for the current Saudi-centered task definition. `Share of class errors` is the proportion of all errors for that true class.

| True Label | Predicted Label | Count | Share of Class Errors |
| --- | --- | ---: | ---: |
| `Saudi` | `Levantine` | 15 | 57.7% |
| `Saudi` | `Maghrebi` | 11 | 42.3% |
| `Egyptian` | `Maghrebi` | 22 | 62.9% |
| `Egyptian` | `Levantine` | 13 | 37.1% |

Context for the denominators:

- `Saudi` has `26` total dev-set errors
- `Egyptian` has `35` total dev-set errors

## Requested Confusion Directions

### Saudi -> Levantine

- Count: `15`
- Share of `Saudi` errors: `57.7%`
- Representative source IDs: `subtask1_dev_295`, `subtask1_dev_546`, `subtask1_dev_757`
- Typical pattern: short reply-like colloquial text, conversational teasing, or emotionally framed language with limited strongly Saudi-local lexical evidence

### Saudi -> Maghrebi

- Count: `11`
- Share of `Saudi` errors: `42.3%`
- Representative source IDs: `subtask1_dev_83`, `subtask1_dev_899`, `subtask1_dev_1666`
- Typical pattern: broad colloquial sentiment, sports or everyday social talk, and weakly localized phrasing that does not anchor clearly to Saudi despite being in-scope

### Egyptian -> Maghrebi

- Count: `22`
- Share of `Egyptian` errors: `62.9%`
- Representative source IDs: `subtask1_dev_50`, `subtask1_dev_104`, `subtask1_dev_509`
- Typical pattern: topic-heavy text, football or public-discourse framing, and colloquial wording that is regionally familiar beyond Egypt rather than distinctly Egyptian

### Egyptian -> Levantine

- Count: `13`
- Share of `Egyptian` errors: `37.1%`
- Representative source IDs: `subtask1_dev_182`, `subtask1_dev_316`, `subtask1_dev_1390`
- Typical pattern: highly conversational short text, interpersonal commentary, and weak-locality everyday phrasing with limited dialect-specific anchors

## Top 10 Off-Diagonal Confusions

| True Label | Predicted Label | Count |
| --- | --- | ---: |
| `Egyptian` | `Maghrebi` | 22 |
| `Levantine` | `Maghrebi` | 21 |
| `Maghrebi` | `Levantine` | 18 |
| `Saudi` | `Levantine` | 15 |
| `Egyptian` | `Levantine` | 13 |
| `Saudi` | `Maghrebi` | 11 |
| `Levantine` | `Egyptian` | 5 |
| `Levantine` | `Saudi` | 5 |
| `Maghrebi` | `Egyptian` | 2 |
| `Maghrebi` | `Saudi` | 1 |

## Likely Causes of Error

### Label Design Asymmetry

The label space is intentionally asymmetric. `Saudi` and `Egyptian` are single-country labels, while `Levantine` and `Maghrebi` are grouped regional labels. That means the grouped labels cover a wider lexical and stylistic range by design. A surface-form baseline can therefore find it easier to map a weak-signal Saudi or Egyptian text into a broader regional bucket than to separate it cleanly as a single-country class.

### Shared Lexical Space / Weak Locality

Many current mistakes involve regionally shared colloquial vocabulary rather than obviously wrong preprocessing or clearly wrong labeling. A substantial portion of the error pool consists of everyday colloquial wording that is familiar across neighboring dialect zones. Some texts are only weakly local even for informed native readers, especially when the text expresses stance, emotion, or banter without strongly regional markers.

This is the main reason the current baseline pulls Saudi toward Levantine or Maghrebi and pulls Egyptian toward Levantine or Maghrebi.

### Quasi-MSA / Softened Colloquial Writing

Some dev examples are not fully MSA, but they are also not strongly local in orthography or vocabulary. They sit in a softened colloquial register: conversational enough to avoid pure MSA, but not dialectally sharp enough to anchor the text to one country-level label. In these cases, TF-IDF features reward broadly shared forms more than subtle locality.

Short texts contribute to this problem, but they are not the dominant cause by themselves. The more important issue is low locality density: some short texts still classify well when they contain distinctive regional cues, while some longer texts remain hard because their wording is broadly shared.

### Topic-Driven Confusion

Some Egyptian errors appear topic-driven rather than purely dialect-driven. Sports talk, public commentary, and generic social-media discourse can introduce lexical patterns that are common across the wider Arabic online space. In those cases, the model may be reacting more to topic clusters than to reliable dialect markers.

### Preprocessing Noise as a Secondary Factor

Preprocessing is not the main explanation for the current four headline confusion directions. Earlier URL and mention cleanup issues were worth fixing, but the remaining pattern is mostly linguistic rather than mechanical. Some Saudi errors reflect Gulf/Mashriqi overlap more than missing preprocessing, and many Egyptian errors reflect shared colloquial or topic-driven language more than tokenization noise.

Preprocessing should therefore be treated as a secondary factor: important for cleanup and consistency, but not the main reason the baseline confuses these classes.

## Inherently Ambiguous Samples

Some short or weak-signal texts are likely hard to classify even for native speakers without metadata. This matters for how later model comparisons should be interpreted.

These cases typically have one or more of the following properties:

- very generic colloquial wording
- reply-style fragments with limited context
- emotion or stance without place-specific lexical evidence
- softened colloquial writing that reduces strong locality
- topic-heavy text where dialect signal is weaker than discourse signal

For later LLM and encoder comparisons, not every error should be read as a model deficiency of the same kind. Some errors are probably recoverable with better modeling; others may reflect genuine label ambiguity under a short-text, no-metadata setting.

## Implications for Later Baselines

- A stronger model should be expected to reduce some `Saudi` and `Egyptian` absorption into broader grouped labels, but not eliminate all such cases.
- Improvement should be judged not only by overall macro F1, but also by whether the four tracked confusion directions shrink meaningfully.
- Later models should be compared against this document, not only against the aggregate confusion matrix, because the current failure mode is structured rather than random.

## Related Documents

- [PROJECT_SCOPE.md](PROJECT_SCOPE.md)
- [DATASET_CARD.md](DATASET_CARD.md)
- [artifacts/reports/classical_baseline_confusion_matrix.csv](artifacts/reports/classical_baseline_confusion_matrix.csv)
- [artifacts/reports/classical_baseline_error_analysis.md](artifacts/reports/classical_baseline_error_analysis.md)
