# Error Analysis

This document records the completed-project error picture while keeping the classical TF-IDF + Logistic Regression baseline as the main qualitative reference. The classical baseline was evaluated on the original `999`-row `dev_core` view and remains the clearest view into the task's lexical failure modes.

Later baselines are now complete. MARBERT on the cleaned `998`-row benchmark-safe dev split materially reduced the tracked Saudi/Egyptian confusion directions, but the analysis below remains useful because it explains why those directions were hard in the first place.

## Classical Reference Result

- Split: original `dev_core` (`999` rows)
- Accuracy: `0.8869`
- Macro F1: `0.8483`

## Confusion Summary

The table below focuses on the four confusion directions that matter most for the Saudi-centered task definition. `Share of class errors` is the proportion of all errors for that true class in the classical baseline.

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

Not every remaining error in the completed project should be read as a model deficiency of the same kind. Some errors are recoverable with stronger modeling; others reflect genuine label ambiguity under a short-text, no-metadata setting.

## Completed-Project Context

The classical reference remains useful because it exposes the main task difficulty, but the completed project state now includes stronger baselines and an encoder result.

On the tracked Saudi/Egyptian confusion directions, MARBERT on the cleaned `998`-row dev split reduced the counts to:

| True Label | Predicted Label | Classical (`999` rows) | MARBERT (`998` rows) |
| --- | --- | ---: | ---: |
| `Saudi` | `Levantine` | 15 | 1 |
| `Saudi` | `Maghrebi` | 11 | 2 |
| `Egyptian` | `Maghrebi` | 22 | 4 |
| `Egyptian` | `Levantine` | 13 | 2 |

Interpretation:

- The contraction in these counts is large enough to support the claim that the encoder materially reduced the tracked failure modes.
- Because the classical and MARBERT counts come from different dev views (`999` vs `998`), treat this as direction-of-change evidence rather than a perfectly like-for-like confusion table.
- The classical analysis still explains the linguistic pressure points that later models had to overcome.

## Implications for the Final Comparison

- Overall macro F1 is important, but the tracked Saudi/Egyptian confusion directions remain the clearest diagnostic for whether a stronger model is improving the Saudi-centered task rather than merely shifting aggregate accuracy.
- MARBERT improved both overall clean performance and the tracked confusion directions.
- OOD degradation and robustness sensitivity remain real constraints even after the encoder improvement.

## Related Documents

- [PROJECT_SCOPE.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/PROJECT_SCOPE.md)
- [DATASET_CARD.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/DATASET_CARD.md)
- [MODEL_CARD.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/MODEL_CARD.md)
- [artifacts/reports/classical_baseline_confusion_matrix.csv](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports/classical_baseline_confusion_matrix.csv)
- [artifacts/reports/marbert_seed_42_confusion_matrix.csv](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports/marbert_seed_42_confusion_matrix.csv)
- [artifacts/reports/final_model_comparison.md](/Users/rahebalmutairi/projects/saudi-centric-dialect-classifier/artifacts/reports/final_model_comparison.md)
