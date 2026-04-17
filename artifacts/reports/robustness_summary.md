# Robustness Summary

Phase 9 Part 3 evaluates deterministic perturbation sensitivity on the cleaned 998-row benchmark-safe dev split. MARBERT uses the verified seed 42 checkpoint in inference-only mode; no encoder retraining is performed here.

## Sanity Checks

- `typo_noise`: `5/5` sanity checks passed.
- `elongation`: `5/5` sanity checks passed.
- `placeholder_noise`: `5/5` sanity checks passed.
- `punctuation_spacing`: `5/5` sanity checks passed.

## Robustness Metrics

| Family | Model | Clean Acc | Perturbed Acc | Acc Drop | Rel Acc Drop % | Clean Macro F1 | Perturbed Macro F1 | Macro F1 Drop | Rel Macro F1 Drop % | Excluded |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| typo_noise | classical | 0.8868 | 0.8447 | 0.0421 | 4.75 | 0.8476 | 0.7975 | 0.0502 | 5.92 | no |
| typo_noise | marbert | 0.9669 | 0.9028 | 0.0641 | 6.63 | 0.9595 | 0.8836 | 0.0760 | 7.92 | no |
| elongation | classical | 0.8868 | 0.8848 | 0.0020 | 0.23 | 0.8476 | 0.8472 | 0.0004 | 0.05 | no |
| elongation | marbert | 0.9669 | 0.9599 | 0.0070 | 0.73 | 0.9595 | 0.9513 | 0.0082 | 0.86 | no |
| placeholder_noise | classical | 0.8868 | 0.8888 | -0.0020 | -0.23 | 0.8476 | 0.8492 | -0.0016 | -0.19 | no |
| placeholder_noise | marbert | 0.9669 | 0.9589 | 0.0080 | 0.83 | 0.9595 | 0.9504 | 0.0092 | 0.96 | no |
| punctuation_spacing | classical | 0.8868 | 0.8727 | 0.0140 | 1.58 | 0.8476 | 0.8275 | 0.0201 | 2.38 | no |
| punctuation_spacing | marbert | 0.9669 | 0.9379 | 0.0291 | 3.01 | 0.9595 | 0.9274 | 0.0321 | 3.35 | no |

## Per-Family Winner Analysis

- `typo_noise`: larger absolute drop `MARBERT`, larger relative drop `MARBERT`, MARBERT advantage `shrinks` under perturbation.
- `elongation`: larger absolute drop `MARBERT`, larger relative drop `MARBERT`, MARBERT advantage `shrinks` under perturbation.
- `placeholder_noise`: larger absolute drop `MARBERT`, larger relative drop `MARBERT`, MARBERT advantage `shrinks` under perturbation.
- `punctuation_spacing`: larger absolute drop `MARBERT`, larger relative drop `MARBERT`, MARBERT advantage `shrinks` under perturbation.

## Aggregate Verdict

- MARBERT more robust on 0 of 4 families.
- Classical more robust on 4 of 4 families.
- Included families in final comparison: `4`.

## Reference Comparison

- Current cleaned MARBERT seed 42 reference: accuracy `0.9669`, macro F1 `0.9595`.
- Current classical baseline reference report: accuracy `0.8869`, macro F1 `0.8483` on the original 999-row dev split.
- Robustness clean classical result on the cleaned 998-row split: accuracy `0.8868`, macro F1 `0.8476`.
