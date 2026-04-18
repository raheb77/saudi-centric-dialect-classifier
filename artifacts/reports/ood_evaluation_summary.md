# OOD Evaluation Summary

Phase 9 Part 2 evaluates the existing v1 classical baseline rerun and MARBERT encoder on standalone NADI 2020 and NADI 2021 dev files only. The same four-label mapping and preprocessing policy used in-domain are applied, and out-of-scope countries are dropped.

## Row Counts After Mapping and Filtering

| Dataset | Raw Rows | Kept Rows | Dropped Out-of-Scope | Saudi | Egyptian | Levantine | Maghrebi |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nadi2020 | 4957 | 3267 | 1690 | 579 | 1070 | 581 | 1037 |
| nadi2021 | 5000 | 3328 | 1672 | 520 | 1041 | 643 | 1124 |

## OOD Metrics

| Dataset | Model | Accuracy | Macro F1 | In-Domain Accuracy | In-Domain Macro F1 | Delta Accuracy | Delta Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nadi2020 | Classical | 0.4763 | 0.4467 | 0.8868 | 0.8476 | -0.4105 | -0.4009 |
| nadi2020 | MARBERT | 0.6122 | 0.5938 | 0.9669 | 0.9595 | -0.3548 | -0.3657 |
| nadi2021 | Classical | 0.5153 | 0.4937 | 0.8868 | 0.8476 | -0.3714 | -0.3539 |
| nadi2021 | MARBERT | 0.6656 | 0.6443 | 0.9669 | 0.9595 | -0.3014 | -0.3153 |

Classical OOD deltas now reference the corrected cleaned `998`-row in-domain split. MARBERT OOD deltas continue to reference the cleaned `998`-row seed `42` in-domain result.

## Recommendation

Proceed to robustness next, but treat OOD weakness as a priority risk: the encoder shows a material macro-F1 drop out of domain and robustness should quantify whether that degradation is systematic or source-specific.
