# MARBERT OOD Summary

This report evaluates the existing v1 MARBERT setup on the standalone NADI2021 dev split after applying the same four-label mapping and preprocessing policy used in-domain.

## Row Counts

- Raw rows: `5000`
- Rows kept after mapping/filtering: `3328`
- Rows dropped as out-of-scope countries: `1672`

| Label | Kept Rows |
| --- | ---: |
| Saudi | 520 |
| Egyptian | 1041 |
| Levantine | 643 |
| Maghrebi | 1124 |

## Metrics

- Accuracy: `0.6656`
- Macro F1: `0.6443`
- In-domain accuracy reference: `0.9669`
- In-domain macro F1 reference: `0.9595`
- Delta accuracy vs in-domain: `-0.3014`
- Delta macro F1 vs in-domain: `-0.3153`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 0.6015 | 0.6212 | 0.6112 | 520 |
| Egyptian | 0.7962 | 0.8146 | 0.8053 | 1041 |
| Levantine | 0.4031 | 0.5334 | 0.4592 | 643 |
| Maghrebi | 0.8011 | 0.6237 | 0.7014 | 1124 |

## Confusion Matrix

| True \ Pred | Saudi | Egyptian | Levantine | Maghrebi |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 323 | 21 | 117 | 59 |
| Egyptian | 15 | 848 | 129 | 49 |
| Levantine | 122 | 112 | 343 | 66 |
| Maghrebi | 77 | 84 | 262 | 701 |
