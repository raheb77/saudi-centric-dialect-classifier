# MARBERT OOD Summary

This report evaluates the existing v1 MARBERT setup on the standalone NADI2020 dev split after applying the same four-label mapping and preprocessing policy used in-domain.

## Row Counts

- Raw rows: `4957`
- Rows kept after mapping/filtering: `3267`
- Rows dropped as out-of-scope countries: `1690`

| Label | Kept Rows |
| --- | ---: |
| Saudi | 579 |
| Egyptian | 1070 |
| Levantine | 581 |
| Maghrebi | 1037 |

## Metrics

- Accuracy: `0.6122`
- Macro F1: `0.5938`
- In-domain accuracy reference: `0.9669`
- In-domain macro F1 reference: `0.9595`
- Delta accuracy vs in-domain: `-0.3548`
- Delta macro F1 vs in-domain: `-0.3657`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 0.5678 | 0.5060 | 0.5352 | 579 |
| Egyptian | 0.7711 | 0.6925 | 0.7297 | 1070 |
| Levantine | 0.3533 | 0.5783 | 0.4386 | 581 |
| Maghrebi | 0.7509 | 0.6075 | 0.6716 | 1037 |

## Confusion Matrix

| True \ Pred | Saudi | Egyptian | Levantine | Maghrebi |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 293 | 39 | 182 | 65 |
| Egyptian | 43 | 741 | 216 | 70 |
| Levantine | 79 | 92 | 336 | 74 |
| Maghrebi | 101 | 89 | 217 | 630 |
