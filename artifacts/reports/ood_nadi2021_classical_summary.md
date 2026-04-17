# Classical Baseline OOD Summary

This report evaluates the existing v1 Classical Baseline setup on the standalone NADI2021 dev split after applying the same four-label mapping and preprocessing policy used in-domain.

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

- Accuracy: `0.5153`
- Macro F1: `0.4937`
- In-domain accuracy reference: `0.8869`
- In-domain macro F1 reference: `0.8483`
- Delta accuracy vs in-domain: `-0.3716`
- Delta macro F1 vs in-domain: `-0.3547`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 0.5875 | 0.2904 | 0.3887 | 520 |
| Egyptian | 0.7655 | 0.4640 | 0.5778 | 1041 |
| Levantine | 0.3067 | 0.5972 | 0.4053 | 643 |
| Maghrebi | 0.5867 | 0.6201 | 0.6029 | 1124 |

## Confusion Matrix

| True \ Pred | Saudi | Egyptian | Levantine | Maghrebi |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 151 | 20 | 206 | 143 |
| Egyptian | 20 | 483 | 334 | 204 |
| Levantine | 51 | 64 | 384 | 144 |
| Maghrebi | 35 | 64 | 328 | 697 |
