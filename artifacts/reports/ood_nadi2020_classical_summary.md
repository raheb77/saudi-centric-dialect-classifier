# Classical Baseline OOD Summary

This report evaluates the existing v1 Classical Baseline setup on the standalone NADI2020 dev split after applying the same four-label mapping and preprocessing policy used in-domain.

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

- Accuracy: `0.4763`
- Macro F1: `0.4467`
- In-domain accuracy reference: `0.8869`
- In-domain macro F1 reference: `0.8483`
- Delta accuracy vs in-domain: `-0.4106`
- Delta macro F1 vs in-domain: `-0.4016`

## Per-Class Metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 0.5126 | 0.2107 | 0.2987 | 579 |
| Egyptian | 0.7918 | 0.4159 | 0.5453 | 1070 |
| Levantine | 0.2700 | 0.5645 | 0.3653 | 581 |
| Maghrebi | 0.5280 | 0.6374 | 0.5775 | 1037 |

## Confusion Matrix

| True \ Pred | Saudi | Egyptian | Levantine | Maghrebi |
| --- | ---: | ---: | ---: | ---: |
| Saudi | 122 | 16 | 223 | 218 |
| Egyptian | 31 | 445 | 373 | 221 |
| Levantine | 43 | 58 | 328 | 152 |
| Maghrebi | 42 | 43 | 291 | 661 |
