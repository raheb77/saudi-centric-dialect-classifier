# MARBERT Error Analysis

This report summarizes the tracked confusion directions for MARBERT on `dev_core` and compares them with the existing benchmark-safe baselines already present in the repository.

## Requested Confusion Directions

| True Label | Predicted Label | MARBERT |
| --- | --- | --- |
| Saudi | Levantine | 3 |
| Saudi | Maghrebi | 2 |
| Egyptian | Maghrebi | 5 |
| Egyptian | Levantine | 3 |

## Top 10 Off-Diagonal Confusions

| True Label | Predicted Label | Count |
| --- | --- | --- |
| Maghrebi | Levantine | 8 |
| Egyptian | Maghrebi | 5 |
| Levantine | Maghrebi | 5 |
| Levantine | Saudi | 5 |
| Egyptian | Levantine | 3 |
| Saudi | Levantine | 3 |
| Levantine | Egyptian | 2 |
| Saudi | Maghrebi | 2 |
| Egyptian | Saudi | 1 |
| Maghrebi | Saudi | 1 |

## Interpretation

- MARBERT should be judged first on overall dev macro F1, then on whether the four tracked Saudi/Egyptian confusion directions shrink relative to the current classical and LLM baselines.
- If MARBERT improves overall macro F1 but keeps the same confusion profile, a second encoder is still useful for architectural diversity rather than simple incremental tuning.
- If MARBERT materially reduces the tracked confusions as well as improving macro F1, it becomes a stronger default encoder baseline for the next phase.
