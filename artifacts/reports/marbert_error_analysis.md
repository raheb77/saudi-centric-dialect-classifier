# MARBERT Error Analysis

This report summarizes the tracked confusion directions for MARBERT on `dev_core` and compares them with the existing benchmark-safe baselines already present in the repository.

## Requested Confusion Directions

| True Label | Predicted Label | MARBERT | Classical | Gemini Zero-Shot | Gemini Few-Shot |
| --- | --- | --- | --- | --- | --- |
| Saudi | Levantine | 1 | 15 | 1 | 1 |
| Saudi | Maghrebi | 1 | 11 | 0 | 0 |
| Egyptian | Maghrebi | 2 | 22 | 1 | 1 |
| Egyptian | Levantine | 2 | 13 | 0 | 2 |

## Top 10 Off-Diagonal Confusions

| True Label | Predicted Label | Count |
| --- | --- | --- |
| Maghrebi | Levantine | 6 |
| Levantine | Maghrebi | 5 |
| Maghrebi | Egyptian | 4 |
| Levantine | Egyptian | 3 |
| Egyptian | Levantine | 2 |
| Egyptian | Maghrebi | 2 |
| Levantine | Saudi | 2 |
| Egyptian | Saudi | 1 |
| Maghrebi | Saudi | 1 |
| Saudi | Levantine | 1 |

## Interpretation

- MARBERT should be judged first on overall dev macro F1, then on whether the four tracked Saudi/Egyptian confusion directions shrink relative to the current classical and LLM baselines.
- If MARBERT improves overall macro F1 but keeps the same confusion profile, a second encoder is still useful for architectural diversity rather than simple incremental tuning.
- If MARBERT materially reduces the tracked confusions as well as improving macro F1, it becomes a stronger default encoder baseline for the next phase.
