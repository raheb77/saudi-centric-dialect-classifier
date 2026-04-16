# Claude Sonnet Hard Subset Error Analysis

This report compares the tracked Saudi/Egyptian absorption confusions on the hard subset only.

## Requested Confusion Directions

| True Label | Predicted Label | Classical | Gemini Zero | Gemini Few | Sonnet Zero | Sonnet Few |
| --- | --- | --- | --- | --- | --- | --- |
| Saudi | Levantine | 15 | 1 | 1 | 1 | 1 |
| Saudi | Maghrebi | 11 | 0 | 0 | 1 | 1 |
| Egyptian | Maghrebi | 22 | 1 | 1 | 2 | 2 |
| Egyptian | Levantine | 13 | 0 | 2 | 0 | 1 |

## Interpretation

- The hard subset contains only rows that were already hard for at least one baseline in the four tracked Saudi/Egyptian absorption directions.
- Classical performance is expected to be near-floor on this subset because the selection rule is anchored on its known failure modes.
- The key question is whether Sonnet reduces those same absorption errors relative to both the classical baseline and Gemini on exactly the same rows.
