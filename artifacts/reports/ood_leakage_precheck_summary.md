# OOD Leakage Pre-check Summary

This summary consolidates the standalone NADI 2020 dev and NADI 2021 DA dev leakage audits against the current benchmark-safe project train/dev references.

## Split Summary

| Split | In-scope rows | Exact-overlap rows | Near-duplicate rows vs train_core | Classification |
| --- | ---: | ---: | ---: | --- |
| `NADI 2020 dev` | 3267 | `0` (0.00%) | `0` (0.00%) | `acceptable as OOD evaluation source` |
| `NADI 2021 DA dev` | 3328 | `0` (0.00%) | `0` (0.00%) | `acceptable as OOD evaluation source` |

## NADI 2020 dev

- Raw label values found: `Algeria`, `Bahrain`, `Djibouti`, `Egypt`, `Iraq`, `Jordan`, `Kuwait`, `Lebanon`, `Libya`, `Mauritania`, `Morocco`, `Oman`, `Palestine`, `Qatar`, `Saudi_Arabia`, `Somalia`, `Sudan`, `Syria`, `Tunisia`, `United_Arab_Emirates`, `Yemen`
- Mapped label distribution: `Saudi` 579 (17.72%), `Egyptian` 1070 (32.75%), `Levantine` 581 (17.78%), `Maghrebi` 1037 (31.74%)
- Exact overlap counts: train `source_id` `0` (0.00%), train `original_text` `0` (0.00%), train `processed_text` `0` (0.00%), dev `source_id` `0` (0.00%), dev `original_text` `0` (0.00%), dev `processed_text` `0` (0.00%)
- Near-duplicate count vs train_core: `0` (0.00%)
- Classification: `acceptable as OOD evaluation source`
- Decision rationale: No candidate rows hit any exact overlap check against the current benchmark-safe train_core/dev_core references.

## NADI 2021 DA dev

- Raw label values found: `Algeria`, `Bahrain`, `Djibouti`, `Egypt`, `Iraq`, `Jordan`, `Kuwait`, `Lebanon`, `Libya`, `Mauritania`, `Morocco`, `Oman`, `Palestine`, `Qatar`, `Saudi_Arabia`, `Somalia`, `Sudan`, `Syria`, `Tunisia`, `United_Arab_Emirates`, `Yemen`
- Mapped label distribution: `Saudi` 520 (15.62%), `Egyptian` 1041 (31.28%), `Levantine` 643 (19.32%), `Maghrebi` 1124 (33.77%)
- Exact overlap counts: train `source_id` `0` (0.00%), train `original_text` `0` (0.00%), train `processed_text` `0` (0.00%), dev `source_id` `0` (0.00%), dev `original_text` `0` (0.00%), dev `processed_text` `0` (0.00%)
- Near-duplicate count vs train_core: `0` (0.00%)
- Classification: `acceptable as OOD evaluation source`
- Decision rationale: No candidate rows hit any exact overlap check against the current benchmark-safe train_core/dev_core references.

## Recommendation

- Phase 9 Part 2 can proceed on the audited OOD sources.
- No split crossed the exact-overlap threshold that would block strict OOD framing.
