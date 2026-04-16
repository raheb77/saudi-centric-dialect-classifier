# Annotation Guidelines

## Purpose
These guidelines define how local raw examples should be accepted, grouped, or dropped for the v1 four-way task:

- `Saudi`
- `Egyptian`
- `Levantine`
- `Maghrebi`

They are project-specific curation rules layered on top of the local NADI and MADAR source labels. They are not official NADI annotation instructions.

## Primary Decision Rule
Use the raw source geography as the starting point, then keep the example only if the text is Arabic-script short-form dialectal Arabic with enough signal for a confident v1 label. If the example is unclear, mixed, or MSA-heavy, drop it rather than forcing a label.

## Label Definitions
| v1 label | Raw source labels to keep | Notes |
| --- | --- | --- |
| `Saudi` | `Saudi_Arabia` | Keep Saudi as its own class. Do not merge other Gulf-country labels into `Saudi`. |
| `Egyptian` | `Egypt` | Keep only clearly Egyptian dialectal text. |
| `Levantine` | `Jordan`, `Lebanon`, `Palestine`, `Syria` | Group these raw country labels under one v1 label. |
| `Maghrebi` | `Algeria`, `Libya`, `Morocco`, `Tunisia` | Group these raw country labels under one v1 label. |

## Keep Criteria
Keep an example only when all of the following are true:

- The main text is written in Arabic script.
- The example is a short text or sentence, not a long document.
- The text contains enough dialectal evidence to support one of the four v1 labels.
- The raw source label maps to one of the four v1 labels.
- The text is not dominated by `USER`, `URL`, hashtags, emojis, or boilerplate.

## Drop Criteria
Drop an example when any of the following apply:

- `non_arabic_script`: Latin script or Arabizi is dominant.
- `msa_heavy`: the text reads primarily as Modern Standard Arabic rather than dialect.
- `mixed_dialect`: more than one dialectal region is materially present.
- `unclear_label`: dialect signal is too weak to justify a v1 label.
- `off_scope_country`: raw source label is outside the v1 mapping.
- `too_short_or_empty`: too little linguistic content remains after normal cleanup.
- `noise_or_template`: repeated tags, spam, copied boilerplate, or mostly placeholder tokens.
- `foreign_dominant`: non-Arabic content dominates the example.

## Special Guidance

### Saudi vs Gulf
Saudi is separate from Gulf in v1 because the local source datasets already distinguish `Saudi_Arabia` from other Gulf-country labels. If a raw example is from `Bahrain`, `Kuwait`, `Oman`, `Qatar`, `UAE` / `United_Arab_Emirates`, or `Yemen`, do not relabel it as `Saudi`. Drop it as `off_scope_country`.

### Levantine Grouping
For v1, `Jordan`, `Lebanon`, `Palestine`, and `Syria` are grouped into `Levantine`. If the raw source label is one of these countries and the text is clearly dialectal, keep it under `Levantine`.

### Maghrebi Grouping
For v1, `Algeria`, `Libya`, `Morocco`, and `Tunisia` are grouped into `Maghrebi`. Do not extend `Maghrebi` to other countries by guesswork. In particular, supporting-source labels such as `Mauritania` remain out of scope in v1.

### Source-Text Tension
If the raw source geography maps to an in-scope v1 label but the text itself looks contradictory, heavily normalized, or ambiguous, drop the example. Do not relabel across countries based on one lexical clue.

### MADAR Examples
`MADAR-2018.tsv` is documented as a reference / future OOD source. If it is used later, keep the same four-label mapping and the same drop rules. Its translated travel domain means it should not silently be treated as equivalent to Twitter data.

### Benchmark Safety and Source Conflicts
`NADI2023_Subtask1_TRAIN.tsv` and `NADI2023_Subtask1_DEV.tsv` are the benchmark anchor. Any exact text overlap between these two files should be removed from dev before benchmark-style evaluation rather than tolerated as harmless duplication.

For augmentation planning, only the bundled `NADI2020-TWT.tsv` and `NADI2021-TWT.tsv` files count as canonical supporting sources. If the same exact text appears across those supporting sources with conflicting labels after normalizing `UAE` and `United_Arab_Emirates` to the same canonical raw label, drop that text from augmentation candidates.

The standalone NADI 2020 and NADI 2021 DA releases remain provenance / auxiliary evaluation assets. They can inform inspection, but they should not be treated as automatic members of the canonical augmentation pool.

## Recommended Review Workflow
1. Check that the raw source label is eligible for one of the four v1 labels.
2. Check script, length, and text quality.
3. Decide whether dialectal evidence is strong enough.
4. Keep with the mapped v1 label or drop with a reason code.

## Suggested Drop Reason Codes
Use these exact codes in later validation or review logs:

- `non_arabic_script`
- `msa_heavy`
- `mixed_dialect`
- `unclear_label`
- `off_scope_country`
- `too_short_or_empty`
- `noise_or_template`
- `foreign_dominant`

## Notes for Future Implementation
- These guidelines intentionally prefer precision over recall.
- Uncertain examples should be discarded, not rescued.
- No automatic relabeling beyond the explicit country-to-v1 mapping is approved for v1.
