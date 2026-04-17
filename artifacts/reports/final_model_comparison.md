# Final Model Comparison

This comparison uses the current full-dev benchmark reports already present in the repository. MARBERT is shown as the existing seed `42` run only because the final leakage audit blocked additional encoder seeds.

## Full-Dev Results

| Model | Setting | Accuracy | Macro F1 | Status |
| --- | --- | --- | --- | --- |
| Classical baseline | TF-IDF + Logistic Regression | `0.8869` | `0.8483` | complete |
| Gemini Flash-Lite | zero-shot | `0.8679` | `0.8330` | complete |
| Gemini Flash-Lite | few-shot | `0.8749` | `0.8414` | complete |
| Claude Sonnet | zero-shot | `0.8268` | `0.7908` | complete |
| Claude Sonnet | few-shot | `0.8408` | `0.8042` | complete |
| MARBERT | seed `42` | `0.9720` | `0.9658` | complete |

## Ranking by Macro F1

1. `MARBERT` seed `42`: `0.9658`
2. `Classical baseline`: `0.8483`
3. `Gemini Flash-Lite` few-shot: `0.8414`
4. `Gemini Flash-Lite` zero-shot: `0.8330`
5. `Claude Sonnet` few-shot: `0.8042`
6. `Claude Sonnet` zero-shot: `0.7908`

## Interpretation

- The current encoder result is materially stronger than the available classical and LLM baselines on full `dev_core`.
- This file should not be read as a completed multi-seed encoder comparison.
- The encoder stability phase remains incomplete because `marbert_leakage_audit.md` found `1` exact `processed_text` overlap in the benchmark-safe core split.

## Current Recommendation

- Treat `marbert_seed_42_*` as the current encoder reference point.
- Do not claim multi-seed MARBERT stability yet.
- Resolve the single train/dev `processed_text` collision before launching seeds `123` and `7`.
