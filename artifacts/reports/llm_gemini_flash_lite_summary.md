# LLM Baseline Summary

This baseline evaluates one chat-completions LLM on `dev_core` with two prompt settings: zero-shot and few-shot.

## Setup

- Provider: `gemini_generate_content`
- Model: `gemini-3.1-flash-lite-preview`
- Train path: `data/processed/train_core.csv`
- Dev path: `data/processed/dev_core.csv`
- Text column: `processed_text`
- Target column: `macro_label`
- Labels: `Saudi, Egyptian, Levantine, Maghrebi`
- Few-shot examples per class: `2`
- Batch size: `25`
- Train rows available for support selection: `10000`
- Dev rows evaluated: `999`

## Overall Metrics

| Mode | Accuracy | Macro F1 | Prompt Tokens | Completion Tokens | Estimated Cost (USD) |
| --- | --- | --- | --- | --- | --- |
| Zero-Shot | 0.8679 | 0.8330 | 41262 | 18142 | N/A |
| Few-Shot | 0.8749 | 0.8414 | 47262 | 18156 | N/A |

## Approximate Latency

| Mode | Requests | Total ms | Avg request ms | Avg row ms |
| --- | --- | --- | --- | --- |
| Zero-Shot | 40 | 81748.0 | 2043.7 | 81.8 |
| Few-Shot | 40 | 77529.3 | 1938.2 | 77.6 |

## Per-Class Metrics

| Label | Zero P | Zero R | Zero F1 | Few P | Few R | Few F1 | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Saudi | 0.5625 | 0.9900 | 0.7174 | 0.5893 | 0.9900 | 0.7388 | 100 |
| Egyptian | 0.7132 | 0.9200 | 0.8035 | 0.7280 | 0.9100 | 0.8089 | 100 |
| Levantine | 0.9599 | 0.8396 | 0.8957 | 0.9526 | 0.8571 | 0.9024 | 399 |
| Maghrebi | 0.9884 | 0.8525 | 0.9154 | 0.9856 | 0.8550 | 0.9157 | 400 |

## Few-Shot Support Set

| Source ID | Label | Processed Text |
| --- | --- | --- |
| subtask1_train_10492 | Saudi | يا قوي يا حرامي ساهر |
| subtask1_train_445 | Saudi | تذكره مجانيه للي يبي يحضر |
| subtask1_train_9718 | Egyptian | انا مشوفتش اوسخ من كده |
| subtask1_train_4298 | Egyptian | وحشتني اوي علي فكره بقه |
| subtask1_train_5745 | Levantine | مش كل شي كل شي |
| subtask1_train_10147 | Levantine | تبا و تب ليش هيك |
| subtask1_train_13697 | Maghrebi | مول كرش باقي ما شبع |
| subtask1_train_3155 | Maghrebi | لي عندو شي فم يربطو |
