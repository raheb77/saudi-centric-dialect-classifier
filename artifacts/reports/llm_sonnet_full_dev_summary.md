# LLM Baseline Summary

This baseline evaluates one chat-completions LLM on `dev_core` with two prompt settings: zero-shot and few-shot.

## Setup

- Provider: `anthropic_messages`
- Model: `claude-sonnet-4-20250514`
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
| Zero-Shot | 0.8268 | 0.7908 | 63593 | 13488 | 0.3931 |
| Few-Shot | 0.8408 | 0.8042 | 71673 | 13592 | 0.4189 |

## Approximate Latency

| Mode | Requests | Total ms | Avg request ms | Avg row ms |
| --- | --- | --- | --- | --- |
| Zero-Shot | 40 | 192887.1 | 4822.2 | 193.1 |
| Few-Shot | 40 | 188076.9 | 4701.9 | 188.3 |

## Per-Class Metrics

| Label | Zero P | Zero R | Zero F1 | Few P | Few R | Few F1 | Support |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Saudi | 0.5839 | 0.8700 | 0.6988 | 0.6385 | 0.8300 | 0.7217 | 100 |
| Egyptian | 0.6013 | 0.9500 | 0.7364 | 0.6025 | 0.9700 | 0.7433 | 100 |
| Levantine | 0.9058 | 0.8195 | 0.8605 | 0.9062 | 0.8471 | 0.8756 | 399 |
| Maghrebi | 0.9577 | 0.7925 | 0.8673 | 0.9612 | 0.8050 | 0.8762 | 400 |

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
