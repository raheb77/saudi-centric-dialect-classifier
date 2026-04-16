from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score


ERROR_ANALYSIS_DIRECTIONS = (
    ("Saudi", "Levantine"),
    ("Saudi", "Maghrebi"),
    ("Egyptian", "Maghrebi"),
    ("Egyptian", "Levantine"),
)

LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
PUNCT_SIGNAL_RE = re.compile(r"[_:/\\]")


@dataclass(frozen=True)
class LLMConfig:
    train_path: Path
    dev_path: Path
    text_column: str
    target_column: str
    label_order: tuple[str, ...]
    api_base: str
    api_key_env: str
    provider_name: str
    model: str
    timeout_seconds: int
    max_retries: int
    temperature: float
    max_completion_tokens: int
    batch_size: int
    input_price_per_1m_tokens: float | None
    output_price_per_1m_tokens: float | None
    few_shot_examples_per_class: int
    report_dir: Path
    report_prefix: str


def load_config(path: Path) -> LLMConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_cfg = payload["data"]
    provider_cfg = payload["provider"]
    inference_cfg = payload["inference"]
    output_cfg = payload["output"]
    few_shot_cfg = payload["few_shot"]
    return LLMConfig(
        train_path=Path(data_cfg["train_path"]),
        dev_path=Path(data_cfg["dev_path"]),
        text_column=data_cfg.get("text_column", "processed_text"),
        target_column=data_cfg.get("target_column", "macro_label"),
        label_order=tuple(payload["labels"]["order"]),
        api_base=str(provider_cfg.get("api_base", "https://api.openai.com/v1/chat/completions")),
        api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
        provider_name=str(provider_cfg.get("name", "openai_chat_completions")),
        model=str(provider_cfg["model"]),
        timeout_seconds=int(provider_cfg.get("timeout_seconds", 60)),
        max_retries=int(provider_cfg.get("max_retries", 2)),
        temperature=float(inference_cfg.get("temperature", 0.0)),
        max_completion_tokens=int(inference_cfg.get("max_completion_tokens", 600)),
        batch_size=int(inference_cfg.get("batch_size", 25)),
        input_price_per_1m_tokens=(
            float(provider_cfg["input_price_per_1m_tokens"])
            if provider_cfg.get("input_price_per_1m_tokens") is not None
            else None
        ),
        output_price_per_1m_tokens=(
            float(provider_cfg["output_price_per_1m_tokens"])
            if provider_cfg.get("output_price_per_1m_tokens") is not None
            else None
        ),
        few_shot_examples_per_class=int(few_shot_cfg.get("examples_per_class", 2)),
        report_dir=Path(output_cfg.get("report_dir", "artifacts/reports")),
        report_prefix=str(output_cfg.get("prefix", "llm_baseline")),
    )


def load_labeled_rows(path: Path, *, text_column: str, target_column: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in (text_column, target_column) if column not in fieldnames]
        if missing:
            missing_text = ", ".join(f"`{column}`" for column in missing)
            raise ValueError(f"Missing required columns in {path.as_posix()}: {missing_text}")
        return list(reader)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _chunks[T](items: list[T], size: int) -> list[list[T]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _support_example_score(text: str) -> tuple[int, int, int, int]:
    tokens = text.split()
    token_count = len(tokens)
    arabic_chars = len(ARABIC_RE.findall(text))
    latin_chars = len(LATIN_RE.findall(text))
    score = 0
    if 3 <= token_count <= 8:
        score += 8
    score -= abs(token_count - 5)
    if "<USER>" not in text:
        score += 4
    if "NUM" not in text:
        score += 3
    if latin_chars == 0:
        score += 3
    if not PUNCT_SIGNAL_RE.search(text):
        score += 2
    if arabic_chars >= max(8, len(text) // 2):
        score += 2
    return (-score, token_count, len(text), latin_chars)


def select_few_shot_support(train_rows: list[dict[str, str]], config: LLMConfig) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for label in config.label_order:
        candidates = [row for row in train_rows if row[config.target_column] == label]
        seen_texts: set[str] = set()
        ordered = sorted(
            candidates,
            key=lambda row: (
                *_support_example_score(row[config.text_column]),
                row.get("source_id", ""),
            ),
        )
        label_examples: list[dict[str, str]] = []
        for row in ordered:
            text = row[config.text_column].strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            label_examples.append(row)
            if len(label_examples) >= config.few_shot_examples_per_class:
                break
        if len(label_examples) < config.few_shot_examples_per_class:
            raise ValueError(f"Not enough support examples found for label `{label}`.")
        selected.extend(label_examples)
    return selected


def build_developer_prompt(config: LLMConfig) -> str:
    labels = ", ".join(config.label_order)
    return "\n".join(
        [
            "You are evaluating a Saudi-centered Arabic dialect classification benchmark.",
            f"Allowed labels: {labels}.",
            "Return labels using these exact strings only.",
            "Label definitions:",
            "- Saudi: dialectal Arabic from Saudi Arabia only. Do not use Saudi for non-Saudi Gulf text.",
            "- Egyptian: dialectal Arabic from Egypt.",
            "- Levantine: dialectal Arabic from Jordan, Lebanon, Palestine, or Syria.",
            "- Maghrebi: dialectal Arabic from Algeria, Libya, Morocco, or Tunisia.",
            "Guidance:",
            "- Many texts are short, weak-signal, or partially quasi-MSA. Use the best dialectal evidence available.",
            "- Prefer local dialectal cues over topic alone.",
            "- Saudi should remain distinct from broader Gulf or Mashriqi overlap unless the text is best read as Saudi.",
        ]
    )


def build_user_prompt(
    batch_rows: list[dict[str, str]],
    *,
    config: LLMConfig,
    mode: str,
    support_rows: list[dict[str, str]] | None = None,
) -> tuple[str, list[str]]:
    item_ids = [str(index + 1) for index in range(len(batch_rows))]
    lines = [
        f"Classify {len(batch_rows)} Arabic short texts.",
        'Return valid JSON only in this format: {"predictions":[{"item_id":"1","label":"Saudi"}]}',
        "Every item_id must appear exactly once.",
        "Each label must be one of: Saudi, Egyptian, Levantine, Maghrebi.",
    ]
    if mode == "few_shot" and support_rows:
        lines.extend(["", "Few-shot support examples:"])
        for row in support_rows:
            lines.append(f'- label="{row[config.target_column]}", text="{row[config.text_column]}"')
    lines.extend(["", "Items:"])
    for item_id, row in zip(item_ids, batch_rows, strict=True):
        lines.append(f'{item_id}\t{row[config.text_column]}')
    return "\n".join(lines), item_ids


def _extract_chat_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices") or []
    if not choices:
        raise ValueError("Missing choices in chat completion response.")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "".join(chunks)
    raise ValueError("Unsupported chat completion content format.")


def _extract_gemini_content(response_payload: dict[str, Any]) -> str:
    candidates = response_payload.get("candidates") or []
    if not candidates:
        prompt_feedback = response_payload.get("promptFeedback") or {}
        raise ValueError(f"Missing candidates in Gemini response: {prompt_feedback}")
    content = (candidates[0].get("content") or {}).get("parts") or []
    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            chunks.append(str(item["text"]))
    if not chunks:
        raise ValueError("Missing text parts in Gemini response.")
    return "".join(chunks)


def _extract_anthropic_content(response_payload: dict[str, Any]) -> str:
    content = response_payload.get("content") or []
    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            chunks.append(str(item.get("text", "")))
    if not chunks:
        raise ValueError("Missing text content in Anthropic response.")
    return "".join(chunks)


def _call_openai_chat_completion(
    *,
    config: LLMConfig,
    developer_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, int], float]:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required API key env var `{config.api_key_env}`.")

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        req = request.Request(config.api_base, data=data, headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                response_payload = json.loads(response.read().decode("utf-8"))
            usage = response_payload.get("usage") or {}
            usage_summary = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
                "total_tokens": int(usage.get("total_tokens", 0)),
            }
            return _extract_chat_content(response_payload), usage_summary, elapsed_ms
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= config.max_retries:
                break
            time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error


def _call_gemini_generate_content(
    *,
    config: LLMConfig,
    developer_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, int], float]:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required API key env var `{config.api_key_env}`.")

    endpoint = f"{config.api_base.rstrip('/')}/v1beta/models/{config.model}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {
            "parts": [{"text": developer_prompt}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_completion_tokens,
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        req = request.Request(endpoint, data=data, headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                response_payload = json.loads(response.read().decode("utf-8"))
            usage = response_payload.get("usageMetadata") or {}
            usage_summary = {
                "prompt_tokens": int(usage.get("promptTokenCount", 0)),
                "completion_tokens": int(usage.get("candidatesTokenCount", 0)),
                "total_tokens": int(usage.get("totalTokenCount", 0)),
            }
            return _extract_gemini_content(response_payload), usage_summary, elapsed_ms
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= config.max_retries:
                break
            time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error


def _call_anthropic_messages(
    *,
    config: LLMConfig,
    developer_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, int], float]:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required API key env var `{config.api_key_env}`.")

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_completion_tokens,
        "system": developer_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        req = request.Request(config.api_base, data=data, headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with request.urlopen(req, timeout=config.timeout_seconds) as response:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                response_payload = json.loads(response.read().decode("utf-8"))
            usage = response_payload.get("usage") or {}
            usage_summary = {
                "prompt_tokens": int(usage.get("input_tokens", 0)),
                "completion_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0)),
            }
            return _extract_anthropic_content(response_payload), usage_summary, elapsed_ms
        except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= config.max_retries:
                break
            time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error


def _call_llm(
    *,
    config: LLMConfig,
    developer_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, int], float]:
    if config.provider_name == "openai_chat_completions":
        return _call_openai_chat_completion(
            config=config,
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
        )
    if config.provider_name == "gemini_generate_content":
        return _call_gemini_generate_content(
            config=config,
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
        )
    if config.provider_name == "anthropic_messages":
        return _call_anthropic_messages(
            config=config,
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
        )
    raise ValueError(f"Unsupported provider `{config.provider_name}`.")


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()


def _coerce_label(value: str, label_order: tuple[str, ...]) -> str:
    normalized = value.strip()
    for label in label_order:
        if normalized == label:
            return label
    lowered = normalized.lower()
    for label in label_order:
        if lowered == label.lower():
            return label
    for label in label_order:
        if label.lower() in lowered:
            return label
    raise ValueError(f"Invalid label returned by model: `{value}`")


def parse_prediction_payload(content: str, *, item_ids: list[str], label_order: tuple[str, ...]) -> list[str]:
    payload = json.loads(_strip_json_fences(content))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("Model response JSON must contain a `predictions` list.")
    by_item_id: dict[str, str] = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("Each prediction entry must be an object.")
        item_id = str(item.get("item_id", "")).strip()
        label_value = str(item.get("label", "")).strip()
        if not item_id:
            raise ValueError("Prediction entry is missing `item_id`.")
        by_item_id[item_id] = _coerce_label(label_value, label_order)
    missing = [item_id for item_id in item_ids if item_id not in by_item_id]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Model response missing predictions for item ids: {missing_text}")
    return [by_item_id[item_id] for item_id in item_ids]


def evaluate_predictions(
    *,
    config: LLMConfig,
    dev_rows: list[dict[str, str]],
    predictions: list[str],
) -> dict[str, Any]:
    dev_labels = [row[config.target_column] for row in dev_rows]
    accuracy = accuracy_score(dev_labels, predictions)
    macro_f1 = f1_score(dev_labels, predictions, labels=list(config.label_order), average="macro")
    report = classification_report(
        dev_labels,
        predictions,
        labels=list(config.label_order),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
    }


def estimate_cost_usd(*, prompt_tokens: int, completion_tokens: int, config: LLMConfig) -> float | None:
    if config.input_price_per_1m_tokens is None or config.output_price_per_1m_tokens is None:
        return None
    input_cost = (prompt_tokens / 1_000_000.0) * config.input_price_per_1m_tokens
    output_cost = (completion_tokens / 1_000_000.0) * config.output_price_per_1m_tokens
    return input_cost + output_cost


def run_prompt_mode(
    *,
    config: LLMConfig,
    dev_rows: list[dict[str, str]],
    mode: str,
    support_rows: list[dict[str, str]] | None,
) -> dict[str, Any]:
    developer_prompt = build_developer_prompt(config)
    predictions: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    request_latencies_ms: list[float] = []
    for batch_rows in _chunks(dev_rows, config.batch_size):
        user_prompt, item_ids = build_user_prompt(
            batch_rows,
            config=config,
            mode=mode,
            support_rows=support_rows,
        )
        content, usage, latency_ms = _call_llm(
            config=config,
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
        )
        batch_predictions = parse_prediction_payload(
            content,
            item_ids=item_ids,
            label_order=config.label_order,
        )
        predictions.extend(batch_predictions)
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]
        request_latencies_ms.append(latency_ms)

    metrics = evaluate_predictions(config=config, dev_rows=dev_rows, predictions=predictions)
    latency_total_ms = sum(request_latencies_ms)
    latency_per_request_ms = latency_total_ms / len(request_latencies_ms) if request_latencies_ms else 0.0
    latency_per_row_ms = latency_total_ms / len(dev_rows) if dev_rows else 0.0
    return {
        "predictions": predictions,
        "metrics": metrics,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "latency": {
            "request_count": len(request_latencies_ms),
            "total_ms": latency_total_ms,
            "avg_request_ms": latency_per_request_ms,
            "avg_row_ms": latency_per_row_ms,
        },
        "estimated_cost_usd": estimate_cost_usd(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            config=config,
        ),
    }


def _prediction_row(
    row: dict[str, str],
    *,
    config: LLMConfig,
    zero_shot_prediction: str,
    few_shot_prediction: str,
) -> dict[str, str]:
    true_label = row.get(config.target_column, "")
    return {
        "source_dataset": row.get("source_dataset", ""),
        "source_file": row.get("source_file", ""),
        "source_id": row.get("source_id", ""),
        "source_row_number": row.get("source_row_number", ""),
        "original_text": row.get("original_text", ""),
        "processed_text": row.get(config.text_column, ""),
        "true_label": true_label,
        "zero_shot_predicted_label": zero_shot_prediction,
        "few_shot_predicted_label": few_shot_prediction,
        "zero_shot_is_error": str(zero_shot_prediction != true_label).lower(),
        "few_shot_is_error": str(few_shot_prediction != true_label).lower(),
    }


def write_dev_predictions_csv(
    path: Path,
    *,
    config: LLMConfig,
    dev_rows: list[dict[str, str]],
    zero_shot_predictions: list[str],
    few_shot_predictions: list[str],
) -> None:
    fieldnames = [
        "source_dataset",
        "source_file",
        "source_id",
        "source_row_number",
        "original_text",
        "processed_text",
        "true_label",
        "zero_shot_predicted_label",
        "few_shot_predicted_label",
        "zero_shot_is_error",
        "few_shot_is_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, zero_shot_prediction, few_shot_prediction in zip(
            dev_rows,
            zero_shot_predictions,
            few_shot_predictions,
            strict=True,
        ):
            writer.writerow(
                _prediction_row(
                    row,
                    config=config,
                    zero_shot_prediction=zero_shot_prediction,
                    few_shot_prediction=few_shot_prediction,
                )
            )


def _confusion_counts(
    *,
    dev_rows: list[dict[str, str]],
    predictions: list[str],
    config: LLMConfig,
) -> Counter[tuple[str, str]]:
    return Counter(
        (row[config.target_column], prediction)
        for row, prediction in zip(dev_rows, predictions, strict=True)
        if row[config.target_column] != prediction
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_classical_confusion_counts(report_dir: Path) -> Counter[tuple[str, str]]:
    path = report_dir / "classical_baseline_dev_predictions.csv"
    if not path.exists():
        return Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return Counter(
        (row["true_label"], row["predicted_label"])
        for row in rows
        if row.get("true_label") and row.get("predicted_label") and row["true_label"] != row["predicted_label"]
    )


def write_error_analysis_markdown(
    path: Path,
    *,
    config: LLMConfig,
    dev_rows: list[dict[str, str]],
    zero_shot_predictions: list[str],
    few_shot_predictions: list[str],
    zero_shot_metrics: dict[str, Any],
    few_shot_metrics: dict[str, Any],
    report_dir: Path,
) -> None:
    zero_confusions = _confusion_counts(dev_rows=dev_rows, predictions=zero_shot_predictions, config=config)
    few_confusions = _confusion_counts(dev_rows=dev_rows, predictions=few_shot_predictions, config=config)
    classical_confusions = _load_classical_confusion_counts(report_dir)
    classical_metrics = _load_json_if_exists(report_dir / "classical_baseline_metrics.json") or {}

    lines = [
        "# LLM Baseline Error Analysis",
        "",
        "This report compares the zero-shot and few-shot LLM baselines against the existing classical TF-IDF baseline on `dev_core`.",
        "",
        "## Requested Confusion Directions",
        "",
    ]
    rows = []
    for true_label, predicted_label in ERROR_ANALYSIS_DIRECTIONS:
        classical_count = classical_confusions.get((true_label, predicted_label), 0)
        zero_count = zero_confusions.get((true_label, predicted_label), 0)
        few_count = few_confusions.get((true_label, predicted_label), 0)
        rows.append(
            [
                true_label,
                predicted_label,
                str(classical_count),
                str(zero_count),
                str(few_count),
            ]
        )
        lines.extend(
            [
                f"### {true_label} -> {predicted_label}",
                "",
                f"- Classical count: `{classical_count}`",
                f"- Zero-shot count: `{zero_count}`",
                f"- Few-shot count: `{few_count}`",
                "",
            ]
        )
    lines.extend(
        _markdown_table(
            ["True Label", "Predicted Label", "Classical", "Zero-Shot", "Few-Shot"],
            rows,
        )
    )
    lines.append("")

    lines.extend(
        [
            "## Overall Comparison",
            "",
            *_markdown_table(
                ["Mode", "Accuracy", "Macro F1"],
                [
                    [
                        "Classical",
                        f"{float(classical_metrics.get('accuracy', 0.0)):.4f}",
                        f"{float(classical_metrics.get('macro_f1', 0.0)):.4f}",
                    ],
                    [
                        "Zero-Shot",
                        f"{zero_shot_metrics['accuracy']:.4f}",
                        f"{zero_shot_metrics['macro_f1']:.4f}",
                    ],
                    [
                        "Few-Shot",
                        f"{few_shot_metrics['accuracy']:.4f}",
                        f"{few_shot_metrics['macro_f1']:.4f}",
                    ],
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- The LLM baseline is evaluated against the same benchmark-safe `train_core` / `dev_core` setup as the classical baseline.",
            "- The requested confusion directions show whether prompting reduces the TF-IDF tendency to absorb Saudi and Egyptian texts into broader regional classes.",
            "- Zero-shot performance reflects the model's raw label understanding; few-shot performance adds only a small, train-derived support set and remains within the benchmark-safe core pool.",
            "- Remaining errors should be read together with `ERROR_ANALYSIS.md`: weak local signal, shared colloquial vocabulary, quasi-MSA writing, and topic-driven cues are still expected to matter even for an LLM baseline.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_markdown(
    path: Path,
    *,
    config: LLMConfig,
    train_rows: int,
    dev_rows: int,
    zero_shot_results: dict[str, Any],
    few_shot_results: dict[str, Any],
    support_rows: list[dict[str, str]],
) -> None:
    per_class_rows: list[list[str]] = []
    for label in config.label_order:
        zero_report = zero_shot_results["metrics"]["classification_report"][label]
        few_report = few_shot_results["metrics"]["classification_report"][label]
        per_class_rows.append(
            [
                label,
                f"{zero_report['precision']:.4f}",
                f"{zero_report['recall']:.4f}",
                f"{zero_report['f1-score']:.4f}",
                f"{few_report['precision']:.4f}",
                f"{few_report['recall']:.4f}",
                f"{few_report['f1-score']:.4f}",
                str(int(few_report["support"])),
            ]
        )
    support_rows_md = [
        [
            row.get("source_id", "") or "-",
            row[config.target_column],
            row[config.text_column],
        ]
        for row in support_rows
    ]
    lines = [
        "# LLM Baseline Summary",
        "",
        "This baseline evaluates one chat-completions LLM on `dev_core` with two prompt settings: zero-shot and few-shot.",
        "",
        "## Setup",
        "",
        f"- Provider: `{config.provider_name}`",
        f"- Model: `{config.model}`",
        f"- Train path: `{config.train_path.as_posix()}`",
        f"- Dev path: `{config.dev_path.as_posix()}`",
        f"- Text column: `{config.text_column}`",
        f"- Target column: `{config.target_column}`",
        f"- Labels: `{', '.join(config.label_order)}`",
        f"- Few-shot examples per class: `{config.few_shot_examples_per_class}`",
        f"- Batch size: `{config.batch_size}`",
        f"- Train rows available for support selection: `{train_rows}`",
        f"- Dev rows evaluated: `{dev_rows}`",
        "",
        "## Overall Metrics",
        "",
        *_markdown_table(
            ["Mode", "Accuracy", "Macro F1", "Prompt Tokens", "Completion Tokens", "Estimated Cost (USD)"],
            [
                [
                    "Zero-Shot",
                    f"{zero_shot_results['metrics']['accuracy']:.4f}",
                    f"{zero_shot_results['metrics']['macro_f1']:.4f}",
                    str(zero_shot_results["usage"]["prompt_tokens"]),
                    str(zero_shot_results["usage"]["completion_tokens"]),
                    (
                        f"{zero_shot_results['estimated_cost_usd']:.4f}"
                        if zero_shot_results["estimated_cost_usd"] is not None
                        else "N/A"
                    ),
                ],
                [
                    "Few-Shot",
                    f"{few_shot_results['metrics']['accuracy']:.4f}",
                    f"{few_shot_results['metrics']['macro_f1']:.4f}",
                    str(few_shot_results["usage"]["prompt_tokens"]),
                    str(few_shot_results["usage"]["completion_tokens"]),
                    (
                        f"{few_shot_results['estimated_cost_usd']:.4f}"
                        if few_shot_results["estimated_cost_usd"] is not None
                        else "N/A"
                    ),
                ],
            ],
        ),
        "",
        "## Approximate Latency",
        "",
        *_markdown_table(
            ["Mode", "Requests", "Total ms", "Avg request ms", "Avg row ms"],
            [
                [
                    "Zero-Shot",
                    str(zero_shot_results["latency"]["request_count"]),
                    f"{zero_shot_results['latency']['total_ms']:.1f}",
                    f"{zero_shot_results['latency']['avg_request_ms']:.1f}",
                    f"{zero_shot_results['latency']['avg_row_ms']:.1f}",
                ],
                [
                    "Few-Shot",
                    str(few_shot_results["latency"]["request_count"]),
                    f"{few_shot_results['latency']['total_ms']:.1f}",
                    f"{few_shot_results['latency']['avg_request_ms']:.1f}",
                    f"{few_shot_results['latency']['avg_row_ms']:.1f}",
                ],
            ],
        ),
        "",
        "## Per-Class Metrics",
        "",
        *_markdown_table(
            [
                "Label",
                "Zero P",
                "Zero R",
                "Zero F1",
                "Few P",
                "Few R",
                "Few F1",
                "Support",
            ],
            per_class_rows,
        ),
        "",
        "## Few-Shot Support Set",
        "",
        *_markdown_table(["Source ID", "Label", "Processed Text"], support_rows_md),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(config: LLMConfig, results: dict[str, Any]) -> dict[str, Path]:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.report_prefix
    summary_path = config.report_dir / f"{prefix}_summary.md"
    metrics_path = config.report_dir / f"{prefix}_metrics.json"
    classification_path = config.report_dir / f"{prefix}_classification_report.json"
    predictions_path = config.report_dir / f"{prefix}_dev_predictions.csv"
    error_analysis_path = config.report_dir / f"{prefix}_error_analysis.md"

    write_summary_markdown(
        summary_path,
        config=config,
        train_rows=results["train_rows"],
        dev_rows=results["dev_rows"],
        zero_shot_results=results["zero_shot"],
        few_shot_results=results["few_shot"],
        support_rows=results["support_rows"],
    )
    metrics_payload = {
        "provider": config.provider_name,
        "model": config.model,
        "train_rows": results["train_rows"],
        "dev_rows": results["dev_rows"],
        "few_shot_support_examples_per_class": config.few_shot_examples_per_class,
        "few_shot_support_examples": [
            {
                "source_id": row.get("source_id", ""),
                "macro_label": row.get(config.target_column, ""),
                "processed_text": row.get(config.text_column, ""),
            }
            for row in results["support_rows"]
        ],
        "zero_shot": {
            "accuracy": results["zero_shot"]["metrics"]["accuracy"],
            "macro_f1": results["zero_shot"]["metrics"]["macro_f1"],
            "usage": results["zero_shot"]["usage"],
            "latency": results["zero_shot"]["latency"],
            "estimated_cost_usd": results["zero_shot"]["estimated_cost_usd"],
        },
        "few_shot": {
            "accuracy": results["few_shot"]["metrics"]["accuracy"],
            "macro_f1": results["few_shot"]["metrics"]["macro_f1"],
            "usage": results["few_shot"]["usage"],
            "latency": results["few_shot"]["latency"],
            "estimated_cost_usd": results["few_shot"]["estimated_cost_usd"],
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    classification_payload = {
        "zero_shot": results["zero_shot"]["metrics"]["classification_report"],
        "few_shot": results["few_shot"]["metrics"]["classification_report"],
    }
    classification_path.write_text(
        json.dumps(classification_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_dev_predictions_csv(
        predictions_path,
        config=config,
        dev_rows=results["dev_rows_data"],
        zero_shot_predictions=results["zero_shot"]["predictions"],
        few_shot_predictions=results["few_shot"]["predictions"],
    )
    write_error_analysis_markdown(
        error_analysis_path,
        config=config,
        dev_rows=results["dev_rows_data"],
        zero_shot_predictions=results["zero_shot"]["predictions"],
        few_shot_predictions=results["few_shot"]["predictions"],
        zero_shot_metrics=results["zero_shot"]["metrics"],
        few_shot_metrics=results["few_shot"]["metrics"],
        report_dir=config.report_dir,
    )
    return {
        "summary_markdown": summary_path,
        "metrics_json": metrics_path,
        "classification_report_json": classification_path,
        "dev_predictions_csv": predictions_path,
        "error_analysis_markdown": error_analysis_path,
    }


def evaluate_llm_baseline(config: LLMConfig) -> dict[str, Any]:
    train_rows = load_labeled_rows(
        config.train_path,
        text_column=config.text_column,
        target_column=config.target_column,
    )
    dev_rows = load_labeled_rows(
        config.dev_path,
        text_column=config.text_column,
        target_column=config.target_column,
    )
    support_rows = select_few_shot_support(train_rows, config)
    zero_shot_results = run_prompt_mode(
        config=config,
        dev_rows=dev_rows,
        mode="zero_shot",
        support_rows=None,
    )
    few_shot_results = run_prompt_mode(
        config=config,
        dev_rows=dev_rows,
        mode="few_shot",
        support_rows=support_rows,
    )
    return {
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "dev_rows_data": dev_rows,
        "support_rows": support_rows,
        "zero_shot": zero_shot_results,
        "few_shot": few_shot_results,
    }


def run_baseline(config_path: Path) -> dict[str, Path]:
    config = load_config(config_path)
    results = evaluate_llm_baseline(config)
    return write_reports(config, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the zero-shot and few-shot LLM baseline on dev_core.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/llm_baseline.yaml"),
        help="Path to the LLM baseline YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_baseline(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
