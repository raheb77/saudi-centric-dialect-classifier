from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "baselines" / "llm_hard_subset.py"
SPEC = importlib.util.spec_from_file_location("project_llm_hard_subset", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_run_hard_subset_writes_expected_reports(tmp_path: Path, monkeypatch) -> None:
    train_path = tmp_path / "data" / "processed" / "train_core.csv"
    dev_path = tmp_path / "data" / "processed" / "dev_core.csv"
    classical_path = tmp_path / "artifacts" / "reports" / "classical_baseline_dev_predictions.csv"
    gemini_path = tmp_path / "artifacts" / "reports" / "llm_gemini_flash_lite_dev_predictions.csv"
    config_path = tmp_path / "configs" / "llm_sonnet_hard_subset.yaml"
    report_dir = tmp_path / "artifacts" / "reports"

    train_rows = [
        {"source_id": "s1", "original_text": "سعودي", "macro_label": "Saudi", "processed_text": "وش عندكم اليوم"},
        {"source_id": "s2", "original_text": "سعودي", "macro_label": "Saudi", "processed_text": "يا حرامي ساهر"},
        {"source_id": "e1", "original_text": "مصري", "macro_label": "Egyptian", "processed_text": "انا مشوفتش اوسخ"},
        {"source_id": "e2", "original_text": "مصري", "macro_label": "Egyptian", "processed_text": "وحشتني اوي بقه"},
        {"source_id": "l1", "original_text": "شامي", "macro_label": "Levantine", "processed_text": "مش كل شي"},
        {"source_id": "l2", "original_text": "شامي", "macro_label": "Levantine", "processed_text": "ليش هيك"},
        {"source_id": "m1", "original_text": "مغاربي", "macro_label": "Maghrebi", "processed_text": "مول كرش"},
        {"source_id": "m2", "original_text": "مغاربي", "macro_label": "Maghrebi", "processed_text": "لي عندو شي"},
    ]
    dev_rows = [
        {"source_id": "d1", "source_dataset": "nadi", "source_file": "dev", "source_row_number": "1", "original_text": "نص سعودي", "macro_label": "Saudi", "processed_text": "وش عندكم"},
        {"source_id": "d2", "source_dataset": "nadi", "source_file": "dev", "source_row_number": "2", "original_text": "نص مصري", "macro_label": "Egyptian", "processed_text": "انا مشوفتش"},
        {"source_id": "d3", "source_dataset": "nadi", "source_file": "dev", "source_row_number": "3", "original_text": "نص سعودي ٢", "macro_label": "Saudi", "processed_text": "مره زحمه"},
        {"source_id": "d4", "source_dataset": "nadi", "source_file": "dev", "source_row_number": "4", "original_text": "نص شامي", "macro_label": "Levantine", "processed_text": "ليش هيك"},
    ]
    classical_rows = [
        {"source_id": "d1", "true_label": "Saudi", "predicted_label": "Levantine"},
        {"source_id": "d2", "true_label": "Egyptian", "predicted_label": "Maghrebi"},
        {"source_id": "d3", "true_label": "Saudi", "predicted_label": "Saudi"},
        {"source_id": "d4", "true_label": "Levantine", "predicted_label": "Levantine"},
    ]
    gemini_rows = [
        {
            "source_id": "d1",
            "true_label": "Saudi",
            "zero_shot_predicted_label": "Saudi",
            "few_shot_predicted_label": "Saudi",
        },
        {
            "source_id": "d2",
            "true_label": "Egyptian",
            "zero_shot_predicted_label": "Egyptian",
            "few_shot_predicted_label": "Levantine",
        },
        {
            "source_id": "d3",
            "true_label": "Saudi",
            "zero_shot_predicted_label": "Maghrebi",
            "few_shot_predicted_label": "Saudi",
        },
        {
            "source_id": "d4",
            "true_label": "Levantine",
            "zero_shot_predicted_label": "Levantine",
            "few_shot_predicted_label": "Levantine",
        },
    ]

    write_csv(train_path, ["source_id", "original_text", "macro_label", "processed_text"], train_rows)
    write_csv(
        dev_path,
        ["source_id", "source_dataset", "source_file", "source_row_number", "original_text", "macro_label", "processed_text"],
        dev_rows,
    )
    write_csv(classical_path, ["source_id", "true_label", "predicted_label"], classical_rows)
    write_csv(
        gemini_path,
        ["source_id", "true_label", "zero_shot_predicted_label", "few_shot_predicted_label"],
        gemini_rows,
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_path": train_path.as_posix(),
                    "dev_path": dev_path.as_posix(),
                    "text_column": "processed_text",
                    "target_column": "macro_label",
                },
                "artifacts": {
                    "classical_predictions_path": classical_path.as_posix(),
                    "gemini_predictions_path": gemini_path.as_posix(),
                },
                "labels": {"order": ["Saudi", "Egyptian", "Levantine", "Maghrebi"]},
                "provider": {
                    "name": "anthropic_messages",
                    "model": "claude-sonnet-4-20250514",
                    "api_base": "https://api.anthropic.com/v1/messages",
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "timeout_seconds": 60,
                    "max_retries": 0,
                    "input_price_per_1m_tokens": 3.0,
                    "output_price_per_1m_tokens": 15.0,
                },
                "inference": {"batch_size": 20, "temperature": 0.0, "max_completion_tokens": 256},
                "few_shot": {"examples_per_class": 2},
                "output": {"report_dir": report_dir.as_posix(), "prefix": "llm_sonnet_hard_subset"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    def fake_run_prompt_mode(*, config, dev_rows, mode, support_rows):
        if mode == "zero_shot":
            predictions = ["Saudi", "Egyptian", "Saudi"]
        else:
            predictions = ["Saudi", "Egyptian", "Maghrebi"]
        metrics = MODULE.evaluate_predictions(config=config, dev_rows=dev_rows, predictions=predictions)
        return {
            "predictions": predictions,
            "metrics": metrics,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "latency": {"request_count": 1, "total_ms": 250.0, "avg_request_ms": 250.0, "avg_row_ms": 83.3},
            "estimated_cost_usd": 0.0010,
        }

    monkeypatch.setattr(MODULE, "run_prompt_mode", fake_run_prompt_mode)

    outputs = MODULE.run_hard_subset(config_path)

    expected_keys = {
        "subset_definition_markdown",
        "summary_markdown",
        "metrics_json",
        "classification_report_json",
        "predictions_csv",
        "error_analysis_markdown",
        "comparison_markdown",
    }
    assert set(outputs) == expected_keys
    for path in outputs.values():
        assert path.exists()

    metrics = json.loads(outputs["metrics_json"].read_text(encoding="utf-8"))
    assert metrics["hard_subset_rows"] == 3
    assert metrics["classical_subset"]["accuracy"] == 1 / 3
    assert metrics["zero_shot"]["accuracy"] == 1.0

    predictions = read_csv_rows(outputs["predictions_csv"])
    assert len(predictions) == 3
    assert {
        "selection_reasons",
        "classical_predicted_label",
        "gemini_zero_shot_predicted_label",
        "sonnet_zero_shot_predicted_label",
        "sonnet_few_shot_predicted_label",
    }.issubset(predictions[0].keys())

    subset_definition = outputs["subset_definition_markdown"].read_text(encoding="utf-8")
    assert "Final unique subset rows: `3`" in subset_definition
    assert "classical" in subset_definition
