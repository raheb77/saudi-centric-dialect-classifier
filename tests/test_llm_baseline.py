from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "baselines" / "llm_baseline.py"
SPEC = importlib.util.spec_from_file_location("project_llm_baseline", MODULE_PATH)
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


def test_run_baseline_writes_expected_reports(tmp_path: Path, monkeypatch) -> None:
    train_path = tmp_path / "data" / "processed" / "train_core.csv"
    dev_path = tmp_path / "data" / "processed" / "dev_core.csv"
    report_dir = tmp_path / "artifacts" / "reports"
    config_path = tmp_path / "configs" / "llm_baseline.yaml"
    fieldnames = ["source_id", "original_text", "macro_label", "processed_text"]

    train_rows = [
        {"source_id": "s1", "original_text": "لهجة سعودية", "macro_label": "Saudi", "processed_text": "الرياض مره جميله"},
        {"source_id": "s2", "original_text": "لهجة سعودية", "macro_label": "Saudi", "processed_text": "وش عندكم اليوم"},
        {"source_id": "e1", "original_text": "لهجة مصرية", "macro_label": "Egyptian", "processed_text": "ايه الاخبار دلوقتي"},
        {"source_id": "e2", "original_text": "لهجة مصرية", "macro_label": "Egyptian", "processed_text": "لسه راجع من الشغل"},
        {"source_id": "l1", "original_text": "لهجة شامية", "macro_label": "Levantine", "processed_text": "شو الاخبار اليوم"},
        {"source_id": "l2", "original_text": "لهجة شامية", "macro_label": "Levantine", "processed_text": "هيدي الطريق مزحومه"},
        {"source_id": "m1", "original_text": "لهجة مغاربية", "macro_label": "Maghrebi", "processed_text": "بغيت نمشي للدرب"},
        {"source_id": "m2", "original_text": "لهجة مغاربية", "macro_label": "Maghrebi", "processed_text": "شكون جا لدار"},
    ]
    dev_rows = [
        {"source_id": "d1", "original_text": "نص سعودي", "macro_label": "Saudi", "processed_text": "وش عندكم"},
        {"source_id": "d2", "original_text": "نص مصري", "macro_label": "Egyptian", "processed_text": "ايه الاخبار"},
        {"source_id": "d3", "original_text": "نص شامي", "macro_label": "Levantine", "processed_text": "شو الاخبار"},
        {"source_id": "d4", "original_text": "نص مغاربي", "macro_label": "Maghrebi", "processed_text": "شكون جا"},
    ]
    write_csv(train_path, fieldnames, train_rows)
    write_csv(dev_path, fieldnames, dev_rows)
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
                "labels": {"order": ["Saudi", "Egyptian", "Levantine", "Maghrebi"]},
                "provider": {
                    "name": "openai_chat_completions",
                    "model": "gpt-4.1-mini",
                    "api_base": "https://api.openai.com/v1/chat/completions",
                    "api_key_env": "OPENAI_API_KEY",
                    "timeout_seconds": 60,
                    "max_retries": 0,
                    "input_price_per_1m_tokens": 0.40,
                    "output_price_per_1m_tokens": 1.60,
                },
                "inference": {"batch_size": 4, "temperature": 0.0, "max_completion_tokens": 256},
                "few_shot": {"examples_per_class": 2},
                "output": {"report_dir": report_dir.as_posix(), "prefix": "test_llm_baseline"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    def fake_chat_completion(*, config, developer_prompt, user_prompt):
        if "Few-shot support examples" in user_prompt:
            labels = ["Saudi", "Egyptian", "Levantine", "Maghrebi"]
        else:
            labels = ["Saudi", "Egyptian", "Maghrebi", "Maghrebi"]
        payload = {
            "predictions": [
                {"item_id": str(index + 1), "label": label}
                for index, label in enumerate(labels)
            ]
        }
        usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return json.dumps(payload, ensure_ascii=False), usage, 250.0

    monkeypatch.setattr(MODULE, "_call_llm", fake_chat_completion)

    outputs = MODULE.run_baseline(config_path)

    expected_keys = {
        "summary_markdown",
        "metrics_json",
        "classification_report_json",
        "dev_predictions_csv",
        "error_analysis_markdown",
    }
    assert set(outputs) == expected_keys
    for path in outputs.values():
        assert path.exists()

    metrics = json.loads(outputs["metrics_json"].read_text(encoding="utf-8"))
    assert metrics["train_rows"] == 8
    assert metrics["dev_rows"] == 4
    assert metrics["zero_shot"]["accuracy"] == 0.75
    assert metrics["few_shot"]["accuracy"] == 1.0

    predictions = read_csv_rows(outputs["dev_predictions_csv"])
    assert len(predictions) == 4
    assert {
        "source_id",
        "original_text",
        "processed_text",
        "true_label",
        "zero_shot_predicted_label",
        "few_shot_predicted_label",
    }.issubset(predictions[0].keys())

    error_analysis = outputs["error_analysis_markdown"].read_text(encoding="utf-8")
    assert "Saudi -> Levantine" in error_analysis
    assert "Classical" in error_analysis


def test_parse_gemini_response_payload() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"predictions":[{"item_id":"1","label":"Saudi"},{"item_id":"2","label":"Levantine"}]}'
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }
    content = MODULE._extract_gemini_content(payload)
    predictions = MODULE.parse_prediction_payload(
        content,
        item_ids=["1", "2"],
        label_order=("Saudi", "Egyptian", "Levantine", "Maghrebi"),
    )
    assert predictions == ["Saudi", "Levantine"]
