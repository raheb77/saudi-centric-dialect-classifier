from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "baselines" / "classical_baseline.py"
SPEC = importlib.util.spec_from_file_location("project_classical_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

run_baseline = MODULE.run_baseline


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


def test_run_baseline_writes_expected_reports(tmp_path: Path) -> None:
    train_path = tmp_path / "data" / "processed" / "train_core.csv"
    dev_path = tmp_path / "data" / "processed" / "dev_core.csv"
    report_dir = tmp_path / "artifacts" / "reports"
    config_path = tmp_path / "configs" / "baseline.yaml"
    fieldnames = ["source_id", "original_text", "macro_label", "processed_text"]

    train_rows = [
        {"source_id": "1", "original_text": "الهلال سعودي الرياض", "macro_label": "Saudi", "processed_text": "الهلال سعودي الرياض"},
        {"source_id": "2", "original_text": "النصر سعودي جدة", "macro_label": "Saudi", "processed_text": "النصر سعودي جدة"},
        {"source_id": "3", "original_text": "القاهرة مصر مصري", "macro_label": "Egyptian", "processed_text": "القاهرة مصر مصري"},
        {"source_id": "4", "original_text": "اسكندرية لهجة مصرية", "macro_label": "Egyptian", "processed_text": "اسكندرية لهجة مصرية"},
        {"source_id": "5", "original_text": "فلسطين شام لهجة", "macro_label": "Levantine", "processed_text": "فلسطين شام لهجة"},
        {"source_id": "6", "original_text": "بيروت شام اردن", "macro_label": "Levantine", "processed_text": "بيروت شام اردن"},
        {"source_id": "7", "original_text": "مغرب الجزائر دارجة", "macro_label": "Maghrebi", "processed_text": "مغرب الجزائر دارجة"},
        {"source_id": "8", "original_text": "تونس ليبيا مغاربي", "macro_label": "Maghrebi", "processed_text": "تونس ليبيا مغاربي"},
    ]
    dev_rows = [
        {"source_id": "9", "original_text": "الرياض سعودي", "macro_label": "Saudi", "processed_text": "الرياض سعودي"},
        {"source_id": "10", "original_text": "مصر القاهرة", "macro_label": "Egyptian", "processed_text": "مصر القاهرة"},
        {"source_id": "11", "original_text": "بيروت شام", "macro_label": "Levantine", "processed_text": "بيروت شام"},
        {"source_id": "12", "original_text": "الجزائر مغرب", "macro_label": "Maghrebi", "processed_text": "الجزائر مغرب"},
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
                "features": {
                    "lowercase": False,
                    "sublinear_tf": True,
                    "word": {"ngram_range": [1, 2], "min_df": 1},
                    "char": {"analyzer": "char_wb", "ngram_range": [3, 5], "min_df": 1},
                },
                "model": {"C": 4.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42},
                "output": {"report_dir": report_dir.as_posix(), "prefix": "test_baseline"},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    outputs = run_baseline(config_path)

    expected_keys = {
        "metrics_json",
        "classification_report_json",
        "classification_report_txt",
        "confusion_matrix_csv",
        "summary_markdown",
        "dev_predictions_csv",
        "error_analysis_markdown",
    }
    assert set(outputs) == expected_keys
    for path in outputs.values():
        assert path.exists()

    metrics = json.loads(outputs["metrics_json"].read_text(encoding="utf-8"))
    assert metrics["train_rows"] == 8
    assert metrics["dev_rows"] == 4
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    predictions = read_csv_rows(outputs["dev_predictions_csv"])
    assert len(predictions) == 4
    assert {
        "source_id",
        "original_text",
        "processed_text",
        "true_label",
        "predicted_label",
    }.issubset(predictions[0].keys())

    error_analysis = outputs["error_analysis_markdown"].read_text(encoding="utf-8")
    assert "Saudi -> Levantine" in error_analysis
    assert "Top 10 Off-Diagonal Confusions" in error_analysis
