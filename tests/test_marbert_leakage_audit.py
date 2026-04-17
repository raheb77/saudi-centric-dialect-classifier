from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "encoders" / "leakage_audit.py"
SPEC = importlib.util.spec_from_file_location("project_marbert_leakage_audit", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_audit_payload_reports_exact_and_near_duplicates() -> None:
    train_rows = [
        {"source_id": "train_1", "original_text": "A", "processed_text": "<USER> a b c", "macro_label": "Saudi"},
        {"source_id": "train_2", "original_text": "B", "processed_text": "dup text", "macro_label": "Egyptian"},
        {"source_id": "train_3", "original_text": "C", "processed_text": "dup text", "macro_label": "Egyptian"},
    ]
    dev_rows = [
        {"source_id": "dev_1", "original_text": "Z", "processed_text": "<USER> a b c", "macro_label": "Saudi"},
        {"source_id": "dev_2", "original_text": "Y", "processed_text": "dup text x", "macro_label": "Egyptian"},
    ]

    payload = MODULE.build_audit_payload(
        train_rows=train_rows,
        dev_rows=dev_rows,
        near_duplicate_threshold=0.6,
    )

    assert payload["hard_checks"]["source_id_overlap"]["count"] == 0
    assert payload["hard_checks"]["original_text_overlap"]["count"] == 0
    assert payload["hard_checks"]["processed_text_overlap"]["count"] == 1
    assert payload["soft_checks"]["train_processed_text_duplicate_groups"]["count"] == 1
    assert payload["soft_checks"]["train_processed_text_duplicate_groups"]["additional_rows"] == 1
    assert payload["soft_checks"]["dev_processed_text_duplicate_groups"]["count"] == 0
    assert payload["soft_checks"]["dev_rows_with_near_duplicate_in_train"]["count"] == 2
    assert payload["status"] == "block"


def test_run_audit_writes_markdown_and_json(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    dev_path = tmp_path / "dev.csv"
    markdown_path = tmp_path / "audit.md"
    json_path = tmp_path / "audit.json"
    train_path.write_text(
        "source_id,original_text,processed_text,macro_label\n"
        "train_1,A,hello world,Saudi\n",
        encoding="utf-8",
    )
    dev_path.write_text(
        "source_id,original_text,processed_text,macro_label\n"
        "dev_1,B,hello world,Saudi\n",
        encoding="utf-8",
    )

    payload = MODULE.run_audit(
        train_path=train_path,
        dev_path=dev_path,
        markdown_out=markdown_path,
        json_out=json_path,
        near_duplicate_threshold=0.9,
    )

    assert payload["hard_checks"]["processed_text_overlap"]["count"] == 1
    assert markdown_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "block"
