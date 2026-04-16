from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "data" / "validation.py"
SPEC = importlib.util.spec_from_file_location("project_validation", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

generate_validation_reports = MODULE.generate_validation_reports
validate_tsv_file = MODULE.validate_tsv_file


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def test_validate_tsv_file_reports_expected_counts(tmp_path: Path) -> None:
    path = tmp_path / "sample.tsv"
    write_tsv(
        path,
        ["#1_id", "#2_content", "#3_label"],
        [
            ["1", "مرحبا يا سعودية", "Saudi_Arabia"],
            ["2", "", "Egypt"],
            ["2", "", "Egypt"],
            ["3", "كلمة", "Jordan"],
        ],
    )

    result = validate_tsv_file(path)

    assert result.schema_name == "nadi2023_st1_labeled"
    assert result.row_count == 4
    assert result.duplicate_row_count == 1
    assert result.text_stats["#2_content"].empty_count == 2
    assert result.text_stats["#2_content"].short_count == 3
    assert result.class_counts["#3_label"]["Egypt"] == 2


def test_generate_validation_reports_writes_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "raw"
    output_dir = tmp_path / "artifacts" / "reports"
    write_tsv(
        data_root / "test.tsv",
        ["#1 tweet_ID", "#2 tweet_content", "#3 country_label", "#4 province_label"],
        [["1", "هذا نص قصير", "Saudi_Arabia", "sa_Riyadh"]],
    )

    reports = generate_validation_reports(data_root, output_dir)

    assert reports["json"].exists()
    assert reports["csv"].exists()
    assert reports["markdown"].exists()


def test_generate_validation_reports_detects_cross_file_overlap_and_leakage(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "raw"
    output_dir = tmp_path / "artifacts" / "reports"
    write_tsv(
        data_root / "nadi2023" / "NADI2023_Release_Train" / "Subtask1" / "NADI2023_Subtask1_TRAIN.tsv",
        ["#1_id", "#2_content", "#3_label"],
        [
            ["1", "نص مشترك جدا", "Saudi_Arabia"],
            ["2", "نص فريد", "Egypt"],
        ],
    )
    write_tsv(
        data_root / "nadi2023" / "NADI2023_Release_Train" / "Subtask1" / "NADI2023_Subtask1_DEV.tsv",
        ["#1_id", "#2_content", "#3_label"],
        [
            ["3", "نص مشترك جدا", "Saudi_Arabia"],
            ["4", "نص تطوير", "Egypt"],
        ],
    )
    write_tsv(
        data_root / "nadi2023" / "NADI2023_Release_Train" / "Subtask1" / "NADI2020-TWT.tsv",
        ["#1 tweet_ID", "#2 tweet_content", "#3 country_label", "#4 province_label"],
        [
            ["A", "نص داعم", "UAE", "ae_Dubai"],
            ["B", "نص آخر", "Saudi_Arabia", "sa_Riyadh"],
        ],
    )
    write_tsv(
        data_root / "nadi2023" / "NADI2023_Release_Train" / "Subtask1" / "NADI2021-TWT.tsv",
        ["#1_tweetid", "#2_tweet", "#3_country_label", "#4_province_label"],
        [
            ["C", "نص داعم", "United_Arab_Emirates", "ae_Abu-Dhabi"],
            ["D", "نص آخر", "Egypt", "eg_Cairo"],
        ],
    )
    write_tsv(
        data_root / "nadi2020" / "NADI_release" / "unlabeled_10M.tsv",
        ["#1 tweet_ID"],
        [["X"], ["Y"]],
    )
    write_tsv(
        data_root / "nadi2020" / "NADI_release" / "train_labeled.tsv",
        ["#1 tweet_ID", "#2 tweet_content", "#3 country_label", "#4 province_label"],
        [["P", "نص مرجعي", "Saudi_Arabia", "sa_Riyadh"]],
    )

    reports = generate_validation_reports(data_root, output_dir)
    payload = json.loads(reports["json"].read_text(encoding="utf-8"))

    assert payload["summary_totals"]["benchmark_anchor_rows_scanned"] == 4
    assert payload["summary_totals"]["canonical_supporting_rows_scanned"] == 4
    assert payload["summary_totals"]["provenance_aux_eval_rows_scanned"] == 1
    assert payload["summary_totals"]["unlabeled_id_only_rows_scanned"] == 2
    assert payload["group_totals"]["benchmark_anchor"]["file_count"] == 2
    assert payload["group_totals"]["canonical_supporting"]["file_count"] == 2
    assert payload["group_totals"]["provenance_aux_eval"]["file_count"] == 1
    assert payload["group_totals"]["unlabeled_id_only"]["file_count"] == 1
    assert payload["benchmark_safety"]["benchmark_relevant_texts_in_multiple_files"] == 3
    assert payload["benchmark_safety"]["benchmark_train_dev_exact_overlap_count"] == 1
    assert payload["benchmark_safety"]["supporting_conflict_case_count"] == 1
