from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "data" / "interim_dataset.py"
SPEC = importlib.util.spec_from_file_location("project_interim_dataset", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

generate_interim_datasets = MODULE.generate_interim_datasets


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generate_interim_datasets_applies_mapping_and_leakage_rules(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "raw"
    subtask1 = data_root / "nadi2023" / "NADI2023_Release_Train" / "Subtask1"
    interim_dir = tmp_path / "data" / "interim"
    report_dir = tmp_path / "artifacts" / "reports"

    write_tsv(
        subtask1 / "NADI2023_Subtask1_TRAIN.tsv",
        ["#1_id", "#2_content", "#3_label"],
        [
            ["t1", "نص تدريب", "Saudi_Arabia"],
            ["t2", "نص خارج النطاق", "Iraq"],
        ],
    )
    write_tsv(
        subtask1 / "NADI2023_Subtask1_DEV.tsv",
        ["#1_id", "#2_content", "#3_label"],
        [
            ["d1", "نص تدريب", "Egypt"],
            ["d2", "نص تطوير", "Syria"],
        ],
    )
    write_tsv(
        subtask1 / "NADI2020-TWT.tsv",
        ["#1 tweet_ID", "#2 tweet_content", "#3 country_label", "#4 province_label"],
        [
            ["a1", "نص تدريب", "Egypt", "eg_Cairo"],
            ["a2", "نص متعارض", "Jordan", "jo_Amman"],
            ["a3", "نص موحد", "UAE", "ae_Dubai"],
            ["a4", "نص صالح", "Saudi_Arabia", "sa_Riyadh"],
        ],
    )
    write_tsv(
        subtask1 / "NADI2021-TWT.tsv",
        ["#1_tweetid", "#2_tweet", "#3_country_label", "#4_province_label"],
        [
            ["b1", "نص متعارض", "Lebanon", "lb_Beirut"],
            ["b2", "نص موحد", "United_Arab_Emirates", "ae_Abu-Dhabi"],
            ["b3", "نص صالح 2", "Egypt", "eg_Faiyum"],
            ["b4", "نص تطوير", "Syria", "sy_Damascus-City"],
        ],
    )

    outputs = generate_interim_datasets(data_root, interim_dir, report_dir)

    train_core = read_csv_rows(outputs["train_core"])
    dev_core = read_csv_rows(outputs["dev_core"])
    train_aug = read_csv_rows(outputs["train_aug_candidates"])
    report = json.loads(outputs["report_json"].read_text(encoding="utf-8"))

    assert len(train_core) == 1
    assert train_core[0]["macro_label"] == "Saudi"
    assert len(dev_core) == 1
    assert dev_core[0]["macro_label"] == "Levantine"
    assert len(train_aug) == 2
    assert {row["source_id"] for row in train_aug} == {"a4", "b3"}
    assert report["outputs"]["train_core"]["dropped_by_reason"]["out_of_scope_country"] == 1
    assert report["outputs"]["dev_core"]["dropped_by_reason"]["benchmark_exact_overlap_with_train"] == 1
    assert report["outputs"]["train_aug_candidates"]["dropped_by_reason"]["overlap_with_train_core"] == 1
    assert report["outputs"]["train_aug_candidates"]["dropped_by_reason"]["overlap_with_dev_core"] == 1
    assert report["outputs"]["train_aug_candidates"]["dropped_by_reason"]["conflicting_supporting_label"] == 2
    assert report["outputs"]["train_aug_candidates"]["dropped_by_reason"]["out_of_scope_country"] == 2
    assert report["overlap_removals"]["aug_dev_core_overlap_text_hash_count"] == 1
    assert report["conflict_removals"]["supporting_conflict_text_hash_count"] == 1
