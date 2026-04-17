from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "ood_leakage_precheck.py"
SPEC = importlib.util.spec_from_file_location("project_ood_leakage_precheck", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


CandidateConfig = MODULE.CandidateConfig
analyze_candidate = MODULE.analyze_candidate
build_benchmark_reference = MODULE.build_benchmark_reference
classification_from_exact_overlap = MODULE.classification_from_exact_overlap
duplicate_processed_text_summary = MODULE.duplicate_processed_text_summary
load_candidate_rows = MODULE.load_candidate_rows
run_precheck = MODULE.run_precheck


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_candidate_rows_applies_mapping_and_drops_out_of_scope(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.tsv"
    write_tsv(
        candidate_path,
        ["id", "text", "country"],
        [
            {"id": "c1", "text": "مرحبا URL", "country": "Saudi_Arabia"},
            {"id": "c2", "text": "اهلا", "country": "Jordan"},
            {"id": "c3", "text": "خارج", "country": "Oman"},
        ],
    )

    rows, raw_label_counts, mapped_raw_labels, dropped_raw_labels = load_candidate_rows(
        CandidateConfig(
            slug="candidate",
            display_name="Candidate",
            path=candidate_path,
            id_column="id",
            text_column="text",
            label_column="country",
        )
    )

    assert [row.source_id for row in rows] == ["c1", "c2"]
    assert rows[0].processed_text == "مرحبا"
    assert rows[1].macro_label == "Levantine"
    assert raw_label_counts["Oman"] == 1
    assert mapped_raw_labels == ["Jordan", "Saudi_Arabia"]
    assert dropped_raw_labels == ["Oman"]


def test_duplicate_processed_text_summary_counts_groups_and_additional_rows() -> None:
    rows = [
        MODULE.CandidateRow(
            source_id="a",
            raw_label="Saudi_Arabia",
            normalized_raw_label="Saudi_Arabia",
            macro_label="Saudi",
            original_text="t1",
            processed_text="same",
        ),
        MODULE.CandidateRow(
            source_id="b",
            raw_label="Saudi_Arabia",
            normalized_raw_label="Saudi_Arabia",
            macro_label="Saudi",
            original_text="t2",
            processed_text="same",
        ),
        MODULE.CandidateRow(
            source_id="c",
            raw_label="Egypt",
            normalized_raw_label="Egypt",
            macro_label="Egyptian",
            original_text="t3",
            processed_text="other",
        ),
    ]

    summary = duplicate_processed_text_summary(rows)

    assert summary.count == 1
    assert summary.additional_rows == 1
    assert summary.representative_groups == (("a", "b"),)


def test_classification_uses_combined_exact_overlap_thresholds() -> None:
    near_duplicates = MODULE.NearDuplicateSummary(
        count=3,
        percentage=1.2,
        representative_pairs=tuple(),
        threshold=0.9,
    )

    classification, rationale, prominent_caveat = classification_from_exact_overlap(
        exact_overlap_count=4,
        total_rows=100,
        near_duplicates=near_duplicates,
    )

    assert classification == "acceptable only as held-out historical evaluation"
    assert prominent_caveat is True
    assert "1%-5% band" in rationale


def test_analyze_candidate_reports_exact_overlaps_and_lev_mapping(tmp_path: Path) -> None:
    train_interim = tmp_path / "data" / "interim" / "train_core.csv"
    train_processed = tmp_path / "data" / "processed" / "train_core.csv"
    dev_interim = tmp_path / "data" / "interim" / "dev_core.csv"
    dev_processed = tmp_path / "data" / "processed" / "dev_core.csv"
    candidate_path = tmp_path / "candidate.tsv"

    write_csv(
        train_interim,
        ["source_id", "text", "macro_label"],
        [
            {"source_id": "train_1", "text": "نص تدريب", "macro_label": "Saudi"},
            {"source_id": "train_2", "text": "نص اخر", "macro_label": "Egyptian"},
        ],
    )
    write_csv(
        train_processed,
        ["source_id", "original_text", "processed_text", "macro_label"],
        [
            {"source_id": "train_1", "original_text": "نص تدريب", "processed_text": "نص تدريب", "macro_label": "Saudi"},
            {"source_id": "train_2", "original_text": "نص اخر", "processed_text": "نص اخر", "macro_label": "Egyptian"},
        ],
    )
    write_csv(
        dev_interim,
        ["source_id", "text", "macro_label"],
        [
            {"source_id": "dev_1", "text": "نص تطوير", "macro_label": "Levantine"},
            {"source_id": "dev_2", "text": "نص مغاربي", "macro_label": "Maghrebi"},
        ],
    )
    write_csv(
        dev_processed,
        ["source_id", "original_text", "processed_text", "macro_label"],
        [
            {"source_id": "dev_1", "original_text": "نص تطوير", "processed_text": "نص تطوير", "macro_label": "Levantine"},
            {"source_id": "dev_2", "original_text": "نص مغاربي", "processed_text": "نص مغاربي", "macro_label": "Maghrebi"},
        ],
    )
    write_tsv(
        candidate_path,
        ["id", "text", "country"],
        [
            {"id": "cand_1", "text": "نص تدريب", "country": "Saudi_Arabia"},
            {"id": "cand_2", "text": "نص جديد", "country": "Jordan"},
            {"id": "cand_3", "text": "خارج", "country": "Oman"},
        ],
    )

    train_reference = build_benchmark_reference(
        name="train_core",
        interim_path=train_interim,
        processed_path=train_processed,
    )
    dev_reference = build_benchmark_reference(
        name="dev_core",
        interim_path=dev_interim,
        processed_path=dev_processed,
    )

    payload = analyze_candidate(
        config=CandidateConfig(
            slug="candidate",
            display_name="Candidate",
            path=candidate_path,
            id_column="id",
            text_column="text",
            label_column="country",
        ),
        train_reference=train_reference,
        dev_reference=dev_reference,
        near_duplicate_threshold=0.9,
    )

    assert payload["row_counts"]["before_filtering"] == 3
    assert payload["row_counts"]["after_dropping_out_of_scope"] == 2
    assert payload["mapped_raw_label_values"] == ["Jordan", "Saudi_Arabia"]
    assert payload["dropped_raw_label_values"] == ["Oman"]
    assert payload["levantine_mapping_confirmation"]["all_present_mapped_to_levantine"] is True
    assert payload["exact_overlap_checks"]["train_core_original_text"]["count"] == 1
    assert payload["exact_overlap_checks"]["train_core_processed_text"]["count"] == 1
    assert payload["combined_exact_overlap"]["count"] == 1
    assert payload["classification"] == "not acceptable until deduplicated"


def test_run_precheck_writes_required_reports(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    report_dir = tmp_path / "artifacts" / "reports"
    train_interim = data_root / "interim" / "train_core.csv"
    dev_interim = data_root / "interim" / "dev_core.csv"
    train_processed = data_root / "processed" / "train_core.csv"
    dev_processed = data_root / "processed" / "dev_core.csv"

    write_csv(
        train_interim,
        ["source_id", "text", "macro_label"],
        [{"source_id": "train_1", "text": "نص تدريب", "macro_label": "Saudi"}],
    )
    write_csv(
        train_processed,
        ["source_id", "original_text", "processed_text", "macro_label"],
        [{"source_id": "train_1", "original_text": "نص تدريب", "processed_text": "نص تدريب", "macro_label": "Saudi"}],
    )
    write_csv(
        dev_interim,
        ["source_id", "text", "macro_label"],
        [{"source_id": "dev_1", "text": "نص تطوير", "macro_label": "Egyptian"}],
    )
    write_csv(
        dev_processed,
        ["source_id", "original_text", "processed_text", "macro_label"],
        [{"source_id": "dev_1", "original_text": "نص تطوير", "processed_text": "نص تطوير", "macro_label": "Egyptian"}],
    )

    write_tsv(
        data_root / "raw" / "nadi2020" / "NADI_release" / "dev_labeled.tsv",
        ["#1 tweet_ID", "#2 tweet_content", "#3 country_label", "#4 province_label"],
        [{"#1 tweet_ID": "c1", "#2 tweet_content": "نص جديد", "#3 country_label": "Saudi_Arabia", "#4 province_label": "sa_Riyadh"}],
    )
    write_tsv(
        data_root / "raw" / "nadi2021" / "NADI2021_DEV.1.0" / "Subtask_1.2+2.2_DA" / "DA_dev_labeled.tsv",
        ["#1_tweetid", "#2_tweet", "#3_country_label", "#4_province_label"],
        [{"#1_tweetid": "c2", "#2_tweet": "نص تطوير", "#3_country_label": "Egypt", "#4_province_label": "eg_Cairo"}],
    )

    cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        payloads = run_precheck(
            train_interim_path=Path("data/interim/train_core.csv"),
            dev_interim_path=Path("data/interim/dev_core.csv"),
            train_processed_path=Path("data/processed/train_core.csv"),
            dev_processed_path=Path("data/processed/dev_core.csv"),
            report_dir=Path("artifacts/reports"),
            near_duplicate_threshold=0.9,
        )
    finally:
        os.chdir(cwd)

    assert len(payloads) == 2
    assert (report_dir / "ood_leakage_precheck_nadi2020.md").exists()
    assert (report_dir / "ood_leakage_precheck_nadi2020.json").exists()
    assert (report_dir / "ood_leakage_precheck_nadi2021.md").exists()
    assert (report_dir / "ood_leakage_precheck_nadi2021.json").exists()
    assert (report_dir / "ood_leakage_precheck_summary.md").exists()
    summary_text = (report_dir / "ood_leakage_precheck_summary.md").read_text(encoding="utf-8")
    assert "Phase 9 Part 2" in summary_text
    report_json = json.loads((report_dir / "ood_leakage_precheck_nadi2021.json").read_text(encoding="utf-8"))
    assert report_json["candidate_split"] == "NADI 2021 DA dev"
