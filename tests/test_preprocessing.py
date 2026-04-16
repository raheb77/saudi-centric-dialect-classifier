from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "data" / "preprocessing.py"
SPEC = importlib.util.spec_from_file_location("project_preprocessing", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

preprocess_csv_file = MODULE.preprocess_csv_file
preprocess_text = MODULE.preprocess_text
preprocess_interim_files = MODULE.preprocess_interim_files


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


def test_preprocess_text_applies_required_rules() -> None:
    text = "USER آلسَّلَام @user مرحببببااا URL https://x.com #السعودية coolلل ى 😀"

    processed = preprocess_text(text)

    assert processed == "<USER> السلام <USER> مرحبباا السعودية coolلل ي 😀"


def test_preprocess_csv_file_preserves_lineage_and_adds_traceability(tmp_path: Path) -> None:
    input_path = tmp_path / "data" / "interim" / "train_core.csv"
    output_path = tmp_path / "data" / "processed" / "train_core.csv"
    write_csv(
        input_path,
        [
            "source_dataset",
            "source_file",
            "source_id",
            "source_row_number",
            "country_label",
            "macro_label",
            "text",
        ],
        [
            {
                "source_dataset": "NADI2023_Subtask1_TRAIN",
                "source_file": "data/raw/example.tsv",
                "source_id": "t1",
                "source_row_number": "2",
                "country_label": "Saudi_Arabia",
                "macro_label": "Saudi",
                "text": "مرحببببا @user #الرياض https://t.co/x",
            }
        ],
    )

    row_count = preprocess_csv_file(input_path, output_path)
    rows = read_csv_rows(output_path)

    assert row_count == 1
    assert rows == [
        {
            "source_dataset": "NADI2023_Subtask1_TRAIN",
            "source_file": "data/raw/example.tsv",
            "source_id": "t1",
                "source_row_number": "2",
                "country_label": "Saudi_Arabia",
                "macro_label": "Saudi",
                "original_text": "مرحببببا @user #الرياض https://t.co/x",
                "processed_text": "مرحببا <USER> الرياض",
            }
        ]


def test_preprocess_interim_files_writes_all_expected_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "data" / "interim"
    output_dir = tmp_path / "data" / "processed"
    fieldnames = [
        "source_dataset",
        "source_file",
        "source_id",
        "source_row_number",
        "raw_label",
        "normalized_raw_label",
        "macro_label",
        "text",
    ]
    row = {
        "source_dataset": "dataset",
        "source_file": "file.tsv",
        "source_id": "1",
        "source_row_number": "2",
        "raw_label": "Egypt",
        "normalized_raw_label": "Egypt",
        "macro_label": "Egyptian",
        "text": "اهلااا #مصر",
    }
    for filename in ("train_core.csv", "dev_core.csv", "train_aug_candidates.csv"):
        write_csv(input_dir / filename, fieldnames, [row])

    outputs = preprocess_interim_files(input_dir, output_dir)

    assert set(outputs) == {"train_core.csv", "dev_core.csv", "train_aug_candidates.csv"}
    for filename in outputs:
        rows = read_csv_rows(output_dir / filename)
        assert rows[0]["processed_text"] == "اهلاا مصر"


def test_preprocess_text_handles_source_placeholders() -> None:
    assert preprocess_text("USER اهلين URL") == "<USER> اهلين"
