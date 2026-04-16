from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


INPUT_FILENAMES = (
    "train_core.csv",
    "dev_core.csv",
    "train_aug_candidates.csv",
)
URL_RE = re.compile(r"(?:https?://|www\.|pic\.twitter\.com/)\S*", re.IGNORECASE)
URL_TOKEN_RE = re.compile(r"(?<!\w)URL(?!\w)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
MENTION_RE = re.compile(r"@[\w_]+", re.UNICODE)
USER_TOKEN_RE = re.compile(r"(?<![\w<])USER(?![\w>])")
ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ALEF_VARIANTS_RE = re.compile(r"[\u0622\u0623\u0625\u0671]")
FINAL_ALEF_MAQSURA_RE = re.compile(r"ى\b")
TATWEEL_RE = re.compile(r"\u0640+")
ELONGATION_RE = re.compile(r"([A-Za-z\u0621-\u064A])\1{2,}")
WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    value = text
    protected_emails: list[str] = []

    def protect_email(match: re.Match[str]) -> str:
        protected_emails.append(match.group(0))
        return f" __EMAIL_{len(protected_emails) - 1}__ "

    value = EMAIL_RE.sub(protect_email, value)
    value = URL_RE.sub(" ", value)
    value = URL_TOKEN_RE.sub(" ", value)
    value = MENTION_RE.sub(" <USER> ", value)
    value = USER_TOKEN_RE.sub(" <USER> ", value)
    value = value.replace("#", "")
    value = ARABIC_DIACRITICS_RE.sub("", value)
    value = ALEF_VARIANTS_RE.sub("ا", value)
    value = FINAL_ALEF_MAQSURA_RE.sub("ي", value)
    value = TATWEEL_RE.sub("", value)
    value = ELONGATION_RE.sub(r"\1\1", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    for index, email in enumerate(protected_emails):
        value = value.replace(f"__EMAIL_{index}__", email)
    return value


def build_output_fieldnames(input_fieldnames: list[str], text_column: str) -> list[str]:
    preserved = [field for field in input_fieldnames if field != text_column]
    return preserved + ["original_text", "processed_text"]


def preprocess_csv_file(input_path: Path, output_path: Path, *, text_column: str = "text") -> int:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_fieldnames = reader.fieldnames or []
        if text_column not in input_fieldnames:
            raise ValueError(f"Missing required text column `{text_column}` in {input_path.as_posix()}")
        output_fieldnames = build_output_fieldnames(input_fieldnames, text_column)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with output_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=output_fieldnames)
            writer.writeheader()
            for row in reader:
                original_text = row.get(text_column, "")
                output_row = {field: row.get(field, "") for field in output_fieldnames}
                output_row["original_text"] = original_text
                output_row["processed_text"] = preprocess_text(original_text)
                writer.writerow(output_row)
                row_count += 1
    return row_count


def preprocess_interim_files(input_dir: Path, output_dir: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for filename in INPUT_FILENAMES:
        input_path = input_dir / filename
        output_path = output_dir / filename
        preprocess_csv_file(input_path, output_path)
        outputs[filename] = output_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess interim CSV files into data/processed.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory containing interim CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed CSV files will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = preprocess_interim_files(args.input_dir, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
