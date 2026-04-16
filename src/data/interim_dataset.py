from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.data.validation import compute_text_fingerprint, normalize_raw_label


INTERIM_FIELDNAMES = [
    "source_dataset",
    "source_file",
    "source_id",
    "source_row_number",
    "text",
    "raw_label",
    "normalized_raw_label",
    "macro_label",
]
RAW_TO_MACRO = {
    "Saudi_Arabia": "Saudi",
    "Egypt": "Egyptian",
    "Jordan": "Levantine",
    "Lebanon": "Levantine",
    "Palestine": "Levantine",
    "Syria": "Levantine",
    "Algeria": "Maghrebi",
    "Libya": "Maghrebi",
    "Morocco": "Maghrebi",
    "Tunisia": "Maghrebi",
}
TRAIN_CORE_NAME = "NADI2023_Subtask1_TRAIN"
DEV_CORE_NAME = "NADI2023_Subtask1_DEV"
AUG_2020_NAME = "NADI2020_TWT"
AUG_2021_NAME = "NADI2021_TWT"
INTERIM_REPORT_MD = "interim_curation_report.md"
INTERIM_REPORT_JSON = "interim_curation_report.json"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: Path
    id_column: str
    text_column: str
    label_column: str


@dataclass
class Record:
    source_dataset: str
    source_file: str
    source_id: str
    source_row_number: int
    text: str
    raw_label: str
    normalized_raw_label: str
    macro_label: str | None
    text_hash: str

    def output_row(self) -> dict[str, Any]:
        return {
            "source_dataset": self.source_dataset,
            "source_file": self.source_file,
            "source_id": self.source_id,
            "source_row_number": self.source_row_number,
            "text": self.text,
            "raw_label": self.raw_label,
            "normalized_raw_label": self.normalized_raw_label,
            "macro_label": self.macro_label or "",
        }


@dataclass
class OutputStats:
    name: str
    path: str
    rows_kept: int = 0
    kept_by_source: Counter[str] = field(default_factory=Counter)
    kept_by_macro_label: Counter[str] = field(default_factory=Counter)
    dropped_by_reason: Counter[str] = field(default_factory=Counter)


@dataclass
class CurationSummary:
    outputs: dict[str, OutputStats]
    overlap_removals: dict[str, Any]
    conflict_removals: dict[str, Any]


def make_source_configs(data_root: Path) -> dict[str, SourceConfig]:
    subtask1_root = data_root / "nadi2023" / "NADI2023_Release_Train" / "Subtask1"
    return {
        TRAIN_CORE_NAME: SourceConfig(
            name=TRAIN_CORE_NAME,
            path=subtask1_root / "NADI2023_Subtask1_TRAIN.tsv",
            id_column="#1_id",
            text_column="#2_content",
            label_column="#3_label",
        ),
        DEV_CORE_NAME: SourceConfig(
            name=DEV_CORE_NAME,
            path=subtask1_root / "NADI2023_Subtask1_DEV.tsv",
            id_column="#1_id",
            text_column="#2_content",
            label_column="#3_label",
        ),
        AUG_2020_NAME: SourceConfig(
            name=AUG_2020_NAME,
            path=subtask1_root / "NADI2020-TWT.tsv",
            id_column="#1 tweet_ID",
            text_column="#2 tweet_content",
            label_column="#3 country_label",
        ),
        AUG_2021_NAME: SourceConfig(
            name=AUG_2021_NAME,
            path=subtask1_root / "NADI2021-TWT.tsv",
            id_column="#1_tweetid",
            text_column="#2_tweet",
            label_column="#3_country_label",
        ),
    }


def load_records(config: SourceConfig) -> list[Record]:
    records: list[Record] = []
    with config.path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            text = row[config.text_column]
            text_hash = compute_text_fingerprint(text.strip())
            raw_label = (row[config.label_column] or "").strip()
            normalized_raw_label = normalize_raw_label(raw_label)
            records.append(
                Record(
                    source_dataset=config.name,
                    source_file=config.path.as_posix(),
                    source_id=(row[config.id_column] or "").strip(),
                    source_row_number=row_number,
                    text=text,
                    raw_label=raw_label,
                    normalized_raw_label=normalized_raw_label,
                    macro_label=RAW_TO_MACRO.get(normalized_raw_label),
                    text_hash=text_hash,
                )
            )
    return records


def write_output_csv(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INTERIM_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(record.output_row())


def summarize_kept(records: list[Record], stats: OutputStats) -> None:
    stats.rows_kept = len(records)
    for record in records:
        stats.kept_by_source[record.source_dataset] += 1
        if record.macro_label is not None:
            stats.kept_by_macro_label[record.macro_label] += 1


def record_drop(stats: OutputStats, reason: str) -> None:
    stats.dropped_by_reason[reason] += 1


def build_train_core(records: list[Record], output_path: Path) -> tuple[list[Record], OutputStats]:
    kept: list[Record] = []
    stats = OutputStats(name="train_core", path=output_path.as_posix())
    for record in records:
        if record.macro_label is None:
            record_drop(stats, "out_of_scope_country")
            continue
        kept.append(record)
    write_output_csv(output_path, kept)
    summarize_kept(kept, stats)
    return kept, stats


def build_dev_core(
    records: list[Record],
    train_text_hashes: set[str],
    output_path: Path,
) -> tuple[list[Record], OutputStats, list[dict[str, Any]]]:
    kept: list[Record] = []
    overlap_examples: list[dict[str, Any]] = []
    stats = OutputStats(name="dev_core", path=output_path.as_posix())
    for record in records:
        if record.text_hash in train_text_hashes:
            record_drop(stats, "benchmark_exact_overlap_with_train")
            if len(overlap_examples) < 20:
                overlap_examples.append(
                    {
                        "text_hash": record.text_hash,
                        "source_dataset": record.source_dataset,
                        "source_id": record.source_id,
                        "source_row_number": record.source_row_number,
                    }
                )
            continue
        if record.macro_label is None:
            record_drop(stats, "out_of_scope_country")
            continue
        kept.append(record)
    write_output_csv(output_path, kept)
    summarize_kept(kept, stats)
    return kept, stats, overlap_examples


def find_supporting_conflicts(records: list[Record]) -> tuple[set[str], list[dict[str, Any]]]:
    by_text_hash: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_text_hash[record.text_hash].append(record)

    conflict_hashes: set[str] = set()
    conflict_examples: list[dict[str, Any]] = []
    for text_hash, bucket in by_text_hash.items():
        labels = sorted({record.normalized_raw_label for record in bucket})
        if len(labels) <= 1:
            continue
        conflict_hashes.add(text_hash)
        if len(conflict_examples) < 20:
            conflict_examples.append(
                {
                    "text_hash": text_hash,
                    "normalized_raw_labels": labels,
                    "occurrences": [
                        {
                            "source_dataset": record.source_dataset,
                            "source_id": record.source_id,
                            "source_row_number": record.source_row_number,
                            "raw_label": record.raw_label,
                            "normalized_raw_label": record.normalized_raw_label,
                        }
                        for record in bucket[:6]
                    ],
                }
            )
    conflict_examples.sort(key=lambda item: (item["text_hash"]))
    return conflict_hashes, conflict_examples


def build_train_aug_candidates(
    records: list[Record],
    train_core_text_hashes: set[str],
    output_path: Path,
) -> tuple[list[Record], OutputStats, dict[str, Any], dict[str, Any]]:
    kept: list[Record] = []
    stats = OutputStats(name="train_aug_candidates", path=output_path.as_posix())
    conflict_hashes, conflict_examples = find_supporting_conflicts(records)
    train_overlap_hashes: set[str] = set()
    train_overlap_examples: list[dict[str, Any]] = []

    for record in records:
        if record.text_hash in conflict_hashes:
            record_drop(stats, "conflicting_supporting_label")
            continue
        if record.text_hash in train_core_text_hashes:
            train_overlap_hashes.add(record.text_hash)
            record_drop(stats, "overlap_with_train_core")
            if len(train_overlap_examples) < 20:
                train_overlap_examples.append(
                    {
                        "text_hash": record.text_hash,
                        "source_dataset": record.source_dataset,
                        "source_id": record.source_id,
                        "source_row_number": record.source_row_number,
                        "normalized_raw_label": record.normalized_raw_label,
                    }
                )
            continue
        if record.macro_label is None:
            record_drop(stats, "out_of_scope_country")
            continue
        kept.append(record)

    write_output_csv(output_path, kept)
    summarize_kept(kept, stats)
    overlap_summary = {
        "train_core_overlap_text_hash_count": len(train_overlap_hashes),
        "train_core_overlap_rows_removed": stats.dropped_by_reason["overlap_with_train_core"],
        "train_core_overlap_examples": train_overlap_examples,
    }
    conflict_summary = {
        "supporting_conflict_text_hash_count": len(conflict_hashes),
        "supporting_conflict_rows_removed": stats.dropped_by_reason["conflicting_supporting_label"],
        "supporting_conflict_examples": conflict_examples,
    }
    return kept, stats, overlap_summary, conflict_summary


def counter_to_sorted_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def build_curation_summary(
    train_stats: OutputStats,
    dev_stats: OutputStats,
    aug_stats: OutputStats,
    dev_overlap_examples: list[dict[str, Any]],
    aug_overlap_summary: dict[str, Any],
    aug_conflict_summary: dict[str, Any],
) -> dict[str, Any]:
    outputs = {
        stats.name: {
            "path": stats.path,
            "rows_kept": stats.rows_kept,
            "kept_by_source": counter_to_sorted_dict(stats.kept_by_source),
            "kept_by_macro_label": counter_to_sorted_dict(stats.kept_by_macro_label),
            "dropped_by_reason": counter_to_sorted_dict(stats.dropped_by_reason),
        }
        for stats in (train_stats, dev_stats, aug_stats)
    }
    overall_kept_by_source: Counter[str] = Counter()
    overall_kept_by_macro_label: Counter[str] = Counter()
    overall_dropped_by_reason: Counter[str] = Counter()
    for stats in (train_stats, dev_stats, aug_stats):
        overall_kept_by_source.update(stats.kept_by_source)
        overall_kept_by_macro_label.update(stats.kept_by_macro_label)
        overall_dropped_by_reason.update(stats.dropped_by_reason)

    return {
        "outputs": outputs,
        "overall_kept_by_source": counter_to_sorted_dict(overall_kept_by_source),
        "overall_kept_by_macro_label": counter_to_sorted_dict(overall_kept_by_macro_label),
        "overall_dropped_by_reason": counter_to_sorted_dict(overall_dropped_by_reason),
        "overlap_removals": {
            "benchmark_train_dev_exact_overlap_text_hash_count": len(dev_overlap_examples),
            "dev_rows_removed_for_benchmark_overlap": dev_stats.dropped_by_reason[
                "benchmark_exact_overlap_with_train"
            ],
            "benchmark_overlap_examples": dev_overlap_examples,
            **aug_overlap_summary,
        },
        "conflict_removals": {
            "label_normalization": {
                "UAE": "UAE",
                "United_Arab_Emirates": "UAE",
            },
            **aug_conflict_summary,
        },
    }


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def render_counter_table(title: str, counter: dict[str, int]) -> list[str]:
    lines = [f"## {title}", ""]
    if not counter:
        lines.append("No rows.")
        lines.append("")
        return lines
    lines.extend(["| Key | Count |", "| --- | ---: |"])
    for key, count in counter.items():
        lines.append(f"| `{key}` | {count} |")
    lines.append("")
    return lines


def render_output_section(name: str, payload: dict[str, Any]) -> list[str]:
    lines = [
        f"## {name}",
        "",
        f"- Output path: `{payload['path']}`",
        f"- Rows kept: `{payload['rows_kept']}`",
        "",
    ]
    lines.extend(render_counter_table(f"{name} Kept by Source", payload["kept_by_source"]))
    lines.extend(render_counter_table(f"{name} Kept by Macro Label", payload["kept_by_macro_label"]))
    lines.extend(render_counter_table(f"{name} Dropped by Reason", payload["dropped_by_reason"]))
    return lines


def render_examples_table(title: str, examples: list[dict[str, Any]], row_formatter: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not examples:
        lines.append("No examples recorded.")
        lines.append("")
        return lines
    lines.extend(row_formatter.splitlines())
    for example in examples:
        lines.append(example["rendered"])
    lines.append("")
    return lines


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Interim Curation Report",
        "",
        "This report describes leakage-aware interim dataset generation from the local NADI 2023 Subtask 1 benchmark anchor and bundled supporting sources only.",
        "",
    ]
    for name, section in payload["outputs"].items():
        lines.extend(render_output_section(name, section))
    lines.extend(render_counter_table("Overall Kept by Source", payload["overall_kept_by_source"]))
    lines.extend(render_counter_table("Overall Kept by Macro Label", payload["overall_kept_by_macro_label"]))
    lines.extend(render_counter_table("Overall Dropped by Reason", payload["overall_dropped_by_reason"]))

    overlap = payload["overlap_removals"]
    lines.extend(
        [
            "## Overlap Removals",
            "",
            f"- Benchmark train/dev exact overlap text hashes: `{overlap['benchmark_train_dev_exact_overlap_text_hash_count']}`",
            f"- Dev rows removed for benchmark overlap: `{overlap['dev_rows_removed_for_benchmark_overlap']}`",
            f"- Aug candidate text hashes already present in train_core: `{overlap['train_core_overlap_text_hash_count']}`",
            f"- Aug candidate rows removed for train_core overlap: `{overlap['train_core_overlap_rows_removed']}`",
            "",
            "### Benchmark Overlap Examples",
            "",
        ]
    )
    if overlap["benchmark_overlap_examples"]:
        lines.extend(["| Text hash | Source dataset | Source ID | Row |", "| --- | --- | --- | ---: |"])
        for example in overlap["benchmark_overlap_examples"]:
            lines.append(
                f"| `{example['text_hash']}` | `{example['source_dataset']}` | "
                f"`{example['source_id']}` | {example['source_row_number']} |"
            )
    else:
        lines.append("No benchmark overlap examples recorded.")
    lines.extend(["", "### Train-Core Overlap Examples", ""])
    if overlap["train_core_overlap_examples"]:
        lines.extend(["| Text hash | Source dataset | Source ID | Row | Label |", "| --- | --- | --- | ---: | --- |"])
        for example in overlap["train_core_overlap_examples"]:
            lines.append(
                f"| `{example['text_hash']}` | `{example['source_dataset']}` | `{example['source_id']}` | "
                f"{example['source_row_number']} | `{example['normalized_raw_label']}` |"
            )
    else:
        lines.append("No train_core overlap examples recorded.")
    lines.append("")

    conflict = payload["conflict_removals"]
    lines.extend(
        [
            "## Conflict Removals",
            "",
            f"- Supporting conflict text hashes: `{conflict['supporting_conflict_text_hash_count']}`",
            f"- Supporting rows removed for conflict: `{conflict['supporting_conflict_rows_removed']}`",
            "- Label normalization used for conflict accounting:",
            f"  - `UAE` -> `{conflict['label_normalization']['UAE']}`",
            "  - `United_Arab_Emirates` -> `UAE`",
            "",
            "### Conflict Examples",
            "",
        ]
    )
    if conflict["supporting_conflict_examples"]:
        lines.extend(["| Text hash | Normalized raw labels | Occurrences |", "| --- | --- | ---: |"])
        for example in conflict["supporting_conflict_examples"]:
            labels = ", ".join(f"`{label}`" for label in example["normalized_raw_labels"])
            lines.append(
                f"| `{example['text_hash']}` | {labels} | {len(example['occurrences'])} |"
            )
    else:
        lines.append("No supporting conflict examples recorded.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_interim_datasets(data_root: Path, interim_dir: Path, report_dir: Path) -> dict[str, Path]:
    configs = make_source_configs(data_root)
    train_records = load_records(configs[TRAIN_CORE_NAME])
    dev_records = load_records(configs[DEV_CORE_NAME])
    aug_records = load_records(configs[AUG_2020_NAME]) + load_records(configs[AUG_2021_NAME])

    train_core_path = interim_dir / "train_core.csv"
    dev_core_path = interim_dir / "dev_core.csv"
    train_aug_path = interim_dir / "train_aug_candidates.csv"
    train_core, train_stats = build_train_core(train_records, train_core_path)
    train_text_hashes = {record.text_hash for record in train_core}
    raw_train_text_hashes = {record.text_hash for record in train_records}
    dev_core, dev_stats, dev_overlap_examples = build_dev_core(
        dev_records,
        raw_train_text_hashes,
        dev_core_path,
    )
    _, aug_stats, aug_overlap_summary, aug_conflict_summary = build_train_aug_candidates(
        aug_records,
        train_text_hashes,
        train_aug_path,
    )

    payload = build_curation_summary(
        train_stats,
        dev_stats,
        aug_stats,
        dev_overlap_examples,
        aug_overlap_summary,
        aug_conflict_summary,
    )
    report_json = report_dir / INTERIM_REPORT_JSON
    report_md = report_dir / INTERIM_REPORT_MD
    write_json_report(report_json, payload)
    write_markdown_report(report_md, payload)
    return {
        "train_core": train_core_path,
        "dev_core": dev_core_path,
        "train_aug_candidates": train_aug_path,
        "report_json": report_json,
        "report_markdown": report_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate leakage-aware interim datasets from local raw data.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw source data.",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory where interim datasets will be written.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory where curation reports will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = generate_interim_datasets(args.data_root, args.interim_dir, args.report_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
