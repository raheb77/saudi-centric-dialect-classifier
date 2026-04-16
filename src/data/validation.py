from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any


LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024
REPORT_JSON_NAME = "data_validation_report.json"
REPORT_MARKDOWN_NAME = "data_validation_summary.md"
REPORT_CSV_NAME = "data_validation_files.csv"
FILE_GROUP_ORDER = (
    "benchmark_anchor",
    "canonical_supporting",
    "provenance_aux_eval",
    "unlabeled_id_only",
    "out_of_scope",
)
FILE_GROUP_TITLES = {
    "benchmark_anchor": "Benchmark Anchor Files",
    "canonical_supporting": "Canonical Supporting Files",
    "provenance_aux_eval": "Provenance / Auxiliary Evaluation Files",
    "unlabeled_id_only": "Unlabeled ID-Only Files",
    "out_of_scope": "Out-of-Scope Files",
}
LABEL_NORMALIZATION = {
    "UAE": "UAE",
    "United_Arab_Emirates": "UAE",
}


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    required_columns: tuple[str, ...]
    text_columns: tuple[str, ...] = ()
    class_columns: tuple[str, ...] = ()


@dataclass
class TextColumnStats:
    empty_count: int = 0
    short_count: int = 0
    sample_empty_rows: list[int] = field(default_factory=list)
    sample_short_rows: list[int] = field(default_factory=list)


@dataclass
class ValidationResult:
    path: str
    schema_name: str
    file_group: str
    file_size_bytes: int
    columns: list[str]
    missing_columns: list[str]
    primary_text_column: str = ""
    primary_label_column: str = ""
    row_count: int = 0
    duplicate_row_count: int = 0
    duplicate_check_mode: str = "full"
    duplicate_row_examples: list[int] = field(default_factory=list)
    text_stats: dict[str, TextColumnStats] = field(default_factory=dict)
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def summary_row(self) -> dict[str, Any]:
        total_empty = sum(stats.empty_count for stats in self.text_stats.values())
        total_short = sum(stats.short_count for stats in self.text_stats.values())
        class_columns = ",".join(self.class_counts.keys())
        text_columns = ",".join(self.text_stats.keys())
        missing_columns = ",".join(self.missing_columns)
        return {
            "path": self.path,
            "schema_name": self.schema_name,
            "file_group": self.file_group,
            "row_count": self.row_count,
            "duplicate_row_count": self.duplicate_row_count,
            "duplicate_check_mode": self.duplicate_check_mode,
            "empty_text_count": total_empty,
            "short_text_count": total_short,
            "missing_columns": missing_columns,
            "text_columns": text_columns,
            "class_columns": class_columns,
        }


SCHEMA_SPECS: tuple[SchemaSpec, ...] = (
    SchemaSpec(
        name="madar_2018",
        required_columns=("sentID.BTEC", "split", "lang", "sent", "city", "country"),
        text_columns=("sent",),
        class_columns=("country", "city"),
    ),
    SchemaSpec(
        name="nadi2020_labeled",
        required_columns=("#1 tweet_ID", "#2 tweet_content", "#3 country_label", "#4 province_label"),
        text_columns=("#2 tweet_content",),
        class_columns=("#3 country_label", "#4 province_label"),
    ),
    SchemaSpec(
        name="nadi2021_labeled",
        required_columns=("#1_tweetid", "#2_tweet", "#3_country_label", "#4_province_label"),
        text_columns=("#2_tweet",),
        class_columns=("#3_country_label", "#4_province_label"),
    ),
    SchemaSpec(
        name="nadi2023_st1_labeled",
        required_columns=("#1_id", "#2_content", "#3_label"),
        text_columns=("#2_content",),
        class_columns=("#3_label",),
    ),
    SchemaSpec(
        name="nadi2023_st1_labeled_checkpoint",
        required_columns=("id", "content", "label"),
        text_columns=("content",),
        class_columns=("label",),
    ),
    SchemaSpec(
        name="nadi2020_unlabeled_ids",
        required_columns=("#1 tweet_ID",),
    ),
    SchemaSpec(
        name="nadi2020_test_unlabeled",
        required_columns=("#1 tweet_ID", "#2 tweet_content"),
        text_columns=("#2 tweet_content",),
    ),
    SchemaSpec(
        name="nadi2023_st1_test_unlabeled",
        required_columns=("#1_id", "#2_content"),
        text_columns=("#2_content",),
    ),
    SchemaSpec(
        name="nadi2023_mt_labeled",
        required_columns=("id", "dialect_id", "source_dialect", "target_msa"),
        text_columns=("source_dialect", "target_msa"),
        class_columns=("dialect_id",),
    ),
    SchemaSpec(
        name="nadi2023_mt_test_unlabeled",
        required_columns=("#1_id", "#2_dialect_id", "#3_source_dialect"),
        text_columns=("#3_source_dialect",),
        class_columns=("#2_dialect_id",),
    ),
)


BENCHMARK_ANCHOR_SUFFIXES = {
    "nadi2023/NADI2023_Release_Train/Subtask1/NADI2023_Subtask1_TRAIN.tsv",
    "nadi2023/NADI2023_Release_Train/Subtask1/NADI2023_Subtask1_DEV.tsv",
}
CANONICAL_SUPPORTING_SUFFIXES = {
    "nadi2023/NADI2023_Release_Train/Subtask1/NADI2020-TWT.tsv",
    "nadi2023/NADI2023_Release_Train/Subtask1/NADI2021-TWT.tsv",
}
PROVENANCE_AUX_EVAL_SUFFIXES = {
    "nadi2020/NADI_release/train_labeled.tsv",
    "nadi2020/NADI_release/dev_labeled.tsv",
    "nadi2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_train_labeled.tsv",
    "nadi2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_dev_labeled.tsv",
}


def match_schema(columns: list[str]) -> SchemaSpec:
    column_set = set(columns)
    best_match: SchemaSpec | None = None
    for spec in SCHEMA_SPECS:
        if set(spec.required_columns).issubset(column_set):
            if best_match is None or len(spec.required_columns) > len(best_match.required_columns):
                best_match = spec
    return best_match or SchemaSpec(name="unknown", required_columns=tuple())


def classify_file_group(path: Path, schema_name: str) -> str:
    normalized_path = path.as_posix()
    if any(normalized_path.endswith(suffix) for suffix in BENCHMARK_ANCHOR_SUFFIXES):
        return "benchmark_anchor"
    if any(normalized_path.endswith(suffix) for suffix in CANONICAL_SUPPORTING_SUFFIXES):
        return "canonical_supporting"
    if any(normalized_path.endswith(suffix) for suffix in PROVENANCE_AUX_EVAL_SUFFIXES):
        return "provenance_aux_eval"
    if schema_name == "nadi2020_unlabeled_ids":
        return "unlabeled_id_only"
    return "out_of_scope"


def normalize_raw_label(label: str) -> str:
    cleaned = label.strip()
    return LABEL_NORMALIZATION.get(cleaned, cleaned)


def compute_row_fingerprint(values: list[str]) -> str:
    payload = "\t".join(values).encode("utf-8", "surrogatepass")
    return hashlib.sha1(payload).hexdigest()


def compute_text_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def token_count(text: str) -> int:
    return len(text.split())


class DuplicateTracker:
    def add(self, fingerprint: str) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        return None


class InMemoryDuplicateTracker(DuplicateTracker):
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def add(self, fingerprint: str) -> bool:
        if fingerprint in self._seen:
            return True
        self._seen.add(fingerprint)
        return False


class SqliteDuplicateTracker(DuplicateTracker):
    def __init__(self) -> None:
        temp_dir = Path(tempfile.mkdtemp(prefix="dialect-validator-"))
        self._db_path = temp_dir / "seen.sqlite3"
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode = OFF")
        self._conn.execute("PRAGMA synchronous = OFF")
        self._conn.execute("PRAGMA temp_store = MEMORY")
        self._conn.execute("CREATE TABLE seen (fingerprint TEXT PRIMARY KEY)")
        self._conn.commit()
        self._conn.execute("BEGIN")

    def add(self, fingerprint: str) -> bool:
        cursor = self._conn.execute(
            "INSERT OR IGNORE INTO seen (fingerprint) VALUES (?)",
            (fingerprint,),
        )
        return cursor.rowcount == 0

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()
        if self._db_path.exists():
            self._db_path.unlink()
        parent = self._db_path.parent
        if parent.exists():
            parent.rmdir()


def build_duplicate_tracker(file_size_bytes: int) -> DuplicateTracker:
    if file_size_bytes >= LARGE_FILE_THRESHOLD_BYTES:
        return SqliteDuplicateTracker()
    return InMemoryDuplicateTracker()


def validate_tsv_file(path: Path) -> ValidationResult:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        schema = match_schema(columns)
        missing_columns = sorted(set(schema.required_columns) - set(columns))
        text_stats = {column: TextColumnStats() for column in schema.text_columns if column in columns}
        class_counters = {column: Counter() for column in schema.class_columns if column in columns}
        primary_text_column = next((column for column in schema.text_columns if column in columns), "")
        primary_label_column = next((column for column in schema.class_columns if column in columns), "")

        result = ValidationResult(
            path=path.as_posix(),
            schema_name=schema.name,
            file_group=classify_file_group(path, schema.name),
            file_size_bytes=path.stat().st_size,
            columns=columns,
            missing_columns=missing_columns,
            primary_text_column=primary_text_column,
            primary_label_column=primary_label_column,
            text_stats=text_stats,
        )

        should_skip_duplicates = (
            result.file_size_bytes >= LARGE_FILE_THRESHOLD_BYTES
            and not text_stats
            and not class_counters
        )
        if should_skip_duplicates:
            result.duplicate_check_mode = "skipped_large_id_only"
            duplicate_tracker: DuplicateTracker | None = None
        else:
            duplicate_tracker = build_duplicate_tracker(result.file_size_bytes)

        try:
            for row_number, row in enumerate(reader, start=2):
                result.row_count += 1

                if duplicate_tracker is not None:
                    values = [(row.get(column) or "") for column in columns]
                    fingerprint = compute_row_fingerprint(values)
                    if duplicate_tracker.add(fingerprint):
                        result.duplicate_row_count += 1
                        if len(result.duplicate_row_examples) < 5:
                            result.duplicate_row_examples.append(row_number)

                for column, counter in class_counters.items():
                    value = (row.get(column) or "").strip() or "<EMPTY>"
                    counter[value] += 1

                for column, stats in text_stats.items():
                    raw_text = row.get(column) or ""
                    text = raw_text.strip()
                    if not text:
                        stats.empty_count += 1
                        if len(stats.sample_empty_rows) < 5:
                            stats.sample_empty_rows.append(row_number)
                    if token_count(text) < 3:
                        stats.short_count += 1
                        if len(stats.sample_short_rows) < 5:
                            stats.sample_short_rows.append(row_number)
        finally:
            if duplicate_tracker is not None:
                duplicate_tracker.close()

    result.class_counts = {
        column: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        for column, counter in class_counters.items()
    }
    return result


def validate_tsv_tree(data_root: Path) -> list[ValidationResult]:
    return [validate_tsv_file(path) for path in sorted(data_root.rglob("*.tsv"))]


def summarize_group(results: list[ValidationResult]) -> dict[str, int]:
    return {
        "file_count": len(results),
        "row_count": sum(result.row_count for result in results),
        "files_with_duplicates": sum(1 for result in results if result.duplicate_row_count),
        "files_with_empty_text": sum(
            1 for result in results if any(stats.empty_count for stats in result.text_stats.values())
        ),
        "files_with_short_text": sum(
            1 for result in results if any(stats.short_count for stats in result.text_stats.values())
        ),
    }


def collect_text_occurrences(result: ValidationResult, *, normalize_labels: bool) -> dict[str, list[dict[str, Any]]]:
    occurrences: dict[str, list[dict[str, Any]]] = {}
    path = Path(result.path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(result.primary_text_column) or "").strip()
            if not text:
                continue
            label = (row.get(result.primary_label_column) or "").strip() or "<EMPTY>"
            stored_label = normalize_raw_label(label) if normalize_labels else label
            text_hash = compute_text_fingerprint(text)
            occurrences.setdefault(text_hash, []).append(
                {
                    "path": result.path,
                    "row_number": row_number,
                    "label": stored_label,
                    "raw_label": label,
                }
            )
    return occurrences


def analyze_benchmark_safety(results: list[ValidationResult]) -> dict[str, Any]:
    benchmark_results = [
        result
        for result in results
        if result.file_group == "benchmark_anchor"
        and result.primary_text_column
        and result.primary_label_column
    ]
    canonical_supporting_results = [
        result
        for result in results
        if result.file_group == "canonical_supporting"
        and result.primary_text_column
        and result.primary_label_column
    ]

    benchmark_relevant_results = benchmark_results + canonical_supporting_results
    benchmark_text_occurrences: dict[str, list[dict[str, Any]]] = {}
    pairwise_counts: Counter[tuple[str, str]] = Counter()
    for result in benchmark_relevant_results:
        file_occurrences = collect_text_occurrences(result, normalize_labels=False)
        for text_hash, occurrences in file_occurrences.items():
            benchmark_text_occurrences.setdefault(text_hash, []).extend(occurrences)

    benchmark_relevant_texts_in_multiple_files = 0
    for text_hash, occurrences in benchmark_text_occurrences.items():
        unique_files = sorted({occurrence["path"] for occurrence in occurrences})
        if len(unique_files) > 1:
            benchmark_relevant_texts_in_multiple_files += 1
            for pair in combinations(unique_files, 2):
                pairwise_counts[pair] += 1

    benchmark_overlap_examples: list[dict[str, Any]] = []
    benchmark_overlap_count = 0
    if len(benchmark_results) >= 2:
        benchmark_indexes = {
            result.path: collect_text_occurrences(result, normalize_labels=False)
            for result in benchmark_results
        }
        for result_a, result_b in combinations(benchmark_results, 2):
            shared_hashes = sorted(
                set(benchmark_indexes[result_a.path]).intersection(benchmark_indexes[result_b.path])
            )
            benchmark_overlap_count += len(shared_hashes)
            for text_hash in shared_hashes[:10]:
                if len(benchmark_overlap_examples) >= 20:
                    break
                benchmark_overlap_examples.append(
                    {
                        "text_hash": text_hash,
                        "files": [result_a.path, result_b.path],
                        "sample_rows": (
                            benchmark_indexes[result_a.path][text_hash][:2]
                            + benchmark_indexes[result_b.path][text_hash][:2]
                        ),
                    }
                )

    supporting_text_occurrences: dict[str, list[dict[str, Any]]] = {}
    for result in canonical_supporting_results:
        file_occurrences = collect_text_occurrences(result, normalize_labels=True)
        for text_hash, occurrences in file_occurrences.items():
            supporting_text_occurrences.setdefault(text_hash, []).extend(occurrences)

    supporting_conflict_cases: list[dict[str, Any]] = []
    supporting_overlap_texts = 0
    for text_hash, occurrences in supporting_text_occurrences.items():
        unique_files = sorted({occurrence["path"] for occurrence in occurrences})
        if len(unique_files) > 1:
            supporting_overlap_texts += 1
        unique_labels = sorted({occurrence["label"] for occurrence in occurrences})
        if len(unique_files) > 1 and len(unique_labels) > 1:
            supporting_conflict_cases.append(
                {
                    "text_hash": text_hash,
                    "normalized_labels": unique_labels,
                    "files": unique_files,
                    "occurrence_count": len(occurrences),
                    "sample_rows": occurrences[:6],
                }
            )

    pairwise_overlap_counts = [
        {
            "file_a": file_a,
            "file_b": file_b,
            "shared_text_count": count,
        }
        for (file_a, file_b), count in sorted(
            pairwise_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]
    supporting_conflict_cases.sort(
        key=lambda case: (-len(case["files"]), -case["occurrence_count"], case["text_hash"])
    )

    return {
        "benchmark_anchor_files": [result.path for result in benchmark_results],
        "canonical_supporting_files": [result.path for result in canonical_supporting_results],
        "benchmark_relevant_file_count": len(benchmark_relevant_results),
        "benchmark_relevant_unique_text_count": len(benchmark_text_occurrences),
        "benchmark_relevant_texts_in_multiple_files": benchmark_relevant_texts_in_multiple_files,
        "benchmark_relevant_pairwise_overlap_counts": pairwise_overlap_counts,
        "benchmark_train_dev_exact_overlap_count": benchmark_overlap_count,
        "benchmark_train_dev_overlap_examples": benchmark_overlap_examples,
        "supporting_overlap_text_count": supporting_overlap_texts,
        "supporting_conflict_case_count": len(supporting_conflict_cases),
        "supporting_conflict_examples": supporting_conflict_cases[:20],
        "label_normalization": LABEL_NORMALIZATION,
    }


def build_report_payload(results: list[ValidationResult], data_root: Path) -> dict[str, Any]:
    total_rows = sum(result.row_count for result in results)
    files_with_duplicates = sum(1 for result in results if result.duplicate_row_count)
    files_with_empty_text = sum(
        1 for result in results if any(stats.empty_count for stats in result.text_stats.values())
    )
    files_with_short_text = sum(
        1 for result in results if any(stats.short_count for stats in result.text_stats.values())
    )
    grouped_results = {
        group: [result for result in results if result.file_group == group]
        for group in FILE_GROUP_ORDER
    }
    group_totals = {
        group: summarize_group(grouped_results[group])
        for group in FILE_GROUP_ORDER
    }
    summary_totals = {
        "benchmark_anchor_rows_scanned": group_totals["benchmark_anchor"]["row_count"],
        "canonical_supporting_rows_scanned": group_totals["canonical_supporting"]["row_count"],
        "provenance_aux_eval_rows_scanned": group_totals["provenance_aux_eval"]["row_count"],
        "unlabeled_id_only_rows_scanned": group_totals["unlabeled_id_only"]["row_count"],
        "out_of_scope_rows_scanned": group_totals["out_of_scope"]["row_count"],
    }
    return {
        "data_root": data_root.as_posix(),
        "files_scanned": len(results),
        "total_rows_scanned": total_rows,
        "files_with_duplicates": files_with_duplicates,
        "files_with_empty_text": files_with_empty_text,
        "files_with_short_text": files_with_short_text,
        "summary_totals": summary_totals,
        "group_totals": group_totals,
        "benchmark_safety": analyze_benchmark_safety(results),
        "results": [asdict(result) for result in results],
    }


def write_json_report(payload: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / REPORT_JSON_NAME
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def write_csv_report(results: list[ValidationResult], output_dir: Path) -> Path:
    report_path = output_dir / REPORT_CSV_NAME
    fieldnames = [
        "path",
        "schema_name",
        "file_group",
        "row_count",
        "duplicate_row_count",
        "duplicate_check_mode",
        "empty_text_count",
        "short_text_count",
        "missing_columns",
        "text_columns",
        "class_columns",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.summary_row())
    return report_path


def render_group_table(results: list[ValidationResult]) -> list[str]:
    lines = [
        "| File | Schema | Rows | Duplicates | Duplicate check | Empty texts | Short texts | Missing columns |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for result in results:
        total_empty = sum(stats.empty_count for stats in result.text_stats.values())
        total_short = sum(stats.short_count for stats in result.text_stats.values())
        missing = ", ".join(result.missing_columns) or "-"
        lines.append(
            f"| `{result.path}` | `{result.schema_name}` | {result.row_count} | "
            f"{result.duplicate_row_count} | `{result.duplicate_check_mode}` | "
            f"{total_empty} | {total_short} | {missing} |"
        )
    if len(lines) == 2:
        lines.append("| _none_ | - | 0 | 0 | - | 0 | 0 | - |")
    return lines


def render_overlap_section(payload: dict[str, Any]) -> list[str]:
    overlap = payload["benchmark_safety"]
    lines = [
        "## Benchmark Safety Checks",
        "",
        "- Scope: benchmark anchor plus canonical supporting sources only",
        "- Provenance / auxiliary evaluation files are excluded from canonical leakage accounting",
        f"- Benchmark-relevant files analyzed: `{overlap['benchmark_relevant_file_count']}`",
        f"- Benchmark-relevant unique exact text hashes: `{overlap['benchmark_relevant_unique_text_count']}`",
        f"- Benchmark-relevant exact text hashes appearing in more than one file: `{overlap['benchmark_relevant_texts_in_multiple_files']}`",
        f"- Exact train/dev overlaps inside the benchmark anchor: `{overlap['benchmark_train_dev_exact_overlap_count']}`",
        f"- Supporting-source same-text conflicting-label cases: `{overlap['supporting_conflict_case_count']}`",
        "- Policy: exact train/dev overlaps in the benchmark anchor should be removed from dev before benchmark-style evaluation",
        "- Policy: same-text conflicting-label cases across the canonical supporting sources should be dropped from augmentation candidates",
        "- Leakage accounting normalizes `UAE` and `United_Arab_Emirates` to `UAE`",
        "",
        "### Benchmark-Relevant Pairwise Overlap Counts",
        "",
    ]

    if overlap["benchmark_relevant_pairwise_overlap_counts"]:
        lines.extend(
            [
                "| File A | File B | Shared exact texts |",
                "| --- | --- | ---: |",
            ]
        )
        for item in overlap["benchmark_relevant_pairwise_overlap_counts"]:
            lines.append(
                f"| `{item['file_a']}` | `{item['file_b']}` | {item['shared_text_count']} |"
            )
    else:
        lines.append("No cross-file text overlaps found in the benchmark-relevant files.")

    lines.extend(["", "### Benchmark Train/Dev Overlap Examples", ""])
    if overlap["benchmark_train_dev_overlap_examples"]:
        lines.extend(
            [
                "| Text hash | Files | Example rows |",
                "| --- | --- | --- |",
            ]
        )
        for item in overlap["benchmark_train_dev_overlap_examples"]:
            files = ", ".join(f"`{Path(path).name}`" for path in item["files"])
            rows = ", ".join(
                f"`{Path(row['path']).name}:{row['row_number']}`"
                for row in item["sample_rows"]
            )
            lines.append(f"| `{item['text_hash']}` | {files} | {rows} |")
    else:
        lines.append("No exact train/dev text overlaps were found inside the benchmark anchor.")

    lines.extend(["", "### Supporting Conflict Examples", ""])
    if overlap["supporting_conflict_examples"]:
        lines.extend(
            [
                "| Text hash | Normalized labels | Files | Occurrences |",
                "| --- | --- | --- | ---: |",
            ]
        )
        for item in overlap["supporting_conflict_examples"]:
            labels = ", ".join(f"`{label}`" for label in item["normalized_labels"])
            files = ", ".join(f"`{Path(path).name}`" for path in item["files"])
            lines.append(
                f"| `{item['text_hash']}` | {labels} | {files} | {item['occurrence_count']} |"
            )
    else:
        lines.append("No canonical supporting-source same-text different-label cases were found.")
    lines.append("")
    return lines


def render_markdown(results: list[ValidationResult], payload: dict[str, Any]) -> str:
    grouped_results = {
        group: [result for result in results if result.file_group == group]
        for group in FILE_GROUP_ORDER
    }
    lines = [
        "# Data Validation Summary",
        "",
        f"- Data root: `{payload['data_root']}`",
        f"- Files scanned: `{payload['files_scanned']}`",
        f"- Total rows scanned: `{payload['total_rows_scanned']}`",
        f"- Benchmark anchor rows scanned: `{payload['summary_totals']['benchmark_anchor_rows_scanned']}`",
        f"- Canonical supporting rows scanned: `{payload['summary_totals']['canonical_supporting_rows_scanned']}`",
        f"- Provenance / auxiliary evaluation rows scanned: `{payload['summary_totals']['provenance_aux_eval_rows_scanned']}`",
        f"- Unlabeled ID-only rows scanned: `{payload['summary_totals']['unlabeled_id_only_rows_scanned']}`",
        f"- Out-of-scope rows scanned: `{payload['summary_totals']['out_of_scope_rows_scanned']}`",
        f"- Files with duplicate rows: `{payload['files_with_duplicates']}`",
        f"- Files with empty texts: `{payload['files_with_empty_text']}`",
        f"- Files with texts under 3 tokens: `{payload['files_with_short_text']}`",
        "",
        "## Grouped File Summary",
        "",
    ]

    for group in FILE_GROUP_ORDER:
        totals = payload["group_totals"][group]
        lines.extend(
            [
                f"### {FILE_GROUP_TITLES[group]}",
                "",
                f"- Files: `{totals['file_count']}`",
                f"- Rows: `{totals['row_count']}`",
                f"- Files with duplicates: `{totals['files_with_duplicates']}`",
                f"- Files with empty texts: `{totals['files_with_empty_text']}`",
                f"- Files with texts under 3 tokens: `{totals['files_with_short_text']}`",
                "",
            ]
        )
        lines.extend(render_group_table(grouped_results[group]))
        lines.append("")

    lines.extend(render_overlap_section(payload))
    lines.extend(["## Per-File Details", ""])
    for group in FILE_GROUP_ORDER:
        for result in grouped_results[group]:
            lines.append(f"### `{result.path}`")
            lines.append(f"- Group: `{result.file_group}`")
            lines.append(f"- Schema: `{result.schema_name}`")
            lines.append(f"- Rows: `{result.row_count}`")
            lines.append(f"- Duplicate rows: `{result.duplicate_row_count}`")
            lines.append(f"- Duplicate check: `{result.duplicate_check_mode}`")
            if result.duplicate_row_examples:
                lines.append(
                    "- Duplicate example rows: "
                    + ", ".join(str(row_number) for row_number in result.duplicate_row_examples)
                )
            if result.missing_columns:
                lines.append(
                    "- Missing required columns: "
                    + ", ".join(f"`{column}`" for column in result.missing_columns)
                )
            for column, stats in result.text_stats.items():
                lines.append(
                    f"- Text column `{column}`: {stats.empty_count} empty, {stats.short_count} under 3 tokens"
                )
                if stats.sample_empty_rows:
                    lines.append(
                        "  sample empty rows: "
                        + ", ".join(str(row_number) for row_number in stats.sample_empty_rows)
                    )
                if stats.sample_short_rows:
                    lines.append(
                        "  sample short rows: "
                        + ", ".join(str(row_number) for row_number in stats.sample_short_rows)
                    )
            for column, counts in result.class_counts.items():
                lines.append(f"- Class counts for `{column}`:")
                for label, count in counts.items():
                    lines.append(f"  - `{label}`: {count}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_markdown_report(results: list[ValidationResult], payload: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / REPORT_MARKDOWN_NAME
    report_path.write_text(render_markdown(results, payload), encoding="utf-8")
    return report_path


def generate_validation_reports(data_root: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = validate_tsv_tree(data_root)
    payload = build_report_payload(results, data_root)
    return {
        "json": write_json_report(payload, output_dir),
        "csv": write_csv_report(results, output_dir),
        "markdown": write_markdown_report(results, payload, output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TSV files under data/raw and write reports.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory where validation reports will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = generate_validation_reports(args.data_root, args.output_dir)
    for name, path in reports.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
