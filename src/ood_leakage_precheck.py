from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.interim_dataset import RAW_TO_MACRO
from src.data.preprocessing import preprocess_text
from src.data.validation import normalize_raw_label


LABEL_ORDER = ("Saudi", "Egyptian", "Levantine", "Maghrebi")
LEVANTINE_SOURCE_LABELS = ("Jordan", "Lebanon", "Palestine", "Syria")
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.9
REPRESENTATIVE_LIMIT = 10


@dataclass(frozen=True)
class CandidateConfig:
    slug: str
    display_name: str
    path: Path
    id_column: str
    text_column: str
    label_column: str


@dataclass(frozen=True)
class CandidateRow:
    source_id: str
    raw_label: str
    normalized_raw_label: str
    macro_label: str
    original_text: str
    processed_text: str


@dataclass(frozen=True)
class BenchmarkReference:
    name: str
    interim_path: Path
    processed_path: Path
    interim_rows: list[dict[str, str]]
    processed_rows: list[dict[str, str]]
    source_id_index: dict[str, tuple[str, ...]]
    original_text_index: dict[str, tuple[str, ...]]
    processed_text_index: dict[str, tuple[str, ...]]
    macro_distribution: dict[str, int]
    missing_processed_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExactOverlapResult:
    count: int
    percentage: float
    representative_pairs: tuple[dict[str, str], ...]
    matched_candidate_source_ids: frozenset[str]


@dataclass(frozen=True)
class DuplicateSummary:
    count: int
    additional_rows: int
    representative_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class NearDuplicateSummary:
    count: int
    percentage: float
    representative_pairs: tuple[dict[str, Any], ...]
    threshold: float


def read_csv_rows(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _sorted_index(index: dict[str, set[str]]) -> dict[str, tuple[str, ...]]:
    return {value: tuple(sorted(source_ids)) for value, source_ids in index.items()}


def build_benchmark_reference(
    *,
    name: str,
    interim_path: Path,
    processed_path: Path,
    interim_text_column: str = "text",
) -> BenchmarkReference:
    interim_rows = read_csv_rows(interim_path)
    processed_rows = read_csv_rows(processed_path)

    source_id_index: dict[str, set[str]] = defaultdict(set)
    original_text_index: dict[str, set[str]] = defaultdict(set)
    processed_text_index: dict[str, set[str]] = defaultdict(set)
    interim_source_ids: set[str] = set()
    processed_source_ids: set[str] = set()
    macro_distribution: Counter[str] = Counter()

    for row in interim_rows:
        source_id = row["source_id"]
        interim_source_ids.add(source_id)
        source_id_index[source_id].add(source_id)
        original_text_index[row[interim_text_column]].add(source_id)
        macro_label = row.get("macro_label", "")
        if macro_label:
            macro_distribution[macro_label] += 1

    for row in processed_rows:
        source_id = row["source_id"]
        processed_source_ids.add(source_id)
        source_id_index[source_id].add(source_id)
        original_text_index[row["original_text"]].add(source_id)
        processed_text_index[row["processed_text"]].add(source_id)

    return BenchmarkReference(
        name=name,
        interim_path=interim_path,
        processed_path=processed_path,
        interim_rows=interim_rows,
        processed_rows=processed_rows,
        source_id_index=_sorted_index(source_id_index),
        original_text_index=_sorted_index(original_text_index),
        processed_text_index=_sorted_index(processed_text_index),
        macro_distribution={label: macro_distribution.get(label, 0) for label in LABEL_ORDER},
        missing_processed_source_ids=tuple(sorted(interim_source_ids - processed_source_ids)),
    )


def load_candidate_rows(config: CandidateConfig) -> tuple[list[CandidateRow], Counter[str], list[str], list[str]]:
    raw_rows = read_csv_rows(config.path, delimiter="\t")
    raw_label_counts: Counter[str] = Counter()
    mapped_raw_labels: set[str] = set()
    dropped_raw_labels: set[str] = set()
    kept_rows: list[CandidateRow] = []

    for row in raw_rows:
        raw_label = (row[config.label_column] or "").strip()
        normalized_raw_label = normalize_raw_label(raw_label)
        raw_label_counts[raw_label] += 1
        macro_label = RAW_TO_MACRO.get(normalized_raw_label)
        if macro_label is None:
            dropped_raw_labels.add(raw_label)
            continue
        mapped_raw_labels.add(raw_label)
        original_text = row[config.text_column] or ""
        kept_rows.append(
            CandidateRow(
                source_id=(row[config.id_column] or "").strip(),
                raw_label=raw_label,
                normalized_raw_label=normalized_raw_label,
                macro_label=macro_label,
                original_text=original_text,
                processed_text=preprocess_text(original_text),
            )
        )

    return kept_rows, raw_label_counts, sorted(mapped_raw_labels), sorted(dropped_raw_labels)


def label_distribution(rows: list[CandidateRow]) -> dict[str, dict[str, float]]:
    counts = Counter(row.macro_label for row in rows)
    total_rows = len(rows)
    return {
        label: {
            "count": counts.get(label, 0),
            "percentage": percentage(counts.get(label, 0), total_rows),
        }
        for label in LABEL_ORDER
    }


def percentage(count: int, total: int) -> float:
    return round((count / total * 100.0) if total else 0.0, 2)


def distribution_delta(
    candidate_distribution: dict[str, dict[str, float]],
    benchmark_distribution: dict[str, int],
    benchmark_total: int,
) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for label in LABEL_ORDER:
        benchmark_count = benchmark_distribution.get(label, 0)
        benchmark_percentage = percentage(benchmark_count, benchmark_total)
        candidate_count = int(candidate_distribution[label]["count"])
        candidate_percentage = float(candidate_distribution[label]["percentage"])
        deltas[label] = {
            "candidate_count": candidate_count,
            "candidate_percentage": candidate_percentage,
            "benchmark_count": benchmark_count,
            "benchmark_percentage": benchmark_percentage,
            "count_delta": candidate_count - benchmark_count,
            "percentage_point_delta": round(candidate_percentage - benchmark_percentage, 2),
        }
    return deltas


def duplicate_processed_text_summary(
    rows: list[CandidateRow],
    *,
    group_limit: int = REPRESENTATIVE_LIMIT,
) -> DuplicateSummary:
    value_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value_to_ids[row.processed_text].append(row.source_id)

    duplicate_groups = [tuple(source_ids) for source_ids in value_to_ids.values() if len(source_ids) > 1]
    duplicate_groups.sort(key=lambda group: (-len(group), group))
    return DuplicateSummary(
        count=len(duplicate_groups),
        additional_rows=sum(len(group) - 1 for group in duplicate_groups),
        representative_groups=tuple(duplicate_groups[:group_limit]),
    )


def exact_overlap_summary(
    candidate_rows: list[CandidateRow],
    *,
    reference_index: dict[str, tuple[str, ...]],
    candidate_field: str,
    pair_limit: int = REPRESENTATIVE_LIMIT,
) -> ExactOverlapResult:
    representative_pairs: list[dict[str, str]] = []
    matched_candidate_source_ids: set[str] = set()

    for row in candidate_rows:
        value = getattr(row, candidate_field)
        benchmark_source_ids = reference_index.get(value, ())
        if not benchmark_source_ids:
            continue
        matched_candidate_source_ids.add(row.source_id)
        for benchmark_source_id in benchmark_source_ids:
            if len(representative_pairs) >= pair_limit:
                break
            representative_pairs.append(
                {
                    "benchmark_source_id": benchmark_source_id,
                    "candidate_source_id": row.source_id,
                }
            )
        if len(representative_pairs) >= pair_limit:
            continue

    return ExactOverlapResult(
        count=len(matched_candidate_source_ids),
        percentage=percentage(len(matched_candidate_source_ids), len(candidate_rows)),
        representative_pairs=tuple(representative_pairs),
        matched_candidate_source_ids=frozenset(matched_candidate_source_ids),
    )


def _tokenize(text: str) -> set[str]:
    return {token for token in text.split() if token}


def near_duplicate_summary(
    train_rows: list[dict[str, str]],
    candidate_rows: list[CandidateRow],
    *,
    threshold: float,
    pair_limit: int = REPRESENTATIVE_LIMIT,
) -> NearDuplicateSummary:
    train_token_sets: list[set[str]] = []
    train_source_ids: list[str] = []
    inverted_index: dict[str, set[int]] = defaultdict(set)

    for index, row in enumerate(train_rows):
        tokens = _tokenize(row["processed_text"])
        train_token_sets.append(tokens)
        train_source_ids.append(row["source_id"])
        for token in tokens:
            inverted_index[token].add(index)

    flagged_candidate_ids: set[str] = set()
    representative_pairs: list[dict[str, Any]] = []

    for row in candidate_rows:
        candidate_tokens = _tokenize(row.processed_text)
        if not candidate_tokens:
            continue
        candidate_indices: set[int] = set()
        for token in candidate_tokens:
            candidate_indices.update(inverted_index.get(token, set()))

        candidate_token_count = len(candidate_tokens)
        for train_index in candidate_indices:
            train_tokens = train_token_sets[train_index]
            if not train_tokens:
                continue
            train_token_count = len(train_tokens)
            if min(candidate_token_count, train_token_count) / max(candidate_token_count, train_token_count) <= threshold:
                continue
            union = candidate_tokens | train_tokens
            similarity = len(candidate_tokens & train_tokens) / len(union) if union else 0.0
            if similarity > threshold:
                flagged_candidate_ids.add(row.source_id)
                if len(representative_pairs) < pair_limit:
                    representative_pairs.append(
                        {
                            "benchmark_source_id": train_source_ids[train_index],
                            "candidate_source_id": row.source_id,
                            "token_jaccard": round(similarity, 4),
                        }
                    )
                break

    return NearDuplicateSummary(
        count=len(flagged_candidate_ids),
        percentage=percentage(len(flagged_candidate_ids), len(candidate_rows)),
        representative_pairs=tuple(representative_pairs),
        threshold=threshold,
    )


def classification_from_exact_overlap(
    *,
    exact_overlap_count: int,
    total_rows: int,
    near_duplicates: NearDuplicateSummary,
) -> tuple[str, str, bool]:
    exact_percentage = percentage(exact_overlap_count, total_rows)
    if exact_overlap_count == 0:
        rationale = (
            "No candidate rows hit any exact overlap check against the current benchmark-safe "
            "train_core/dev_core references."
        )
        if near_duplicates.count:
            rationale += (
                f" Near-duplicate risk remains at {near_duplicates.count} rows "
                f"({near_duplicates.percentage:.2f}%) against train_core, so the split is technically "
                "overlap-free but should still be interpreted in light of lexical similarity."
            )
        return "acceptable as OOD evaluation source", rationale, False
    if exact_percentage < 1.0:
        rationale = (
            f"{exact_overlap_count} candidate rows ({exact_percentage:.2f}%) hit at least one exact overlap check. "
            "That rules out strict OOD framing but still fits held-out historical evaluation."
        )
        return "acceptable only as held-out historical evaluation", rationale, False
    if exact_percentage <= 5.0:
        rationale = (
            f"{exact_overlap_count} candidate rows ({exact_percentage:.2f}%) hit at least one exact overlap check. "
            "That falls in the 1%-5% band, so any use should be downgraded to held-out historical evaluation "
            "with a prominent caveat."
        )
        return "acceptable only as held-out historical evaluation", rationale, True
    rationale = (
        f"{exact_overlap_count} candidate rows ({exact_percentage:.2f}%) hit at least one exact overlap check, "
        "which exceeds the 5% ceiling."
    )
    return "not acceptable until deduplicated", rationale, True


def analyze_candidate(
    *,
    config: CandidateConfig,
    train_reference: BenchmarkReference,
    dev_reference: BenchmarkReference,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    raw_rows = read_csv_rows(config.path, delimiter="\t")
    candidate_rows, raw_label_counts, mapped_raw_labels, dropped_raw_labels = load_candidate_rows(config)
    candidate_distribution = label_distribution(candidate_rows)
    dev_distribution_delta = distribution_delta(
        candidate_distribution,
        dev_reference.macro_distribution,
        len(dev_reference.interim_rows),
    )

    exact_checks = {
        "train_core_source_id": exact_overlap_summary(
            candidate_rows,
            reference_index=train_reference.source_id_index,
            candidate_field="source_id",
        ),
        "train_core_original_text": exact_overlap_summary(
            candidate_rows,
            reference_index=train_reference.original_text_index,
            candidate_field="original_text",
        ),
        "train_core_processed_text": exact_overlap_summary(
            candidate_rows,
            reference_index=train_reference.processed_text_index,
            candidate_field="processed_text",
        ),
        "dev_core_source_id": exact_overlap_summary(
            candidate_rows,
            reference_index=dev_reference.source_id_index,
            candidate_field="source_id",
        ),
        "dev_core_original_text": exact_overlap_summary(
            candidate_rows,
            reference_index=dev_reference.original_text_index,
            candidate_field="original_text",
        ),
        "dev_core_processed_text": exact_overlap_summary(
            candidate_rows,
            reference_index=dev_reference.processed_text_index,
            candidate_field="processed_text",
        ),
    }
    combined_exact_overlap_source_ids = frozenset().union(
        *(result.matched_candidate_source_ids for result in exact_checks.values())
    )
    exact_overlap_count = len(combined_exact_overlap_source_ids)

    duplicate_summary = duplicate_processed_text_summary(candidate_rows)
    near_duplicates = near_duplicate_summary(
        train_reference.processed_rows,
        candidate_rows,
        threshold=near_duplicate_threshold,
    )
    classification, rationale, prominent_caveat = classification_from_exact_overlap(
        exact_overlap_count=exact_overlap_count,
        total_rows=len(candidate_rows),
        near_duplicates=near_duplicates,
    )

    levantine_presence = {
        label: raw_label_counts[label]
        for label in LEVANTINE_SOURCE_LABELS
        if raw_label_counts[label]
    }
    levantine_confirmation = {
        "present_raw_labels": list(levantine_presence.keys()),
        "counts": levantine_presence,
        "all_present_mapped_to_levantine": all(
            RAW_TO_MACRO.get(normalize_raw_label(label)) == "Levantine"
            for label in levantine_presence
        ),
    }

    payload = {
        "candidate_split": config.display_name,
        "candidate_path": config.path.as_posix(),
        "row_counts": {
            "before_filtering": len(raw_rows),
            "after_dropping_out_of_scope": len(candidate_rows),
            "dropped_out_of_scope": len(raw_rows) - len(candidate_rows),
        },
        "raw_label_values": {
            label: raw_label_counts[label]
            for label in sorted(raw_label_counts)
        },
        "mapped_raw_label_values": mapped_raw_labels,
        "dropped_raw_label_values": dropped_raw_labels,
        "levantine_mapping_confirmation": levantine_confirmation,
        "mapped_label_distribution": candidate_distribution,
        "dev_core_distribution_reference": {
            "source_path": dev_reference.interim_path.as_posix(),
            "rows": len(dev_reference.interim_rows),
            "distribution": {
                label: {
                    "count": dev_reference.macro_distribution.get(label, 0),
                    "percentage": percentage(
                        dev_reference.macro_distribution.get(label, 0),
                        len(dev_reference.interim_rows),
                    ),
                }
                for label in LABEL_ORDER
            },
            "candidate_minus_dev_core": dev_distribution_delta,
        },
        "benchmark_reference_note": {
            "train_core": {
                "interim_path": train_reference.interim_path.as_posix(),
                "interim_rows": len(train_reference.interim_rows),
                "processed_path": train_reference.processed_path.as_posix(),
                "processed_rows": len(train_reference.processed_rows),
                "source_id_reference_size": len(train_reference.source_id_index),
                "original_text_reference_size": len(train_reference.original_text_index),
                "processed_text_reference_size": len(train_reference.processed_text_index),
                "missing_processed_source_ids": list(train_reference.missing_processed_source_ids),
            },
            "dev_core": {
                "interim_path": dev_reference.interim_path.as_posix(),
                "interim_rows": len(dev_reference.interim_rows),
                "processed_path": dev_reference.processed_path.as_posix(),
                "processed_rows": len(dev_reference.processed_rows),
                "source_id_reference_size": len(dev_reference.source_id_index),
                "original_text_reference_size": len(dev_reference.original_text_index),
                "processed_text_reference_size": len(dev_reference.processed_text_index),
                "missing_processed_source_ids": list(dev_reference.missing_processed_source_ids),
            },
        },
        "exact_overlap_checks": {
            name: {
                "count": result.count,
                "percentage": result.percentage,
                "representative_pairs": list(result.representative_pairs),
            }
            for name, result in exact_checks.items()
        },
        "combined_exact_overlap": {
            "count": exact_overlap_count,
            "percentage": percentage(exact_overlap_count, len(candidate_rows)),
        },
        "duplicate_processed_text_within_candidate": {
            "count": duplicate_summary.count,
            "additional_rows": duplicate_summary.additional_rows,
            "representative_groups": [list(group) for group in duplicate_summary.representative_groups],
        },
        "near_duplicate_vs_train_core": {
            "count": near_duplicates.count,
            "percentage": near_duplicates.percentage,
            "threshold": near_duplicates.threshold,
            "representative_pairs": list(near_duplicates.representative_pairs),
        },
        "classification": classification,
        "requires_prominent_caveat": prominent_caveat,
        "decision_rationale": rationale,
    }
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_percentage(value: float) -> str:
    return f"{value:.2f}%"


def _format_count_and_percentage(count: int, pct: float) -> str:
    return f"`{count}` ({_format_percentage(pct)})"


def _write_distribution_table(
    lines: list[str],
    *,
    candidate_distribution: dict[str, dict[str, float]],
    benchmark_delta: dict[str, dict[str, float]],
) -> None:
    lines.extend(
        [
            "| Label | Candidate count | Candidate % | Current dev_core count | Current dev_core % | Count delta | Percentage-point delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in LABEL_ORDER:
        delta = benchmark_delta[label]
        lines.append(
            f"| `{label}` | {int(candidate_distribution[label]['count'])} | "
            f"{_format_percentage(float(candidate_distribution[label]['percentage']))} | "
            f"{delta['benchmark_count']} | {_format_percentage(float(delta['benchmark_percentage']))} | "
            f"{int(delta['count_delta'])} | {delta['percentage_point_delta']:+.2f} pp |"
        )
    lines.append("")


def _write_exact_overlap_table(lines: list[str], exact_checks: dict[str, Any]) -> None:
    order = (
        ("train_core_source_id", "train_core `source_id`"),
        ("train_core_original_text", "train_core `original_text`"),
        ("train_core_processed_text", "train_core `processed_text`"),
        ("dev_core_source_id", "dev_core `source_id`"),
        ("dev_core_original_text", "dev_core `original_text`"),
        ("dev_core_processed_text", "dev_core `processed_text`"),
    )
    lines.extend(
        [
            "| Check | Candidate rows with exact overlap |",
            "| --- | ---: |",
        ]
    )
    for key, label in order:
        result = exact_checks[key]
        lines.append(f"| {label} | {_format_count_and_percentage(result['count'], float(result['percentage']))} |")
    lines.append("")


def write_candidate_markdown(path: Path, payload: dict[str, Any]) -> None:
    row_counts = payload["row_counts"]
    benchmark_note = payload["benchmark_reference_note"]
    exact_checks = payload["exact_overlap_checks"]
    duplicate_summary = payload["duplicate_processed_text_within_candidate"]
    near_duplicates = payload["near_duplicate_vs_train_core"]
    lines = [
        f"# OOD Leakage Pre-check: {payload['candidate_split']}",
        "",
        "This audit checks whether the standalone split can be treated as an OOD evaluation source relative to the current benchmark-safe core data already present in the project.",
        "",
        "## Scope",
        "",
        f"- Candidate path: `{payload['candidate_path']}`",
        f"- Rows before filtering: `{row_counts['before_filtering']}`",
        f"- Rows after dropping out-of-scope labels: `{row_counts['after_dropping_out_of_scope']}`",
        f"- Out-of-scope rows dropped: `{row_counts['dropped_out_of_scope']}`",
        "",
        "## Raw Label Inventory",
        "",
        "| Raw label | Count |",
        "| --- | ---: |",
    ]
    for label, count in payload["raw_label_values"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            "",
            "## Mapping Outcome",
            "",
            f"- Successfully mapped raw labels: {', '.join(f'`{label}`' for label in payload['mapped_raw_label_values'])}",
            f"- Dropped out-of-scope raw labels: {', '.join(f'`{label}`' for label in payload['dropped_raw_label_values'])}",
        ]
    )

    levantine_confirmation = payload["levantine_mapping_confirmation"]
    present_labels = levantine_confirmation["present_raw_labels"]
    if present_labels:
        present_summary = ", ".join(
            f"`{label}` ({levantine_confirmation['counts'][label]})" for label in present_labels
        )
        lines.append(
            f"- Levantine source-label confirmation: all present labels mapped to `Levantine` = "
            f"`{levantine_confirmation['all_present_mapped_to_levantine']}`; present labels: {present_summary}"
        )
    else:
        lines.append("- Levantine source-label confirmation: no expected Levantine raw labels were present.")

    lines.extend(
        [
            "",
            "## Benchmark Reference Note",
            "",
            f"- `train_core` interim rows: `{benchmark_note['train_core']['interim_rows']}`; processed rows: `{benchmark_note['train_core']['processed_rows']}`",
            f"- `dev_core` interim rows: `{benchmark_note['dev_core']['interim_rows']}`; processed rows: `{benchmark_note['dev_core']['processed_rows']}`",
            "- `source_id` and `original_text` overlap checks use the union of the current interim and processed benchmark references.",
            "- `processed_text` overlap checks use the current processed benchmark files as stored.",
        ]
    )
    missing_processed_dev_ids = benchmark_note["dev_core"]["missing_processed_source_ids"]
    if missing_processed_dev_ids:
        lines.append(
            f"- Current processed `dev_core` is missing source_ids present in interim: "
            f"{', '.join(f'`{source_id}`' for source_id in missing_processed_dev_ids)}"
        )
    lines.extend(
        [
            "",
            "## Mapped Label Distribution vs Current dev_core",
            "",
        ]
    )
    _write_distribution_table(
        lines,
        candidate_distribution=payload["mapped_label_distribution"],
        benchmark_delta=payload["dev_core_distribution_reference"]["candidate_minus_dev_core"],
    )

    lines.extend(
        [
            "## Exact Leakage Checks",
            "",
        ]
    )
    _write_exact_overlap_table(lines, exact_checks)

    for key, title in (
        ("train_core_source_id", "Representative Pairs: train_core `source_id`"),
        ("train_core_original_text", "Representative Pairs: train_core `original_text`"),
        ("train_core_processed_text", "Representative Pairs: train_core `processed_text`"),
        ("dev_core_source_id", "Representative Pairs: dev_core `source_id`"),
        ("dev_core_original_text", "Representative Pairs: dev_core `original_text`"),
        ("dev_core_processed_text", "Representative Pairs: dev_core `processed_text`"),
    ):
        pairs = exact_checks[key]["representative_pairs"]
        if not pairs:
            continue
        lines.extend(
            [
                f"### {title}",
                "",
                "| Benchmark source_id | Candidate source_id |",
                "| --- | --- |",
            ]
        )
        for pair in pairs:
            lines.append(f"| `{pair['benchmark_source_id']}` | `{pair['candidate_source_id']}` |")
        lines.append("")

    lines.extend(
        [
            "## Duplicate and Near-Duplicate Diagnostics",
            "",
            f"- Duplicate `processed_text` groups within the candidate split: `{duplicate_summary['count']}`",
            f"- Additional rows beyond first occurrence inside duplicate groups: `{duplicate_summary['additional_rows']}`",
            f"- Train-core near-duplicate rows at token Jaccard > `{near_duplicates['threshold']}`: "
            f"{_format_count_and_percentage(int(near_duplicates['count']), float(near_duplicates['percentage']))}",
        ]
    )
    if duplicate_summary["representative_groups"]:
        lines.extend(
            [
                "",
                "### Representative Duplicate Groups",
                "",
            ]
        )
        for group in duplicate_summary["representative_groups"]:
            lines.append(f"- {', '.join(f'`{source_id}`' for source_id in group)}")
    if near_duplicates["representative_pairs"]:
        lines.extend(
            [
                "",
                "### Representative Near-Duplicate Pairs",
                "",
                "| Train source_id | Candidate source_id | Token Jaccard |",
                "| --- | --- | ---: |",
            ]
        )
        for pair in near_duplicates["representative_pairs"]:
            lines.append(
                f"| `{pair['benchmark_source_id']}` | `{pair['candidate_source_id']}` | `{pair['token_jaccard']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Combined exact-overlap rows across all six checks: "
            f"{_format_count_and_percentage(payload['combined_exact_overlap']['count'], float(payload['combined_exact_overlap']['percentage']))}",
            f"- Classification: `{payload['classification']}`",
            f"- Rationale: {payload['decision_rationale']}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_markdown(path: Path, payloads: list[dict[str, Any]]) -> None:
    stop_required = any(payload["classification"] == "not acceptable until deduplicated" for payload in payloads)
    acceptable_ood = [payload for payload in payloads if payload["classification"] == "acceptable as OOD evaluation source"]
    historical_only = [
        payload for payload in payloads if payload["classification"] == "acceptable only as held-out historical evaluation"
    ]

    lines = [
        "# OOD Leakage Pre-check Summary",
        "",
        "This summary consolidates the standalone NADI 2020 dev and NADI 2021 DA dev leakage audits against the current benchmark-safe project train/dev references.",
        "",
        "## Split Summary",
        "",
        "| Split | In-scope rows | Exact-overlap rows | Near-duplicate rows vs train_core | Classification |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for payload in payloads:
        lines.append(
            f"| `{payload['candidate_split']}` | {payload['row_counts']['after_dropping_out_of_scope']} | "
            f"{_format_count_and_percentage(payload['combined_exact_overlap']['count'], float(payload['combined_exact_overlap']['percentage']))} | "
            f"{_format_count_and_percentage(payload['near_duplicate_vs_train_core']['count'], float(payload['near_duplicate_vs_train_core']['percentage']))} | "
            f"`{payload['classification']}` |"
        )

    for payload in payloads:
        lines.extend(
            [
                "",
                f"## {payload['candidate_split']}",
                "",
                f"- Raw label values found: {', '.join(f'`{label}`' for label in payload['raw_label_values'].keys())}",
                "- Mapped label distribution: "
                + ", ".join(
                    f"`{label}` {payload['mapped_label_distribution'][label]['count']} "
                    f"({_format_percentage(float(payload['mapped_label_distribution'][label]['percentage']))})"
                    for label in LABEL_ORDER
                ),
                "- Exact overlap counts: "
                + ", ".join(
                    f"{label} {_format_count_and_percentage(payload['exact_overlap_checks'][key]['count'], float(payload['exact_overlap_checks'][key]['percentage']))}"
                    for key, label in (
                        ("train_core_source_id", "train `source_id`"),
                        ("train_core_original_text", "train `original_text`"),
                        ("train_core_processed_text", "train `processed_text`"),
                        ("dev_core_source_id", "dev `source_id`"),
                        ("dev_core_original_text", "dev `original_text`"),
                        ("dev_core_processed_text", "dev `processed_text`"),
                    )
                ),
                f"- Near-duplicate count vs train_core: {_format_count_and_percentage(payload['near_duplicate_vs_train_core']['count'], float(payload['near_duplicate_vs_train_core']['percentage']))}",
                f"- Classification: `{payload['classification']}`",
                f"- Decision rationale: {payload['decision_rationale']}",
            ]
        )
        if payload["requires_prominent_caveat"]:
            lines.append("- Caveat: this split should carry a prominent leakage caveat if used as held-out historical evaluation.")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
        ]
    )
    if stop_required:
        lines.extend(
            [
                "- Phase 9 Part 2 should not proceed yet.",
                "- Option A: run deduplication as a separate sub-task",
                "- Option B: drop this source and use only acceptable ones",
                "- Option C: downgrade the framing to held-out historical evaluation with caveat",
                "",
            ]
        )
    elif acceptable_ood and not historical_only:
        lines.extend(
            [
                "- Phase 9 Part 2 can proceed on the audited OOD sources.",
                "- No split crossed the exact-overlap threshold that would block strict OOD framing.",
                "",
            ]
        )
    elif acceptable_ood and historical_only:
        accepted_names = ", ".join(f"`{payload['candidate_split']}`" for payload in acceptable_ood)
        historical_names = ", ".join(f"`{payload['candidate_split']}`" for payload in historical_only)
        lines.extend(
            [
                "- Phase 9 Part 2 can proceed only for the splits that remained overlap-free under the strict OOD framing.",
                f"- Acceptable as strict OOD sources: {accepted_names}",
                f"- Historical-only sources: {historical_names}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- Phase 9 Part 2 should not proceed under a strict OOD framing.",
                "- The audited sources are usable only as held-out historical evaluation sources under the current overlap profile.",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit standalone NADI dev splits for OOD leakage against the current benchmark-safe core data.")
    parser.add_argument("--train-interim-path", type=Path, default=Path("data/interim/train_core.csv"))
    parser.add_argument("--dev-interim-path", type=Path, default=Path("data/interim/dev_core.csv"))
    parser.add_argument("--train-processed-path", type=Path, default=Path("data/processed/train_core.csv"))
    parser.add_argument("--dev-processed-path", type=Path, default=Path("data/processed/dev_core.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--near-duplicate-threshold", type=float, default=DEFAULT_NEAR_DUPLICATE_THRESHOLD)
    return parser.parse_args()


def run_precheck(
    *,
    train_interim_path: Path,
    dev_interim_path: Path,
    train_processed_path: Path,
    dev_processed_path: Path,
    report_dir: Path,
    near_duplicate_threshold: float,
) -> list[dict[str, Any]]:
    train_reference = build_benchmark_reference(
        name="train_core",
        interim_path=train_interim_path,
        processed_path=train_processed_path,
    )
    dev_reference = build_benchmark_reference(
        name="dev_core",
        interim_path=dev_interim_path,
        processed_path=dev_processed_path,
    )

    configs = (
        CandidateConfig(
            slug="nadi2020",
            display_name="NADI 2020 dev",
            path=Path("data/raw/nadi2020/NADI_release/dev_labeled.tsv"),
            id_column="#1 tweet_ID",
            text_column="#2 tweet_content",
            label_column="#3 country_label",
        ),
        CandidateConfig(
            slug="nadi2021",
            display_name="NADI 2021 DA dev",
            path=Path("data/raw/nadi2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_dev_labeled.tsv"),
            id_column="#1_tweetid",
            text_column="#2_tweet",
            label_column="#3_country_label",
        ),
    )

    payloads: list[dict[str, Any]] = []
    for config in configs:
        payload = analyze_candidate(
            config=config,
            train_reference=train_reference,
            dev_reference=dev_reference,
            near_duplicate_threshold=near_duplicate_threshold,
        )
        write_candidate_markdown(
            report_dir / f"ood_leakage_precheck_{config.slug}.md",
            payload,
        )
        write_json(
            report_dir / f"ood_leakage_precheck_{config.slug}.json",
            payload,
        )
        payloads.append(payload)

    write_summary_markdown(report_dir / "ood_leakage_precheck_summary.md", payloads)
    return payloads


def main() -> int:
    args = parse_args()
    payloads = run_precheck(
        train_interim_path=args.train_interim_path,
        dev_interim_path=args.dev_interim_path,
        train_processed_path=args.train_processed_path,
        dev_processed_path=args.dev_processed_path,
        report_dir=args.report_dir,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    for payload in payloads:
        print(
            f"{payload['candidate_split']}: {payload['classification']} "
            f"(exact overlap rows={payload['combined_exact_overlap']['count']}, "
            f"near duplicates={payload['near_duplicate_vs_train_core']['count']})"
        )
    print(f"summary: {(args.report_dir / 'ood_leakage_precheck_summary.md').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
