from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DuplicateSummary:
    count: int
    additional_rows: int
    representative_groups: list[list[str]]


@dataclass(frozen=True)
class OverlapSummary:
    count: int
    representative_pairs: list[dict[str, str]]


@dataclass(frozen=True)
class NearDuplicateSummary:
    count: int
    representative_pairs: list[dict[str, Any]]
    threshold: float


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_overlap_summary(
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    *,
    field: str,
    pair_limit: int = 5,
) -> OverlapSummary:
    train_index: dict[str, list[str]] = defaultdict(list)
    for row in train_rows:
        train_index[row[field]].append(row["source_id"])

    overlap_values: set[str] = set()
    representative_pairs: list[dict[str, str]] = []
    for dev_row in dev_rows:
        value = dev_row[field]
        train_ids = train_index.get(value, [])
        if not train_ids:
            continue
        overlap_values.add(value)
        for train_id in train_ids:
            if len(representative_pairs) >= pair_limit:
                break
            representative_pairs.append(
                {
                    "train_source_id": train_id,
                    "dev_source_id": dev_row["source_id"],
                }
            )
        if len(representative_pairs) >= pair_limit:
            continue

    return OverlapSummary(
        count=len(overlap_values),
        representative_pairs=representative_pairs,
    )


def duplicate_processed_text_summary(
    rows: list[dict[str, str]],
    *,
    field: str = "processed_text",
    group_limit: int = 5,
) -> DuplicateSummary:
    value_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        value_to_ids[row[field]].append(row["source_id"])

    duplicate_groups = [source_ids for source_ids in value_to_ids.values() if len(source_ids) > 1]
    duplicate_groups.sort(key=lambda group: (-len(group), group))
    return DuplicateSummary(
        count=len(duplicate_groups),
        additional_rows=sum(len(group) - 1 for group in duplicate_groups),
        representative_groups=duplicate_groups[:group_limit],
    )


def _tokenize(text: str) -> set[str]:
    return {token for token in text.split() if token}


def near_duplicate_summary(
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    *,
    field: str = "processed_text",
    threshold: float = 0.9,
    pair_limit: int = 5,
) -> NearDuplicateSummary:
    train_token_sets: list[set[str]] = []
    inverted_index: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(train_rows):
        tokens = _tokenize(row[field])
        train_token_sets.append(tokens)
        for token in tokens:
            inverted_index[token].add(index)

    representative_pairs: list[dict[str, Any]] = []
    flagged_dev_rows = 0
    for dev_row in dev_rows:
        dev_tokens = _tokenize(dev_row[field])
        if not dev_tokens:
            continue
        candidate_indices: set[int] = set()
        for token in dev_tokens:
            candidate_indices.update(inverted_index.get(token, set()))

        matched = False
        dev_len = len(dev_tokens)
        for train_index in candidate_indices:
            train_tokens = train_token_sets[train_index]
            if not train_tokens:
                continue
            train_len = len(train_tokens)
            if min(dev_len, train_len) / max(dev_len, train_len) <= threshold:
                continue
            intersection = len(dev_tokens & train_tokens)
            union = len(dev_tokens | train_tokens)
            similarity = intersection / union if union else 0.0
            if similarity > threshold:
                flagged_dev_rows += 1
                matched = True
                if len(representative_pairs) < pair_limit:
                    representative_pairs.append(
                        {
                            "train_source_id": train_rows[train_index]["source_id"],
                            "dev_source_id": dev_row["source_id"],
                            "token_jaccard": round(similarity, 4),
                        }
                    )
                break
        if matched:
            continue

    return NearDuplicateSummary(
        count=flagged_dev_rows,
        representative_pairs=representative_pairs,
        threshold=threshold,
    )


def build_audit_payload(
    *,
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    source_id_overlap = exact_overlap_summary(train_rows, dev_rows, field="source_id")
    original_text_overlap = exact_overlap_summary(train_rows, dev_rows, field="original_text")
    processed_text_overlap = exact_overlap_summary(train_rows, dev_rows, field="processed_text")
    train_duplicates = duplicate_processed_text_summary(train_rows)
    dev_duplicates = duplicate_processed_text_summary(dev_rows)
    near_duplicates = near_duplicate_summary(
        train_rows,
        dev_rows,
        threshold=near_duplicate_threshold,
    )
    hard_failures = [
        source_id_overlap.count,
        original_text_overlap.count,
        processed_text_overlap.count,
    ]
    return {
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "hard_checks": {
            "source_id_overlap": {
                "count": source_id_overlap.count,
                "representative_pairs": source_id_overlap.representative_pairs,
            },
            "original_text_overlap": {
                "count": original_text_overlap.count,
                "representative_pairs": original_text_overlap.representative_pairs,
            },
            "processed_text_overlap": {
                "count": processed_text_overlap.count,
                "representative_pairs": processed_text_overlap.representative_pairs,
            },
        },
        "soft_checks": {
            "train_processed_text_duplicate_groups": {
                "count": train_duplicates.count,
                "additional_rows": train_duplicates.additional_rows,
                "representative_groups": train_duplicates.representative_groups,
            },
            "dev_processed_text_duplicate_groups": {
                "count": dev_duplicates.count,
                "additional_rows": dev_duplicates.additional_rows,
                "representative_groups": dev_duplicates.representative_groups,
            },
            "dev_rows_with_near_duplicate_in_train": {
                "count": near_duplicates.count,
                "threshold": near_duplicates.threshold,
                "representative_pairs": near_duplicates.representative_pairs,
            },
        },
        "status": "block" if any(hard_failures) else "pass",
    }


def write_audit_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_audit_markdown(path: Path, payload: dict[str, Any]) -> None:
    hard_checks = payload["hard_checks"]
    soft_checks = payload["soft_checks"]
    lines = [
        "# MARBERT Leakage Audit",
        "",
        "This audit checks exact train/dev overlap on the processed benchmark-safe core split used by the MARBERT encoder pipeline.",
        "",
        "## Scope",
        "",
        f"- Train rows: `{payload['train_rows']}`",
        f"- Dev rows: `{payload['dev_rows']}`",
        "- Hard checks 1-3 use exact string equality on `source_id`, `original_text`, and `processed_text`.",
        "- Soft checks 4-5 report duplicate `processed_text` groups within each split.",
        "- Soft check 6 reports dev rows that have at least one train near-duplicate by token Jaccard similarity greater than `0.9`.",
        "",
        "## Hard Checks",
        "",
        "| Check | Count | Status |",
        "| --- | --- | --- |",
        f"| 1. `source_id` overlap train vs dev | `{hard_checks['source_id_overlap']['count']}` | {'block' if hard_checks['source_id_overlap']['count'] else 'pass'} |",
        f"| 2. exact `original_text` overlap train vs dev | `{hard_checks['original_text_overlap']['count']}` | {'block' if hard_checks['original_text_overlap']['count'] else 'pass'} |",
        f"| 3. exact `processed_text` overlap train vs dev | `{hard_checks['processed_text_overlap']['count']}` | {'block' if hard_checks['processed_text_overlap']['count'] else 'pass'} |",
        "",
    ]

    for check_name, label in (
        ("source_id_overlap", "`source_id`"),
        ("original_text_overlap", "`original_text`"),
        ("processed_text_overlap", "`processed_text`"),
    ):
        representative_pairs = hard_checks[check_name]["representative_pairs"]
        if representative_pairs:
            lines.extend(
                [
                    f"### Representative Pairs For {label}",
                    "",
                    "| Train source_id | Dev source_id |",
                    "| --- | --- |",
                ]
            )
            for pair in representative_pairs:
                lines.append(f"| `{pair['train_source_id']}` | `{pair['dev_source_id']}` |")
            lines.append("")

    lines.extend(
        [
            "## Soft Checks",
            "",
            "| Check | Count | Notes |",
            "| --- | --- | --- |",
            f"| 4. duplicate `processed_text` groups within train | `{soft_checks['train_processed_text_duplicate_groups']['count']}` | additional rows beyond first: `{soft_checks['train_processed_text_duplicate_groups']['additional_rows']}` |",
            f"| 5. duplicate `processed_text` groups within dev | `{soft_checks['dev_processed_text_duplicate_groups']['count']}` | additional rows beyond first: `{soft_checks['dev_processed_text_duplicate_groups']['additional_rows']}` |",
            f"| 6. dev rows with a train near-duplicate by token Jaccard > `0.9` | `{soft_checks['dev_rows_with_near_duplicate_in_train']['count']}` | threshold: `{soft_checks['dev_rows_with_near_duplicate_in_train']['threshold']}` |",
            "",
        ]
    )

    for split_name in ("train_processed_text_duplicate_groups", "dev_processed_text_duplicate_groups"):
        groups = soft_checks[split_name]["representative_groups"]
        if groups:
            title = "train" if split_name.startswith("train") else "dev"
            lines.extend(
                [
                    f"### Representative Duplicate Groups In {title}",
                    "",
                ]
            )
            for group in groups:
                source_ids = ", ".join(f"`{source_id}`" for source_id in group)
                lines.append(f"- {source_ids}")
            lines.append("")

    near_duplicate_pairs = soft_checks["dev_rows_with_near_duplicate_in_train"]["representative_pairs"]
    if near_duplicate_pairs:
        lines.extend(
            [
                "### Representative Near-Duplicate Pairs",
                "",
                "| Train source_id | Dev source_id | Token Jaccard |",
                "| --- | --- | --- |",
            ]
        )
        for pair in near_duplicate_pairs:
            lines.append(
                f"| `{pair['train_source_id']}` | `{pair['dev_source_id']}` | `{pair['token_jaccard']:.4f}` |"
            )
        lines.append("")

    lines.extend(
        [
            "## Decision",
            "",
            "- Hard checks 1-3 are stop conditions for benchmark-safe reruns.",
            "- Soft checks 4-6 are diagnostic flags only and do not block reruns by themselves.",
            f"- Overall audit status: `{payload['status']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    train_path: Path,
    dev_path: Path,
    markdown_out: Path,
    json_out: Path,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    train_rows = load_rows(train_path)
    dev_rows = load_rows(dev_path)
    payload = build_audit_payload(
        train_rows=train_rows,
        dev_rows=dev_rows,
        near_duplicate_threshold=near_duplicate_threshold,
    )
    write_audit_markdown(markdown_out, payload)
    write_audit_json(json_out, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a benchmark-safe leakage audit for MARBERT train/dev processed splits.")
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train_core.csv"))
    parser.add_argument("--dev-path", type=Path, default=Path("data/processed/dev_core.csv"))
    parser.add_argument("--markdown-out", type=Path, default=Path("artifacts/reports/marbert_leakage_audit.md"))
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/reports/marbert_leakage_audit.json"))
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_audit(
        train_path=args.train_path,
        dev_path=args.dev_path,
        markdown_out=args.markdown_out,
        json_out=args.json_out,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    print(f"status: {payload['status']}")
    print(f"markdown: {args.markdown_out}")
    print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
