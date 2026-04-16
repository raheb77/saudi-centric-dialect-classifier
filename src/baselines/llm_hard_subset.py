from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.baselines.llm_baseline import (
    ERROR_ANALYSIS_DIRECTIONS,
    LLMConfig,
    _markdown_table,
    evaluate_predictions,
    load_config as load_llm_config,
    load_labeled_rows,
    run_prompt_mode,
    select_few_shot_support,
)


@dataclass(frozen=True)
class HardSubsetConfig:
    llm_config: LLMConfig
    classical_predictions_path: Path
    gemini_predictions_path: Path


def load_config(path: Path) -> HardSubsetConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    artifacts_cfg = payload["artifacts"]
    return HardSubsetConfig(
        llm_config=load_llm_config(path),
        classical_predictions_path=Path(artifacts_cfg["classical_predictions_path"]),
        gemini_predictions_path=Path(artifacts_cfg["gemini_predictions_path"]),
    )


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_hard_subset(
    *,
    dev_rows: list[dict[str, str]],
    classical_rows: list[dict[str, str]],
    gemini_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    target_directions = set(ERROR_ANALYSIS_DIRECTIONS)
    reasons_by_source_id: dict[str, set[str]] = {}
    for row in classical_rows:
        if (row["true_label"], row["predicted_label"]) in target_directions:
            reasons_by_source_id.setdefault(row["source_id"], set()).add("classical")
    for row in gemini_rows:
        if (row["true_label"], row["zero_shot_predicted_label"]) in target_directions:
            reasons_by_source_id.setdefault(row["source_id"], set()).add("gemini_zero")
        if (row["true_label"], row["few_shot_predicted_label"]) in target_directions:
            reasons_by_source_id.setdefault(row["source_id"], set()).add("gemini_few")
    subset_rows = [row for row in dev_rows if row.get("source_id", "") in reasons_by_source_id]
    return subset_rows, reasons_by_source_id


def evaluate_existing_predictions(
    *,
    subset_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    prediction_column: str,
    config: LLMConfig,
) -> dict[str, Any]:
    by_source_id = {row["source_id"]: row for row in prediction_rows}
    predictions = [by_source_id[row["source_id"]][prediction_column] for row in subset_rows]
    metrics = evaluate_predictions(config=config, dev_rows=subset_rows, predictions=predictions)
    return {
        "predictions": predictions,
        "metrics": metrics,
    }


def _confusion_counts(
    *,
    rows: list[dict[str, str]],
    predictions: list[str],
    config: LLMConfig,
) -> Counter[tuple[str, str]]:
    return Counter(
        (row[config.target_column], prediction)
        for row, prediction in zip(rows, predictions, strict=True)
        if row[config.target_column] != prediction
    )


def write_subset_definition_markdown(
    path: Path,
    *,
    config: HardSubsetConfig,
    dev_rows: list[dict[str, str]],
    subset_rows: list[dict[str, str]],
    reasons_by_source_id: dict[str, set[str]],
    classical_rows: list[dict[str, str]],
    gemini_rows: list[dict[str, str]],
) -> None:
    classical_selected = {
        row["source_id"]
        for row in classical_rows
        if (row["true_label"], row["predicted_label"]) in set(ERROR_ANALYSIS_DIRECTIONS)
    }
    gemini_zero_selected = {
        row["source_id"]
        for row in gemini_rows
        if (row["true_label"], row["zero_shot_predicted_label"]) in set(ERROR_ANALYSIS_DIRECTIONS)
    }
    gemini_few_selected = {
        row["source_id"]
        for row in gemini_rows
        if (row["true_label"], row["few_shot_predicted_label"]) in set(ERROR_ANALYSIS_DIRECTIONS)
    }
    selector_combo_counts = Counter(
        ", ".join(sorted(reasons_by_source_id[row["source_id"]]))
        for row in subset_rows
    )
    true_label_counts = Counter(row[config.llm_config.target_column] for row in subset_rows)

    lines = [
        "# Claude Sonnet Hard Subset Definition",
        "",
        "This subset is built from `data/processed/dev_core.csv` only.",
        "",
        "## Selection Rule",
        "",
        "Keep a dev-set row if at least one existing baseline prediction places it in one of these tracked confusion directions:",
        "",
    ]
    for true_label, predicted_label in ERROR_ANALYSIS_DIRECTIONS:
        lines.append(f"- `{true_label} -> {predicted_label}`")
    lines.extend(
        [
            "",
            "Baselines checked for inclusion:",
            "",
            "- classical baseline (`predicted_label`)",
            "- Gemini Flash-Lite zero-shot (`zero_shot_predicted_label`)",
            "- Gemini Flash-Lite few-shot (`few_shot_predicted_label`)",
            "",
            "The final subset is the union of matching rows, deduplicated by `source_id` and kept in original `dev_core` order.",
            "",
            "## Counts",
            "",
            f"- Full `dev_core` rows: `{len(dev_rows)}`",
            f"- Classical matches: `{len(classical_selected)}`",
            f"- Gemini zero-shot matches: `{len(gemini_zero_selected)}`",
            f"- Gemini few-shot matches: `{len(gemini_few_selected)}`",
            f"- Final unique subset rows: `{len(subset_rows)}`",
            "",
            "## True-Label Composition",
            "",
            *_markdown_table(
                ["True Label", "Rows"],
                [[label, str(true_label_counts.get(label, 0))] for label in config.llm_config.label_order],
            ),
            "",
            "## Selector Overlap",
            "",
            *_markdown_table(
                ["Selection Source(s)", "Rows"],
                [[combo, str(count)] for combo, count in sorted(selector_combo_counts.items())],
            ),
            "",
            "## Implication",
            "",
            "Because all tracked confusion directions start from `Saudi` or `Egyptian`, the resulting hard subset contains only those two true labels. Metrics therefore stay comparable across models on the same rows, but 4-class macro F1 will be depressed by the zero-support `Levantine` and `Maghrebi` true classes in this subset.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_predictions_csv(
    path: Path,
    *,
    subset_rows: list[dict[str, str]],
    reasons_by_source_id: dict[str, set[str]],
    classical_rows: list[dict[str, str]],
    gemini_rows: list[dict[str, str]],
    sonnet_zero_predictions: list[str],
    sonnet_few_predictions: list[str],
    config: LLMConfig,
) -> None:
    classical_by_source_id = {row["source_id"]: row for row in classical_rows}
    gemini_by_source_id = {row["source_id"]: row for row in gemini_rows}
    fieldnames = [
        "source_dataset",
        "source_file",
        "source_id",
        "source_row_number",
        "selection_reasons",
        "original_text",
        "processed_text",
        "true_label",
        "classical_predicted_label",
        "gemini_zero_shot_predicted_label",
        "gemini_few_shot_predicted_label",
        "sonnet_zero_shot_predicted_label",
        "sonnet_few_shot_predicted_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, sonnet_zero_prediction, sonnet_few_prediction in zip(
            subset_rows,
            sonnet_zero_predictions,
            sonnet_few_predictions,
            strict=True,
        ):
            source_id = row["source_id"]
            writer.writerow(
                {
                    "source_dataset": row.get("source_dataset", ""),
                    "source_file": row.get("source_file", ""),
                    "source_id": source_id,
                    "source_row_number": row.get("source_row_number", ""),
                    "selection_reasons": ",".join(sorted(reasons_by_source_id[source_id])),
                    "original_text": row.get("original_text", ""),
                    "processed_text": row.get(config.text_column, ""),
                    "true_label": row.get(config.target_column, ""),
                    "classical_predicted_label": classical_by_source_id[source_id]["predicted_label"],
                    "gemini_zero_shot_predicted_label": gemini_by_source_id[source_id]["zero_shot_predicted_label"],
                    "gemini_few_shot_predicted_label": gemini_by_source_id[source_id]["few_shot_predicted_label"],
                    "sonnet_zero_shot_predicted_label": sonnet_zero_prediction,
                    "sonnet_few_shot_predicted_label": sonnet_few_prediction,
                }
            )


def write_summary_markdown(
    path: Path,
    *,
    config: HardSubsetConfig,
    subset_rows: list[dict[str, str]],
    support_rows: list[dict[str, str]],
    classical_subset: dict[str, Any],
    gemini_zero_subset: dict[str, Any],
    gemini_few_subset: dict[str, Any],
    sonnet_zero: dict[str, Any],
    sonnet_few: dict[str, Any],
) -> None:
    llm_config = config.llm_config
    per_class_rows = []
    for label in llm_config.label_order:
        zero_report = sonnet_zero["metrics"]["classification_report"][label]
        few_report = sonnet_few["metrics"]["classification_report"][label]
        per_class_rows.append(
            [
                label,
                f"{zero_report['precision']:.4f}",
                f"{zero_report['recall']:.4f}",
                f"{zero_report['f1-score']:.4f}",
                f"{few_report['precision']:.4f}",
                f"{few_report['recall']:.4f}",
                f"{few_report['f1-score']:.4f}",
                str(int(few_report["support"])),
            ]
        )
    support_rows_md = [
        [row.get("source_id", "") or "-", row[llm_config.target_column], row[llm_config.text_column]]
        for row in support_rows
    ]
    lines = [
        "# Claude Sonnet Hard Subset Summary",
        "",
        "This report evaluates Claude Sonnet on the hard subset only, then compares Sonnet against the classical baseline and Gemini Flash-Lite on the same rows.",
        "",
        "## Setup",
        "",
        f"- Provider: `{llm_config.provider_name}`",
        f"- Model: `{llm_config.model}`",
        f"- Train path: `{llm_config.train_path.as_posix()}`",
        f"- Dev path: `{llm_config.dev_path.as_posix()}`",
        f"- Hard subset rows: `{len(subset_rows)}`",
        f"- Text column: `{llm_config.text_column}`",
        f"- Target column: `{llm_config.target_column}`",
        f"- Few-shot examples per class: `{llm_config.few_shot_examples_per_class}`",
        "",
        "## Overall Comparison On The Hard Subset",
        "",
        *_markdown_table(
            ["Mode", "Accuracy", "Macro F1"],
            [
                ["Classical", f"{classical_subset['metrics']['accuracy']:.4f}", f"{classical_subset['metrics']['macro_f1']:.4f}"],
                ["Gemini Zero-Shot", f"{gemini_zero_subset['metrics']['accuracy']:.4f}", f"{gemini_zero_subset['metrics']['macro_f1']:.4f}"],
                ["Gemini Few-Shot", f"{gemini_few_subset['metrics']['accuracy']:.4f}", f"{gemini_few_subset['metrics']['macro_f1']:.4f}"],
                ["Sonnet Zero-Shot", f"{sonnet_zero['metrics']['accuracy']:.4f}", f"{sonnet_zero['metrics']['macro_f1']:.4f}"],
                ["Sonnet Few-Shot", f"{sonnet_few['metrics']['accuracy']:.4f}", f"{sonnet_few['metrics']['macro_f1']:.4f}"],
            ],
        ),
        "",
        "## Sonnet Latency And Cost",
        "",
        *_markdown_table(
            ["Mode", "Requests", "Total ms", "Avg request ms", "Avg row ms", "Estimated Cost (USD)"],
            [
                [
                    "Zero-Shot",
                    str(sonnet_zero["latency"]["request_count"]),
                    f"{sonnet_zero['latency']['total_ms']:.1f}",
                    f"{sonnet_zero['latency']['avg_request_ms']:.1f}",
                    f"{sonnet_zero['latency']['avg_row_ms']:.1f}",
                    f"{sonnet_zero['estimated_cost_usd']:.4f}" if sonnet_zero["estimated_cost_usd"] is not None else "N/A",
                ],
                [
                    "Few-Shot",
                    str(sonnet_few["latency"]["request_count"]),
                    f"{sonnet_few['latency']['total_ms']:.1f}",
                    f"{sonnet_few['latency']['avg_request_ms']:.1f}",
                    f"{sonnet_few['latency']['avg_row_ms']:.1f}",
                    f"{sonnet_few['estimated_cost_usd']:.4f}" if sonnet_few["estimated_cost_usd"] is not None else "N/A",
                ],
            ],
        ),
        "",
        "## Sonnet Per-Class Metrics",
        "",
        *_markdown_table(
            ["Label", "Zero P", "Zero R", "Zero F1", "Few P", "Few R", "Few F1", "Support"],
            per_class_rows,
        ),
        "",
        "## Few-Shot Support Set",
        "",
        *_markdown_table(["Source ID", "Label", "Processed Text"], support_rows_md),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_error_analysis_markdown(
    path: Path,
    *,
    config: HardSubsetConfig,
    subset_rows: list[dict[str, str]],
    classical_subset: dict[str, Any],
    gemini_zero_subset: dict[str, Any],
    gemini_few_subset: dict[str, Any],
    sonnet_zero: dict[str, Any],
    sonnet_few: dict[str, Any],
) -> None:
    llm_config = config.llm_config
    lines = [
        "# Claude Sonnet Hard Subset Error Analysis",
        "",
        "This report compares the tracked Saudi/Egyptian absorption confusions on the hard subset only.",
        "",
        "## Requested Confusion Directions",
        "",
    ]
    rows = []
    classical_confusions = _confusion_counts(
        rows=subset_rows,
        predictions=classical_subset["predictions"],
        config=llm_config,
    )
    gemini_zero_confusions = _confusion_counts(
        rows=subset_rows,
        predictions=gemini_zero_subset["predictions"],
        config=llm_config,
    )
    gemini_few_confusions = _confusion_counts(
        rows=subset_rows,
        predictions=gemini_few_subset["predictions"],
        config=llm_config,
    )
    sonnet_zero_confusions = _confusion_counts(
        rows=subset_rows,
        predictions=sonnet_zero["predictions"],
        config=llm_config,
    )
    sonnet_few_confusions = _confusion_counts(
        rows=subset_rows,
        predictions=sonnet_few["predictions"],
        config=llm_config,
    )
    for true_label, predicted_label in ERROR_ANALYSIS_DIRECTIONS:
        rows.append(
            [
                true_label,
                predicted_label,
                str(classical_confusions.get((true_label, predicted_label), 0)),
                str(gemini_zero_confusions.get((true_label, predicted_label), 0)),
                str(gemini_few_confusions.get((true_label, predicted_label), 0)),
                str(sonnet_zero_confusions.get((true_label, predicted_label), 0)),
                str(sonnet_few_confusions.get((true_label, predicted_label), 0)),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "True Label",
                "Predicted Label",
                "Classical",
                "Gemini Zero",
                "Gemini Few",
                "Sonnet Zero",
                "Sonnet Few",
            ],
            rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The hard subset contains only rows that were already hard for at least one baseline in the four tracked Saudi/Egyptian absorption directions.",
            "- Classical performance is expected to be near-floor on this subset because the selection rule is anchored on its known failure modes.",
            "- The key question is whether Sonnet reduces those same absorption errors relative to both the classical baseline and Gemini on exactly the same rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_markdown(
    path: Path,
    *,
    subset_rows: list[dict[str, str]],
    classical_subset: dict[str, Any],
    gemini_zero_subset: dict[str, Any],
    gemini_few_subset: dict[str, Any],
    sonnet_zero: dict[str, Any],
    sonnet_few: dict[str, Any],
) -> None:
    lines = [
        "# Hard Subset Comparison",
        "",
        f"This comparison uses the same `{len(subset_rows)}` hard-subset rows for every model.",
        "",
        "## Overall Metrics",
        "",
        *_markdown_table(
            ["Mode", "Accuracy", "Macro F1"],
            [
                ["Classical", f"{classical_subset['metrics']['accuracy']:.4f}", f"{classical_subset['metrics']['macro_f1']:.4f}"],
                ["Gemini Zero-Shot", f"{gemini_zero_subset['metrics']['accuracy']:.4f}", f"{gemini_zero_subset['metrics']['macro_f1']:.4f}"],
                ["Gemini Few-Shot", f"{gemini_few_subset['metrics']['accuracy']:.4f}", f"{gemini_few_subset['metrics']['macro_f1']:.4f}"],
                ["Sonnet Zero-Shot", f"{sonnet_zero['metrics']['accuracy']:.4f}", f"{sonnet_zero['metrics']['macro_f1']:.4f}"],
                ["Sonnet Few-Shot", f"{sonnet_few['metrics']['accuracy']:.4f}", f"{sonnet_few['metrics']['macro_f1']:.4f}"],
            ],
        ),
        "",
        "## Comparison Readout",
        "",
        "- Use this document to decide whether Claude Sonnet is strong enough on the targeted hard cases to justify a full-dev run.",
        "- If Sonnet materially improves over both Gemini and the classical baseline on this subset, a full-dev evaluation is justified.",
        "- If Sonnet only matches Gemini or improves marginally, the extra full-dev cost is harder to justify before moving on to the encoder phase.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(config: HardSubsetConfig, results: dict[str, Any]) -> dict[str, Path]:
    report_dir = config.llm_config.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.llm_config.report_prefix
    subset_definition_path = report_dir / f"{prefix}_definition.md"
    summary_path = report_dir / f"{prefix}_summary.md"
    metrics_path = report_dir / f"{prefix}_metrics.json"
    classification_report_path = report_dir / f"{prefix}_classification_report.json"
    predictions_path = report_dir / f"{prefix}_predictions.csv"
    error_analysis_path = report_dir / f"{prefix}_error_analysis.md"
    comparison_path = report_dir / "llm_hard_subset_comparison.md"

    write_subset_definition_markdown(
        subset_definition_path,
        config=config,
        dev_rows=results["dev_rows_data"],
        subset_rows=results["subset_rows"],
        reasons_by_source_id=results["reasons_by_source_id"],
        classical_rows=results["classical_rows"],
        gemini_rows=results["gemini_rows"],
    )
    write_summary_markdown(
        summary_path,
        config=config,
        subset_rows=results["subset_rows"],
        support_rows=results["support_rows"],
        classical_subset=results["classical_subset"],
        gemini_zero_subset=results["gemini_zero_subset"],
        gemini_few_subset=results["gemini_few_subset"],
        sonnet_zero=results["sonnet_zero"],
        sonnet_few=results["sonnet_few"],
    )
    metrics_payload = {
        "provider": config.llm_config.provider_name,
        "model": config.llm_config.model,
        "hard_subset_rows": len(results["subset_rows"]),
        "classical_subset": {
            "accuracy": results["classical_subset"]["metrics"]["accuracy"],
            "macro_f1": results["classical_subset"]["metrics"]["macro_f1"],
        },
        "gemini_zero_subset": {
            "accuracy": results["gemini_zero_subset"]["metrics"]["accuracy"],
            "macro_f1": results["gemini_zero_subset"]["metrics"]["macro_f1"],
        },
        "gemini_few_subset": {
            "accuracy": results["gemini_few_subset"]["metrics"]["accuracy"],
            "macro_f1": results["gemini_few_subset"]["metrics"]["macro_f1"],
        },
        "zero_shot": {
            "accuracy": results["sonnet_zero"]["metrics"]["accuracy"],
            "macro_f1": results["sonnet_zero"]["metrics"]["macro_f1"],
            "usage": results["sonnet_zero"]["usage"],
            "latency": results["sonnet_zero"]["latency"],
            "estimated_cost_usd": results["sonnet_zero"]["estimated_cost_usd"],
        },
        "few_shot": {
            "accuracy": results["sonnet_few"]["metrics"]["accuracy"],
            "macro_f1": results["sonnet_few"]["metrics"]["macro_f1"],
            "usage": results["sonnet_few"]["usage"],
            "latency": results["sonnet_few"]["latency"],
            "estimated_cost_usd": results["sonnet_few"]["estimated_cost_usd"],
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    classification_payload = {
        "classical_subset": results["classical_subset"]["metrics"]["classification_report"],
        "gemini_zero_subset": results["gemini_zero_subset"]["metrics"]["classification_report"],
        "gemini_few_subset": results["gemini_few_subset"]["metrics"]["classification_report"],
        "zero_shot": results["sonnet_zero"]["metrics"]["classification_report"],
        "few_shot": results["sonnet_few"]["metrics"]["classification_report"],
    }
    classification_report_path.write_text(
        json.dumps(classification_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_predictions_csv(
        predictions_path,
        subset_rows=results["subset_rows"],
        reasons_by_source_id=results["reasons_by_source_id"],
        classical_rows=results["classical_rows"],
        gemini_rows=results["gemini_rows"],
        sonnet_zero_predictions=results["sonnet_zero"]["predictions"],
        sonnet_few_predictions=results["sonnet_few"]["predictions"],
        config=config.llm_config,
    )
    write_error_analysis_markdown(
        error_analysis_path,
        config=config,
        subset_rows=results["subset_rows"],
        classical_subset=results["classical_subset"],
        gemini_zero_subset=results["gemini_zero_subset"],
        gemini_few_subset=results["gemini_few_subset"],
        sonnet_zero=results["sonnet_zero"],
        sonnet_few=results["sonnet_few"],
    )
    write_comparison_markdown(
        comparison_path,
        subset_rows=results["subset_rows"],
        classical_subset=results["classical_subset"],
        gemini_zero_subset=results["gemini_zero_subset"],
        gemini_few_subset=results["gemini_few_subset"],
        sonnet_zero=results["sonnet_zero"],
        sonnet_few=results["sonnet_few"],
    )
    return {
        "subset_definition_markdown": subset_definition_path,
        "summary_markdown": summary_path,
        "metrics_json": metrics_path,
        "classification_report_json": classification_report_path,
        "predictions_csv": predictions_path,
        "error_analysis_markdown": error_analysis_path,
        "comparison_markdown": comparison_path,
    }


def evaluate_hard_subset(config: HardSubsetConfig) -> dict[str, Any]:
    llm_config = config.llm_config
    train_rows = load_labeled_rows(
        llm_config.train_path,
        text_column=llm_config.text_column,
        target_column=llm_config.target_column,
    )
    dev_rows = load_labeled_rows(
        llm_config.dev_path,
        text_column=llm_config.text_column,
        target_column=llm_config.target_column,
    )
    classical_rows = load_prediction_rows(config.classical_predictions_path)
    gemini_rows = load_prediction_rows(config.gemini_predictions_path)
    subset_rows, reasons_by_source_id = select_hard_subset(
        dev_rows=dev_rows,
        classical_rows=classical_rows,
        gemini_rows=gemini_rows,
    )
    support_rows = select_few_shot_support(train_rows, llm_config)
    classical_subset = evaluate_existing_predictions(
        subset_rows=subset_rows,
        prediction_rows=classical_rows,
        prediction_column="predicted_label",
        config=llm_config,
    )
    gemini_zero_subset = evaluate_existing_predictions(
        subset_rows=subset_rows,
        prediction_rows=gemini_rows,
        prediction_column="zero_shot_predicted_label",
        config=llm_config,
    )
    gemini_few_subset = evaluate_existing_predictions(
        subset_rows=subset_rows,
        prediction_rows=gemini_rows,
        prediction_column="few_shot_predicted_label",
        config=llm_config,
    )
    sonnet_zero = run_prompt_mode(
        config=llm_config,
        dev_rows=subset_rows,
        mode="zero_shot",
        support_rows=None,
    )
    sonnet_few = run_prompt_mode(
        config=llm_config,
        dev_rows=subset_rows,
        mode="few_shot",
        support_rows=support_rows,
    )
    return {
        "dev_rows_data": dev_rows,
        "classical_rows": classical_rows,
        "gemini_rows": gemini_rows,
        "subset_rows": subset_rows,
        "reasons_by_source_id": reasons_by_source_id,
        "support_rows": support_rows,
        "classical_subset": classical_subset,
        "gemini_zero_subset": gemini_zero_subset,
        "gemini_few_subset": gemini_few_subset,
        "sonnet_zero": sonnet_zero,
        "sonnet_few": sonnet_few,
    }


def run_hard_subset(config_path: Path) -> dict[str, Path]:
    config = load_config(config_path)
    results = evaluate_hard_subset(config)
    return write_reports(config, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Claude Sonnet on the hard subset only.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/llm_sonnet_hard_subset.yaml"),
        help="Path to the hard-subset YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_hard_subset(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
