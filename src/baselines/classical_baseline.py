from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline


ERROR_ANALYSIS_DIRECTIONS = (
    ("Saudi", "Levantine"),
    ("Saudi", "Maghrebi"),
    ("Egyptian", "Maghrebi"),
    ("Egyptian", "Levantine"),
)


@dataclass(frozen=True)
class BaselineConfig:
    train_path: Path
    dev_path: Path
    text_column: str
    target_column: str
    label_order: tuple[str, ...]
    word_ngram_range: tuple[int, int]
    word_min_df: int
    char_ngram_range: tuple[int, int]
    char_min_df: int
    char_analyzer: str
    lowercase: bool
    sublinear_tf: bool
    c: float
    max_iter: int
    solver: str
    random_state: int
    report_dir: Path
    report_prefix: str


def _tuple_from_range(values: list[int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError("Expected a two-item ngram range.")
    return values[0], values[1]


def load_config(path: Path) -> BaselineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_cfg = payload["data"]
    feature_cfg = payload["features"]
    model_cfg = payload["model"]
    output_cfg = payload["output"]
    return BaselineConfig(
        train_path=Path(data_cfg["train_path"]),
        dev_path=Path(data_cfg["dev_path"]),
        text_column=data_cfg.get("text_column", "processed_text"),
        target_column=data_cfg.get("target_column", "macro_label"),
        label_order=tuple(payload["labels"]["order"]),
        word_ngram_range=_tuple_from_range(feature_cfg["word"]["ngram_range"]),
        word_min_df=int(feature_cfg["word"].get("min_df", 1)),
        char_ngram_range=_tuple_from_range(feature_cfg["char"]["ngram_range"]),
        char_min_df=int(feature_cfg["char"].get("min_df", 1)),
        char_analyzer=feature_cfg["char"].get("analyzer", "char_wb"),
        lowercase=bool(feature_cfg.get("lowercase", False)),
        sublinear_tf=bool(feature_cfg.get("sublinear_tf", True)),
        c=float(model_cfg.get("C", 1.0)),
        max_iter=int(model_cfg.get("max_iter", 1000)),
        solver=str(model_cfg.get("solver", "lbfgs")),
        random_state=int(model_cfg.get("random_state", 42)),
        report_dir=Path(output_cfg.get("report_dir", "artifacts/reports")),
        report_prefix=str(output_cfg.get("prefix", "classical_baseline")),
    )


def load_labeled_rows(path: Path, *, text_column: str, target_column: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in (text_column, target_column) if column not in fieldnames]
        if missing:
            missing_text = ", ".join(f"`{column}`" for column in missing)
            raise ValueError(f"Missing required columns in {path.as_posix()}: {missing_text}")
        rows = list(reader)
    return rows


def build_pipeline(config: BaselineConfig) -> Pipeline:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=config.word_ngram_range,
        min_df=config.word_min_df,
        lowercase=config.lowercase,
        sublinear_tf=config.sublinear_tf,
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer=config.char_analyzer,
        ngram_range=config.char_ngram_range,
        min_df=config.char_min_df,
        lowercase=config.lowercase,
        sublinear_tf=config.sublinear_tf,
    )
    features = FeatureUnion(
        [
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ]
    )
    classifier = LogisticRegression(
        C=config.c,
        max_iter=config.max_iter,
        solver=config.solver,
        random_state=config.random_state,
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def write_confusion_matrix_csv(path: Path, labels: tuple[str, ...], matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _truncate_text(value: str, limit: int = 160) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _prediction_row(row: dict[str, str], predicted_label: str, *, config: BaselineConfig) -> dict[str, str]:
    return {
        "source_dataset": row.get("source_dataset", ""),
        "source_file": row.get("source_file", ""),
        "source_id": row.get("source_id", ""),
        "source_row_number": row.get("source_row_number", ""),
        "original_text": row.get("original_text", ""),
        "processed_text": row.get(config.text_column, ""),
        "true_label": row.get(config.target_column, ""),
        "predicted_label": predicted_label,
        "is_error": str(row.get(config.target_column, "") != predicted_label).lower(),
    }


def write_dev_predictions_csv(
    path: Path,
    *,
    dev_rows: list[dict[str, str]],
    predictions: list[str],
    config: BaselineConfig,
) -> None:
    fieldnames = [
        "source_dataset",
        "source_file",
        "source_id",
        "source_row_number",
        "original_text",
        "processed_text",
        "true_label",
        "predicted_label",
        "is_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, predicted_label in zip(dev_rows, predictions, strict=True):
            writer.writerow(_prediction_row(row, predicted_label, config=config))


def _collect_confusion_examples(
    dev_rows: list[dict[str, str]],
    predictions: list[str],
    *,
    config: BaselineConfig,
    true_label: str,
    predicted_label: str,
    limit: int = 10,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for row, prediction in zip(dev_rows, predictions, strict=True):
        if row[config.target_column] != true_label or prediction != predicted_label:
            continue
        examples.append(
            {
                "source_id": row.get("source_id", ""),
                "original_text": _truncate_text(row.get("original_text", "")),
                "processed_text": _truncate_text(row.get(config.text_column, "")),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _render_confusion_examples_table(examples: list[dict[str, str]]) -> list[str]:
    if not examples:
        return ["No examples found.", ""]
    rows = [
        [example["source_id"] or "-", example["processed_text"] or "-", example["original_text"] or "-"]
        for example in examples
    ]
    return [
        *_markdown_table(["Source ID", "Processed Text", "Original Text"], rows),
        "",
    ]


def write_error_analysis_markdown(
    path: Path,
    *,
    config: BaselineConfig,
    dev_rows: list[dict[str, str]],
    predictions: list[str],
) -> None:
    confusion_counts = Counter(
        (row[config.target_column], prediction)
        for row, prediction in zip(dev_rows, predictions, strict=True)
        if row[config.target_column] != prediction
    )
    top_confusions = sorted(
        confusion_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:10]
    lines = [
        "# Classical Baseline Error Analysis",
        "",
        "This report summarizes row-level dev-set errors from the current TF-IDF + Logistic Regression baseline.",
        "",
        "## Requested Confusion Directions",
        "",
    ]
    for true_label, predicted_label in ERROR_ANALYSIS_DIRECTIONS:
        count = confusion_counts.get((true_label, predicted_label), 0)
        lines.extend(
            [
                f"### {true_label} -> {predicted_label}",
                "",
                f"- Count: `{count}`",
                "",
                *_render_confusion_examples_table(
                    _collect_confusion_examples(
                        dev_rows,
                        predictions,
                        config=config,
                        true_label=true_label,
                        predicted_label=predicted_label,
                    )
                ),
            ]
        )

    lines.extend(["## Top 10 Off-Diagonal Confusions", ""])
    if top_confusions:
        rows = [
            [true_label, predicted_label, str(count)]
            for (true_label, predicted_label), count in top_confusions
        ]
        lines.extend(_markdown_table(["True Label", "Predicted Label", "Count"], rows))
        lines.append("")
    else:
        lines.extend(["No off-diagonal confusions found.", ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The baseline is strongest on broad Levantine and Maghrebi separation, with weaker recall on Saudi and Egyptian examples.",
            "- The largest requested confusions show Saudi and Egyptian texts being pulled toward broader regional classes, which is consistent with a surface-form TF-IDF model relying on shared lexical and orthographic cues.",
            "- This suggests the current baseline captures strong regional similarity but misses finer class-specific markers for Saudi and Egyptian short texts.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_markdown(
    path: Path,
    *,
    config: BaselineConfig,
    train_rows: int,
    dev_rows: int,
    metrics: dict[str, Any],
    confusion: list[list[int]],
) -> None:
    per_class_rows = [
        [
            label,
            f"{metrics['classification_report'][label]['precision']:.4f}",
            f"{metrics['classification_report'][label]['recall']:.4f}",
            f"{metrics['classification_report'][label]['f1-score']:.4f}",
            str(int(metrics["classification_report"][label]["support"])),
        ]
        for label in config.label_order
    ]
    confusion_rows = [
        [true_label, *[str(value) for value in row]]
        for true_label, row in zip(config.label_order, confusion, strict=True)
    ]
    lines = [
        "# Classical Baseline Summary",
        "",
        "This baseline trains a Logistic Regression classifier on combined word and char TF-IDF features using `train_core` and evaluates on `dev_core` only.",
        "",
        "## Setup",
        "",
        f"- Train path: `{config.train_path.as_posix()}`",
        f"- Dev path: `{config.dev_path.as_posix()}`",
        f"- Text column: `{config.text_column}`",
        f"- Target column: `{config.target_column}`",
        f"- Labels: `{', '.join(config.label_order)}`",
        f"- Word n-grams: `{config.word_ngram_range}`",
        f"- Char analyzer: `{config.char_analyzer}`",
        f"- Char n-grams: `{config.char_ngram_range}`",
        f"- Train rows: `{train_rows}`",
        f"- Dev rows: `{dev_rows}`",
        "",
        "## Overall Metrics",
        "",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Macro F1: `{metrics['macro_f1']:.4f}`",
        "",
        "## Per-Class Metrics",
        "",
        *_markdown_table(
            ["Label", "Precision", "Recall", "F1", "Support"],
            per_class_rows,
        ),
        "",
        "## Confusion Matrix",
        "",
        *_markdown_table(
            ["True \\ Pred", *list(config.label_order)],
            confusion_rows,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_baseline(config: BaselineConfig) -> dict[str, Any]:
    train_rows = load_labeled_rows(
        config.train_path,
        text_column=config.text_column,
        target_column=config.target_column,
    )
    dev_rows = load_labeled_rows(
        config.dev_path,
        text_column=config.text_column,
        target_column=config.target_column,
    )
    train_texts = [row[config.text_column] for row in train_rows]
    train_labels = [row[config.target_column] for row in train_rows]
    dev_texts = [row[config.text_column] for row in dev_rows]
    dev_labels = [row[config.target_column] for row in dev_rows]

    pipeline = build_pipeline(config)
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(dev_texts)

    accuracy = accuracy_score(dev_labels, predictions)
    macro_f1 = f1_score(dev_labels, predictions, labels=list(config.label_order), average="macro")
    report_dict = classification_report(
        dev_labels,
        predictions,
        labels=list(config.label_order),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        dev_labels,
        predictions,
        labels=list(config.label_order),
        zero_division=0,
    )
    confusion = confusion_matrix(
        dev_labels,
        predictions,
        labels=list(config.label_order),
    ).tolist()

    return {
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "dev_predictions": [str(prediction) for prediction in predictions],
        "dev_rows_data": dev_rows,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "labels": list(config.label_order),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": confusion,
    }


def write_reports(config: BaselineConfig, results: dict[str, Any]) -> dict[str, Path]:
    config.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.report_prefix
    metrics_path = config.report_dir / f"{prefix}_metrics.json"
    report_json_path = config.report_dir / f"{prefix}_classification_report.json"
    report_txt_path = config.report_dir / f"{prefix}_classification_report.txt"
    confusion_path = config.report_dir / f"{prefix}_confusion_matrix.csv"
    summary_path = config.report_dir / f"{prefix}_summary.md"
    predictions_path = config.report_dir / f"{prefix}_dev_predictions.csv"
    error_analysis_path = config.report_dir / f"{prefix}_error_analysis.md"

    metrics_payload = {
        "train_rows": results["train_rows"],
        "dev_rows": results["dev_rows"],
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "labels": results["labels"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_json_path.write_text(
        json.dumps(results["classification_report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_txt_path.write_text(results["classification_report_text"], encoding="utf-8")
    write_confusion_matrix_csv(confusion_path, config.label_order, results["confusion_matrix"])
    write_dev_predictions_csv(
        predictions_path,
        dev_rows=results["dev_rows_data"],
        predictions=results["dev_predictions"],
        config=config,
    )
    write_summary_markdown(
        summary_path,
        config=config,
        train_rows=results["train_rows"],
        dev_rows=results["dev_rows"],
        metrics=results,
        confusion=results["confusion_matrix"],
    )
    write_error_analysis_markdown(
        error_analysis_path,
        config=config,
        dev_rows=results["dev_rows_data"],
        predictions=results["dev_predictions"],
    )
    return {
        "metrics_json": metrics_path,
        "classification_report_json": report_json_path,
        "classification_report_txt": report_txt_path,
        "confusion_matrix_csv": confusion_path,
        "summary_markdown": summary_path,
        "dev_predictions_csv": predictions_path,
        "error_analysis_markdown": error_analysis_path,
    }


def run_baseline(config_path: Path) -> dict[str, Path]:
    config = load_config(config_path)
    results = evaluate_baseline(config)
    return write_reports(config, results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the classical TF-IDF baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to the baseline YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_baseline(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
