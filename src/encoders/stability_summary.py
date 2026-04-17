from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


LABELS = ("Saudi", "Egyptian", "Levantine", "Maghrebi")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_mean_std(values: list[float]) -> str:
    return f"{mean(values):.4f} +/- {stdev(values):.4f}"


def collect_seed_runs(report_dir: Path, *, seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        payload = load_json(report_dir / f"marbert_seed_{seed}_metrics.json")
        payload["seed"] = seed
        runs.append(payload)
    return runs


def aggregate_seed_metrics(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy_values = [float(run["accuracy"]) for run in seed_runs]
    macro_f1_values = [float(run["macro_f1"]) for run in seed_runs]
    per_class_values = {
        label: [float(run["per_class_metrics"][label]["f1"]) for run in seed_runs]
        for label in LABELS
    }
    return {
        "n_seeds": len(seed_runs),
        "accuracy": {
            "mean": mean(accuracy_values),
            "std": stdev(accuracy_values),
            "values": accuracy_values,
        },
        "macro_f1": {
            "mean": mean(macro_f1_values),
            "std": stdev(macro_f1_values),
            "values": macro_f1_values,
        },
        "per_class_f1": {
            label: {
                "mean": mean(values),
                "std": stdev(values),
                "values": values,
            }
            for label, values in per_class_values.items()
        },
    }


def stability_interpretation(macro_f1_std: float) -> str:
    if macro_f1_std < 0.005:
        return "highly stable"
    if macro_f1_std <= 0.015:
        return "acceptable stability"
    return "unstable, requires investigation"


def write_stability_summary_markdown(
    path: Path,
    *,
    seed_runs: list[dict[str, Any]],
    aggregates: dict[str, Any],
    baselines: dict[str, Any],
) -> None:
    lines = [
        "# MARBERT Stability Summary",
        "",
        "This summary aggregates the cleaned post-dedup MARBERT runs on the 998-row `dev_core` split.",
        "",
        "## Seed Runs",
        "",
        "| Seed | Dev rows | Accuracy | Macro F1 | Saudi F1 | Egyptian F1 | Levantine F1 | Maghrebi F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in seed_runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{run['seed']}`",
                    f"`{run['dev_rows']}`",
                    f"`{float(run['accuracy']):.4f}`",
                    f"`{float(run['macro_f1']):.4f}`",
                    f"`{float(run['per_class_metrics']['Saudi']['f1']):.4f}`",
                    f"`{float(run['per_class_metrics']['Egyptian']['f1']):.4f}`",
                    f"`{float(run['per_class_metrics']['Levantine']['f1']):.4f}`",
                    f"`{float(run['per_class_metrics']['Maghrebi']['f1']):.4f}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Mean ± Std",
            "",
            "| Metric | Mean ± Std |",
            "| --- | --- |",
            f"| Accuracy | `{format_mean_std(aggregates['accuracy']['values'])}` |",
            f"| Macro F1 | `{format_mean_std(aggregates['macro_f1']['values'])}` |",
            f"| Saudi F1 | `{format_mean_std(aggregates['per_class_f1']['Saudi']['values'])}` |",
            f"| Egyptian F1 | `{format_mean_std(aggregates['per_class_f1']['Egyptian']['values'])}` |",
            f"| Levantine F1 | `{format_mean_std(aggregates['per_class_f1']['Levantine']['values'])}` |",
            f"| Maghrebi F1 | `{format_mean_std(aggregates['per_class_f1']['Maghrebi']['values'])}` |",
            "",
            "## Interpretation",
            "",
            f"- Macro F1 stability: `{stability_interpretation(float(aggregates['macro_f1']['std']))}`",
            "- Thresholds used: `< 0.005` highly stable, `0.005-0.015` acceptable stability, `> 0.015` unstable.",
            "- The MARBERT multi-seed pass is intended to establish encoder stability on the cleaned benchmark-safe split.",
            "",
            "## Baseline Context",
            "",
            "| Model | Setting | Accuracy | Macro F1 | Run Type |",
            "| --- | --- | --- | --- | --- |",
            f"| Classical baseline | TF-IDF + Logistic Regression | `{float(baselines['classical']['accuracy']):.4f}` | `{float(baselines['classical']['macro_f1']):.4f}` | single run |",
            f"| Gemini Flash-Lite | zero-shot | `{float(baselines['gemini']['zero_shot']['accuracy']):.4f}` | `{float(baselines['gemini']['zero_shot']['macro_f1']):.4f}` | single run |",
            f"| Gemini Flash-Lite | few-shot | `{float(baselines['gemini']['few_shot']['accuracy']):.4f}` | `{float(baselines['gemini']['few_shot']['macro_f1']):.4f}` | single run |",
            f"| Claude Sonnet | zero-shot | `{float(baselines['sonnet']['zero_shot']['accuracy']):.4f}` | `{float(baselines['sonnet']['zero_shot']['macro_f1']):.4f}` | single run |",
            f"| Claude Sonnet | few-shot | `{float(baselines['sonnet']['few_shot']['accuracy']):.4f}` | `{float(baselines['sonnet']['few_shot']['macro_f1']):.4f}` | single run |",
            f"| MARBERT | mean +/- std (n=3 seeds) | `{format_mean_std(aggregates['accuracy']['values'])}` | `{format_mean_std(aggregates['macro_f1']['values'])}` | multi-seed |",
            "",
            "- Classical, Gemini, and Sonnet were not multi-seed rerun here. Their single-run values are shown for task context, not as a claim of LLM instability.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_stability_summary_json(path: Path, *, seed_runs: list[dict[str, Any]], aggregates: dict[str, Any]) -> None:
    payload = {
        "seed_runs": seed_runs,
        "aggregates": aggregates,
        "macro_f1_stability_label": stability_interpretation(float(aggregates["macro_f1"]["std"])),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_final_model_comparison(
    path: Path,
    *,
    aggregates: dict[str, Any],
    baselines: dict[str, Any],
) -> None:
    lines = [
        "# Final Model Comparison",
        "",
        "| Model | Setting | Accuracy | Macro F1 | Evaluation Split |",
        "| --- | --- | --- | --- | --- |",
        f"| Classical baseline | TF-IDF + Logistic Regression | `{float(baselines['classical']['accuracy']):.4f}` | `{float(baselines['classical']['macro_f1']):.4f}` | original dev (999 rows) |",
        f"| Gemini Flash-Lite | zero-shot | `{float(baselines['gemini']['zero_shot']['accuracy']):.4f}` | `{float(baselines['gemini']['zero_shot']['macro_f1']):.4f}` | original dev (999 rows) |",
        f"| Gemini Flash-Lite | few-shot | `{float(baselines['gemini']['few_shot']['accuracy']):.4f}` | `{float(baselines['gemini']['few_shot']['macro_f1']):.4f}` | original dev (999 rows) |",
        f"| Claude Sonnet | zero-shot | `{float(baselines['sonnet']['zero_shot']['accuracy']):.4f}` | `{float(baselines['sonnet']['zero_shot']['macro_f1']):.4f}` | original dev (999 rows) |",
        f"| Claude Sonnet | few-shot | `{float(baselines['sonnet']['few_shot']['accuracy']):.4f}` | `{float(baselines['sonnet']['few_shot']['macro_f1']):.4f}` | original dev (999 rows) |",
        f"| MARBERT | mean +/- std (n=3 seeds) | `{format_mean_std(aggregates['accuracy']['values'])}` | `{format_mean_std(aggregates['macro_f1']['values'])}` | cleaned dev (998 rows) |",
        "",
        "## Recommendation",
        "",
        "- MARBERT is strong enough to stand as the v1 encoder result if its multi-seed stability remains in the highly stable or acceptable range.",
        "- A second encoder is optional rather than required; it is justified only if you want architectural diversity or a stronger comparison story.",
        "",
        "MARBERT was re-evaluated on 998 rows after removing one dev row due to train/dev processed-text overlap. Classical, Gemini, and Sonnet were evaluated on the original 999-row dev split. The post-dedup MARBERT metrics are reported explicitly and should be interpreted as the benchmark-safe encoder result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed MARBERT results into stability and comparison reports.")
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/reports"))
    parser.add_argument("--stability-markdown-out", type=Path, default=Path("artifacts/reports/marbert_stability_summary.md"))
    parser.add_argument("--stability-json-out", type=Path, default=Path("artifacts/reports/marbert_stability_summary.json"))
    parser.add_argument("--comparison-out", type=Path, default=Path("artifacts/reports/final_model_comparison.md"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir
    baselines = {
        "classical": load_json(report_dir / "classical_baseline_metrics.json"),
        "gemini": load_json(report_dir / "llm_gemini_flash_lite_metrics.json"),
        "sonnet": load_json(report_dir / "llm_sonnet_full_dev_metrics.json"),
    }
    seed_runs = collect_seed_runs(report_dir, seeds=tuple(args.seeds))
    aggregates = aggregate_seed_metrics(seed_runs)
    write_stability_summary_markdown(
        args.stability_markdown_out,
        seed_runs=seed_runs,
        aggregates=aggregates,
        baselines=baselines,
    )
    write_stability_summary_json(
        args.stability_json_out,
        seed_runs=seed_runs,
        aggregates=aggregates,
    )
    write_final_model_comparison(
        args.comparison_out,
        aggregates=aggregates,
        baselines=baselines,
    )
    print(f"stability_markdown: {args.stability_markdown_out}")
    print(f"stability_json: {args.stability_json_out}")
    print(f"comparison_markdown: {args.comparison_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
