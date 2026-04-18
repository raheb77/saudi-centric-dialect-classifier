from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "encoders" / "stability_summary.py"
SPEC = importlib.util.spec_from_file_location("project_stability_summary", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_write_final_model_comparison_uses_dynamic_split_labels(tmp_path: Path) -> None:
    comparison_path = tmp_path / "final_model_comparison.md"
    baselines = {
        "classical": {"accuracy": 0.8867735470941884, "macro_f1": 0.8476041092943942, "dev_rows": 998},
        "gemini": {
            "dev_rows": 999,
            "zero_shot": {"accuracy": 0.8678678678678678, "macro_f1": 0.8330107302184784},
            "few_shot": {"accuracy": 0.8748748748748749, "macro_f1": 0.8414330449563122},
        },
        "sonnet": {
            "dev_rows": 999,
            "zero_shot": {"accuracy": 0.8268268268268268, "macro_f1": 0.790765166649751},
            "few_shot": {"accuracy": 0.8408408408408409, "macro_f1": 0.8042180735440324},
        },
    }
    aggregates = {
        "accuracy": {"values": [0.966933867735471, 0.9719438877755511, 0.9649298597194389]},
        "macro_f1": {"values": [0.959520211095972, 0.9682733122263675, 0.9561414061130634]},
    }

    MODULE.write_final_model_comparison(
        comparison_path,
        aggregates=aggregates,
        baselines=baselines,
        reference_dev_rows=998,
    )

    comparison_text = comparison_path.read_text(encoding="utf-8")

    assert "cleaned benchmark-safe dev (998 rows)" in comparison_text
    assert "original full-dev (999 rows)" in comparison_text
    assert "apples-to-apples on 998 rows" in comparison_text


def test_parse_args_does_not_overwrite_final_model_comparison_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["stability_summary.py"])

    args = MODULE.parse_args()

    assert args.comparison_out is None
