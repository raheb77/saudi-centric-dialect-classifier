from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "src" / "encoders" / "encoder_baseline.py"
SPEC = importlib.util.spec_from_file_location("project_encoder_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def __call__(self, text, *, truncation, padding, max_length, return_tensors=None):
        del truncation, return_tensors
        if isinstance(text, list):
            return {
                "input_ids": [[1, 2, 3] for _ in text],
                "attention_mask": [[1, 1, 1] for _ in text],
            }
        if padding == "max_length":
            return {
                "input_ids": torch.arange(max_length, dtype=torch.long).unsqueeze(0),
                "attention_mask": torch.ones((1, max_length), dtype=torch.long),
            }
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }

    def pad(self, features, *, padding, return_tensors):
        del padding, return_tensors
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        for feature in features:
            ids = feature["input_ids"]
            mask = feature["attention_mask"]
            pad_len = max_len - len(ids)
            input_ids.append(torch.tensor(list(ids) + [0] * pad_len, dtype=torch.long))
            attention_mask.append(torch.tensor(list(mask) + [0] * pad_len, dtype=torch.long))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
        }


def test_config_parsing_and_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "marbert.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_path": "data/processed/train_core.csv",
                    "dev_path": "data/processed/dev_core.csv",
                    "text_column": "processed_text",
                    "target_column": "macro_label",
                },
                "labels": {"order": ["Saudi", "Egyptian", "Levantine", "Maghrebi"]},
                "model": {"name_or_path": "UBC-NLP/MARBERT", "num_labels": 4, "max_length": 128},
                "training": {
                    "batch_size": 32,
                    "eval_batch_size": 64,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "num_epochs": 5,
                    "warmup_ratio": 0.1,
                    "gradient_accumulation_steps": 1,
                    "max_grad_norm": 1.0,
                    "num_workers": 0,
                },
                "optimizer": {"type": "adamw_torch"},
                "scheduler": {"type": "linear_with_warmup"},
                "early_stopping": {
                    "metric": "eval_macro_f1",
                    "mode": "max",
                    "patience": 2,
                    "min_delta": 0.001,
                },
                "loss": {"type": "cross_entropy", "class_weights": "balanced"},
                "seed": 42,
                "reproducibility": {
                    "deterministic_algorithms": True,
                    "cudnn_deterministic": True,
                    "cudnn_benchmark": False,
                },
                "output": {
                    "report_dir": "artifacts/reports",
                    "checkpoint_dir": "artifacts/checkpoints/marbert_base",
                    "prefix": "marbert",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = MODULE.load_config(config_path)

    assert config.model.name_or_path == "UBC-NLP/MARBERT"
    assert config.model.num_labels == 4
    assert config.loss.class_weights == "balanced"
    assert config.seed == 42


def test_dataset_returns_expected_fields_and_shapes() -> None:
    rows = [
        {
            "processed_text": "هذا نص سعودي",
            "macro_label": "Saudi",
        }
    ]
    dataset = MODULE.TextClassificationDataset(
        rows,
        tokenizer=FakeTokenizer(),
        text_column="processed_text",
        label_column="macro_label",
        label_to_id={"Saudi": 0, "Egyptian": 1, "Levantine": 2, "Maghrebi": 3},
        max_length=16,
    )

    item = dataset[0]

    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (3,)
    assert item["attention_mask"].shape == (3,)
    assert item["labels"].shape == ()
    assert item["labels"].item() == 0


def test_forward_pass_outputs_batch_logits_shape() -> None:
    model = MODULE.build_tiny_test_model(num_labels=4)
    batch = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "attention_mask": torch.ones((2, 8), dtype=torch.long),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    loss_fn = torch.nn.CrossEntropyLoss()

    loss, logits = MODULE.forward_model_batch(model, batch, loss_fn)

    assert loss.ndim == 0
    assert logits.shape == (2, 4)
