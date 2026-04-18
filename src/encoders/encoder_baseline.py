from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    get_scheduler,
)


ERROR_ANALYSIS_DIRECTIONS = (
    ("Saudi", "Levantine"),
    ("Saudi", "Maghrebi"),
    ("Egyptian", "Maghrebi"),
    ("Egyptian", "Levantine"),
)


@dataclass(frozen=True)
class DataConfig:
    train_path: Path
    dev_path: Path
    text_column: str
    target_column: str


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    num_labels: int
    max_length: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_ratio: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    num_workers: int


@dataclass(frozen=True)
class OptimizerConfig:
    type: str


@dataclass(frozen=True)
class SchedulerConfig:
    type: str


@dataclass(frozen=True)
class EarlyStoppingConfig:
    metric: str
    mode: str
    patience: int
    min_delta: float


@dataclass(frozen=True)
class LossConfig:
    type: str
    class_weights: str | None


@dataclass(frozen=True)
class ReproducibilityConfig:
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool


@dataclass(frozen=True)
class OutputConfig:
    report_dir: Path
    checkpoint_dir: Path
    prefix: str


@dataclass(frozen=True)
class EncoderConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    early_stopping: EarlyStoppingConfig
    loss: LossConfig
    reproducibility: ReproducibilityConfig
    label_order: tuple[str, ...]
    seed: int
    output: OutputConfig


class TextClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        tokenizer: Any,
        text_column: str,
        label_column: str,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.label_to_id = label_to_id
        self.max_length = max_length
        texts = [row[self.text_column] for row in self.rows]
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        self.encoded_rows = [
            {key: value[index] for key, value in encoded.items()}
            for index in range(len(self.rows))
        ]
        self.labels = torch.tensor(
            [self.label_to_id[row[self.label_column]] for row in self.rows],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value, dtype=torch.long) for key, value in self.encoded_rows[index].items()}
        item["labels"] = self.labels[index]
        return item


def load_config(path: Path) -> EncoderConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_cfg = payload["data"]
    model_cfg = payload["model"]
    training_cfg = payload["training"]
    optimizer_cfg = payload["optimizer"]
    scheduler_cfg = payload["scheduler"]
    early_cfg = payload["early_stopping"]
    loss_cfg = payload["loss"]
    output_cfg = payload["output"]
    reproducibility_cfg = payload.get("reproducibility", {})
    label_order = tuple(payload["labels"]["order"])
    config = EncoderConfig(
        data=DataConfig(
            train_path=Path(data_cfg["train_path"]),
            dev_path=Path(data_cfg["dev_path"]),
            text_column=data_cfg.get("text_column", "processed_text"),
            target_column=data_cfg.get("target_column", "macro_label"),
        ),
        model=ModelConfig(
            name_or_path=str(model_cfg["name_or_path"]),
            num_labels=int(model_cfg["num_labels"]),
            max_length=int(model_cfg["max_length"]),
        ),
        training=TrainingConfig(
            batch_size=int(training_cfg["batch_size"]),
            eval_batch_size=int(training_cfg["eval_batch_size"]),
            learning_rate=float(training_cfg["learning_rate"]),
            weight_decay=float(training_cfg["weight_decay"]),
            num_epochs=int(training_cfg["num_epochs"]),
            warmup_ratio=float(training_cfg["warmup_ratio"]),
            gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
            max_grad_norm=float(training_cfg["max_grad_norm"]),
            num_workers=int(training_cfg.get("num_workers", 0)),
        ),
        optimizer=OptimizerConfig(type=str(optimizer_cfg["type"])),
        scheduler=SchedulerConfig(type=str(scheduler_cfg["type"])),
        early_stopping=EarlyStoppingConfig(
            metric=str(early_cfg["metric"]),
            mode=str(early_cfg["mode"]),
            patience=int(early_cfg["patience"]),
            min_delta=float(early_cfg["min_delta"]),
        ),
        loss=LossConfig(
            type=str(loss_cfg["type"]),
            class_weights=(
                str(loss_cfg["class_weights"])
                if loss_cfg.get("class_weights") is not None
                else None
            ),
        ),
        reproducibility=ReproducibilityConfig(
            deterministic_algorithms=bool(reproducibility_cfg.get("deterministic_algorithms", True)),
            cudnn_deterministic=bool(reproducibility_cfg.get("cudnn_deterministic", True)),
            cudnn_benchmark=bool(reproducibility_cfg.get("cudnn_benchmark", False)),
        ),
        label_order=label_order,
        seed=int(payload.get("seed", 42)),
        output=OutputConfig(
            report_dir=Path(output_cfg.get("report_dir", "artifacts/reports")),
            checkpoint_dir=Path(output_cfg.get("checkpoint_dir", "artifacts/checkpoints/marbert_base")),
            prefix=str(output_cfg.get("prefix", "marbert")),
        ),
    )
    validate_config(config)
    return config


def validate_config(config: EncoderConfig) -> None:
    if len(config.label_order) != config.model.num_labels:
        raise ValueError("`model.num_labels` must match the number of labels in `labels.order`.")
    if config.optimizer.type != "adamw_torch":
        raise ValueError("Only `adamw_torch` is supported in v1.")
    if config.scheduler.type != "linear_with_warmup":
        raise ValueError("Only `linear_with_warmup` is supported in v1.")
    if config.loss.type != "cross_entropy":
        raise ValueError("Only `cross_entropy` loss is supported in v1.")
    if config.loss.class_weights not in {None, "balanced", "none"}:
        raise ValueError("`loss.class_weights` must be `balanced`, `none`, or omitted.")
    if config.early_stopping.metric != "eval_macro_f1":
        raise ValueError("Only `eval_macro_f1` early stopping is supported in v1.")
    if config.early_stopping.mode not in {"max", "min"}:
        raise ValueError("`early_stopping.mode` must be `max` or `min`.")
    if config.training.gradient_accumulation_steps < 1:
        raise ValueError("`gradient_accumulation_steps` must be at least 1.")
    if not 0.0 <= config.training.warmup_ratio <= 1.0:
        raise ValueError("`warmup_ratio` must be in [0, 1].")


def set_seed(seed: int, reproducibility: ReproducibilityConfig) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = reproducibility.cudnn_deterministic
        torch.backends.cudnn.benchmark = reproducibility.cudnn_benchmark
    try:
        torch.use_deterministic_algorithms(reproducibility.deterministic_algorithms, warn_only=True)
    except Exception:
        pass
    if not torch.cuda.is_available() and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        cpu_threads = os.cpu_count() or 1
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(cpu_threads, 8)))
        except RuntimeError:
            pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def load_tokenizer(name_or_path: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=False)


def build_model(config: EncoderConfig, *, device: torch.device) -> Any:
    label_to_id = {label: index for index, label in enumerate(config.label_order)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    model_kwargs: dict[str, Any] = {}
    if device.type == "mps":
        # MPS on this machine cannot fit MARBERT fine-tuning in fp32 at the requested batch size.
        # Loading fp16 parameters keeps the benchmark setup intact while reducing optimizer-state memory.
        model_kwargs["torch_dtype"] = torch.float16
    return AutoModelForSequenceClassification.from_pretrained(
        config.model.name_or_path,
        num_labels=config.model.num_labels,
        label2id=label_to_id,
        id2label=id_to_label,
        **model_kwargs,
    )


def build_dataloaders(
    config: EncoderConfig,
    *,
    tokenizer: Any,
    train_rows: list[dict[str, str]],
    dev_rows: list[dict[str, str]],
) -> tuple[DataLoader[dict[str, torch.Tensor]], DataLoader[dict[str, torch.Tensor]], dict[str, int]]:
    label_to_id = {label: index for index, label in enumerate(config.label_order)}
    train_dataset = TextClassificationDataset(
        train_rows,
        tokenizer=tokenizer,
        text_column=config.data.text_column,
        label_column=config.data.target_column,
        label_to_id=label_to_id,
        max_length=config.model.max_length,
    )
    dev_dataset = TextClassificationDataset(
        dev_rows,
        tokenizer=tokenizer,
        text_column=config.data.text_column,
        label_column=config.data.target_column,
        label_to_id=label_to_id,
        max_length=config.model.max_length,
    )
    def collate_fn(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = torch.stack([feature["labels"] for feature in features])
        model_features = [{key: value for key, value in feature.items() if key != "labels"} for feature in features]
        batch = tokenizer.pad(model_features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        worker_init_fn=seed_worker if config.training.num_workers > 0 else None,
        generator=generator,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        worker_init_fn=seed_worker if config.training.num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    return train_loader, dev_loader, label_to_id


def compute_loss_weights(
    config: EncoderConfig,
    *,
    train_rows: list[dict[str, str]],
    label_to_id: dict[str, int],
    device: torch.device,
) -> torch.Tensor | None:
    if config.loss.class_weights in {None, "none"}:
        return None
    labels = np.array([label_to_id[row[config.data.target_column]] for row in train_rows])
    classes = np.arange(len(config.label_order))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_loss_fn(weights: torch.Tensor | None) -> nn.Module:
    return nn.CrossEntropyLoss(weight=weights)


def create_optimizer(model: Any, config: EncoderConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def create_scheduler(optimizer: AdamW, *, config: EncoderConfig, total_training_steps: int) -> Any:
    scheduler_name = "linear" if config.scheduler.type == "linear_with_warmup" else config.scheduler.type
    warmup_steps = int(total_training_steps * config.training.warmup_ratio)
    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def maybe_clear_mps_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def forward_model_batch(model: Any, batch: dict[str, torch.Tensor], loss_fn: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**model_inputs)
    logits = outputs.logits
    loss = loss_fn(logits.float(), labels)
    return loss, logits


def train_one_epoch(
    model: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    optimizer: AdamW,
    scheduler: Any,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        loss, _ = forward_model_batch(model, batch, loss_fn)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        if step % gradient_accumulation_steps == 0 or step == len(loader):
            maybe_clear_mps_cache(device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            maybe_clear_mps_cache(device)
        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size * gradient_accumulation_steps
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def evaluate_model(
    model: Any,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    loss_fn: nn.Module,
    device: torch.device,
    label_order: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss, logits = forward_model_batch(model, batch, loss_fn)
            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, labels=list(range(len(label_order))), average="macro")
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(label_order))),
        target_names=list(label_order),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(label_order))),
    ).tolist()
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
        "labels": labels,
    }


def is_improved(current: float, best: float | None, *, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


def save_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    checkpoint_dir: Path,
    epoch: int,
    metric_value: float,
    history: list[dict[str, Any]],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "best_epoch": epoch,
        "best_metric": metric_value,
        "history": history,
    }
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_best_model(checkpoint_dir: Path, device: torch.device) -> Any:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    return model


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions_if_exists(path: Path, *, prediction_keys: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if all(key in row for key in prediction_keys)]


def compute_tracked_confusion_counts(
    true_labels: list[str],
    predicted_labels: list[str],
) -> dict[str, int]:
    counts = Counter(zip(true_labels, predicted_labels, strict=True))
    return {
        f"{true_label}->{predicted_label}": int(counts.get((true_label, predicted_label), 0))
        for true_label, predicted_label in ERROR_ANALYSIS_DIRECTIONS
    }


def build_per_class_metrics(
    report: dict[str, Any],
    *,
    label_order: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    return {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in label_order
    }


def build_comparison_rows(report_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    classical = _load_json_if_exists(report_dir / "classical_baseline_metrics.json")
    gemini = _load_json_if_exists(report_dir / "llm_gemini_flash_lite_metrics.json")
    sonnet = _load_json_if_exists(report_dir / "llm_sonnet_full_dev_metrics.json")
    if classical:
        rows.append(
            [
                "Classical",
                f"{float(classical.get('accuracy', 0.0)):.4f}",
                f"{float(classical.get('macro_f1', 0.0)):.4f}",
            ]
        )
    if gemini:
        rows.extend(
            [
                [
                    "Gemini Zero-Shot",
                    f"{float(gemini.get('zero_shot', {}).get('accuracy', 0.0)):.4f}",
                    f"{float(gemini.get('zero_shot', {}).get('macro_f1', 0.0)):.4f}",
                ],
                [
                    "Gemini Few-Shot",
                    f"{float(gemini.get('few_shot', {}).get('accuracy', 0.0)):.4f}",
                    f"{float(gemini.get('few_shot', {}).get('macro_f1', 0.0)):.4f}",
                ],
            ]
        )
    if sonnet:
        rows.extend(
            [
                [
                    "Sonnet Zero-Shot",
                    f"{float(sonnet.get('zero_shot', {}).get('accuracy', 0.0)):.4f}",
                    f"{float(sonnet.get('zero_shot', {}).get('macro_f1', 0.0)):.4f}",
                ],
                [
                    "Sonnet Few-Shot",
                    f"{float(sonnet.get('few_shot', {}).get('accuracy', 0.0)):.4f}",
                    f"{float(sonnet.get('few_shot', {}).get('macro_f1', 0.0)):.4f}",
                ],
            ]
        )
    return rows


def write_confusion_matrix_csv(path: Path, labels: tuple[str, ...], matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def write_summary_markdown(
    path: Path,
    *,
    config: EncoderConfig,
    train_rows: int,
    dev_rows: int,
    results: dict[str, Any],
    comparison_rows: list[list[str]],
) -> None:
    report = results["eval"]["classification_report"]
    per_class_rows = [
        [
            label,
            f"{report[label]['precision']:.4f}",
            f"{report[label]['recall']:.4f}",
            f"{report[label]['f1-score']:.4f}",
            str(int(report[label]["support"])),
        ]
        for label in config.label_order
    ]
    history_rows = [
        [
            str(item["epoch"]),
            f"{item['train_loss']:.4f}",
            f"{item['eval_loss']:.4f}",
            f"{item['eval_accuracy']:.4f}",
            f"{item['eval_macro_f1']:.4f}",
        ]
        for item in results["history"]
    ]
    lines = [
        "# MARBERT Summary",
        "",
        "This run fine-tunes `UBC-NLP/MARBERT` on `train_core` and evaluates on `dev_core` only.",
        "",
        "## Setup",
        "",
        f"- Model: `{config.model.name_or_path}`",
        f"- Train path: `{config.data.train_path.as_posix()}`",
        f"- Dev path: `{config.data.dev_path.as_posix()}`",
        f"- Text column: `{config.data.text_column}`",
        f"- Target column: `{config.data.target_column}`",
        f"- Labels: `{', '.join(config.label_order)}`",
        f"- Max length: `{config.model.max_length}`",
        f"- Train batch size: `{config.training.batch_size}`",
        f"- Eval batch size: `{config.training.eval_batch_size}`",
        f"- Learning rate: `{config.training.learning_rate}`",
        f"- Weight decay: `{config.training.weight_decay}`",
        f"- Epochs requested: `{config.training.num_epochs}`",
        f"- Warmup ratio: `{config.training.warmup_ratio}`",
        f"- Optimizer: `{config.optimizer.type}`",
        f"- Scheduler: `{config.scheduler.type}`",
        f"- Loss: `{config.loss.type}`",
        f"- Class weights: `{config.loss.class_weights or 'none'}`",
        f"- Seed: `{config.seed}`",
        f"- Device: `{results['device']}`",
        f"- Parameter dtype: `{results['parameter_dtype']}`",
        f"- Best checkpoint: `{results['checkpoint_dir'].as_posix()}`",
        f"- Best epoch: `{results['best_epoch']}`",
        f"- Train rows: `{train_rows}`",
        f"- Dev rows: `{dev_rows}`",
        "",
        "## Dev Metrics",
        "",
        f"- Accuracy: `{results['eval']['accuracy']:.4f}`",
        f"- Macro F1: `{results['eval']['macro_f1']:.4f}`",
        f"- Eval loss: `{results['eval']['loss']:.4f}`",
        "",
        "## Per-Class Metrics",
        "",
        *_markdown_table(["Label", "Precision", "Recall", "F1", "Support"], per_class_rows),
        "",
        "## Training History",
        "",
        *_markdown_table(["Epoch", "Train Loss", "Eval Loss", "Eval Accuracy", "Eval Macro F1"], history_rows),
        "",
    ]
    if comparison_rows:
        lines.extend(
            [
                "## Comparison Note",
                "",
                *_markdown_table(["Baseline", "Accuracy", "Macro F1"], comparison_rows + [["MARBERT", f"{results['eval']['accuracy']:.4f}", f"{results['eval']['macro_f1']:.4f}"]]),
                "",
                "- Use this table as a benchmark-safe comparison against the current classical baseline and as task-context reference against the current Gemini full-dev and Sonnet full-dev runs already present in the repository.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_error_analysis_markdown(
    path: Path,
    *,
    config: EncoderConfig,
    dev_rows: list[dict[str, str]],
    prediction_labels: list[str],
    results: dict[str, Any],
    report_dir: Path,
) -> None:
    confusion_counts = Counter(
        (row[config.data.target_column], prediction)
        for row, prediction in zip(dev_rows, prediction_labels, strict=True)
        if row[config.data.target_column] != prediction
    )
    top_confusions = sorted(
        confusion_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:10]

    comparison_sources = {
        "Classical": _load_predictions_if_exists(
            report_dir / "classical_baseline_dev_predictions.csv",
            prediction_keys=("true_label", "predicted_label"),
        ),
        "Gemini Zero-Shot": _load_predictions_if_exists(
            report_dir / "llm_gemini_flash_lite_dev_predictions.csv",
            prediction_keys=("true_label", "zero_shot_predicted_label"),
        ),
        "Gemini Few-Shot": _load_predictions_if_exists(
            report_dir / "llm_gemini_flash_lite_dev_predictions.csv",
            prediction_keys=("true_label", "few_shot_predicted_label"),
        ),
        "Sonnet Zero-Shot": _load_predictions_if_exists(
            report_dir / "llm_sonnet_full_dev_predictions.csv",
            prediction_keys=("true_label", "zero_shot_predicted_label"),
        ),
        "Sonnet Few-Shot": _load_predictions_if_exists(
            report_dir / "llm_sonnet_full_dev_predictions.csv",
            prediction_keys=("true_label", "few_shot_predicted_label"),
        ),
    }
    comparison_counts: dict[str, Counter[tuple[str, str]]] = {}
    for name, rows in comparison_sources.items():
        if not rows:
            continue
        if name == "Classical":
            comparison_counts[name] = Counter(
                (row["true_label"], row["predicted_label"])
                for row in rows
                if row["true_label"] != row["predicted_label"]
            )
        elif name.endswith("Zero-Shot"):
            comparison_counts[name] = Counter(
                (row["true_label"], row["zero_shot_predicted_label"])
                for row in rows
                if row["true_label"] != row["zero_shot_predicted_label"]
            )
        else:
            comparison_counts[name] = Counter(
                (row["true_label"], row["few_shot_predicted_label"])
                for row in rows
                if row["true_label"] != row["few_shot_predicted_label"]
            )

    lines = [
        "# MARBERT Error Analysis",
        "",
        "This report summarizes the tracked confusion directions for MARBERT on `dev_core` and compares them with the existing benchmark-safe baselines already present in the repository.",
        "",
        "## Requested Confusion Directions",
        "",
    ]
    comparison_headers = ["True Label", "Predicted Label", "MARBERT"]
    comparison_headers.extend(comparison_counts.keys())
    comparison_rows: list[list[str]] = []
    for true_label, predicted_label in ERROR_ANALYSIS_DIRECTIONS:
        row = [
            true_label,
            predicted_label,
            str(confusion_counts.get((true_label, predicted_label), 0)),
        ]
        for name in comparison_counts:
            row.append(str(comparison_counts[name].get((true_label, predicted_label), 0)))
        comparison_rows.append(row)
    lines.extend(_markdown_table(comparison_headers, comparison_rows))
    lines.extend(["", "## Top 10 Off-Diagonal Confusions", ""])
    if top_confusions:
        lines.extend(
            _markdown_table(
                ["True Label", "Predicted Label", "Count"],
                [[true_label, predicted_label, str(count)] for (true_label, predicted_label), count in top_confusions],
            )
        )
        lines.append("")
    else:
        lines.extend(["No off-diagonal confusions found.", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- MARBERT should be judged first on overall dev macro F1, then on whether the four tracked Saudi/Egyptian confusion directions shrink relative to the current classical and LLM baselines.",
            "- If MARBERT improves overall macro F1 but keeps the same confusion profile, a second encoder is still useful for architectural diversity rather than simple incremental tuning.",
            "- If MARBERT materially reduces the tracked confusions as well as improving macro F1, it becomes a stronger default encoder baseline for the next phase.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    config: EncoderConfig,
    results: dict[str, Any],
    dev_rows: list[dict[str, str]],
) -> dict[str, Path]:
    config.output.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output.prefix
    summary_path = config.output.report_dir / f"{prefix}_summary.md"
    metrics_path = config.output.report_dir / f"{prefix}_metrics.json"
    report_path = config.output.report_dir / f"{prefix}_classification_report.json"
    confusion_path = config.output.report_dir / f"{prefix}_confusion_matrix.csv"
    error_analysis_path = config.output.report_dir / f"{prefix}_error_analysis.md"

    label_predictions = [config.label_order[index] for index in results["eval"]["predictions"]]
    true_label_names = [config.label_order[index] for index in results["eval"]["labels"]]
    tracked_confusion_counts = compute_tracked_confusion_counts(true_label_names, label_predictions)
    per_class_metrics = build_per_class_metrics(
        results["eval"]["classification_report"],
        label_order=config.label_order,
    )
    metrics_payload = {
        "model_name_or_path": config.model.name_or_path,
        "train_rows": results["train_rows"],
        "dev_rows": results["dev_rows"],
        "seed": config.seed,
        "device": results["device"],
        "parameter_dtype": results["parameter_dtype"],
        "best_epoch": results["best_epoch"],
        "best_eval_macro_f1": results["best_eval_macro_f1"],
        "checkpoint_dir": results["checkpoint_dir"].as_posix(),
        "class_weights": config.loss.class_weights or "none",
        "eval_loss": results["eval"]["loss"],
        "accuracy": results["eval"]["accuracy"],
        "macro_f1": results["eval"]["macro_f1"],
        "per_class_metrics": per_class_metrics,
        "tracked_confusion_counts": tracked_confusion_counts,
        "config_deviations": {},
        "training_history": results["history"],
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps(results["eval"]["classification_report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_confusion_matrix_csv(confusion_path, config.label_order, results["eval"]["confusion_matrix"])
    comparison_rows = build_comparison_rows(config.output.report_dir)
    write_summary_markdown(
        summary_path,
        config=config,
        train_rows=results["train_rows"],
        dev_rows=results["dev_rows"],
        results=results,
        comparison_rows=comparison_rows,
    )
    write_error_analysis_markdown(
        error_analysis_path,
        config=config,
        dev_rows=dev_rows,
        prediction_labels=label_predictions,
        results=results,
        report_dir=config.output.report_dir,
    )
    return {
        "summary_markdown": summary_path,
        "metrics_json": metrics_path,
        "classification_report_json": report_path,
        "confusion_matrix_csv": confusion_path,
        "error_analysis_markdown": error_analysis_path,
    }


def train_and_evaluate(config: EncoderConfig) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    set_seed(config.seed, config.reproducibility)
    train_rows = load_labeled_rows(
        config.data.train_path,
        text_column=config.data.text_column,
        target_column=config.data.target_column,
    )
    dev_rows = load_labeled_rows(
        config.data.dev_path,
        text_column=config.data.text_column,
        target_column=config.data.target_column,
    )

    tokenizer = load_tokenizer(config.model.name_or_path)
    train_loader, dev_loader, label_to_id = build_dataloaders(
        config,
        tokenizer=tokenizer,
        train_rows=train_rows,
        dev_rows=dev_rows,
    )

    device = resolve_device()
    model = build_model(config, device=device)
    model.to(device)

    weights = compute_loss_weights(config, train_rows=train_rows, label_to_id=label_to_id, device=device)
    loss_fn = create_loss_fn(weights)
    optimizer = create_optimizer(model, config)

    steps_per_epoch = math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * config.training.num_epochs
    scheduler = create_scheduler(optimizer, config=config, total_training_steps=total_training_steps)

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_dir = config.output.checkpoint_dir

    for epoch in range(1, config.training.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
        )
        eval_results = evaluate_model(
            model,
            dev_loader,
            loss_fn=loss_fn,
            device=device,
            label_order=config.label_order,
        )
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_results["loss"],
            "eval_accuracy": eval_results["accuracy"],
            "eval_macro_f1": eval_results["macro_f1"],
        }
        history.append(epoch_summary)
        current_metric = eval_results["macro_f1"]
        if is_improved(
            current_metric,
            best_metric,
            mode=config.early_stopping.mode,
            min_delta=config.early_stopping.min_delta,
        ):
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                metric_value=current_metric,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping.patience:
            break

    best_model = load_best_model(checkpoint_dir, device)
    final_eval = evaluate_model(
        best_model,
        dev_loader,
        loss_fn=loss_fn,
        device=device,
        label_order=config.label_order,
    )
    return (
        {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "device": str(device),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "best_epoch": best_epoch,
            "best_eval_macro_f1": best_metric if best_metric is not None else final_eval["macro_f1"],
            "checkpoint_dir": checkpoint_dir,
            "history": history,
            "eval": final_eval,
        },
        train_rows,
        dev_rows,
    )


def run_experiment(config_path: Path) -> dict[str, Path]:
    config = load_config(config_path)
    results, _, dev_rows = train_and_evaluate(config)
    return write_reports(config=config, results=results, dev_rows=dev_rows)


def build_tiny_test_model(num_labels: int) -> Any:
    config = BertConfig(
        vocab_size=128,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=48,
        num_labels=num_labels,
    )
    return AutoModelForSequenceClassification.from_config(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a benchmark-safe Arabic encoder baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/marbert_base.yaml"),
        help="Path to the encoder YAML config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = run_experiment(args.config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
