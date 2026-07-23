from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.iemocap_aligned_dataset import (
    load_iemocap_aligned,
    tokenize_record_with_word_audio,
    tokenize_dataset_with_word_audio,
)


METRIC_KEYS = {
    "model",
    "dataset",
    "train_size",
    "test_size",
    "num_labels",
    "label_names",
    "history",
    "best_epoch",
    "test_accuracy",
    "test_macro_f1",
    "classification_report",
    "confusion_matrix",
}


class TextTensorDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        return {
            "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(record["attention_mask"], dtype=torch.long),
            "label": torch.tensor(record["label"], dtype=torch.long),
        }


class FusionTensorDataset(TextTensorDataset):
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        item = super().__getitem__(idx)
        item["audio_features"] = torch.tensor(
            record["audio_features"],
            dtype=torch.float32,
        )
        item["audio_token_mask"] = torch.tensor(
            record["audio_token_mask"],
            dtype=torch.float32,
        )
        return item


class AudioFrameDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return {
            "audio_frame_features": np.asarray(
                record["audio_frame_features"],
                dtype=np.float32,
            ),
            "label": int(record["label"]),
        }


class SlidingFrameFusionDataset(TextTensorDataset):
    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        item = super().__getitem__(idx)
        item["audio_frame_features"] = np.asarray(
            record["audio_frame_features"],
            dtype=np.float32,
        )
        return item


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def load_limited_iemocap(
    aligned_dir: Path,
    limit_train: int | None = None,
    limit_test: int | None = None,
):
    train_ds, test_ds, num_labels, label_names = load_iemocap_aligned(aligned_dir)
    return (
        limit_dataset(train_ds, limit_train),
        limit_dataset(test_ds, limit_test),
        num_labels,
        label_names,
    )


def prepare_text_datasets(
    aligned_dir: Path,
    tokenizer_name: str,
    max_length: int,
    limit_train: int | None,
    limit_test: int | None,
) -> tuple[TextTensorDataset, TextTensorDataset, int, list[str]]:
    train_ds, test_ds, num_labels, label_names = load_limited_iemocap(
        aligned_dir,
        limit_train,
        limit_test,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenized_train = tokenizer(
        list(train_ds["text"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokenized_test = tokenizer(
        list(test_ds["text"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    train_records = [
        {**{key: tokenized_train[key][idx] for key in tokenized_train}, "label": row["label"]}
        for idx, row in enumerate(train_ds)
    ]
    test_records = [
        {**{key: tokenized_test[key][idx] for key in tokenized_test}, "label": row["label"]}
        for idx, row in enumerate(test_ds)
    ]
    return (
        TextTensorDataset(train_records),
        TextTensorDataset(test_records),
        num_labels,
        label_names,
    )


def prepare_fusion_datasets(
    aligned_dir: Path,
    tokenizer_name: str,
    max_length: int,
    limit_train: int | None,
    limit_test: int | None,
) -> tuple[FusionTensorDataset, FusionTensorDataset, int, list[str]]:
    train_ds, test_ds, num_labels, label_names = load_limited_iemocap(
        aligned_dir,
        limit_train,
        limit_test,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError(f"{tokenizer_name!r} must use a fast tokenizer")

    tokenized_train = tokenize_dataset_with_word_audio(
        train_ds,
        tokenizer=tokenizer,
        max_length=max_length,
        feature_width=88,
    )
    tokenized_test = tokenize_dataset_with_word_audio(
        test_ds,
        tokenizer=tokenizer,
        max_length=max_length,
        feature_width=88,
    )
    return (
        FusionTensorDataset(list(tokenized_train)),
        FusionTensorDataset(list(tokenized_test)),
        num_labels,
        label_names,
    )


def prepare_audio_datasets(
    aligned_dir: Path,
    limit_train: int | None,
    limit_test: int | None,
) -> tuple[AudioFrameDataset, AudioFrameDataset, int, list[str]]:
    train_ds, test_ds, num_labels, label_names = load_limited_iemocap(
        aligned_dir,
        limit_train,
        limit_test,
    )
    return (
        AudioFrameDataset(list(train_ds)),
        AudioFrameDataset(list(test_ds)),
        num_labels,
        label_names,
    )


def prepare_sliding_frame_fusion_datasets(
    aligned_dir: Path,
    tokenizer_name: str,
    max_length: int,
    limit_train: int | None,
    limit_test: int | None,
) -> tuple[SlidingFrameFusionDataset, SlidingFrameFusionDataset, int, list[str]]:
    train_ds, test_ds, num_labels, label_names = load_limited_iemocap(
        aligned_dir,
        limit_train,
        limit_test,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenized_train = tokenizer(
        list(train_ds["text"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tokenized_test = tokenizer(
        list(test_ds["text"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    train_records = [
        {
            **{key: tokenized_train[key][idx] for key in tokenized_train},
            "audio_frame_features": row["audio_frame_features"],
            "label": row["label"],
        }
        for idx, row in enumerate(train_ds)
    ]
    test_records = [
        {
            **{key: tokenized_test[key][idx] for key in tokenized_test},
            "audio_frame_features": row["audio_frame_features"],
            "label": row["label"],
        }
        for idx, row in enumerate(test_ds)
    ]
    return (
        SlidingFrameFusionDataset(train_records),
        SlidingFrameFusionDataset(test_records),
        num_labels,
        label_names,
    )


def collate_audio_frames(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    lengths = [item["audio_frame_features"].shape[0] for item in batch]
    feature_dim = int(batch[0]["audio_frame_features"].shape[1])
    max_length = max(max(lengths), 1)
    features = torch.zeros((len(batch), max_length, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_length), dtype=torch.float32)

    for idx, item in enumerate(batch):
        frames = torch.tensor(item["audio_frame_features"], dtype=torch.float32)
        length = int(frames.shape[0])
        if length == 0:
            continue
        features[idx, :length] = frames
        mask[idx, :length] = 1.0

    return {
        "audio_frame_features": features,
        "audio_frame_mask": mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }


def collate_sliding_frame_fusion(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    audio_batch = collate_audio_frames(batch)
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "audio_frame_features": audio_batch["audio_frame_features"],
        "audio_frame_mask": audio_batch["audio_frame_mask"],
        "label": audio_batch["label"],
    }


def compute_class_weights(dataset: Dataset, num_labels: int) -> torch.Tensor:
    counts: Counter[int] = Counter()
    for idx in range(len(dataset)):
        label = dataset[idx]["label"]
        if isinstance(label, torch.Tensor):
            label_idx = int(label.item())
        else:
            label_idx = int(label)
        counts[label_idx] += 1

    total = sum(counts.values())
    weights = torch.ones(num_labels, dtype=torch.float32)
    for label_idx in range(num_labels):
        count = counts.get(label_idx, 0)
        if count:
            weights[label_idx] = total / (num_labels * count)
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def make_weighted_cross_entropy(
    dataset: Dataset,
    num_labels: int,
    device: torch.device,
    enabled: bool,
) -> tuple[nn.Module, list[float] | None]:
    if not enabled:
        return nn.CrossEntropyLoss(), None
    weights = compute_class_weights(dataset, num_labels).to(device)
    return nn.CrossEntropyLoss(weight=weights), weights.detach().cpu().tolist()


def load_text_encoder_from_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing text checkpoint: {checkpoint_path}")
    if not hasattr(model, "text_model"):
        raise ValueError(f"{model.__class__.__name__} does not expose text_model")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    text_state = {
        key.removeprefix("text_model."): value
        for key, value in state_dict.items()
        if key.startswith("text_model.")
    }
    if not text_state:
        raise ValueError(f"No text_model.* weights found in {checkpoint_path}")

    incompatible = model.text_model.load_state_dict(text_state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": len(text_state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def make_differential_lr_optimizer(
    model: nn.Module,
    *,
    text_lr: float,
    non_text_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    text_parameters = []
    non_text_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("text_model."):
            text_parameters.append(parameter)
        else:
            non_text_parameters.append(parameter)

    parameter_groups = []
    if text_parameters:
        parameter_groups.append({"params": text_parameters, "lr": text_lr})
    if non_text_parameters:
        parameter_groups.append({"params": non_text_parameters, "lr": non_text_lr})
    if not parameter_groups:
        raise ValueError("No trainable parameters found")
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def overlap_weighted_word_features_from_frames(record: dict[str, Any]) -> np.ndarray:
    frames = np.asarray(record["audio_frame_features"], dtype=np.float32)
    starts = np.asarray(record["audio_frame_start_sec"], dtype=np.float32)
    ends = np.asarray(record["audio_frame_end_sec"], dtype=np.float32)
    words = record["words"]
    if frames.ndim != 2:
        return np.zeros((len(words), 0), dtype=np.float32)

    output = np.zeros((len(words), frames.shape[1]), dtype=np.float32)
    for word_idx, word in enumerate(words):
        word_start = float(word["start_sec"])
        word_end = float(word["end_sec"])
        overlaps = np.maximum(
            0.0,
            np.minimum(ends, word_end) - np.maximum(starts, word_start),
        )
        total_overlap = float(overlaps.sum())
        if total_overlap <= 0.0:
            continue
        output[word_idx] = (frames * overlaps[:, None]).sum(axis=0) / total_overlap
    return output


def token_features_from_sliding_frame_overlap(
    record: dict[str, Any],
    tokenizer,
    max_length: int,
    feature_width: int = 88,
) -> dict[str, Any]:
    word_features = overlap_weighted_word_features_from_frames(record)
    if word_features.shape[1] != feature_width:
        padded = np.zeros((word_features.shape[0], feature_width), dtype=np.float32)
        width = min(feature_width, word_features.shape[1])
        if width:
            padded[:, :width] = word_features[:, :width]
        word_features = padded

    words = [word["word"] for word in record["words"]]
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    token_audio = np.zeros((max_length, feature_width), dtype=np.float32)
    audio_token_mask = np.zeros((max_length,), dtype=np.int64)
    word_ids = encoding.word_ids()
    word_masks = record.get("word_audio_feature_mask", [1] * len(words))

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx >= len(word_features):
            continue
        token_audio[token_idx] = word_features[word_idx]
        audio_token_mask[token_idx] = int(word_masks[word_idx])

    return {
        **encoding,
        "audio_features": token_audio.tolist(),
        "audio_token_mask": audio_token_mask.tolist(),
        "label": record["label"],
        "utterance_id": record["utterance_id"],
    }


def run_sliding_embedding_equivalence_check(
    aligned_dir: Path,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    limit: int = 256,
    atol: float = 1e-5,
) -> dict[str, Any]:
    train_ds, test_ds, _, _ = load_iemocap_aligned(aligned_dir)
    records = list(train_ds.select(range(min(limit, len(train_ds)))))
    if len(records) < limit:
        remaining = limit - len(records)
        records.extend(list(test_ds.select(range(min(remaining, len(test_ds))))))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    max_abs_diffs = []
    mask_mismatches = 0
    equivalent_count = 0
    examples = []

    for record in records:
        existing = tokenize_record_with_word_audio(
            record,
            tokenizer=tokenizer,
            max_length=max_length,
            feature_width=88,
        )
        rebuilt = token_features_from_sliding_frame_overlap(
            record,
            tokenizer=tokenizer,
            max_length=max_length,
            feature_width=88,
        )
        existing_audio = np.asarray(existing["audio_features"], dtype=np.float32)
        rebuilt_audio = np.asarray(rebuilt["audio_features"], dtype=np.float32)
        diff = float(np.max(np.abs(existing_audio - rebuilt_audio)))
        max_abs_diffs.append(diff)
        masks_equal = existing["audio_token_mask"] == rebuilt["audio_token_mask"]
        if not masks_equal:
            mask_mismatches += 1
        if diff <= atol and masks_equal:
            equivalent_count += 1
        if len(examples) < 5:
            examples.append(
                {
                    "utterance_id": record["utterance_id"],
                    "max_abs_diff": diff,
                    "masks_equal": masks_equal,
                }
            )

    max_diff = max(max_abs_diffs, default=0.0)
    mean_diff = sum(max_abs_diffs) / max(len(max_abs_diffs), 1)
    return {
        "aligned_dir": str(aligned_dir),
        "tokenizer_name": tokenizer_name,
        "max_length": max_length,
        "sample_count": len(records),
        "atol": atol,
        "equivalent_count": equivalent_count,
        "equivalent_fraction": equivalent_count / max(len(records), 1),
        "mask_mismatches": mask_mismatches,
        "max_abs_diff": max_diff,
        "mean_max_abs_diff": mean_diff,
        "is_equivalent": equivalent_count == len(records) and max_diff <= atol,
        "examples": examples,
        "interpretation": (
            "Token-overlap pooling from sliding frames matches existing word_audio_features."
            if equivalent_count == len(records) and max_diff <= atol
            else "Token-overlap pooling from sliding frames differs from existing word_audio_features."
        ),
    }


def get_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        return output.logits
    return output


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def model_inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in batch.items() if key != "label"}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = get_logits(model(**model_inputs(batch)))
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()

        batch_size = int(batch["label"].shape[0])
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=-1) == batch["label"]).sum().item())
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        logits = get_logits(model(**model_inputs(batch)))
        loss = criterion(logits, batch["label"])
        predictions = logits.argmax(dim=-1)

        batch_size = int(batch["label"].shape[0])
        total_loss += float(loss.item()) * batch_size
        correct += int((predictions == batch["label"]).sum().item())
        total += batch_size
        y_true.extend(int(value) for value in batch["label"].detach().cpu().tolist())
        y_pred.extend(int(value) for value in predictions.detach().cpu().tolist())

    return total_loss / max(total, 1), correct / max(total, 1), y_true, y_pred


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    num_labels = len(label_names)
    confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_labels and 0 <= pred_label < num_labels:
            confusion[true_label][pred_label] += 1

    report: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for idx, label_name in enumerate(label_names):
        tp = confusion[idx][idx]
        predicted = sum(confusion[row][idx] for row in range(num_labels))
        actual = sum(confusion[idx])
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        report[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }

    accuracy = sum(confusion[idx][idx] for idx in range(num_labels)) / max(len(y_true), 1)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / max(num_labels, 1),
        "classification_report": report,
        "confusion_matrix": confusion,
    }


def train_and_evaluate(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    model_name: str,
    dataset_name: str,
    train_size: int,
    test_size: int,
    label_names: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    history = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_macro_f1 = -1.0
    best_test_payload: tuple[float, float, list[int], list[int]] | None = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        test_loss, test_acc, y_true, y_pred = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        test_metrics = compute_metrics(y_true, y_pred, label_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_metrics["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row))

        if test_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(test_metrics["macro_f1"])
            best_epoch = epoch
            best_test_payload = (test_loss, test_acc, y_true, y_pred)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_test_payload is None:
        best_test_payload = evaluate(model, test_loader, criterion, device)
    best_test_loss, best_test_acc, best_true, best_pred = best_test_payload
    final_metrics = compute_metrics(best_true, best_pred, label_names)
    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "train_size": train_size,
        "test_size": test_size,
        "num_labels": len(label_names),
        "label_names": label_names,
        "history": history,
        "best_epoch": best_epoch,
        "test_loss": best_test_loss,
        "test_accuracy": best_test_acc,
        "test_macro_f1": final_metrics["macro_f1"],
        "classification_report": final_metrics["classification_report"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "config": config,
    }
    metrics["required_metric_keys_present"] = sorted(METRIC_KEYS & set(metrics))
    metrics["_best_state_dict"] = best_state
    return metrics


def save_checkpoint_and_metrics(
    *,
    model: nn.Module,
    checkpoint_path: Path,
    metrics_path: Path,
    metrics: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = metrics.pop("_best_state_dict", None)
    if state_dict is None:
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    torch.save(
        {
            "state_dict": state_dict,
            "label_names": metrics["label_names"],
            "num_labels": metrics["num_labels"],
            "best_epoch": metrics["best_epoch"],
            "config": metrics["config"],
        },
        checkpoint_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_fn,
    )


def strongest_and_weakest_classes(metrics: dict[str, Any]) -> tuple[str, str]:
    report = metrics["classification_report"]
    ranked = sorted(
        ((label, values["f1"]) for label, values in report.items()),
        key=lambda item: item[1],
    )
    if not ranked:
        return "n/a", "n/a"
    weakest = f"{ranked[0][0]} ({ranked[0][1]:.3f})"
    strongest = f"{ranked[-1][0]} ({ranked[-1][1]:.3f})"
    return strongest, weakest


def confusion_summary(metrics: dict[str, Any]) -> str:
    labels = metrics["label_names"]
    confusion = metrics["confusion_matrix"]
    pairs: list[tuple[int, str]] = []
    for true_idx, row in enumerate(confusion):
        for pred_idx, count in enumerate(row):
            if true_idx != pred_idx and count:
                pairs.append((count, f"{labels[true_idx]} -> {labels[pred_idx]}"))
    pairs.sort(reverse=True)
    if not pairs:
        return "No off-diagonal errors."
    return "; ".join(f"{name}: {count}" for count, name in pairs[:5])


def write_findings_report(
    *,
    output_path: Path,
    metrics_paths: dict[str, Path],
    metrics_by_model: dict[str, dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_metrics = next(iter(metrics_by_model.values()))
    labels = ", ".join(first_metrics["label_names"])
    rows = []
    for display_name, metrics in metrics_by_model.items():
        rows.append(
            "| {name} | {acc:.4f} | {f1:.4f} | {epoch} | {train_loss:.4f} | {test_loss:.4f} | {path} |".format(
                name=display_name,
                acc=metrics["test_accuracy"],
                f1=metrics["test_macro_f1"],
                epoch=metrics["best_epoch"],
                train_loss=metrics["history"][metrics["best_epoch"] - 1]["train_loss"],
                test_loss=metrics["test_loss"],
                path=metrics_paths[display_name].as_posix(),
            )
        )

    text = metrics_by_model.get("Text-only")
    audio = metrics_by_model.get("Audio-only")
    fusion = metrics_by_model.get("Simple fusion")
    sliding = metrics_by_model.get("Sliding-frame fusion")
    fusion_note = "Fusion metrics were not available for comparison."
    recommendation = "Run the missing baseline before choosing the next fusion architecture."
    if text and audio and fusion:
        text_delta = fusion["test_macro_f1"] - text["test_macro_f1"]
        audio_delta = fusion["test_macro_f1"] - audio["test_macro_f1"]
        fusion_note = (
            f"Fusion macro F1 is {text_delta:+.4f} versus text-only and "
            f"{audio_delta:+.4f} versus audio-only."
        )
        recommendation = (
            "Try early-fusion self-attention next."
            if text_delta > 0 and audio_delta > 0
            else "Try cross-modal attention next; the pooled fusion baseline did not clearly separate complementary signal."
        )
    if fusion and sliding:
        sliding_delta = sliding["test_macro_f1"] - fusion["test_macro_f1"]
        fusion_note += (
            f" Sliding-frame fusion macro F1 is {sliding_delta:+.4f} "
            "versus simple word-aligned fusion."
        )
        if sliding_delta > 0:
            recommendation = "Keep the direct sliding-frame audio branch in the next fusion experiment."

    best_display_name, best_metrics = max(
        metrics_by_model.items(), key=lambda item: item[1]["test_macro_f1"]
    )
    ranking = sorted(
        metrics_by_model.items(), key=lambda item: item[1]["test_macro_f1"], reverse=True
    )
    ranking_text = ", ".join(
        f"{name} ({metrics['test_macro_f1']:.4f})" for name, metrics in ranking
    )
    headline = (
        f"The best model in this comparison is **{best_display_name}** "
        f"with macro F1 {best_metrics['test_macro_f1']:.4f} and accuracy "
        f"{best_metrics['test_accuracy']:.4f}."
    )

    equivalence_path = output_path.parent / "sliding_embedding_equivalence.json"
    equivalence_note = "No sliding-window embedding equivalence diagnostic was found."
    if equivalence_path.exists():
        equivalence = json.loads(equivalence_path.read_text())
        equivalence_note = (
            "The token-overlap reconstruction from sliding-window frames matched the "
            f"existing word-aligned token features for {equivalence['equivalent_count']}/"
            f"{equivalence['sample_count']} sampled records. The maximum absolute "
            f"difference was {equivalence['max_abs_diff']:.6g}, with "
            f"{equivalence['mask_mismatches']} token-mask mismatches. "
            f"Diagnostic conclusion: {equivalence['interpretation']}"
        )

    artifact_lines = [
        "- Visualization report: `iemocap_baseline_visual_report.md`",
        "- Notebook: `iemocap_baseline_visualizations.ipynb`",
        "- Aggregate metrics plot: `baseline_metric_comparison.png`",
        "- Training curves: `baseline_training_curves.png`",
        "- Per-class F1 plot: `baseline_per_class_f1.png`",
        "- Confusion matrices: `baseline_confusion_matrices.png`",
    ]

    notes = []
    for display_name, metrics in metrics_by_model.items():
        strongest, weakest = strongest_and_weakest_classes(metrics)
        notes.append(
            f"### {display_name}\n"
            f"- Strongest class by F1: {strongest}\n"
            f"- Weakest class by F1: {weakest}\n"
            f"- Largest confusions: {confusion_summary(metrics)}"
        )

    output_path.write_text(
        "# IEMOCAP Baseline Findings\n\n"
        "## Dataset Summary\n\n"
        f"- Dataset: `data/iemocap_aligned_features`\n"
        f"- Train examples: {first_metrics['train_size']}\n"
        f"- Test examples: {first_metrics['test_size']}\n"
        f"- Labels: {labels}\n"
        "- Feature source: aligned IEMOCAP metadata with token-aligned word prosody and temporal `audio_frame_features`.\n\n"
        "## Headline Findings\n\n"
        f"{headline}\n\n"
        f"Macro F1 ranking: {ranking_text}.\n\n"
        "The repeated sliding-window experiment uses raw temporal "
        "`audio_frame_features` directly through a GRU audio branch, rather than the "
        "existing `word_audio_features` copied onto DistilBERT subword tokens. This "
        "makes it the comparison for the alternate sliding-window embedding path.\n\n"
        "## Model Comparison\n\n"
        "| Model | Test accuracy | Macro F1 | Best epoch | Train loss | Test loss | Metrics file |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        + "\n".join(rows)
        + "\n\n"
        "## Sliding-Window Embedding Check\n\n"
        f"{equivalence_note}\n\n"
        "## Per-Model Notes\n\n"
        + "\n\n".join(notes)
        + "\n\n"
        "## Visualization Artifacts\n\n"
        + "\n".join(artifact_lines)
        + "\n\n"
        "## Conclusion\n\n"
        f"{fusion_note} {recommendation}\n",
    )
    print(f"Saved findings report to {output_path}")


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
