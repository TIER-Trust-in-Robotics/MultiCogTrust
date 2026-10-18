"""Full frozen-encoder MVP baseline; Session 5 is never encoded or evaluated."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.emotion_checkpoint import (
    EMOTION_LABELS,
    PreprocessingConfig,
    save_checkpoint,
    read_checkpoint,
    load_checkpoint_weights,
)
from src.iemocap_embedding_cache import (
    build_cache,
    CachedDataset,
    collate_embeddings,
    sha256,
    write_json,
)
from src.models import CrossModalTransformerFusionClassifier
from src.prepare_iemocap import DEFAULT_METADATA
from src.train_cross_modal_finetune_iemocap import load_samples, choose_device


def metrics_from_predictions(labels, probabilities):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

    predicted = np.argmax(probabilities, axis=1)
    classes = list(range(4))
    return {
        "count": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(
            f1_score(
                labels, predicted, labels=classes, average="macro", zero_division=0
            )
        ),
        "uar": float(
            recall_score(
                labels, predicted, labels=classes, average="macro", zero_division=0
            )
        ),
        "per_class_recall": dict(
            zip(
                EMOTION_LABELS,
                recall_score(
                    labels, predicted, labels=classes, average=None, zero_division=0
                ).tolist(),
            )
        ),
        "confusion": confusion_matrix(labels, predicted, labels=classes).tolist(),
    }


def forward(model, batch, device):
    text, tm, audio, am, labels = [x.to(device) for x in batch]
    return model.forward_from_embeddings(text, audio, tm, am).logits, labels


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        logits, truth = forward(model, batch, device)
        labels.extend(truth.cpu().tolist())
        probabilities.extend(logits.softmax(-1).cpu().tolist())
    return metrics_from_predictions(labels, probabilities), probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/mvp_day3_baseline.json"
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "data/processed/iemocap_encoder_cache"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "models/mvp/audio_text_baseline.pt"
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config["train_sessions"] != [1, 2, 3]
        or config["validation_session"] != 4
        or config["test_session"] != 5
        or config["test_evaluated"]
        or not config["freeze_audio_encoder"]
        or not config["freeze_text_encoder"]
        or config["selection_metric"] != "macro_f1"
    ):
        raise ValueError(
            "MVP baseline requires frozen encoders, fixed sessions and validation macro F1 selection"
        )
    if args.output.exists():
        raise FileExistsError(
            f"Preserving existing run: {args.output}; use another --output"
        )
    started = time.perf_counter()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = choose_device(args.device)
    samples = load_samples(str(args.metadata), str(ROOT))
    training = [s for s in samples if s.session in config["train_sessions"]]
    validation = [s for s in samples if s.session == config["validation_session"]]
    if not training or not validation:
        raise ValueError("Empty development split")
    preprocessing = PreprocessingConfig(
        max_audio_sec=config["max_audio_sec"], max_tokens=config["max_tokens"]
    )
    model = CrossModalTransformerFusionClassifier(
        num_labels=4,
        fusion_head_output_type=config["fusion_head"],
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
        local_files_only=args.offline,
        **{
            k: config[k]
            for k in (
                "text_model_name",
                "text_revision",
                "audio_model_name",
                "audio_revision",
            )
        },
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config["text_model_name"],
        revision=config["text_revision"],
        local_files_only=args.offline,
    )
    print(
        f"Device {device}; training {len(training)}; validation {len(validation)}; Session 5 excluded",
        flush=True,
    )
    build_cache(
        training + validation,
        model,
        tokenizer,
        preprocessing,
        args.cache,
        sha256(args.metadata),
        device,
    )
    train_loader = DataLoader(
        CachedDataset(training, args.cache),
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_embeddings,
    )
    val_loader = DataLoader(
        CachedDataset(validation, args.cache),
        batch_size=config["batch_size"],
        collate_fn=collate_embeddings,
    )
    label_smoothing = float(config.get("label_smoothing", 0.0))
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    provenance = {
        "kind": "full_development_baseline",
        "seed": config["seed"],
        "train_count": len(training),
        "validation_count": len(validation),
        "train_utterance_ids": [s.utterance_id for s in training],
        "validation_utterance_ids": [s.utterance_id for s in validation],
        "validation_session": 4,
        "test_session": 5,
        "test_evaluated": False,
        "metadata_sha256": sha256(args.metadata),
        "config": config,
        "config_sha256": sha256(args.config),
        "cache_index_sha256": sha256(args.cache / "index.json"),
        "transcript_source": "iemocap_reference",
        "device": device,
        "trainer_sha256": sha256(Path(__file__)),
    }
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config["learning_rate"])
    weights = None
    if config["balanced_loss"]:
        counts = np.bincount([s.label for s in training], minlength=4)
        weights = torch.tensor(
            len(training) / (4 * counts), dtype=torch.float32, device=device
        )
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=label_smoothing
    )
    best, history = -1, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_start, total_loss, seen = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, labels = forward(model, batch, device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        metrics, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation": metrics,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: loss={row['train_loss']:.4f}, val F1={metrics['macro_f1']:.4f}, UAR={metrics['uar']:.4f}, WA={metrics['accuracy']:.4f} ({row['seconds']:.1f}s)",
            flush=True,
        )
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            save_checkpoint(
                args.output,
                model,
                preprocessing=preprocessing,
                include_encoders=False,
                provenance={**provenance, "epoch": epoch, "validation": metrics},
            )
        write_json(
            args.output.with_suffix(".progress.json"),
            {"finished": False, "history": history, "provenance": provenance},
        )
    checkpoint = read_checkpoint(args.output, "iemocap_emotion_4")
    load_checkpoint_weights(model, checkpoint)
    metrics, probabilities = evaluate(model, val_loader, device)
    if abs(metrics["macro_f1"] - best) > 1e-10:
        raise AssertionError(
            "Reloaded best checkpoint does not reproduce selection metric"
        )
    majority_label = int(np.bincount([s.label for s in training], minlength=4).argmax())
    majority = np.zeros((len(validation), 4))
    majority[:, majority_label] = 1
    report = {
        "finished": True,
        "test_evaluated": False,
        "best_epoch": checkpoint["provenance"]["epoch"],
        "selection_metric": "validation_macro_f1",
        "validation": metrics,
        "history": history,
        "majority_baseline": metrics_from_predictions(
            [s.label for s in validation], majority
        ),
        "majority_label_from_training": EMOTION_LABELS[majority_label],
        "provenance": provenance,
        "preprocessing": asdict(preprocessing),
        "checkpoint_sha256": sha256(args.output),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json(args.output.with_suffix(".metrics.json"), report)
    with args.output.with_suffix(".validation.jsonl").open("w") as handle:
        for sample, probs in zip(validation, probabilities):
            handle.write(
                json.dumps(
                    {
                        "utterance_id": sample.utterance_id,
                        "label": sample.label,
                        "prediction": int(np.argmax(probs)),
                        "probabilities": probs,
                    }
                )
                + "\n"
            )
    write_json(
        args.output.with_suffix(".progress.json"),
        {"finished": True, "history": history},
    )
    print(
        json.dumps(
            {
                "best_epoch": report["best_epoch"],
                "validation": metrics,
                "test_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
