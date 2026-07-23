from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_baseline_training import (  # noqa: E402
    choose_device,
    make_dataloader,
    prepare_text_datasets,
    save_checkpoint_and_metrics,
    train_and_evaluate,
)
from src.models.baselines import TextOnlyClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT as a text-only IEMOCAP benchmark."
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ROOT / "data" / "iemocap_aligned_features",
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "baselines" / "text_finetuned_distilbert.pt",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "models" / "baselines" / "text_finetuned_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    train_ds, test_ds, num_labels, label_names = prepare_text_datasets(
        aligned_dir=args.aligned_dir,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    train_loader = make_dataloader(train_ds, args.batch_size, shuffle=True)
    test_loader = make_dataloader(test_ds, args.batch_size, shuffle=False)

    model = TextOnlyClassifier(
        num_labels=num_labels,
        text_model_name=args.model_name,
        dropout=args.dropout,
        freeze_text_encoder=False,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["device"] = str(device)
    config["freeze_text_encoder"] = False
    metrics = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        model_name="text_finetuned_distilbert",
        dataset_name="iemocap_aligned_features",
        train_size=len(train_ds),
        test_size=len(test_ds),
        label_names=label_names,
        config=config,
    )
    save_checkpoint_and_metrics(
        model=model,
        checkpoint_path=args.output,
        metrics_path=args.metrics_output,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
