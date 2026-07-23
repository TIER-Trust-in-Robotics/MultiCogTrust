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
    collate_audio_frames,
    make_dataloader,
    prepare_audio_datasets,
    save_checkpoint_and_metrics,
    train_and_evaluate,
)
from src.models.baselines import AudioOnlyGRUClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a temporal GRU audio-only IEMOCAP baseline."
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ROOT / "data" / "iemocap_aligned_features",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--audio-feature-dim", type=int, default=88)
    parser.add_argument("--audio-hidden-dim", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "baselines" / "audio_only_gru.pt",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "models" / "baselines" / "audio_only_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    train_ds, test_ds, num_labels, label_names = prepare_audio_datasets(
        aligned_dir=args.aligned_dir,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    train_loader = make_dataloader(
        train_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=collate_audio_frames,
    )
    test_loader = make_dataloader(
        test_ds,
        args.batch_size,
        shuffle=False,
        collate_fn=collate_audio_frames,
    )

    model = AudioOnlyGRUClassifier(
        num_labels=num_labels,
        audio_feature_dim=args.audio_feature_dim,
        audio_hidden_dim=args.audio_hidden_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        dropout=args.dropout,
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
    metrics = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        model_name="audio_only_gru",
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
