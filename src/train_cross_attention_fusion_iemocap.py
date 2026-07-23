from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_baseline_training import (  # noqa: E402
    choose_device,
    collate_sliding_frame_fusion,
    load_text_encoder_from_checkpoint,
    make_dataloader,
    make_differential_lr_optimizer,
    make_weighted_cross_entropy,
    prepare_sliding_frame_fusion_datasets,
    save_checkpoint_and_metrics,
    train_and_evaluate,
)
from src.models.baselines import TextAudioCrossAttentionClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IEMOCAP text-to-audio cross-attention fusion."
    )
    parser.add_argument(
        "--aligned-dir",
        type=Path,
        default=ROOT / "data" / "iemocap_aligned_features",
    )
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument(
        "--text-checkpoint",
        type=Path,
        default=ROOT / "models" / "baselines" / "text_finetuned_distilbert.pt",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-lr", type=float, default=1e-5)
    parser.add_argument("--non-text-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--audio-feature-dim", type=int, default=88)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--classifier-hidden-dim", type=int, default=256)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "baselines" / "cross_attention_fusion.pt",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "models" / "baselines" / "cross_attention_fusion_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    train_ds, test_ds, num_labels, label_names = prepare_sliding_frame_fusion_datasets(
        aligned_dir=args.aligned_dir,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    train_loader = make_dataloader(
        train_ds,
        args.batch_size,
        shuffle=True,
        collate_fn=collate_sliding_frame_fusion,
    )
    test_loader = make_dataloader(
        test_ds,
        args.batch_size,
        shuffle=False,
        collate_fn=collate_sliding_frame_fusion,
    )

    model = TextAudioCrossAttentionClassifier(
        num_labels=num_labels,
        text_model_name=args.model_name,
        audio_feature_dim=args.audio_feature_dim,
        num_attention_heads=args.attention_heads,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        freeze_text_encoder=False,
    )
    checkpoint_summary = load_text_encoder_from_checkpoint(model, args.text_checkpoint)
    model = model.to(device)

    criterion, class_weights = make_weighted_cross_entropy(
        train_ds,
        num_labels,
        device,
        enabled=not args.no_class_weights,
    )
    optimizer = make_differential_lr_optimizer(
        model,
        text_lr=args.text_lr,
        non_text_lr=args.non_text_lr,
        weight_decay=args.weight_decay,
    )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["device"] = str(device)
    config["freeze_text_encoder"] = False
    config["embedding_variant"] = "cross_attention_text_queries_audio_frames"
    config["text_checkpoint_load"] = checkpoint_summary
    config["class_weights"] = class_weights
    metrics = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        model_name="cross_attention_fusion",
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
