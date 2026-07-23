from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_baseline_training import (  # noqa: E402
    choose_device,
    collate_sliding_frame_fusion,
    confusion_summary,
    make_dataloader,
    save_checkpoint_and_metrics,
    strongest_and_weakest_classes,
    train_and_evaluate,
)
from src.models.baselines import SlidingFrameTextFusionClassifier  # noqa: E402
from src.msp_aligned_dataset import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    LazyMSPSlidingFrameFusionDataset,
    load_msp_sliding_frame_fusion_datasets,
)


class ShardLocalSampler(Sampler[int]):
    """Shuffle MSP examples while keeping adjacent batches local to audio shards."""

    def __init__(self, dataset: LazyMSPSlidingFrameFusionDataset) -> None:
        self.dataset = dataset
        self.indices_by_shard: dict[int, list[int]] = {}
        for idx, record in enumerate(dataset.records):
            self.indices_by_shard.setdefault(int(record["feature_shard"]), []).append(idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator()
        generator.seed()
        shard_ids = list(self.indices_by_shard)
        shard_order = torch.randperm(len(shard_ids), generator=generator).tolist()
        for shard_order_idx in shard_order:
            shard_id = shard_ids[shard_order_idx]
            indices = self.indices_by_shard[shard_id]
            local_order = torch.randperm(len(indices), generator=generator).tolist()
            for local_idx in local_order:
                yield indices[local_idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen DistilBERT + sliding-frame prosody fusion on MSP-PODCAST."
    )
    parser.add_argument("--aligned-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--audio-feature-dim", type=int, default=88)
    parser.add_argument("--audio-hidden-dim", type=int, default=128)
    parser.add_argument("--audio-gru-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--unfreeze-text", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--shard-cache-size", type=int, default=8)
    parser.add_argument(
        "--random-shuffle",
        action="store_true",
        help="Use fully random DataLoader shuffling instead of shard-local sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "baselines" / "msp_sliding_frame_fusion.pt",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT
        / "models"
        / "baselines"
        / "msp_sliding_frame_fusion_metrics.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "models" / "baselines" / "msp_baseline_findings.md",
    )
    return parser.parse_args()


def format_label_counts(counts: dict[str, Any] | None) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{label}: {count}" for label, count in sorted(counts.items()))


def write_msp_findings_report(
    *,
    output_path: Path,
    metrics_path: Path,
    metrics: dict[str, Any],
    aligned_dir: Path,
) -> None:
    report_path = aligned_dir / "alignment_report.json"
    alignment_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    split_counts = alignment_report.get("split_counts", {})
    label_counts = alignment_report.get("label_counts", {})
    discovered = alignment_report.get("utterances_discovered", "n/a")
    included = alignment_report.get("included_utterances", "n/a")
    invalid_spans = alignment_report.get("invalid_word_spans", "n/a")
    feature_failures = alignment_report.get("feature_extraction_failures", "n/a")
    words_without_overlap = alignment_report.get("words_without_audio_overlap", "n/a")
    transcript_alignment = alignment_report.get("transcript_alignment", {})

    best = metrics["history"][metrics["best_epoch"] - 1]
    strongest, weakest = strongest_and_weakest_classes(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# MSP-PODCAST Sliding-Frame Fusion Findings\n\n"
        "## Dataset Summary\n\n"
        f"- Dataset: `{aligned_dir.as_posix()}`\n"
        f"- Processed examples: {included} of {discovered} discovered Train/Test1 candidates\n"
        f"- Train examples used: {metrics['train_size']}\n"
        f"- Test examples used: {metrics['test_size']}\n"
        f"- Labels: {', '.join(metrics['label_names'])}\n"
        f"- Split counts in artifact: {split_counts}\n"
        f"- Train label counts: {format_label_counts(label_counts.get('train'))}\n"
        f"- Test label counts: {format_label_counts(label_counts.get('test'))}\n"
        f"- Invalid alignment spans skipped during preprocessing: {invalid_spans}\n"
        f"- Feature extraction failures: {feature_failures}\n"
        f"- Words without audio overlap: {words_without_overlap}\n"
        f"- Same-normalized transcript/alignment ratio: "
        f"{transcript_alignment.get('same_normalized_text_ratio', 'n/a')}\n\n"
        "## Model Result\n\n"
        "| Model | Test accuracy | Macro F1 | Best epoch | Train loss | Test loss | Metrics file |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        f"| Sliding-frame fusion | {metrics['test_accuracy']:.4f} | "
        f"{metrics['test_macro_f1']:.4f} | {metrics['best_epoch']} | "
        f"{best['train_loss']:.4f} | {metrics['test_loss']:.4f} | "
        f"{metrics_path.as_posix()} |\n\n"
        "## Per-Class Notes\n\n"
        f"- Strongest class by F1: {strongest}\n"
        f"- Weakest class by F1: {weakest}\n"
        f"- Largest confusions: {confusion_summary(metrics)}\n\n"
        "## Conclusion\n\n"
        "This run trains the MSP counterpart of the best IEMOCAP fusion variant: "
        "frozen DistilBERT pooled text plus a GRU over raw sliding-window "
        "`audio_frame_features`.\n",
    )
    print(f"Saved MSP findings report to {output_path}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    train_ds, test_ds, num_labels, label_names = load_msp_sliding_frame_fusion_datasets(
        output_dir=args.aligned_dir,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        shard_cache_size=args.shard_cache_size,
    )
    if args.random_shuffle:
        train_loader = make_dataloader(
            train_ds,
            args.batch_size,
            shuffle=True,
            collate_fn=collate_sliding_frame_fusion,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=ShardLocalSampler(train_ds),
            num_workers=0,
            collate_fn=collate_sliding_frame_fusion,
        )
    test_loader = make_dataloader(
        test_ds,
        args.batch_size,
        shuffle=False,
        collate_fn=collate_sliding_frame_fusion,
    )

    model = SlidingFrameTextFusionClassifier(
        num_labels=num_labels,
        text_model_name=args.model_name,
        audio_feature_dim=args.audio_feature_dim,
        audio_hidden_dim=args.audio_hidden_dim,
        audio_gru_hidden_dim=args.audio_gru_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        freeze_text_encoder=not args.unfreeze_text,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["device"] = str(device)
    config["embedding_variant"] = "direct_sliding_frame_sequence"
    config["loader"] = "lazy_msp_audio_frame_shards"
    metrics = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        model_name="msp_sliding_frame_fusion",
        dataset_name="msp_aligned_features",
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
    write_msp_findings_report(
        output_path=args.report_output,
        metrics_path=args.metrics_output,
        metrics=metrics,
        aligned_dir=args.aligned_dir,
    )


if __name__ == "__main__":
    main()
