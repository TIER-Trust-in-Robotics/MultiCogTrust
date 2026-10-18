"""Benchmark the distilled MemoCMT fusion on CMU-MOSEI (audio + text).

Native-data benchmark: DistilBERT + DistilHuBERT frozen, the ~6M-param
bidirectional cross-attention fusion trained on MOSEI's standard
train/valid/test folds with 7-class sentiment (round(label) + 3).

Because MOSEI is ~4x IEMOCAP, frozen-encoder embeddings are precomputed once
into fp16 shards on disk and streamed per epoch instead of held in RAM.

Reported metrics follow the MOSEI conventions:
    Acc-7            7-class accuracy on rounded sentiment
    Acc-2 / F1 non0  binary (pos vs neg) excluding label == 0 clips;
                     prediction positive iff predicted class value > 0
    Acc-2 / F1 has0  binary (non-neg vs neg) on all clips
    MAE              |predicted class value - continuous label|

Usage:
    uv run python src/train_cross_modal_mosei.py --precompute
    uv run python src/train_cross_modal_mosei.py --epochs 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import CrossModalTransformerFusionClassifier  # noqa: E402
from src.mosei_dataset import DEFAULT_OUTPUT_DIR, load_mosei_samples  # noqa: E402
from src.train_cross_modal_transformer_iemocap import load_audio  # noqa: E402

NUM_CLASSES = 7  # sentiment -3..3 rounded
SPLITS = ("train", "valid", "test")


def sentiment_to_class(sentiment: float) -> int:
    return int(np.clip(round(sentiment), -3, 3)) + 3


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def shard_paths(embeddings_dir: Path, split: str) -> list[Path]:
    return sorted(embeddings_dir.glob(f"{split}_*.pt"))


def precompute_shards(
    samples_by_split: dict[str, list],
    model: CrossModalTransformerFusionClassifier,
    tokenizer,
    embeddings_dir: Path,
    max_seconds: float,
    shard_size: int,
    device: str,
) -> None:
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for split, samples in samples_by_split.items():
        n_shards = (len(samples) + shard_size - 1) // shard_size
        for shard_idx in range(n_shards):
            shard_path = embeddings_dir / f"{split}_{shard_idx:03d}.pt"
            if shard_path.exists():
                continue
            chunk = samples[shard_idx * shard_size : (shard_idx + 1) * shard_size]
            items = []
            start = time.time()
            with torch.no_grad():
                for sample in chunk:
                    encoded = tokenizer(
                        sample.text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=297,
                    )
                    text_embedding = model.encode_text(
                        encoded["input_ids"].to(device),
                        encoded["attention_mask"].to(device),
                    )
                    waveform = (
                        load_audio(sample.audio_path, max_seconds)
                        .unsqueeze(0)
                        .to(device)
                    )
                    audio_embedding = model.encode_audio(waveform)
                    items.append(
                        {
                            "text": text_embedding.cpu().half(),
                            "audio": audio_embedding.cpu().half(),
                            "sentiment": sample.sentiment,
                        }
                    )
            torch.save(items, shard_path)
            rate = len(items) / max(1e-6, time.time() - start)
            print(
                f"  {shard_path.name}: {len(items)} utts ({rate:.1f} utt/s), "
                f"shard {shard_idx + 1}/{n_shards} of {split}"
            )


def iter_shard_items(embeddings_dir: Path, split: str, rng: random.Random | None = None):
    paths = shard_paths(embeddings_dir, split)
    if rng is not None:
        paths = list(paths)
        rng.shuffle(paths)
    for path in paths:
        items = torch.load(path, weights_only=False)
        if rng is not None:
            rng.shuffle(items)
        yield from items


def evaluate(
    model: CrossModalTransformerFusionClassifier,
    embeddings_dir: Path,
    split: str,
    device: str,
) -> dict[str, float]:
    model.eval()
    true_values: list[float] = []
    pred_classes: list[int] = []
    with torch.no_grad():
        for item in iter_shard_items(embeddings_dir, split):
            output = model.forward_from_embeddings(
                item["text"].float().to(device),
                item["audio"].float().to(device),
            )
            pred_classes.append(int(output.logits.argmax(dim=-1).item()))
            true_values.append(float(item["sentiment"]))

    true_arr = np.array(true_values)
    pred_value = np.array(pred_classes) - 3  # class -> sentiment value
    true_class = np.array([sentiment_to_class(v) for v in true_values])

    non0 = true_arr != 0
    non0_true = true_arr[non0] > 0
    non0_pred = pred_value[non0] > 0
    has0_true = true_arr >= 0
    has0_pred = pred_value >= 0

    return {
        "n": len(true_arr),
        "acc7": float((pred_value + 3 == true_class).mean()),
        "acc2_non0": float((non0_true == non0_pred).mean()),
        "f1_non0": float(f1_score(non0_true, non0_pred, average="weighted")),
        "acc2_has0": float((has0_true == has0_pred).mean()),
        "f1_has0": float(f1_score(has0_true, has0_pred, average="weighted")),
        "mae": float(np.abs(pred_value - true_arr).mean()),
    }


def trainable_state_dict(model: nn.Module) -> dict:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(("text_encoder.", "audio_encoder."))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", default=str(DEFAULT_OUTPUT_DIR / "metadata.jsonl")
    )
    parser.add_argument(
        "--embeddings-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "embeddings"
    )
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-audio-sec", type=float, default=8.0)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=0, help="debug cap per split")
    parser.add_argument("--fusion-head", default="min", choices=["cls", "mean", "max", "min"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "mosei")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    print("Loading MOSEI samples...")
    samples = load_mosei_samples(args.metadata)
    samples_by_split = {split: [s for s in samples if s.split == split] for split in SPLITS}
    if args.max_samples > 0:
        for split in SPLITS:
            rng.shuffle(samples_by_split[split])
            samples_by_split[split] = samples_by_split[split][: args.max_samples]
    print({split: len(items) for split, items in samples_by_split.items()})

    model = CrossModalTransformerFusionClassifier(
        num_labels=NUM_CLASSES,
        fusion_head_output_type=args.fusion_head,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
    ).to(args.device)

    if args.precompute:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        print(f"Pre-computing frozen embeddings to {args.embeddings_dir} ...")
        precompute_shards(
            samples_by_split,
            model,
            tokenizer,
            args.embeddings_dir,
            args.max_audio_sec,
            args.shard_size,
            args.device,
        )
        print("Precompute done.")
        return

    if not shard_paths(args.embeddings_dir, "train"):
        raise FileNotFoundError(
            f"No embedding shards under {args.embeddings_dir}; run with --precompute first"
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = {"epoch": 0, "valid": None, "state": None}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_items = 0
        start = time.time()
        for item in iter_shard_items(args.embeddings_dir, "train", rng=rng):
            optimizer.zero_grad()
            output = model.forward_from_embeddings(
                item["text"].float().to(args.device),
                item["audio"].float().to(args.device),
            )
            label = torch.tensor(
                [sentiment_to_class(item["sentiment"])], device=args.device
            )
            loss = criterion(output.logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_items += 1
        train_loss = total_loss / max(1, n_items)
        valid_metrics = evaluate(model, args.embeddings_dir, "valid", args.device)
        history.append({"epoch": epoch, "train_loss": train_loss, **valid_metrics})
        print(
            f"Epoch {epoch:2d}  train_loss={train_loss:.4f}  "
            f"valid acc7={valid_metrics['acc7']:.3f}  "
            f"acc2_non0={valid_metrics['acc2_non0']:.3f}  "
            f"mae={valid_metrics['mae']:.3f}  "
            f"({time.time() - start:.0f}s)"
        )
        if best["valid"] is None or valid_metrics["acc7"] > best["valid"]["acc7"]:
            best = {
                "epoch": epoch,
                "valid": valid_metrics,
                "state": {k: v.cpu().clone() for k, v in trainable_state_dict(model).items()},
            }

    model.load_state_dict(best["state"], strict=False)
    test_metrics = evaluate(model, args.embeddings_dir, "test", args.device)
    print(f"\nBest epoch {best['epoch']} (valid acc7={best['valid']['acc7']:.3f})")
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    torch.save(
        {"trainable_state_dict": best["state"], "num_labels": NUM_CLASSES,
         "fusion_head_output_type": args.fusion_head},
        args.out_dir / "mosei_cross_modal.pt",
    )
    with open(args.out_dir / "mosei_cross_modal_metrics.json", "w") as handle:
        json.dump(
            {
                "model": "distilled_memocmt_frozen",
                "dataset": "cmu_mosei_native",
                "protocol": "standard folds, 7-class CE, batch 1, frozen encoders",
                "best_epoch": best["epoch"],
                "valid": best["valid"],
                "test": test_metrics,
                "history": history,
                "args": {k: str(v) for k, v in vars(args).items()},
            },
            handle,
            indent=2,
        )
    print(f"Saved checkpoint and metrics to {args.out_dir}")


if __name__ == "__main__":
    main()
