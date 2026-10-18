"""Train the MemoCMT-style cross-modal transformer on IEMOCAP (4-class).

Lightweight adaptation of MemoCMT (Khan et al., 2025): DistilBERT + DistilHuBERT
feeding the bidirectional cross-attention fusion in
``src.models.CrossModalTransformerFusionClassifier``.

Label setup follows the original: 4 classes with excited merged into happiness
    ang -> 0, hap -> 1, exc -> 1, sad -> 2, neu -> 3

Sanity mode (default) freezes both encoders and pre-computes their embeddings
once, then trains only the ~6M-param fusion + head on the cached embeddings.
This gives fast feedback that the pipeline learns before committing to a long,
fully fine-tuned run.

Usage:
    uv run python src/train_cross_modal_transformer_iemocap.py            # sanity run
    uv run python src/train_cross_modal_transformer_iemocap.py --max-samples 800 --epochs 25
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_aligned_dataset import resolve_data_path  # noqa: E402
from src.models import CrossModalTransformerFusionClassifier  # noqa: E402

LABEL_MAP = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
LABEL_NAMES = ["anger", "happiness", "sadness", "neutral"]
SAMPLE_RATE = 16000


@dataclass
class Sample:
    text: str
    audio_path: str
    label: int


def load_samples(metadata_path: str, data_root: str) -> list[Sample]:
    samples: list[Sample] = []
    missing = 0
    with open(metadata_path) as handle:
        for line in handle:
            record = json.loads(line)
            code = record["label_code"]
            if code not in LABEL_MAP:
                continue
            local_path = resolve_data_path(
                record["audio_path"], Path(data_root) / "data"
            )
            if not local_path.exists():
                missing += 1
                continue
            samples.append(
                Sample(
                    text=record["text"],
                    audio_path=str(local_path),
                    label=LABEL_MAP[code],
                )
            )
    if not samples:
        raise FileNotFoundError(
            f"No audio from {metadata_path} resolved under {data_root} "
            f"({missing} paths missing); restore data/IEMOCAP_full_release/ "
            "or pass --data-root"
        )
    if missing:
        print(f"Warning: skipped {missing} utterances with missing audio")
    return samples


def stratified_subsample(
    samples: list[Sample], max_samples: int, rng: random.Random
) -> list[Sample]:
    if max_samples <= 0 or max_samples >= len(samples):
        rng.shuffle(samples)
        return samples
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    per_label = max(1, max_samples // len(by_label))
    chosen: list[Sample] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        chosen.extend(label_samples[:per_label])
    rng.shuffle(chosen)
    return chosen


def load_audio(path: str, max_seconds: float) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio)
    if sr != SAMPLE_RATE:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    max_len = int(max_seconds * SAMPLE_RATE)
    if waveform.numel() > max_len:
        waveform = waveform[:max_len]
    return waveform


@dataclass
class CachedEmbedding:
    text_embedding: torch.Tensor  # [1, T_text, dim]
    audio_embedding: torch.Tensor  # [1, T_audio, dim]
    label: int


def precompute_embeddings(
    samples: list[Sample],
    model: CrossModalTransformerFusionClassifier,
    tokenizer,
    max_seconds: float,
    device: str,
) -> list[CachedEmbedding]:
    """Run the frozen encoders once and cache token/frame embeddings."""
    cache: list[CachedEmbedding] = []
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, sample in enumerate(samples):
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
            waveform = load_audio(sample.audio_path, max_seconds).unsqueeze(0).to(device)
            audio_embedding = model.encode_audio(waveform)
            cache.append(
                CachedEmbedding(
                    text_embedding=text_embedding.cpu(),
                    audio_embedding=audio_embedding.cpu(),
                    label=sample.label,
                )
            )
            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                rate = (i + 1) / (time.time() - start)
                print(f"  encoded {i + 1}/{len(samples)}  ({rate:.1f} utt/s)")
    return cache


def evaluate(
    model: CrossModalTransformerFusionClassifier,
    cache: list[CachedEmbedding],
    device: str,
) -> tuple[float, np.ndarray]:
    model.eval()
    correct = 0
    confusion = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=int)
    with torch.no_grad():
        for item in cache:
            output = model.forward_from_embeddings(
                item.text_embedding.to(device),
                item.audio_embedding.to(device),
            )
            prediction = int(output.logits.argmax(dim=-1).item())
            confusion[item.label, prediction] += 1
            correct += int(prediction == item.label)
    return correct / max(1, len(cache)), confusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata", default="data/iemocap_aligned_features/metadata.jsonl"
    )
    parser.add_argument("--data-root", default=os.path.abspath("."))
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-audio-sec", type=float, default=6.0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--fusion-head", default="min", choices=["cls", "mean", "max", "min"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    print("Loading IEMOCAP 4-class samples...")
    samples = load_samples(args.metadata, args.data_root)
    samples = stratified_subsample(samples, args.max_samples, rng)
    counts = {name: 0 for name in LABEL_NAMES}
    for sample in samples:
        counts[LABEL_NAMES[sample.label]] += 1
    print(f"Using {len(samples)} utterances: {counts}")

    split = int(len(samples) * (1 - args.val_frac))
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = CrossModalTransformerFusionClassifier(
        num_labels=len(LABEL_NAMES),
        fusion_head_output_type=args.fusion_head,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,  # sanity run: train only fusion + head
    ).to(args.device)

    print("Pre-computing encoder embeddings (frozen)...")
    train_cache = precompute_embeddings(
        train_samples, model, tokenizer, args.max_audio_sec, args.device
    )
    val_cache = precompute_embeddings(
        val_samples, model, tokenizer, args.max_audio_sec, args.device
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"Trainable params (fusion + head): {n_trainable / 1e6:.2f}M")

    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    val_acc, _ = evaluate(model, val_cache, args.device)
    print(f"Epoch 0 (untrained)  val_acc={val_acc:.3f}  (chance={1/len(LABEL_NAMES):.3f})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_cache)
        total_loss = 0.0
        for item in train_cache:
            optimizer.zero_grad()
            output = model.forward_from_embeddings(
                item.text_embedding.to(args.device),
                item.audio_embedding.to(args.device),
            )
            label = torch.tensor([item.label], device=args.device)
            loss = criterion(output.logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_cache)
        val_acc, _ = evaluate(model, val_cache, args.device)
        print(f"Epoch {epoch:2d}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    _, confusion = evaluate(model, val_cache, args.device)
    print("\nFinal val confusion (rows=true, cols=pred):")
    print("        " + "  ".join(f"{n[:4]:>5}" for n in LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name[:6]:>6}  " + "  ".join(f"{confusion[i, j]:5d}" for j in range(len(LABEL_NAMES))))


if __name__ == "__main__":
    main()
