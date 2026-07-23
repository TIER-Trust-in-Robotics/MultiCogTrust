"""Fine-tune the cross-modal transformer on IEMOCAP (4-class, session-split).

Unlike ``train_cross_modal_transformer_iemocap.py`` (frozen encoders +
pre-computed embeddings, for sanity checks), this script trains with the audio
encoder (DistilHuBERT) unfrozen, so the encoders run inside the training loop
with real padded batches. DistilBERT stays frozen by default to bound memory.

Split is speaker/session-independent: one IEMOCAP session is held out for test
(default Session 5), the rest are train, with a slice of train used for
validation/model selection. This is the rigorous IEMOCAP protocol; the paper's
random 80/10/10 leaks speakers across splits.

Designed to scale (full 5530 utterances, CUDA/MPS), but ``--max-samples`` and
``--epochs`` let you run a fast bounded check locally first.

Examples:
    # bounded local check (subset, 2 epochs, CPU)
    uv run python src/train_cross_modal_finetune_iemocap.py \
        --max-samples 400 --epochs 2 --batch-size 4 --device cpu

    # full run on a GPU box
    uv run python src/train_cross_modal_finetune_iemocap.py \
        --epochs 20 --batch-size 8 --device cuda --balanced-loss
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import CrossModalTransformerFusionClassifier  # noqa: E402

LABEL_MAP = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
LABEL_NAMES = ["anger", "happiness", "sadness", "neutral"]
SAMPLE_RATE = 16000


@dataclass
class Sample:
    text: str
    audio_path: str
    label: int
    session: int


def load_samples(metadata_path: str, data_root: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(metadata_path) as handle:
        for line in handle:
            record = json.loads(line)
            code = record["label_code"]
            if code not in LABEL_MAP:
                continue
            raw_path = record["audio_path"]
            idx = raw_path.find("data/IEMOCAP_full_release/")
            local_path = os.path.join(data_root, raw_path[idx:])
            if not os.path.exists(local_path):
                continue
            samples.append(
                Sample(
                    text=record["text"],
                    audio_path=local_path,
                    label=LABEL_MAP[code],
                    session=int(record["session"]),
                )
            )
    return samples


def cap_per_class(samples: list[Sample], max_samples: int, rng: random.Random) -> list[Sample]:
    if max_samples <= 0 or max_samples >= len(samples):
        return samples
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    per_label = max(1, max_samples // len(by_label))
    chosen: list[Sample] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        chosen.extend(label_samples[:per_label])
    return chosen


class IemocapDataset(Dataset):
    def __init__(self, samples: list[Sample], max_seconds: float) -> None:
        self.samples = samples
        self.max_len = int(max_seconds * SAMPLE_RATE)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        audio, sr = sf.read(sample.audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio)
        if sr != SAMPLE_RATE:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.numel() > self.max_len:
            waveform = waveform[: self.max_len]
        return sample.text, waveform, sample.label


def make_collate(tokenizer):
    def collate(batch):
        texts, waveforms, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=297,
        )
        max_audio = max(w.numel() for w in waveforms)
        audio = torch.zeros(len(waveforms), max_audio)
        for i, waveform in enumerate(waveforms):
            audio[i, : waveform.numel()] = waveform
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            audio,
            torch.tensor(labels, dtype=torch.long),
        )

    return collate


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, device) -> tuple[float, float, np.ndarray]:
    model.eval()
    confusion = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=int)
    with torch.no_grad():
        for input_ids, attention_mask, audio, labels in loader:
            output = model(
                input_ids.to(device),
                attention_mask.to(device),
                audio.to(device),
            )
            predictions = output.logits.argmax(dim=-1).cpu()
            for true, pred in zip(labels, predictions):
                confusion[int(true), int(pred)] += 1
    total = confusion.sum()
    weighted_acc = confusion.trace() / max(1, total)  # overall accuracy
    recalls = confusion.diagonal() / confusion.sum(axis=1).clip(min=1)
    unweighted_acc = float(recalls.mean())  # mean per-class recall
    return weighted_acc, unweighted_acc, confusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/iemocap_aligned_features/metadata.jsonl")
    parser.add_argument("--data-root", default=os.path.abspath("."))
    parser.add_argument("--test-session", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all (per-class cap otherwise)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-audio-sec", type=float, default=6.0)
    parser.add_argument("--fusion-head", default="min", choices=["cls", "mean", "max", "min"])
    parser.add_argument("--freeze-text", action="store_true", default=True)
    parser.add_argument("--unfreeze-text", dest="freeze_text", action="store_false")
    parser.add_argument("--freeze-audio", action="store_true", default=False)
    parser.add_argument("--balanced-loss", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="models/cross_modal_finetune_iemocap.pt")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")

    samples = load_samples(args.metadata, args.data_root)
    samples = cap_per_class(samples, args.max_samples, rng)

    test_samples = [s for s in samples if s.session == args.test_session]
    train_pool = [s for s in samples if s.session != args.test_session]
    rng.shuffle(train_pool)
    n_val = int(len(train_pool) * args.val_frac)
    val_samples, train_samples = train_pool[:n_val], train_pool[n_val:]

    def dist(items: list[Sample]) -> dict[str, int]:
        counts = {name: 0 for name in LABEL_NAMES}
        for item in items:
            counts[LABEL_NAMES[item.label]] += 1
        return counts

    print(f"Train {len(train_samples)} {dist(train_samples)}")
    print(f"Val   {len(val_samples)} {dist(val_samples)}")
    print(f"Test  {len(test_samples)} (Session {args.test_session}) {dist(test_samples)}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate = make_collate(tokenizer)

    def make_loader(items: list[Sample], shuffle: bool) -> DataLoader:
        return DataLoader(
            IemocapDataset(items, args.max_audio_sec),
            batch_size=args.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=args.num_workers,
        )

    train_loader = make_loader(train_samples, True)
    val_loader = make_loader(val_samples, False)
    test_loader = make_loader(test_samples, False)

    model = CrossModalTransformerFusionClassifier(
        num_labels=len(LABEL_NAMES),
        fusion_head_output_type=args.fusion_head,
        freeze_text_encoder=args.freeze_text,
        freeze_audio_encoder=args.freeze_audio,
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Trainable params: {n_trainable / 1e6:.1f}M "
        f"(text {'frozen' if args.freeze_text else 'trainable'}, "
        f"audio {'frozen' if args.freeze_audio else 'trainable'})"
    )

    if args.balanced_loss:
        counts = np.bincount([s.label for s in train_samples], minlength=len(LABEL_NAMES))
        weights = torch.tensor(counts.sum() / (len(counts) * counts.clip(min=1)), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        print(f"Class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_ua = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()
        for input_ids, attention_mask, audio, labels in train_loader:
            optimizer.zero_grad()
            output = model(input_ids.to(device), attention_mask.to(device), audio.to(device))
            loss = criterion(output.logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_loss = total_loss / max(1, len(train_loader))
        val_wa, val_ua, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:2d}  loss={train_loss:.4f}  "
            f"val_WA={val_wa:.3f}  val_UA={val_ua:.3f}  ({time.time() - start:.0f}s)"
        )
        if val_ua > best_val_ua:
            best_val_ua = val_ua
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_names": LABEL_NAMES,
                    "fusion_head": args.fusion_head,
                    "epoch": epoch,
                    "val_ua": val_ua,
                },
                args.out,
            )

    test_wa, test_ua, confusion = evaluate(model, test_loader, device)
    print(f"\nTEST (Session {args.test_session})  WA={test_wa:.3f}  UA={test_ua:.3f}")
    print("Confusion (rows=true, cols=pred):")
    print("        " + "  ".join(f"{n[:4]:>5}" for n in LABEL_NAMES))
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name[:6]:>6}  " + "  ".join(f"{confusion[i, j]:5d}" for j in range(len(LABEL_NAMES))))
    print(f"\nBest checkpoint (val_UA={best_val_ua:.3f}) saved to {args.out}")


if __name__ == "__main__":
    main()
