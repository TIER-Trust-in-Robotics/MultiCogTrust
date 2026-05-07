"""Train audio-only GRU baseline on IEMOCAP."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Add repo root so "from src.*" resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.iemocap_aligned_dataset import load_iemocap_aligned  # noqa: E402
from src.models.baselines import AudioOnlyGRUClassifier  # noqa: E402


def collate_fn(batch):
    """Pad variable-length audio_frame_features (T x 88) to longest in batch."""
    raw = [np.array(b["audio_frame_features"], dtype=np.float32) for b in batch]

    max_frames = max(f.shape[0] for f in raw)
    feat_dim = raw[0].shape[1]

    audio_features = torch.zeros(len(batch), max_frames, feat_dim)
    audio_mask = torch.zeros(len(batch), max_frames, dtype=torch.bool)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    for i, f in enumerate(raw):
        n = f.shape[0]
        audio_features[i, :n] = torch.from_numpy(f)
        audio_mask[i, :n] = True

    return {"audio_features": audio_features, "audio_mask": audio_mask, "labels": labels}


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            audio_features = batch["audio_features"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(audio_features, audio_mask)
            total_loss += criterion(logits, labels).item() * len(labels)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    loss = total_loss / len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss, acc, macro_f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path("models/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, num_labels, label_names = load_iemocap_aligned()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # Infer feature dim from first sample
    input_dim = np.array(train_ds[0]["audio_frame_features"], dtype=np.float32).shape[-1]

    model = AudioOnlyGRUClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1, best_epoch = -1.0, 0
    best_preds, best_labels = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            audio_features = batch["audio_features"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(audio_features, audio_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        train_loss /= len(train_ds)
        test_loss, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"test_loss={test_loss:.4f} | acc={test_acc:.4f} | macro_f1={test_f1:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "audio_only_gru.pt")
            best_preds, best_labels = preds, labels

    report = classification_report(best_labels, best_preds, target_names=label_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(best_labels, best_preds).tolist()

    metrics = {
        "model": "audio_only_gru",
        "dataset": "iemocap_aligned",
        "num_labels": num_labels,
        "label_names": label_names,
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "best_epoch": best_epoch,
        "test_accuracy": report["accuracy"],
        "test_macro_f1": best_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    with open(out_dir / "audio_only_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBest macro F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
