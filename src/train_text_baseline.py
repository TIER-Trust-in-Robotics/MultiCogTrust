"""Train text-only DistilBERT baseline on IEMOCAP."""

import argparse
import json
import sys
from pathlib import Path

# Add repo root so "from src.models..." resolves when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from src.iemocap_aligned_dataset import load_iemocap_aligned
from src.models.baselines import TextOnlyClassifier


def tokenize_dataset(dataset, tokenizer, max_length: int):
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )


def collate_fn(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            total_loss += criterion(logits, labels).item() * len(labels)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    loss = total_loss / len(all_labels)
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return loss, acc, macro_f1, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--unfreeze-after", type=int, default=None,
                        help="Epoch to unfreeze encoder (None = stay frozen)")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path("models/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, num_labels, label_names = load_iemocap_aligned()

    # use_fast=True gives per-token/subword offsets and faster tokenization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_ds = tokenize_dataset(train_ds, tokenizer, args.max_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, args.max_length)

    train_ds.set_format(type="python", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="python", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    model = TextOnlyClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        freeze_encoder=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_f1, best_epoch = -1.0, 0
    best_preds, best_labels = [], []

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_after and epoch == args.unfreeze_after:
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            print(f"Epoch {epoch}: encoder unfrozen")

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
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
            torch.save(model.state_dict(), out_dir / "text_only_distilbert.pt")
            best_preds, best_labels = preds, labels

    report = classification_report(best_labels, best_preds, target_names=label_names,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(best_labels, best_preds).tolist()

    metrics = {
        "model": "text_only_distilbert",
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

    with open(out_dir / "text_only_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBest macro F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
