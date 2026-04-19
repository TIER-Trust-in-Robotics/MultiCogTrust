from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import loader


class MLP_gate(nn.Module):
    """Two-layer MLP that predicts neutral vs non-neutral speech from prosodic cues."""

    def __init__(self, n_features: int = 19, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def _dataset_to_loader(ds, batch_size: int, shuffle: bool, feature_key: str = "trust_features") -> DataLoader:
    """Convert a Hugging Face Dataset object into a PyTorch DataLoader."""
    vectors = [
        np.asarray(sample[feature_key], dtype=np.float32)
        for sample in ds["audio_features"]
    ]
    labels = np.asarray(ds["labels"], dtype=np.float32)

    features = torch.from_numpy(np.stack(vectors))
    targets = torch.from_numpy(labels).unsqueeze(1)
    tensor_ds = TensorDataset(features, targets)
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=shuffle)


class MLP_gate_full(nn.Module):
    """Larger MLP using all 88 eGeMAPSv02 features. Returns logits (no sigmoid)."""

    def __init__(self, n_features: int = 88, hidden_dims: tuple = (256, 128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits. Apply sigmoid for probabilities."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def _evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.BCELoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            preds = (outputs >= 0.5).float()

            total_loss += loss.item() * inputs.size(0)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_mlp(
    dataset_path: str,
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    use_full_features: bool = False,
) -> tuple[MLP_gate | MLP_gate_full, dict]:
    """Train an MLP gate on IEMOCAP prosodic features.

    Args:
        use_full_features: If True, train MLP_gate_full on all 88 eGeMAPSv02 features.
                           If False, train MLP_gate on the 19-dim trust subset.
    """
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds, test_ds, _, _ = loader.load_dataset_npz(dataset_path, binary=True)
    feature_key = "all_features" if use_full_features else "trust_features"
    train_loader = _dataset_to_loader(train_ds, batch_size=batch_size, shuffle=True, feature_key=feature_key)
    test_loader = _dataset_to_loader(test_ds, batch_size=batch_size, shuffle=False, feature_key=feature_key)

    model = (MLP_gate_full() if use_full_features else MLP_gate()).to(device_obj)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device_obj)
            targets = targets.to(device_obj)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_acc = _evaluate(model, test_loader, device_obj)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | test_acc={test_acc:.3f}"
        )

    return model, history


def _evaluate_full(
    model: MLP_gate_full,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_loss += loss.item() * inputs.size(0)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _load_arrays(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) float32 arrays from a .npz file."""
    train_ds, test_ds, _, _ = loader.load_dataset_npz(dataset_path, binary=True)
    X_train = np.stack([np.asarray(s["all_features"], dtype=np.float32) for s in train_ds["audio_features"]])
    y_train = np.asarray(train_ds["labels"], dtype=np.float32)
    X_test  = np.stack([np.asarray(s["all_features"], dtype=np.float32) for s in test_ds["audio_features"]])
    y_test  = np.asarray(test_ds["labels"], dtype=np.float32)
    return X_train, y_train, X_test, y_test


def train_mlp_full(
    dataset_path: str | list[str],
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 3e-4,
    device: str | None = None,
) -> tuple[MLP_gate_full, StandardScaler, dict]:
    """Train MLP_gate_full with StandardScaler, pos_weight, and LR scheduling.

    Args:
        dataset_path: Path (or list of paths) to .npz files produced by data/loader.py.
                      When a list is given the train splits are concatenated; each
                      dataset keeps its own test split (also concatenated).
    """
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    paths = [dataset_path] if isinstance(dataset_path, str) else dataset_path
    arrays = [_load_arrays(p) for p in paths]
    X_train = np.concatenate([a[0] for a in arrays], axis=0)
    y_train = np.concatenate([a[1] for a in arrays], axis=0)
    X_test  = np.concatenate([a[2] for a in arrays], axis=0)
    y_test  = np.concatenate([a[3] for a in arrays], axis=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # Weight neutral class (pos) by n_non_neutral/n_neutral to counteract imbalance.
    n_neutral     = int(y_train.sum())
    n_non_neutral = len(y_train) - n_neutral
    pos_weight = torch.tensor([n_non_neutral / n_neutral], dtype=torch.float32).to(device_obj)
    print(f"Train — non-neutral: {n_non_neutral}, neutral: {n_neutral}, pos_weight: {pos_weight.item():.3f}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1)),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).unsqueeze(1)),
        batch_size=batch_size, shuffle=False,
    )

    model = MLP_gate_full().to(device_obj)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, min_lr=1e-5)

    history: dict = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device_obj), targets.to(device_obj)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_loss, test_acc = _evaluate_full(model, test_loader, device_obj, criterion)
        scheduler.step(test_loss)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | test={test_loss:.4f} | "
            f"acc={test_acc:.3f} | lr={lr_now:.1e}"
        )

    return model, scaler, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the prosodic MLP gate on 19 eGeMAPS features from IEMOCAP."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/iemocap_features.npz",
        help="Path to the cached eGeMAPS feature archive.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (cpu, cuda, mps). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--output",
        default="models/trust_mlp.pt",
        help="Where to save the trained state_dict.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, history = train_mlp(
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "history": history,
            "config": vars(args),
        },
        output_path,
    )
    print(f"Saved trained model to {output_path}")


if __name__ == "__main__":
    main()
