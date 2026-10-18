from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = ROOT / "models" / "baselines"
DEFAULT_METRICS = DEFAULT_BASELINE_DIR / "msp_sliding_frame_fusion_metrics.json"


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing MSP metrics file: {path}")
    return json.loads(path.read_text())


def save_training_curves(metrics: dict[str, Any], output_path: Path) -> None:
    history = metrics["history"]
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["test_loss"] for row in history], marker="o", label="Test")
    axes[0].set_title("MSP Sliding-Frame Fusion Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, [row["test_accuracy"] for row in history], marker="o", label="Accuracy")
    axes[1].plot(epochs, [row["test_macro_f1"] for row in history], marker="o", label="Macro F1")
    axes[1].set_title("MSP Test Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 0.65)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_class_f1(metrics: dict[str, Any], output_path: Path) -> None:
    labels = metrics["label_names"]
    values = [metrics["classification_report"][label]["f1"] for label in labels]
    colors = ["#4c78a8" if value > 0 else "#bab0ac" for value in values]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(np.arange(len(labels)), values, color=colors)
    ax.set_title("MSP Per-Class F1")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 0.75)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(metrics: dict[str, Any], output_path: Path) -> None:
    labels = metrics["label_names"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=max(0.45, float(normalized.max())))
    ax.set_title("MSP Normalized Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def ranked_classes(metrics: dict[str, Any]) -> tuple[tuple[str, float], tuple[str, float]]:
    ranked = sorted(
        (
            (label, values["f1"])
            for label, values in metrics["classification_report"].items()
        ),
        key=lambda item: item[1],
    )
    return ranked[-1], ranked[0]


def confusion_pairs(metrics: dict[str, Any], top_k: int = 5) -> list[tuple[str, int]]:
    labels = metrics["label_names"]
    matrix = metrics["confusion_matrix"]
    pairs = []
    for actual_idx, row in enumerate(matrix):
        for predicted_idx, count in enumerate(row):
            if actual_idx != predicted_idx and count:
                pairs.append((f"{labels[actual_idx]} -> {labels[predicted_idx]}", count))
    return sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]


def write_visual_report(
    metrics: dict[str, Any],
    figures: dict[str, Path],
    output_path: Path,
    baseline_dir: Path,
) -> None:
    strongest, weakest = ranked_classes(metrics)
    confusion = ", ".join(f"{pair} ({count})" for pair, count in confusion_pairs(metrics))
    figure_rows = ["| Figure | File |", "| --- | --- |"]
    image_lines = []
    for title, path in figures.items():
        rel = path.relative_to(baseline_dir)
        figure_rows.append(f"| {title} | `{rel.as_posix()}` |")
        image_lines.extend([f"![{title}]({rel.as_posix()})", ""])

    report = "\n".join(
        [
            "# MSP-PODCAST Baseline Visual Report",
            "",
            "## Scope",
            "",
            "These figures summarize the full MSP-PODCAST sliding-frame fusion run over "
            f"`{metrics['dataset']}` with {metrics['train_size']} train examples and "
            f"{metrics['test_size']} test examples.",
            "",
            "## Result",
            "",
            f"- Best epoch: {metrics['best_epoch']}",
            f"- Test accuracy: {metrics['test_accuracy']:.4f}",
            f"- Test macro F1: {metrics['test_macro_f1']:.4f}",
            f"- Strongest class by F1: {strongest[0]} ({strongest[1]:.3f})",
            f"- Weakest class by F1: {weakest[0]} ({weakest[1]:.3f})",
            f"- Largest confusions: {confusion}",
            "",
            "## Files",
            "",
            "\n".join(figure_rows),
            "",
            "## Figures",
            "",
            *image_lines,
        ]
    )
    output_path.write_text(report)


def update_findings_report(findings_path: Path, figures: dict[str, Path], baseline_dir: Path) -> None:
    if not findings_path.exists():
        return
    marker = "## Visualization Artifacts"
    original = findings_path.read_text()
    base = original.split(marker)[0].rstrip()
    rows = [marker, ""]
    for title, path in figures.items():
        rows.append(f"- {title}: `{path.relative_to(baseline_dir).as_posix()}`")
    findings_path.write_text(base + "\n\n" + "\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_BASELINE_DIR / "msp_baseline_visual_report.md",
    )
    parser.add_argument(
        "--findings",
        type=Path,
        default=DEFAULT_BASELINE_DIR / "msp_baseline_findings.md",
    )
    args = parser.parse_args()

    args.baseline_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.metrics)
    figures = {
        "Training curves": args.baseline_dir / "msp_training_curves.png",
        "Per-class F1": args.baseline_dir / "msp_per_class_f1.png",
        "Normalized confusion matrix": args.baseline_dir / "msp_confusion_matrix.png",
    }
    save_training_curves(metrics, figures["Training curves"])
    save_per_class_f1(metrics, figures["Per-class F1"])
    save_confusion_matrix(metrics, figures["Normalized confusion matrix"])
    write_visual_report(metrics, figures, args.report_output, args.baseline_dir)
    update_findings_report(args.findings, figures, args.baseline_dir)

    for path in [*figures.values(), args.report_output]:
        print(path)


if __name__ == "__main__":
    main()
