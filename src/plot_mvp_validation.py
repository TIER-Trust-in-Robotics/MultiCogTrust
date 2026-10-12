"""Plot the completed baseline's validation history and selected confusion matrix."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(metrics_path, output):
    report = json.loads(metrics_path.read_text())
    if not report["finished"] or report["test_evaluated"]:
        raise ValueError("Expected a completed development-only run")
    history = report["history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    epochs = [r["epoch"] for r in history]
    for metric, label in (
        ("macro_f1", "Macro F1"),
        ("uar", "UAR"),
        ("accuracy", "Accuracy"),
    ):
        axes[0].plot(
            epochs, [r["validation"][metric] for r in history], ".-", label=label
        )
    axes[0].axvline(
        report["best_epoch"],
        color="#444444",
        linestyle="--",
        label=f"Selected epoch {report['best_epoch']}",
    )
    axes[0].set(xlabel="Epoch", ylabel="Session 4 validation metric", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    matrix = np.array(report["validation"]["confusion"])
    normalized = matrix / matrix.sum(1, keepdims=True)
    image = axes[1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            axes[1].text(
                j,
                i,
                f"{matrix[i, j]}\n{normalized[i, j]:.1%}",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.55 else "black",
                fontsize=9,
            )
    labels = ["Anger", "Happiness", "Sadness", "Neutral"]
    axes[1].set(
        xticks=range(4),
        xticklabels=labels,
        yticks=range(4),
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="Reference",
    )
    fig.colorbar(image, ax=axes[1], shrink=0.7, label="Recall by reference class")
    fig.suptitle(
        "Full frozen audio/text baseline | reference transcripts | Session 5 unevaluated",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("models/mvp/audio_text_baseline.metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mvp/day4_baseline_validation.png"),
    )
    args = parser.parse_args()
    plot(args.metrics, args.output)
