from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = ROOT / "models" / "baselines"

MODEL_FILES = {
    "Text-only": "text_only_metrics.json",
    "Fine-tuned text": "text_finetuned_metrics.json",
    "Audio-only": "audio_only_metrics.json",
    "Simple fusion": "simple_fusion_metrics.json",
    "Fine-tuned simple fusion": "simple_fusion_finetuned_text_metrics.json",
    "Sliding-frame fusion": "sliding_frame_fusion_metrics.json",
    "Fine-tuned sliding fusion": "sliding_frame_fusion_finetuned_text_metrics.json",
    "Cross-attention fusion": "cross_attention_fusion_metrics.json",
}


def load_metrics(baseline_dir: Path) -> dict[str, dict[str, Any]]:
    metrics = {}
    for model_name, filename in MODEL_FILES.items():
        path = baseline_dir / filename
        if not path.exists() and model_name not in {
            "Fine-tuned text",
            "Fine-tuned simple fusion",
            "Sliding-frame fusion",
            "Fine-tuned sliding fusion",
            "Cross-attention fusion",
        }:
            raise FileNotFoundError(f"Missing metrics file: {path}")
        if not path.exists():
            continue
        metrics[model_name] = json.loads(path.read_text())
    return metrics


def save_metric_comparison(
    metrics: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    names = list(metrics)
    accuracy = [metrics[name]["test_accuracy"] for name in names]
    macro_f1 = [metrics[name]["test_macro_f1"] for name in names]
    x = np.arange(len(names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11, 5.5))
    acc_bars = ax.bar(x - width / 2, accuracy, width, label="Test accuracy", color="#4c78a8")
    f1_bars = ax.bar(x + width / 2, macro_f1, width, label="Macro F1", color="#f58518")
    ax.set_title("IEMOCAP Baseline Test Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, max(max(accuracy), max(macro_f1)) + 0.08)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(acc_bars, fmt="%.3f", padding=3)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_training_curves(
    metrics: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False)
    for name, data in metrics.items():
        history = data["history"]
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label=name)
        axes[1].plot(epochs, [row["test_macro_f1"] for row in history], marker="o", label=name)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Test Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_class_f1(
    metrics: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    labels = next(iter(metrics.values()))["label_names"]
    x = np.arange(len(labels))
    width = min(0.22, 0.8 / max(len(metrics), 1))

    fig, ax = plt.subplots(figsize=(14, 6.0))
    offsets = np.linspace(-width, width, num=len(metrics))
    for offset, (name, data) in zip(offsets, metrics.items()):
        f1 = [data["classification_report"][label]["f1"] for label in labels]
        ax.bar(x + offset, f1, width, label=name)

    ax.set_title("Per-Class F1 by Baseline")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def normalize_confusion(confusion: list[list[int]]) -> np.ndarray:
    matrix = np.asarray(confusion, dtype=np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def save_confusion_matrices(
    metrics: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    labels = next(iter(metrics.values()))["label_names"]
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(max(16, 4 * len(metrics)), 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, metrics.items()):
        matrix = normalize_confusion(data["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=max(0.45, matrix.max()))
        ax.set_title(name)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Row-normalized rate")
    fig.suptitle("Normalized Confusion Matrices")
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


def confusion_pairs(metrics: dict[str, Any], top_k: int = 6) -> list[tuple[str, int]]:
    labels = metrics["label_names"]
    matrix = metrics["confusion_matrix"]
    pairs = []
    for actual_idx, row in enumerate(matrix):
        for predicted_idx, count in enumerate(row):
            if actual_idx == predicted_idx or count == 0:
                continue
            pairs.append((f"{labels[actual_idx]} -> {labels[predicted_idx]}", count))
    return sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]


def figure_table(figures: dict[str, Path], baseline_dir: Path) -> str:
    rows = ["| Figure | File |", "| --- | --- |"]
    for title, path in figures.items():
        rel_path = path.relative_to(baseline_dir)
        rows.append(f"| {title} | `{rel_path.as_posix()}` |")
    return "\n".join(rows)


def write_markdown_report(
    metrics: dict[str, dict[str, Any]],
    figures: dict[str, Path],
    baseline_dir: Path,
    output_path: Path,
) -> None:
    text = metrics["Text-only"]
    finetuned = metrics.get("Fine-tuned text")
    audio = metrics["Audio-only"]
    fusion = metrics["Simple fusion"]
    fusion_text_delta = fusion["test_macro_f1"] - text["test_macro_f1"]
    fusion_audio_delta = fusion["test_macro_f1"] - audio["test_macro_f1"]
    finetuned_note = "Fine-tuned text metrics are not present in this report."
    if finetuned:
        finetuned_note = (
            "Fine-tuned DistilBERT directly updates the text encoder on IEMOCAP. "
            f"Its macro F1 is **{finetuned['test_macro_f1']:.4f}**, "
            f"which is **{finetuned['test_macro_f1'] - text['test_macro_f1']:+.4f}** "
            "versus the frozen text baseline."
        )
    sliding = metrics.get("Sliding-frame fusion")
    best_model_name, best_model_metrics = max(
        metrics.items(),
        key=lambda item: item[1]["test_macro_f1"],
    )
    sliding_note = "Sliding-frame fusion metrics are not present in this report."
    if sliding:
        sliding_fusion_delta = sliding["test_macro_f1"] - fusion["test_macro_f1"]
        sliding_note = (
            "Sliding-frame fusion uses the raw `audio_frame_features` sequence directly. "
            f"Its macro F1 is **{sliding_fusion_delta:+.4f}** versus the existing "
            "word-aligned simple fusion baseline."
        )

    equivalence_path = baseline_dir / "sliding_embedding_equivalence.json"
    equivalence_note = "No sliding embedding equivalence diagnostic was found."
    if equivalence_path.exists():
        equivalence = json.loads(equivalence_path.read_text())
        equivalence_note = (
            f"Token-pooled sliding-window features matched existing word-aligned features for "
            f"{equivalence['equivalent_count']}/{equivalence['sample_count']} sampled records "
            f"(max absolute difference {equivalence['max_abs_diff']:.6g}). "
            f"{equivalence['interpretation']}"
        )

    comparison_rows = [
        "| Model | Accuracy | Macro F1 | Best epoch | Train loss | Test loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in metrics.items():
        best = data["history"][data["best_epoch"] - 1]
        comparison_rows.append(
            f"| {name} | {data['test_accuracy']:.4f} | {data['test_macro_f1']:.4f} | "
            f"{data['best_epoch']} | {best['train_loss']:.4f} | {data['test_loss']:.4f} |"
        )

    notes = []
    for name, data in metrics.items():
        strongest, weakest = ranked_classes(data)
        confusion = ", ".join(
            f"{pair} ({count})" for pair, count in confusion_pairs(data)
        )
        notes.append(
            f"### {name}\n\n"
            f"- Best class by F1: **{strongest[0]}** ({strongest[1]:.3f}).\n"
            f"- Weakest class by F1: **{weakest[0]}** ({weakest[1]:.3f}).\n"
            f"- Most common off-diagonal confusions: {confusion}.\n"
        )

    comparison_table = "\n".join(comparison_rows)
    detail_notes = "\n".join(notes)
    ranking = sorted(
        metrics.items(), key=lambda item: item[1]["test_macro_f1"], reverse=True
    )
    ranking_sentence = ", ".join(
        f"{name} ({data['test_macro_f1']:.4f})" for name, data in ranking
    )
    report = "\n".join(
        [
            "# IEMOCAP Baseline Visualization Notes",
            "",
            "## Scope",
            "",
            "These visualizations summarize the full baseline run over `data/iemocap_aligned_features`.",
            f"The split has {text['train_size']} train examples and {text['test_size']} test examples,",
            "using the existing 11-label IEMOCAP mapping:",
            f"{', '.join(text['label_names'])}.",
            "",
            "## Files",
            "",
            figure_table(figures, baseline_dir),
            "",
            "## Figure Previews",
            "",
            "![Aggregate metric comparison](baseline_metric_comparison.png)",
            "",
            "![Training curves](baseline_training_curves.png)",
            "",
            "![Per-class F1](baseline_per_class_f1.png)",
            "",
            "![Normalized confusion matrices](baseline_confusion_matrices.png)",
            "",
            "## Headline Result",
            "",
            f"Best macro F1 in this comparison is **{best_model_name}** "
            f"({best_model_metrics['test_macro_f1']:.4f}).",
            f"Existing simple fusion improves macro F1 by **{fusion_text_delta:+.4f}** over text-only and",
            f"**{fusion_audio_delta:+.4f}** over audio-only. That is a modest but real signal that",
            "token-aligned prosody adds useful information in the pooled-fusion setup.",
            finetuned_note,
            sliding_note,
            "",
            "## Sliding-Window Embedding Check",
            "",
            equivalence_note,
            "",
            "## Model Comparison",
            "",
            comparison_table,
            "",
            "## Visual Reading Guide",
            "",
            "- `baseline_metric_comparison.png` is the quickest view. Macro F1 ranks as:",
            f"  {ranking_sentence}.",
            "- `baseline_training_curves.png` shows learning dynamics across the frozen, fusion,",
            "  and fine-tuned text settings.",
            "- `baseline_per_class_f1.png` shows the class imbalance pain points. Sadness is easiest",
            "  across all models, while rare or ambiguous labels such as disgust, fear, surprise,",
            "  happiness, and other remain difficult.",
            "- `baseline_confusion_matrices.png` shows where mistakes concentrate. The models frequently",
            "  confuse undefined, frustration, neutral, and excitement, which are also semantically and",
            "  annotationally close in parts of IEMOCAP.",
            "",
            "## Per-Model Details",
            "",
            detail_notes,
            "## Interpretation",
            "",
            "The text-only classifier is already a strong baseline despite the frozen DistilBERT encoder,",
            "which suggests lexical content carries a lot of emotion signal in this split. The fine-tuned",
            "text benchmark tests whether updating DistilBERT should be treated as the overall text",
            "ceiling before adding multimodal fusion. The audio-only GRU captures some useful prosodic",
            "structure, especially for sadness, but struggles with several low-support or overlapping",
            "classes. The word-aligned and sliding-frame fusion models test whether prosody is more",
            "useful after word-level aggregation or as a separate temporal sequence.",
            "",
            "The next experiment should preserve token-level text/prosody alignment but replace the",
            "pooled concatenation with early-fusion self-attention or a small cross-modal attention block.",
            "The goal is to let the model decide which words should attend to which prosodic frames",
            "instead of averaging each modality independently before classification.",
            "",
        ]
    )
    output_path.write_text(report)


def notebook_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def write_notebook(output_path: Path) -> None:
    notebook = {
        "cells": [
            notebook_cell(
                "markdown",
                "# IEMOCAP Baseline Visualizations\n\n"
                "This notebook reloads the saved baseline metrics and reproduces the PNG figures.",
            ),
            notebook_cell(
                "code",
                "from pathlib import Path\n"
                "import sys\n\n"
                "def find_repo_root(start: Path) -> Path:\n"
                "    for candidate in [start, *start.parents]:\n"
                "        if (candidate / 'src' / 'plot_iemocap_baseline_results.py').exists():\n"
                "            return candidate\n"
                "    raise RuntimeError('Could not find MultiCogTrust repo root')\n\n"
                "ROOT = find_repo_root(Path.cwd())\n"
                "if str(ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT))\n\n"
                "from src.plot_iemocap_baseline_results import load_metrics\n\n"
                "baseline_dir = ROOT / 'models' / 'baselines'\n"
                "metrics = load_metrics(baseline_dir)\n"
                "[(name, data['test_accuracy'], data['test_macro_f1']) for name, data in metrics.items()]",
            ),
            notebook_cell(
                "markdown",
                "## Generated Figures\n\n"
                "Run the next cell to regenerate all PNG figures in `models/baselines/`.",
            ),
            notebook_cell(
                "code",
                "from src.plot_iemocap_baseline_results import main\n\n"
                "main(['--baseline-dir', str(baseline_dir)])",
            ),
            notebook_cell(
                "markdown",
                "## Static Figure Preview\n\n"
                "![Metric comparison](baseline_metric_comparison.png)\n\n"
                "![Training curves](baseline_training_curves.png)\n\n"
                "![Per-class F1](baseline_per_class_f1.png)\n\n"
                "![Confusion matrices](baseline_confusion_matrices.png)",
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, indent=2) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visualizations and narrative notes for IEMOCAP baselines."
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Defaults to <baseline-dir>/iemocap_baseline_visual_report.md.",
    )
    parser.add_argument(
        "--notebook-output",
        type=Path,
        default=None,
        help="Defaults to <baseline-dir>/iemocap_baseline_visualizations.ipynb.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    baseline_dir = args.baseline_dir
    baseline_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(baseline_dir)

    figures = {
        "Aggregate metric comparison": baseline_dir / "baseline_metric_comparison.png",
        "Training curves": baseline_dir / "baseline_training_curves.png",
        "Per-class F1": baseline_dir / "baseline_per_class_f1.png",
        "Normalized confusion matrices": baseline_dir / "baseline_confusion_matrices.png",
    }
    save_metric_comparison(metrics, figures["Aggregate metric comparison"])
    save_training_curves(metrics, figures["Training curves"])
    save_per_class_f1(metrics, figures["Per-class F1"])
    save_confusion_matrices(metrics, figures["Normalized confusion matrices"])

    markdown_output = (
        args.markdown_output
        if args.markdown_output is not None
        else baseline_dir / "iemocap_baseline_visual_report.md"
    )
    notebook_output = (
        args.notebook_output
        if args.notebook_output is not None
        else baseline_dir / "iemocap_baseline_visualizations.ipynb"
    )
    write_markdown_report(metrics, figures, baseline_dir, markdown_output)
    write_notebook(notebook_output)

    for path in [*figures.values(), markdown_output, notebook_output]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
