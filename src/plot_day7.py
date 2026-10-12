"""Render the completed development comparisons with their evaluation limits."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models/mvp/visual_day5"
OUTPUT = ROOT / "data/processed/mvp/day5_7"


def main():
    comparison = json.loads((MODELS / "comparison.json").read_text())
    deployment = json.loads((MODELS / "deployment.selection.json").read_text())
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    figure.suptitle(
        "Visual branches: a small reference-text gain, no ASR gain", fontsize=15
    )
    modes = ["blendshapes", "image", "combined"]
    summaries = [comparison["by_representation"][m] for m in modes]
    axes[0].errorbar(
        range(3),
        [s["mean_macro_f1"] for s in summaries],
        yerr=[s["std_macro_f1"] for s in summaries],
        fmt="o",
        capsize=5,
        color="#177a72",
        label="Three seeds: mean ± sample SD",
    )
    axes[0].axhline(
        comparison["baseline"]["macro_f1"],
        ls="--",
        color="#667788",
        label="Frozen audio/text baseline",
    )
    axes[0].set(
        xticks=range(3),
        xticklabels=["Blendshapes", "Image", "Combined"],
        ylabel="Validation macro F1",
        ylim=(0.735, 0.765),
        title="Reference transcripts · compact heads",
    )
    axes[0].legend(loc="upper left", fontsize=8)
    values = deployment["candidates"]
    axes[1].plot(
        [r["alpha"] for r in values],
        [r["metrics"]["macro_f1"] for r in values],
        "o-",
        color="#a75b26",
        label="Same combined head + Whisper tiny.en",
    )
    axes[1].scatter(
        [0],
        [values[0]["metrics"]["macro_f1"]],
        s=100,
        facecolors="none",
        edgecolors="#177a72",
        lw=2,
    )
    axes[1].annotate(
        "Deployment weight = 0",
        (0, values[0]["metrics"]["macro_f1"]),
        xytext=(0.30, 0.755),
        arrowprops={"arrowstyle": "->", "color": "#667788"},
        fontsize=9,
    )
    axes[1].set(
        xlabel="Visual probability weight",
        ylabel="Validation macro F1",
        ylim=(0.35, 0.79),
        title="Actual ASR · deployment weight selection",
    )
    axes[1].legend(loc="lower left", fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    figure.supxlabel(
        "Session 4: 1,031 utterances. Selection scores, not independent test results. Session 5 remains unused.",
        fontsize=9,
    )
    path = OUTPUT / "visual_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(path)


if __name__ == "__main__":
    main()
