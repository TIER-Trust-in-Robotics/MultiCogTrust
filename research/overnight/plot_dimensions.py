"""Standalone scientific figure from the completed, fixed dimension report."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from harness import OUT


def main():
    result = json.loads((OUT / "phase3_summary.json").read_text())
    assert result["finished"]
    dimensions = ("valence", "arousal", "dominance")
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    x = np.arange(3)
    for offset, key, label, color in ((-1, "text", "Text", "#CC7A00"),
                                     (0, "audio", "Audio", "#3073A7"),
                                     (1, "both", "Text + audio", "#37816E")):
        values = [result["iemocap"][key][name]["session4"]["pearson_r"] for name in dimensions]
        bars = ax.bar(x + offset * .24, values, width=.23, label=label, color=color)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=9)
    ax.set_xticks(x, [name.capitalize() for name in dimensions])
    ax.set_ylim(0, .85)
    ax.set_ylabel("Correlation with annotated dimension (Pearson r)")
    ax.set_title("Which affect dimensions do the frozen features retain?", pad=14)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "IEMOCAP Session 4 · 2,103 utterances · ridge selected OOF on Sessions 1–3\n"
             "Representation probes; no causal attribution to the fusion head and no trust target", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .095, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"phase3_dimension_correlations.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
