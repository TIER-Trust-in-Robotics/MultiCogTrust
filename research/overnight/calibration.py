"""Phase 4: OOF-only temperature fitting and frozen selective prediction rules."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from metrics import LABEL, RUNS, f1, load, oof, scored
from harness import OUT
from src.iemocap_embedding_cache import sha256, write_json

SEEDS = (42, 43, 44, 45, 46)
COVERAGES = (1., .9, .8, .7, .5)


def scaled(probabilities, temperature):
    logits = np.log(np.clip(probabilities, 1e-12, 1.)) / temperature
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def calibration_metrics(truth, probabilities):
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == truth
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = 0.
    reliability = []
    for index in range(10):
        selected = bins == index
        if selected.any():
            mean_confidence = float(confidence[selected].mean())
            accuracy = float(correct[selected].mean())
            ece += float(selected.mean()) * abs(mean_confidence - accuracy)
            reliability.append({"bin": index, "count": int(selected.sum()), "confidence": mean_confidence, "accuracy": accuracy})
    return {"count": len(truth), "accuracy": float(correct.mean()), "nll": float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1.)).mean()),
            "brier": float(np.square(probabilities - np.eye(4)[truth]).sum(1).mean()), "ece10": ece, "bins": reliability}


def evaluate_set(ids, predictions, temperature, thresholds):
    available = [uid for uid in ids if uid in predictions]
    probabilities = np.array([predictions[uid] for uid in available], dtype=np.float64)
    truth = np.array([LABEL[uid] for uid in available])
    calibrated = scaled(probabilities, temperature)
    if not np.array_equal(probabilities.argmax(1), calibrated.argmax(1)):
        raise AssertionError("Temperature scaling changed labels")
    confidence = calibrated.max(1)
    correct = calibrated.argmax(1) == truth
    coverage = []
    for target, threshold in zip(COVERAGES, thresholds):
        keep = confidence >= threshold
        coverage.append({"target_oof_coverage": target, "threshold": threshold, "retained": int(keep.sum()),
                         "coverage": float(keep.sum() / len(ids)), "accuracy": float(correct[keep].mean()) if keep.any() else None,
                         "per_class_coverage": [float(((truth == label) & keep).sum() / sum(LABEL[uid] == label for uid in ids)) for label in range(4)]})
    return {"scored_count": len(ids), "available_count": len(available), "abstentions_no_transcript": len(ids) - len(available),
            "macro_f1_all_scored": f1(ids, predictions), "raw": calibration_metrics(truth, probabilities),
            "calibrated": calibration_metrics(truth, calibrated), "selective": coverage}


def main():
    if (OUT / "phase4_summary.json").exists():
        raise FileExistsError("Preserving completed calibration analysis")
    train_ids, val_ids = scored((1, 2, 3)), scored((4,))
    runs, hashes = {}, {}
    for recipe, reference, asr in (("ref", "reference", "asr"), ("prev", "prevref", "prevasr")):
        results = []
        for seed in SEEDS:
            predictions = oof(recipe, seed, reference)
            if set(predictions) != set(train_ids):
                raise ValueError("Incomplete OOF calibration coverage")
            probabilities = np.array([predictions[uid] for uid in train_ids], dtype=np.float64)
            truth = np.array([LABEL[uid] for uid in train_ids])
            objective = lambda log_temperature: -np.log(np.clip(scaled(probabilities, np.exp(log_temperature))[np.arange(len(truth)), truth], 1e-12, 1.)).mean()
            fit = minimize_scalar(objective, bounds=(-3., 3.), method="bounded")
            if not fit.success:
                raise ValueError("Temperature fitting failed")
            temperature = float(np.exp(fit.x))
            confidence = scaled(probabilities, temperature).max(1)
            thresholds = [0. if coverage == 1 else float(np.quantile(confidence, 1 - coverage)) for coverage in COVERAGES]
            # Freeze all fitting outputs before loading Session 4 predictions.
            fit_path = OUT / "calibration_fits" / f"{recipe}_s{seed}.json"
            write_json(fit_path, {"temperature": temperature, "thresholds": thresholds, "target_coverages": COVERAGES,
                                 "fit_ids": train_ids, "fit_sessions": [1, 2, 3], "seed": seed, "source": "OOF reference only"})
            row = {"seed": seed, "temperature": temperature, "fit_sha256": sha256(fit_path),
                   "oof_reference": evaluate_set(train_ids, predictions, temperature, thresholds)}
            for condition, source in (("reference", reference), ("asr", asr)):
                row[f"session4_{condition}"] = evaluate_set(val_ids, load(f"full_{recipe}_s{seed}", f"4:{source}"), temperature, thresholds)
            results.append(row)
            for tag in [f"full_{recipe}_s{seed}", *[f"loso{held}_{recipe}_s{seed}" for held in (1, 2, 3)]]:
                hashes[tag] = sha256(RUNS / f"{tag}.npz")
        runs[recipe] = results
    write_json(OUT / "phase4_summary.json", {"finished": True, "runs": runs, "prediction_hashes": hashes,
               "source_sha256": sha256(Path(__file__)), "test_model_inference": False,
               "protocol": "Temperatures and thresholds fitted only to Sessions1-3 OOF reference predictions, separately per recipe/seed. S4 ASR assesses transfer of that calibration. Ten fixed equal-width ECE bins.",
               "limitations": ["OOF post-fit calibration metrics are in-sample to the temperature fit; S4 is the transfer report.", "Four empty ASR transcripts are always abstentions; calibration metrics use available predictions while coverage uses all 1031 targets.", "Selective accuracy concerns retained utterances; rejecting difficult classes can inflate it. Class coverage is reported.", "Seed variation is not uncertainty across human populations; calibrated confidence is not validated trust."]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for axis, condition in zip(axes, ("session4_reference", "session4_asr")):
        for recipe, label in (("ref", "A: current line"), ("prev", "B: previous + current")):
            curves = [row[condition]["selective"] for row in runs[recipe]]
            x = np.array([[point["coverage"] for point in curve] for curve in curves])
            y = np.array([[point["accuracy"] for point in curve] for curve in curves])
            axis.errorbar(x.mean(0), y.mean(0), yerr=y.std(0, ddof=1), marker="o", capsize=3, label=label)
        axis.set_title("Reference transcripts" if condition.endswith("reference") else "Whisper transcripts")
        axis.set_xlabel("Fraction of all utterances retained")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy among retained utterances")
    fig.suptitle("Session 4: thresholds chosen OOF; error bars = seed SD")
    fig.tight_layout()
    fig.savefig(OUT / "phase4_selective_accuracy.png", dpi=160)
    plt.close(fig)
    for recipe, rows in runs.items():
        print(recipe, "temperature", [round(row["temperature"], 3) for row in rows], flush=True)
        for condition in ("oof_reference", "session4_reference", "session4_asr"):
            print(condition, "ECE raw/cal", np.mean([row[condition]["raw"]["ece10"] for row in rows]), np.mean([row[condition]["calibrated"]["ece10"] for row in rows]), flush=True)


if __name__ == "__main__":
    main()
