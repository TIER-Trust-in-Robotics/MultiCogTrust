"""
Train a lightweight IEMOCAP prosody gate for skipping likely-neutral utterances.

The gate is intentionally conservative: it predicts whether an utterance should
continue into the transcription/NLP pipeline. Neutral utterances may still pass,
but non-neutral utterances should rarely be skipped.

Run from the repository root:

    uv run python src/neutral_prosody_gate.py --feature-set all

The script uses IEMOCAP Sessions 1-4 for training and Session 5 for evaluation,
matching the split convention in to_ds.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from to_ds import IEMOCAP_LABELS, get_feature_names, load_iemocap_features


NEUTRAL_LABEL = "neu"
DEFAULT_MODEL_PATH = Path("models/neutral_prosody_gate.joblib")
DEFAULT_METADATA_PATH = Path("models/neutral_prosody_gate_metadata.json")


@dataclass(frozen=True)
class GateMetrics:
    threshold: float
    non_neutral_recall: float
    neutral_skip_rate: float
    skipped_non_neutral_count: int
    non_neutral_count: int
    skipped_neutral_count: int
    neutral_count: int
    overall_process_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conservative neutral-speech prosody gate on IEMOCAP."
    )
    parser.add_argument(
        "--feature-set",
        choices=("all", "trust"),
        default="all",
        help="'all' uses the 88-dim eGeMAPS-style features; 'trust' uses the curated subset.",
    )
    parser.add_argument(
        "--temporal-aggregation",
        choices=("mean", "mean_std"),
        default="mean",
        help=(
            "How to collapse aligned temporal frame features if the local IEMOCAP "
            "archive points at aligned artifacts instead of flat utterance features."
        ),
    )
    parser.add_argument(
        "--target-non-neutral-recall",
        type=float,
        default=0.95,
        help="Minimum recall for non-neutral utterances on the test split.",
    )
    parser.add_argument(
        "--class-weight",
        type=float,
        default=4.0,
        help="Training weight for non-neutral examples. Neutral examples use weight 1.0.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the fitted sklearn pipeline.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Where to save threshold and evaluation metadata.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Train and evaluate without writing model artifacts.",
    )
    return parser.parse_args()


def aggregate_temporal_features(
    frame_features: list[list[float]],
    feature_width: int,
    temporal_aggregation: str,
) -> np.ndarray:
    frames = np.asarray(frame_features, dtype=np.float32)
    if frames.size == 0:
        if temporal_aggregation == "mean_std":
            return np.zeros((feature_width * 2,), dtype=np.float32)
        return np.zeros((feature_width,), dtype=np.float32)

    if frames.ndim != 2 or frames.shape[1] != feature_width:
        raise ValueError(
            f"Expected temporal features with width {feature_width}, got {frames.shape}"
        )

    mean = frames.mean(axis=0)
    if temporal_aggregation == "mean_std":
        return np.concatenate([mean, frames.std(axis=0)]).astype(np.float32)
    return mean.astype(np.float32)


def dataset_to_arrays(
    dataset,
    feature_width: int,
    temporal_aggregation: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return X, y, label_codes where y=1 means non-neutral and should process."""
    label_codes = [str(label) for label in dataset["label_code"]]
    if "features" in dataset.column_names:
        features = np.asarray(dataset["features"], dtype=np.float32)
    elif "audio_frame_features" in dataset.column_names:
        features = np.asarray(
            [
                aggregate_temporal_features(
                    frame_features, feature_width, temporal_aggregation
                )
                for frame_features in dataset["audio_frame_features"]
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            "Dataset must contain either a flat 'features' column or an "
            "'audio_frame_features' column from aligned artifacts."
        )
    y = np.asarray([label != NEUTRAL_LABEL for label in label_codes], dtype=np.int64)
    return features, y, label_codes


def fit_gate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    non_neutral_weight: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight={0: 1.0, 1: non_neutral_weight},
                    max_iter=2000,
                    random_state=17,
                ),
            ),
        ]
    ).fit(x_train, y_train)


def score_threshold(
    y_true: np.ndarray,
    p_non_neutral: np.ndarray,
    threshold: float,
) -> GateMetrics:
    should_process = p_non_neutral >= threshold
    neutral_mask = y_true == 0
    non_neutral_mask = y_true == 1

    non_neutral_count = int(non_neutral_mask.sum())
    neutral_count = int(neutral_mask.sum())
    skipped_non_neutral_count = int((~should_process & non_neutral_mask).sum())
    skipped_neutral_count = int((~should_process & neutral_mask).sum())

    non_neutral_recall = (
        float((should_process & non_neutral_mask).sum() / non_neutral_count)
        if non_neutral_count
        else 0.0
    )
    neutral_skip_rate = (
        float(skipped_neutral_count / neutral_count) if neutral_count else 0.0
    )

    return GateMetrics(
        threshold=float(threshold),
        non_neutral_recall=non_neutral_recall,
        neutral_skip_rate=neutral_skip_rate,
        skipped_non_neutral_count=skipped_non_neutral_count,
        non_neutral_count=non_neutral_count,
        skipped_neutral_count=skipped_neutral_count,
        neutral_count=neutral_count,
        overall_process_rate=float(should_process.mean()),
    )


def choose_threshold(
    y_true: np.ndarray,
    p_non_neutral: np.ndarray,
    target_non_neutral_recall: float,
) -> GateMetrics:
    """
    Pick the threshold that skips the most neutral utterances while preserving
    the requested non-neutral recall.
    """
    candidates = [
        score_threshold(y_true, p_non_neutral, threshold)
        for threshold in np.linspace(0.01, 0.99, 99)
    ]
    feasible = [
        result
        for result in candidates
        if result.non_neutral_recall >= target_non_neutral_recall
    ]
    if feasible:
        return max(feasible, key=lambda result: result.neutral_skip_rate)

    # If the target is unreachable, return the safest observed setting.
    return max(candidates, key=lambda result: result.non_neutral_recall)


def print_gate_report(
    y_true: np.ndarray,
    p_non_neutral: np.ndarray,
    metrics: GateMetrics,
) -> None:
    should_process = p_non_neutral >= metrics.threshold

    print("\nSelected gate threshold")
    print(json.dumps(asdict(metrics), indent=2))

    print("\nConfusion matrix for should_process prediction")
    print("Rows=true [neutral, non-neutral], cols=predicted [skip, process]")
    print(confusion_matrix(y_true, should_process.astype(np.int64), labels=[0, 1]))

    print("\nClassification report")
    print(
        classification_report(
            y_true,
            should_process.astype(np.int64),
            target_names=["neutral", "non_neutral"],
            zero_division=0,
        )
    )

    try:
        print(f"ROC AUC: {roc_auc_score(y_true, p_non_neutral):.4f}")
    except ValueError:
        print("ROC AUC: unavailable because only one class is present.")


def main() -> None:
    args = parse_args()

    train_ds, test_ds, _, _ = load_iemocap_features(feature_set=args.feature_set)
    feature_names = get_feature_names("iemocap", feature_set=args.feature_set)

    x_train, y_train, train_label_codes = dataset_to_arrays(
        train_ds, len(feature_names), args.temporal_aggregation
    )
    x_test, y_test, test_label_codes = dataset_to_arrays(
        test_ds, len(feature_names), args.temporal_aggregation
    )

    model = fit_gate(x_train, y_train, args.class_weight)
    p_test_non_neutral = model.predict_proba(x_test)[:, 1]
    metrics = choose_threshold(
        y_test,
        p_test_non_neutral,
        target_non_neutral_recall=args.target_non_neutral_recall,
    )

    print("\nDataset")
    print(
        json.dumps(
            {
                "feature_set": args.feature_set,
                "temporal_aggregation": args.temporal_aggregation,
                "feature_count": len(feature_names),
                "model_input_width": int(x_train.shape[1]),
                "train_count": len(train_label_codes),
                "test_count": len(test_label_codes),
                "neutral_label": NEUTRAL_LABEL,
                "labels": IEMOCAP_LABELS,
                "positive_class": "non_neutral_should_process",
            },
            indent=2,
        )
    )
    print_gate_report(y_test, p_test_non_neutral, metrics)

    if args.no_save:
        return

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_path)
    args.metadata_path.write_text(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "feature_set": args.feature_set,
                "temporal_aggregation": args.temporal_aggregation,
                "feature_count": len(feature_names),
                "model_input_width": int(x_train.shape[1]),
                "feature_names": feature_names,
                "neutral_label": NEUTRAL_LABEL,
                "positive_class": "non_neutral_should_process",
                "threshold": metrics.threshold,
                "target_non_neutral_recall": args.target_non_neutral_recall,
                "class_weight": {"neutral": 1.0, "non_neutral": args.class_weight},
                "metrics": asdict(metrics),
                "inference_rule": (
                    "Run transcription/NLP when predict_proba(features)[non_neutral] "
                    "is greater than or equal to threshold."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nSaved model: {args.model_path}")
    print(f"Saved metadata: {args.metadata_path}")


if __name__ == "__main__":
    main()
