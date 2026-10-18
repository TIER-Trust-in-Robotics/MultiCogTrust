"""Native MOSEI regression metrics, following MMSA's published definitions.

Acc-7 clips to [-3, 3] and uses NumPy's round-to-even convention. Binary F1
is support-weighted. MAE and Pearson correlation use the unrounded scores.
An undefined correlation or empty nonzero subset is represented by None.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def mosei_metrics(truth, predicted):
    truth = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth):
        raise ValueError("Expected equal, nonempty one-dimensional score arrays")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("Sentiment scores must be finite")
    if (np.abs(truth) > 3).any():
        raise ValueError("MOSEI labels must be in [-3, 3]")

    nonzero = truth != 0
    corr = None
    if len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(predicted) > 0:
        corr = float(np.corrcoef(truth, predicted)[0, 1])
    result = {
        "n": len(truth),
        "n_nonzero": int(nonzero.sum()),
        "mae": float(np.abs(predicted - truth).mean()),
        "pearson_r": corr,
        "acc7": float(
            (np.rint(np.clip(predicted, -3, 3)) == np.rint(truth)).mean()
        ),
    }
    for suffix, actual, guess in (
        ("non0", truth[nonzero] > 0, predicted[nonzero] > 0),
        ("has0", truth >= 0, predicted >= 0),
    ):
        result[f"acc2_{suffix}"] = float((actual == guess).mean()) if len(actual) else None
        result[f"f1_{suffix}"] = (
            float(f1_score(actual, guess, average="weighted", zero_division=0))
            if len(actual) else None
        )
    return result


def class_scores(probabilities):
    """Return distinct argmax and expected-score projections of a 7-way head."""
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        p.ndim != 2 or p.shape[1] != 7 or not len(p)
        or not np.isfinite(p).all() or (p < 0).any()
        or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("Expected finite, normalized N x 7 probabilities")
    return {"argmax": p.argmax(axis=1) - 3, "expected": p @ np.arange(-3, 4)}
