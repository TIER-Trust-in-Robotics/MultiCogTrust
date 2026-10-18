"""A restricted visual intervention; no probability-calibration claim."""

import numpy as np


def gated_labels(base, visual, available, visual_threshold, base_confidence_limit):
    base, visual = np.asarray(base), np.asarray(visual)
    available = np.asarray(available, dtype=bool)
    if base.shape != visual.shape or base.ndim != 2 or base.shape[1] != 4 or available.shape != (len(base),):
        raise ValueError("Expected aligned four-class probability arrays and availability")
    labels = base.argmax(1).copy()
    change = available & (labels != 1) & (visual.argmax(1) == 1) & (visual[:, 1] >= visual_threshold) & (base.max(1) <= base_confidence_limit)
    labels[change] = 1
    return labels, change
