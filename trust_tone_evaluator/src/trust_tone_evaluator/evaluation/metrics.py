"""Evaluation metrics for trust assessment."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TrustMetricsCalculator:
    """Computes classification, ordinal, regression, and per-indicator metrics."""

    TRUST_LEVEL_NAMES = ["Very Low", "Low", "Moderate", "High", "Very High"]
    INDICATOR_NAMES = [
        "confidence",
        "hesitation",
        "stress",
        "engagement",
        "sincerity",
        "cognitive_load",
        "valence",
        "arousal",
        "dominance",
    ]

    def __init__(self):
        """Initialize metrics calculator."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for metrics computation")

    def compute_all(
        self,
        pred_logits: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_indicators: torch.Tensor,
        target_levels: torch.Tensor,
        target_scores: torch.Tensor,
        target_indicators: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        metrics = {}

        # Convert to numpy
        pred_levels = pred_logits.argmax(dim=-1).cpu().numpy()
        pred_scores_np = pred_scores.cpu().numpy()
        pred_ind = pred_indicators.cpu().numpy()

        true_levels = target_levels.cpu().numpy()
        true_scores_np = target_scores.cpu().numpy()
        true_ind = target_indicators.cpu().numpy()

        # Classification metrics
        classification_metrics = self.compute_classification_metrics(
            pred_levels, true_levels
        )
        metrics.update(classification_metrics)

        # Regression metrics for trust score
        regression_metrics = self.compute_regression_metrics(
            pred_scores_np, true_scores_np, prefix="score"
        )
        metrics.update(regression_metrics)

        # Per-indicator metrics
        indicator_metrics = self.compute_indicator_metrics(pred_ind, true_ind)
        metrics.update(indicator_metrics)

        return metrics

    def compute_classification_metrics(
        self,
        pred_levels: np.ndarray,
        true_levels: np.ndarray,
    ) -> Dict[str, float]:
        """Compute accuracy, F1, kappa, and one-off accuracy for trust levels."""
        metrics = {}

        # Basic accuracy
        metrics["level_accuracy"] = accuracy_score(true_levels, pred_levels)

        # F1 scores
        metrics["level_f1_macro"] = f1_score(
            true_levels, pred_levels, average="macro", zero_division=0
        )
        metrics["level_f1_weighted"] = f1_score(
            true_levels, pred_levels, average="weighted", zero_division=0
        )

        # Precision and recall
        metrics["level_precision_macro"] = precision_score(
            true_levels, pred_levels, average="macro", zero_division=0
        )
        metrics["level_recall_macro"] = recall_score(
            true_levels, pred_levels, average="macro", zero_division=0
        )

        # Cohen's Kappa (accounts for chance agreement)
        # Quadratic weights for ordinal data
        try:
            metrics["level_kappa"] = cohen_kappa_score(
                true_levels, pred_levels, weights="quadratic"
            )
        except ValueError:
            metrics["level_kappa"] = 0.0

        # One-off accuracy (within 1 level)
        metrics["level_one_off_accuracy"] = np.mean(
            np.abs(pred_levels - true_levels) <= 1
        )

        # Mean Absolute Error for ordinal prediction
        metrics["level_mae"] = mean_absolute_error(true_levels, pred_levels)

        return metrics

    def compute_regression_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and Pearson correlation."""
        prefix = f"{prefix}_" if prefix else ""
        metrics = {}

        # Mean Absolute Error
        metrics[f"{prefix}mae"] = mean_absolute_error(targets, predictions)

        # Root Mean Squared Error
        metrics[f"{prefix}rmse"] = np.sqrt(mean_squared_error(targets, predictions))

        # Pearson correlation
        if len(np.unique(predictions)) > 1 and len(np.unique(targets)) > 1:
            correlation = np.corrcoef(predictions, targets)[0, 1]
            metrics[f"{prefix}correlation"] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics[f"{prefix}correlation"] = 0.0

        return metrics

    def compute_indicator_metrics(
        self,
        pred_indicators: np.ndarray,
        true_indicators: np.ndarray,
    ) -> Dict[str, float]:
        """Compute per-indicator MAE and correlation."""
        metrics = {}

        # Overall indicator MAE
        metrics["indicator_mae_mean"] = mean_absolute_error(
            true_indicators, pred_indicators
        )

        # Per-indicator metrics
        for i, name in enumerate(self.INDICATOR_NAMES):
            if i >= pred_indicators.shape[1]:
                break

            pred = pred_indicators[:, i]
            true = true_indicators[:, i]

            # MAE
            mae = mean_absolute_error(true, pred)
            metrics[f"indicator_{name}_mae"] = mae

            # Correlation
            if len(np.unique(pred)) > 1 and len(np.unique(true)) > 1:
                corr = np.corrcoef(pred, true)[0, 1]
                metrics[f"indicator_{name}_corr"] = corr if not np.isnan(corr) else 0.0
            else:
                metrics[f"indicator_{name}_corr"] = 0.0

        return metrics

    def compute_confusion_matrix(
        self,
        pred_levels: np.ndarray,
        true_levels: np.ndarray,
    ) -> np.ndarray:
        """Return confusion matrix for trust levels."""
        return confusion_matrix(true_levels, pred_levels)

    def get_classification_report(
        self,
        pred_levels: np.ndarray,
        true_levels: np.ndarray,
    ) -> str:
        """Return formatted sklearn classification report."""
        return classification_report(
            true_levels,
            pred_levels,
            target_names=self.TRUST_LEVEL_NAMES,
            zero_division=0,
        )

    def summarize(
        self,
        metrics: Dict[str, float],
        show_indicators: bool = False,
    ) -> str:
        """Return a human-readable metrics summary string."""
        lines = [
            "=" * 50,
            "Trust Assessment Evaluation Summary",
            "=" * 50,
            "",
            "Classification Metrics (Trust Levels):",
            f"  Accuracy:         {metrics.get('level_accuracy', 0):.4f}",
            f"  One-off Accuracy: {metrics.get('level_one_off_accuracy', 0):.4f}",
            f"  F1 (macro):       {metrics.get('level_f1_macro', 0):.4f}",
            f"  F1 (weighted):    {metrics.get('level_f1_weighted', 0):.4f}",
            f"  Cohen's Kappa:    {metrics.get('level_kappa', 0):.4f}",
            "",
            "Regression Metrics (Trust Score):",
            f"  MAE:              {metrics.get('score_mae', 0):.4f}",
            f"  RMSE:             {metrics.get('score_rmse', 0):.4f}",
            f"  Correlation:      {metrics.get('score_correlation', 0):.4f}",
            "",
            f"Indicator MAE (mean): {metrics.get('indicator_mae_mean', 0):.4f}",
        ]

        if show_indicators:
            lines.extend(["", "Per-Indicator Metrics:"])
            for name in self.INDICATOR_NAMES:
                mae = metrics.get(f"indicator_{name}_mae", 0)
                corr = metrics.get(f"indicator_{name}_corr", 0)
                lines.append(f"  {name:15s}: MAE={mae:.4f}, Corr={corr:.4f}")

        lines.append("=" * 50)

        return "\n".join(lines)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    emotion_to_trust_map: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    """Evaluate a trust model on a dataset. Returns all metrics."""
    model = model.to(device)
    model.eval()

    all_pred_logits = []
    all_pred_scores = []
    all_pred_indicators = []
    all_true_levels = []
    all_true_scores = []
    all_true_indicators = []
    all_emotion_logits = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            outputs = model(features)

            all_pred_logits.append(outputs["trust_logits"].cpu())
            all_pred_scores.append(outputs["trust_score"].cpu())
            all_pred_indicators.append(outputs["indicators"].cpu())

            if "emotion_logits" in outputs:
                all_emotion_logits.append(outputs["emotion_logits"].cpu())

            all_true_levels.append(batch["trust_level"])
            all_true_scores.append(batch["trust_score"])
            all_true_indicators.append(batch["indicators"])

    # Concatenate
    pred_logits = torch.cat(all_pred_logits)
    pred_scores = torch.cat(all_pred_scores)
    pred_indicators = torch.cat(all_pred_indicators)
    true_levels = torch.cat(all_true_levels)
    true_scores = torch.cat(all_true_scores)
    true_indicators = torch.cat(all_true_indicators)

    # If emotion-to-trust mapping is provided, derive trust predictions from emotions
    if emotion_to_trust_map and all_emotion_logits:
        emotion_logits = torch.cat(all_emotion_logits)
        pred_emotions = emotion_logits.argmax(dim=-1)

        # Create synthetic trust logits from emotion predictions
        # This maps each predicted emotion to its trust level
        num_trust_levels = pred_logits.size(-1)
        mapped_logits = torch.zeros_like(pred_logits)
        for emo_idx, trust_idx in emotion_to_trust_map.items():
            mask = pred_emotions == emo_idx
            if mask.any():
                mapped_logits[mask, trust_idx] = 1.0
        pred_logits = mapped_logits

    # Compute metrics
    calculator = TrustMetricsCalculator()
    metrics = calculator.compute_all(
        pred_logits,
        pred_scores,
        pred_indicators,
        true_levels,
        true_scores,
        true_indicators,
    )

    return metrics
