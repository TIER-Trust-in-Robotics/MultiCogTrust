"""Loss functions for multi-task trust prediction."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalLoss(nn.Module):
    """Ordinal regression loss for ordered trust levels (VERY_LOW < ... < VERY_HIGH)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ordinal loss via cumulative binary targets."""
        batch_size = logits.size(0)
        device = logits.device

        # Create cumulative binary targets
        # For class k, binary targets are [1,1,...,1,0,0,...,0] with k ones
        cum_targets = torch.zeros(batch_size, self.num_classes - 1, device=device)
        for i in range(self.num_classes - 1):
            cum_targets[:, i] = (targets > i).float()

        # Use first num_classes-1 logits for cumulative predictions
        # Apply sigmoid to get cumulative probabilities
        cum_probs = torch.sigmoid(logits[:, : self.num_classes - 1])

        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy(cum_probs, cum_targets, reduction="mean")

        return loss


class FocalLoss(nn.Module):
    """Focal loss for class imbalance — down-weights easy examples."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TrustMultiTaskLoss(nn.Module):
    """Weighted sum of classification, regression, and indicator losses."""

    def __init__(
        self,
        level_weight: float = 1.0,
        score_weight: float = 1.0,
        indicator_weight: float = 0.5,
        emotion_weight: float = 0.0,
        use_ordinal_loss: bool = True,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        num_classes: int = 5,
        num_emotions: int = 8,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.level_weight = level_weight
        self.score_weight = score_weight
        self.indicator_weight = indicator_weight
        self.emotion_weight = emotion_weight
        self.use_ordinal_loss = use_ordinal_loss
        self.use_focal_loss = use_focal_loss
        self.num_classes = num_classes

        # Classification loss
        if use_ordinal_loss:
            self.classification_loss = OrdinalLoss(num_classes)
        elif use_focal_loss:
            self.classification_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.classification_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )

        # Emotion classification loss (separate from trust level)
        if emotion_weight > 0:
            self.emotion_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Regression loss for trust score
        self.regression_loss = nn.MSELoss()

        # Multi-label BCE for indicators
        self.indicator_loss = nn.BCELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss. Returns dict with individual losses and total_loss."""
        losses = {}

        # Trust level classification loss
        level_loss = self.classification_loss(
            predictions["trust_logits"],
            targets["trust_level"],
        )
        losses["level_loss"] = level_loss * self.level_weight

        # Trust score regression loss
        score_loss = self.regression_loss(
            predictions["trust_score"],
            targets["trust_score"],
        )
        losses["score_loss"] = score_loss * self.score_weight

        # Trust indicator loss
        indicator_loss = self.indicator_loss(
            predictions["indicators"],
            targets["indicators"],
        )
        losses["indicator_loss"] = indicator_loss * self.indicator_weight

        # Emotion classification loss (if available)
        if (
            self.emotion_weight > 0
            and "emotion_logits" in predictions
            and "emotion_idx" in targets
        ):
            emotion_loss = self.emotion_loss(
                predictions["emotion_logits"],
                targets["emotion_idx"],
            )
            losses["emotion_loss"] = emotion_loss * self.emotion_weight
        else:
            losses["emotion_loss"] = torch.tensor(0.0, device=predictions["trust_logits"].device)

        # Total loss
        losses["total_loss"] = (
            losses["level_loss"] + losses["score_loss"]
            + losses["indicator_loss"] + losses["emotion_loss"]
        )

        return losses

    def __repr__(self) -> str:
        return (
            f"TrustMultiTaskLoss(\n"
            f"  level_weight={self.level_weight},\n"
            f"  score_weight={self.score_weight},\n"
            f"  indicator_weight={self.indicator_weight},\n"
            f"  use_ordinal={self.use_ordinal_loss}\n"
            f")"
        )


class ConsistencyLoss(nn.Module):
    """Penalizes divergence between continuous trust_score and predicted trust_level."""

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        trust_score: torch.Tensor,
        trust_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Get predicted class probabilities
        probs = F.softmax(trust_logits, dim=-1)

        # Expected score from class probabilities
        # Classes are 0-4, normalize to 0-1
        num_classes = trust_logits.size(-1)
        class_scores = torch.arange(num_classes, device=trust_logits.device).float()
        class_scores = class_scores / (num_classes - 1)

        expected_score = (probs * class_scores).sum(dim=-1)

        # MSE between predicted score and expected score
        loss = F.mse_loss(trust_score, expected_score)

        return loss * self.weight
