"""MLP model for trust assessment from eGeMAPS functionals. Recommended for small datasets (<5000 samples)."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from trust_tone_evaluator.models.base_model import BaseTrustModel


class MLPTrustModel(BaseTrustModel):
    """
    Simple MLP for trust assessment from utterance-level features.

    Architecture:
    1. Input normalization (BatchNorm)
    2. Hidden layers with ReLU, BatchNorm, Dropout
    3. Shared embedding layer
    4. Multi-head outputs for trust dimensions
    """

    def __init__(
        self,
        input_dim: int = 88,
        hidden_dims: List[int] = None,
        num_trust_levels: int = 5,
        num_emotions: int = 8,
        num_indicators: int = 9,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        use_emotion_head: bool = False,
    ):
        config = {
            "num_trust_levels": num_trust_levels,
            "num_indicators": num_indicators,
            "embedding_dim": embedding_dim,
        }
        super().__init__(config)

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.input_dim = input_dim
        self.use_emotion_head = use_emotion_head

        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)

        # Build hidden layers
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Embedding projection
        self.embedding_proj = nn.Linear(in_dim, embedding_dim)

        # Trust level classification head
        self.trust_level_head = nn.Linear(embedding_dim, num_trust_levels)

        # Emotion classification head (for emotion-first training)
        if use_emotion_head:
            self.emotion_head = nn.Linear(embedding_dim, num_emotions)

        # Trust score regression head
        self.trust_score_head = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

        # Trust indicators head
        self.indicator_head = nn.Sequential(
            nn.Linear(embedding_dim, num_indicators),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Handle 3D input (batch, seq, features) by averaging over time
        if x.dim() == 3:
            x = x.mean(dim=1)

        # x is now (batch, features)
        x = self.input_norm(x)
        x = self.backbone(x)

        embedding = self.embedding_proj(x)

        trust_logits = self.trust_level_head(embedding)
        trust_score = self.trust_score_head(embedding).squeeze(-1)
        indicators = self.indicator_head(embedding)

        result = {
            "trust_logits": trust_logits,
            "trust_score": trust_score,
            "indicators": indicators,
            "embedding": embedding,
        }

        if self.use_emotion_head:
            result["emotion_logits"] = self.emotion_head(embedding)

        return result

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            outputs["trust_level_pred"] = outputs["trust_logits"].argmax(dim=-1)
        return outputs

    def __repr__(self) -> str:
        return (
            f"MLPTrustModel(\n"
            f"  input_dim={self.input_dim},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  num_params={self.get_num_parameters():,}\n"
            f")"
        )


def create_mlp_model(
    input_dim: int = 88,
    preset: str = "medium",
    **kwargs,
) -> MLPTrustModel:
    presets = {
        "small": {
            "hidden_dims": [64, 32],
            "embedding_dim": 32,
            "dropout": 0.2,
        },
        "medium": {
            "hidden_dims": [128, 64],
            "embedding_dim": 64,
            "dropout": 0.3,
        },
        "large": {
            "hidden_dims": [256, 128, 64],
            "embedding_dim": 128,
            "dropout": 0.4,
        },
    }

    config = presets.get(preset, presets["medium"])
    config.update(kwargs)

    return MLPTrustModel(input_dim=input_dim, **config)
