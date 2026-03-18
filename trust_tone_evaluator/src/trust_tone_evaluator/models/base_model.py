"""Base class for trust assessment models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseTrustModel(ABC, nn.Module):
    """Abstract base class for trust assessment models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_trust_levels = config.get("num_trust_levels", 5)
        self.num_indicators = config.get("num_indicators", 9)
        self.embedding_dim = config.get("embedding_dim", 256)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass. Returns dict with:
        trust_logits, trust_score, indicators, embedding.
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the fusion embedding dimension."""
        pass

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_layers(self, layer_names: list) -> None:
        """Freeze layers matching the given name prefixes."""
        for name, param in self.named_parameters():
            for layer_name in layer_names:
                if name.startswith(layer_name):
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
