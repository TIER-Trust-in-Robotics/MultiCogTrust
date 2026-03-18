"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)

    @abstractmethod
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from audio. Returns (feature_dim,) or (num_frames, feature_dim)."""
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the feature dimensionality."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return names of each feature dimension."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sample_rate={self.sample_rate})"
