"""OpenSMILE-based feature extraction for prosodic analysis."""

from typing import Any, Dict, List, Optional

import numpy as np

try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False

from trust_tone_evaluator.features.base_extractor import BaseFeatureExtractor


class OpenSMILEExtractor(BaseFeatureExtractor):
    """
    OpenSMILE-based feature extraction.

    Feature sets: eGeMAPS (88), GeMAPSv01b (62), ComParE_2016 (6373).
    eGeMAPS recommended for trust assessment.
    """

    # Available feature sets
    FEATURE_SETS = {
        "egemaps": "eGeMAPSv02",
        "gemaps": "GeMAPSv01b",
        "compare": "ComParE_2016",
    }

    # Trust-relevant features from eGeMAPS
    TRUST_RELEVANT_FEATURES = {
        # Pitch (F0) - correlates with confidence, stress
        "F0semitoneFrom27.5Hz_sma3nz_amean": "pitch_mean",
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "pitch_variability",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2": "pitch_range",
        # Energy/Loudness - correlates with engagement, dominance
        "loudness_sma3_amean": "loudness_mean",
        "loudness_sma3_stddevNorm": "loudness_variability",
        # Voice Quality - correlates with stress, sincerity
        "jitterLocal_sma3nz_amean": "jitter",
        "shimmerLocaldB_sma3nz_amean": "shimmer",
        "HNRdBACF_sma3nz_amean": "hnr",
        # Spectral - correlates with emotional state
        "alphaRatio_sma3_amean": "alpha_ratio",
        "hammarbergIndex_sma3_amean": "hammarberg_index",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_set: str = "egemaps",
        feature_level: str = "functionals",
    ):
        if not OPENSMILE_AVAILABLE:
            raise ImportError(
                "opensmile package not installed. Install with: pip install opensmile"
            )

        config = config or {}
        super().__init__(config)

        self.feature_set_name = feature_set.lower()
        self.feature_level = feature_level

        if self.feature_set_name not in self.FEATURE_SETS:
            raise ValueError(
                f"Unknown feature set: {feature_set}. "
                f"Available: {list(self.FEATURE_SETS.keys())}"
            )

        # Initialize opensmile
        feature_set_enum = getattr(opensmile.FeatureSet, self.FEATURE_SETS[self.feature_set_name])
        feature_level_enum = (
            opensmile.FeatureLevel.Functionals
            if feature_level == "functionals"
            else opensmile.FeatureLevel.LowLevelDescriptors
        )

        self.smile = opensmile.Smile(
            feature_set=feature_set_enum,
            feature_level=feature_level_enum,
        )

        self._feature_names = list(self.smile.feature_names)
        self._feature_dim = len(self._feature_names)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract openSMILE features. Returns (feature_dim,) for functionals or (frames, feature_dim) for LLD."""
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # Extract features
        features_df = self.smile.process_signal(audio, self.sample_rate)

        # Convert to numpy
        features = features_df.values

        # Flatten for functionals (single row)
        if self.feature_level == "functionals":
            features = features.flatten()

        return features.astype(np.float32)

    def extract_with_names(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features as a name->value dictionary."""
        features = self.extract(audio)
        return dict(zip(self._feature_names, features))

    def extract_trust_relevant(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract trust-relevant features with readable names."""
        all_features = self.extract_with_names(audio)

        trust_features = {}
        for opensmile_name, readable_name in self.TRUST_RELEVANT_FEATURES.items():
            if opensmile_name in all_features:
                trust_features[readable_name] = all_features[opensmile_name]

        return trust_features

    def get_feature_dim(self) -> int:
        """Return feature dimensionality."""
        return self._feature_dim

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self._feature_names

    def __repr__(self) -> str:
        return (
            f"OpenSMILEExtractor(feature_set='{self.feature_set_name}', "
            f"feature_level='{self.feature_level}', "
            f"feature_dim={self._feature_dim})"
        )


def create_opensmile_extractor(
    feature_set: str = "egemaps",
    sample_rate: int = 16000,
) -> OpenSMILEExtractor:
    """Factory function for OpenSMILEExtractor."""
    config = {"sample_rate": sample_rate}
    return OpenSMILEExtractor(config=config, feature_set=feature_set)
