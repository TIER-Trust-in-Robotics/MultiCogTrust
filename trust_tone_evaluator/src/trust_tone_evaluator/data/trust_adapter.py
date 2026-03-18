"""Emotion-to-trust label mapping for adapting emotion datasets."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class TrustLabels:
    """Trust labels derived from emotion."""

    trust_level: int  # 1-5 scale
    trust_score: float  # 0.0-1.0 continuous
    confidence: float  # Speaker certainty
    hesitation: float  # Uncertainty markers
    stress: float  # Psychological tension
    engagement: float  # Attentiveness
    sincerity: float  # Perceived authenticity
    cognitive_load: float  # Mental effort (default)
    valence: float  # Emotional valence (-1 to 1)
    arousal: float  # Activation level (0-1)
    dominance: float  # Assertiveness (0-1)

    def to_indicator_vector(self) -> np.ndarray:
        """Convert to (9,) indicator vector matching TrustIndicators order."""
        return np.array(
            [
                self.confidence,
                self.hesitation,
                self.stress,
                self.engagement,
                self.sincerity,
                self.cognitive_load,
                (self.valence + 1) / 2,  # Normalize to 0-1
                self.arousal,
                self.dominance,
            ],
            dtype=np.float32,
        )

    @property
    def trust_level_index(self) -> int:
        """Return 0-indexed trust level for classification."""
        return self.trust_level - 1


class EmotionToTrustMapper:
    """
    Maps emotion labels to trust-relevant dimensions.

    High trust: calm, neutral, happy. Low trust: angry, fearful, sad, disgust.
    """

    # IEMOCAP emotion mapping
    IEMOCAP_TRUST_MAPPING: Dict[str, Dict[str, float]] = {
        "neutral": {
            "trust_level": 4,  # High
            "confidence": 0.7,
            "hesitation": 0.2,
            "stress": 0.2,
            "engagement": 0.5,
            "sincerity": 0.8,
            "cognitive_load": 0.3,
            "valence": 0.0,
            "arousal": 0.3,
            "dominance": 0.5,
        },
        "happy": {
            "trust_level": 4,
            "confidence": 0.8,
            "hesitation": 0.1,
            "stress": 0.2,
            "engagement": 0.9,
            "sincerity": 0.7,
            "cognitive_load": 0.3,
            "valence": 0.8,
            "arousal": 0.7,
            "dominance": 0.6,
        },
        "sad": {
            "trust_level": 2,  # Low
            "confidence": 0.3,
            "hesitation": 0.5,
            "stress": 0.6,
            "engagement": 0.4,
            "sincerity": 0.8,
            "cognitive_load": 0.6,
            "valence": -0.7,
            "arousal": 0.3,
            "dominance": 0.3,
        },
        "angry": {
            "trust_level": 2,
            "confidence": 0.9,  # High confidence, but low trust
            "hesitation": 0.1,
            "stress": 0.9,
            "engagement": 0.9,
            "sincerity": 0.5,
            "cognitive_load": 0.4,
            "valence": -0.6,
            "arousal": 0.9,
            "dominance": 0.8,
        },
        "fearful": {
            "trust_level": 1,  # Very low
            "confidence": 0.2,
            "hesitation": 0.9,
            "stress": 0.9,
            "engagement": 0.7,
            "sincerity": 0.7,
            "cognitive_load": 0.8,
            "valence": -0.7,
            "arousal": 0.8,
            "dominance": 0.2,
        },
        "surprised": {
            "trust_level": 3,  # Moderate
            "confidence": 0.4,
            "hesitation": 0.6,
            "stress": 0.5,
            "engagement": 0.8,
            "sincerity": 0.7,
            "cognitive_load": 0.6,
            "valence": 0.2,
            "arousal": 0.8,
            "dominance": 0.4,
        },
        "disgust": {
            "trust_level": 1,
            "confidence": 0.6,
            "hesitation": 0.3,
            "stress": 0.7,
            "engagement": 0.6,
            "sincerity": 0.4,
            "cognitive_load": 0.5,
            "valence": -0.8,
            "arousal": 0.6,
            "dominance": 0.5,
        },
        "frustrated": {
            "trust_level": 2,
            "confidence": 0.7,
            "hesitation": 0.3,
            "stress": 0.8,
            "engagement": 0.7,
            "sincerity": 0.6,
            "cognitive_load": 0.6,
            "valence": -0.5,
            "arousal": 0.7,
            "dominance": 0.5,
        },
        "excited": {
            "trust_level": 4,
            "confidence": 0.8,
            "hesitation": 0.1,
            "stress": 0.3,
            "engagement": 0.9,
            "sincerity": 0.7,
            "cognitive_load": 0.4,
            "valence": 0.7,
            "arousal": 0.9,
            "dominance": 0.6,
        },
    }

    # RAVDESS mapping (includes calm)
    RAVDESS_TRUST_MAPPING: Dict[str, Dict[str, float]] = {
        **IEMOCAP_TRUST_MAPPING,
        "calm": {
            "trust_level": 5,  # Very high - calm is ideal for trust
            "confidence": 0.8,
            "hesitation": 0.1,
            "stress": 0.1,
            "engagement": 0.6,
            "sincerity": 0.9,
            "cognitive_load": 0.2,
            "valence": 0.3,
            "arousal": 0.2,
            "dominance": 0.6,
        },
    }

    def __init__(self, dataset: str = "iemocap"):
        dataset_lower = dataset.lower()

        if dataset_lower == "iemocap":
            self.mapping = self.IEMOCAP_TRUST_MAPPING.copy()
        elif dataset_lower == "ravdess":
            self.mapping = self.RAVDESS_TRUST_MAPPING.copy()
        else:
            # Use RAVDESS as default (most complete)
            self.mapping = self.RAVDESS_TRUST_MAPPING.copy()

        self.dataset = dataset_lower

    def map_emotion(self, emotion: str) -> Dict[str, float]:
        """Map an emotion label to trust dimension values."""
        emotion_lower = emotion.lower().strip()

        if emotion_lower not in self.mapping:
            # Default to neutral for unknown emotions
            return self.mapping.get("neutral", self._default_mapping())

        return self.mapping[emotion_lower].copy()

    def get_trust_labels(
        self,
        emotion: str,
        valence: Optional[float] = None,
        arousal: Optional[float] = None,
        dominance: Optional[float] = None,
    ) -> TrustLabels:
        """Get TrustLabels from emotion, optionally overriding VAD dimensions."""
        mapping = self.map_emotion(emotion)

        # Override with dimensional labels if provided
        if valence is not None:
            mapping["valence"] = valence
        if arousal is not None:
            mapping["arousal"] = arousal
        if dominance is not None:
            mapping["dominance"] = dominance

        # Compute trust score from level
        trust_level = int(mapping["trust_level"])
        trust_score = (trust_level - 1) / 4.0

        return TrustLabels(
            trust_level=trust_level,
            trust_score=trust_score,
            confidence=mapping["confidence"],
            hesitation=mapping["hesitation"],
            stress=mapping["stress"],
            engagement=mapping["engagement"],
            sincerity=mapping["sincerity"],
            cognitive_load=mapping["cognitive_load"],
            valence=mapping["valence"],
            arousal=mapping["arousal"],
            dominance=mapping["dominance"],
        )

    def get_labels_tuple(
        self, emotion: str
    ) -> Tuple[int, float, np.ndarray]:
        """Return (trust_level, trust_score, indicators_array) for dataset loading."""
        labels = self.get_trust_labels(emotion)
        return (
            labels.trust_level,
            labels.trust_score,
            labels.to_indicator_vector(),
        )

    def _default_mapping(self) -> Dict[str, float]:
        """Return default (neutral) mapping."""
        return {
            "trust_level": 3,
            "confidence": 0.5,
            "hesitation": 0.5,
            "stress": 0.5,
            "engagement": 0.5,
            "sincerity": 0.5,
            "cognitive_load": 0.5,
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.5,
        }

    def enhance_with_dimensional(
        self,
        emotion: str,
        valence: float,
        arousal: float,
        dominance: float,
    ) -> TrustLabels:
        """Blend categorical emotion with dimensional VAD annotations (useful for IEMOCAP)."""
        labels = self.get_trust_labels(emotion)

        if valence >= 0 and valence <= 1:
            valence = valence * 2 - 1  # Convert 0-1 to -1 to 1

        valence_effect = valence * 0.15
        adjusted_trust_score = np.clip(labels.trust_score + valence_effect, 0, 1)
        adjusted_stress = np.clip(labels.stress + (arousal - 0.5) * 0.2, 0, 1)
        adjusted_confidence = np.clip(labels.confidence + (dominance - 0.5) * 0.2, 0, 1)
        adjusted_engagement = np.clip(arousal, 0, 1)

        adjusted_trust_level = int(np.round(adjusted_trust_score * 4 + 1))
        adjusted_trust_level = np.clip(adjusted_trust_level, 1, 5)

        return TrustLabels(
            trust_level=adjusted_trust_level,
            trust_score=adjusted_trust_score,
            confidence=adjusted_confidence,
            hesitation=labels.hesitation,
            stress=adjusted_stress,
            engagement=adjusted_engagement,
            sincerity=labels.sincerity,
            cognitive_load=labels.cognitive_load,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )

    @property
    def available_emotions(self) -> list:
        """Return list of emotions this mapper handles."""
        return list(self.mapping.keys())

    @property
    def num_trust_levels(self) -> int:
        """Return number of trust levels (5)."""
        return 5

    def __repr__(self) -> str:
        return f"EmotionToTrustMapper(dataset='{self.dataset}', emotions={len(self.mapping)})"
