"""Trust metrics output dataclasses for multimodal fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class TrustLevel(Enum):
    """Discrete trust level classification (1-5 scale)."""

    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5

    @classmethod
    def from_score(cls, score: float) -> "TrustLevel":
        """Convert continuous score (0-1) to discrete level."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH

    def to_score(self) -> float:
        """Convert discrete level to continuous score (0-1)."""
        return (self.value - 1) / 4.0


@dataclass
class ProsodyFeatures:
    """Raw prosodic feature values (pitch, energy, timing, voice quality)."""

    # Pitch (F0) features
    pitch_mean: float = 0.0  # Mean fundamental frequency (Hz)
    pitch_std: float = 0.0  # Standard deviation of F0
    pitch_range: float = 0.0  # F0 max - min
    pitch_slope: float = 0.0  # Rising/falling intonation (positive = rising)

    # Energy/loudness features
    energy_mean: float = 0.0  # Mean RMS energy
    energy_std: float = 0.0  # Energy variation
    energy_contour: Optional[np.ndarray] = None  # Time series (optional)

    # Temporal features
    speaking_rate: float = 0.0  # Estimated syllables per second
    pause_ratio: float = 0.0  # Ratio of silence to speech
    pause_count: int = 0  # Number of pauses
    pause_durations: List[float] = field(default_factory=list)  # Duration of each pause

    # Voice quality features
    jitter: float = 0.0  # Pitch perturbation (cycle-to-cycle variation)
    shimmer: float = 0.0  # Amplitude perturbation
    hnr: float = 0.0  # Harmonics-to-noise ratio (dB)
    formant_dispersion: float = 0.0  # Voice quality indicator

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding numpy arrays for JSON serialization."""
        return {
            "pitch_mean": float(self.pitch_mean),
            "pitch_std": float(self.pitch_std),
            "pitch_range": float(self.pitch_range),
            "pitch_slope": float(self.pitch_slope),
            "energy_mean": float(self.energy_mean),
            "energy_std": float(self.energy_std),
            "speaking_rate": float(self.speaking_rate),
            "pause_ratio": float(self.pause_ratio),
            "pause_count": int(self.pause_count),
            "pause_durations": [float(d) for d in self.pause_durations],
            "jitter": float(self.jitter),
            "shimmer": float(self.shimmer),
            "hnr": float(self.hnr),
            "formant_dispersion": float(self.formant_dispersion),
        }

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array(
            [
                self.pitch_mean,
                self.pitch_std,
                self.pitch_range,
                self.pitch_slope,
                self.energy_mean,
                self.energy_std,
                self.speaking_rate,
                self.pause_ratio,
                self.pause_count,
                self.jitter,
                self.shimmer,
                self.hnr,
                self.formant_dispersion,
            ],
            dtype=np.float32,
        )


@dataclass
class TrustIndicators:
    """Trust-relevant psychological indicators (0.0-1.0 unless noted)."""

    # Primary trust dimensions
    confidence: float = 0.5  # Speaker certainty/self-assurance
    hesitation: float = 0.5  # Uncertainty markers (filled pauses, restarts)
    stress: float = 0.5  # Psychological stress/tension
    engagement: float = 0.5  # Attentiveness and involvement
    sincerity: float = 0.5  # Perceived authenticity/honesty

    # Secondary indicators
    cognitive_load: float = 0.5  # Mental effort estimation

    # Emotional dimensions (from VAD model)
    emotional_valence: float = 0.0  # Positive/negative affect (-1.0 to 1.0)
    emotional_arousal: float = 0.5  # Activation level (0.0 to 1.0)
    dominance: float = 0.5  # Assertiveness/control (0.0 to 1.0)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "confidence": float(self.confidence),
            "hesitation": float(self.hesitation),
            "stress": float(self.stress),
            "engagement": float(self.engagement),
            "sincerity": float(self.sincerity),
            "cognitive_load": float(self.cognitive_load),
            "emotional_valence": float(self.emotional_valence),
            "emotional_arousal": float(self.emotional_arousal),
            "dominance": float(self.dominance),
        }

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for model output comparison."""
        return np.array(
            [
                self.confidence,
                self.hesitation,
                self.stress,
                self.engagement,
                self.sincerity,
                self.cognitive_load,
                (self.emotional_valence + 1) / 2,  # Normalize valence to 0-1
                self.emotional_arousal,
                self.dominance,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "TrustIndicators":
        """Create from numpy array (model output)."""
        return cls(
            confidence=float(vector[0]),
            hesitation=float(vector[1]),
            stress=float(vector[2]),
            engagement=float(vector[3]),
            sincerity=float(vector[4]),
            cognitive_load=float(vector[5]),
            emotional_valence=float(vector[6] * 2 - 1),  # Convert back to -1 to 1
            emotional_arousal=float(vector[7]),
            dominance=float(vector[8]),
        )

    @classmethod
    def num_indicators(cls) -> int:
        """Return the number of indicator dimensions."""
        return 9


@dataclass
class TrustMetrics:
    """Complete output from the tone evaluator, designed for multimodal fusion."""

    # Temporal identification
    timestamp: datetime = field(default_factory=datetime.now)
    segment_id: str = ""
    audio_duration_ms: int = 0

    # Trust assessment outputs
    trust_level: TrustLevel = TrustLevel.MODERATE
    trust_score: float = 0.5  # Continuous 0.0-1.0
    trust_confidence: float = 0.5  # Model confidence in prediction

    # Detailed indicators
    indicators: TrustIndicators = field(default_factory=TrustIndicators)

    # Raw prosodic features (for analysis and fusion algorithms)
    prosody: ProsodyFeatures = field(default_factory=ProsodyFeatures)

    # Embedding for multimodal fusion (256-dimensional latent representation)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))

    # Metadata
    model_version: str = "0.1.0"
    processing_time_ms: float = 0.0

    # Optional: frame-level predictions for temporal analysis
    frame_predictions: Optional[np.ndarray] = None
    frame_timestamps: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON/API output."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "segment_id": self.segment_id,
            "audio_duration_ms": self.audio_duration_ms,
            "trust_level": self.trust_level.name,
            "trust_level_value": self.trust_level.value,
            "trust_score": float(self.trust_score),
            "trust_confidence": float(self.trust_confidence),
            "indicators": self.indicators.to_dict(),
            "prosody": self.prosody.to_dict(),
            "embedding": self.embedding.tolist(),
            "model_version": self.model_version,
            "processing_time_ms": float(self.processing_time_ms),
        }

        if self.frame_predictions is not None:
            result["frame_predictions"] = self.frame_predictions.tolist()
        if self.frame_timestamps is not None:
            result["frame_timestamps"] = self.frame_timestamps.tolist()

        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustMetrics":
        """Deserialize from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            segment_id=data["segment_id"],
            audio_duration_ms=data["audio_duration_ms"],
            trust_level=TrustLevel[data["trust_level"]],
            trust_score=data["trust_score"],
            trust_confidence=data["trust_confidence"],
            indicators=TrustIndicators(**data["indicators"]),
            prosody=ProsodyFeatures(**{k: v for k, v in data["prosody"].items()}),
            embedding=np.array(data["embedding"], dtype=np.float32),
            model_version=data["model_version"],
            processing_time_ms=data["processing_time_ms"],
        )

    def to_fusion_output(self) -> "FusionReadyOutput":
        """Convert to lightweight fusion-ready format."""
        return FusionReadyOutput(
            modality="tone",
            timestamp_ms=int(self.timestamp.timestamp() * 1000),
            trust_score=self.trust_score,
            confidence=self.trust_confidence,
            embedding=self.embedding.copy(),
            indicators={
                "confidence": self.indicators.confidence,
                "hesitation": self.indicators.hesitation,
                "stress": self.indicators.stress,
                "engagement": self.indicators.engagement,
                "valence": self.indicators.emotional_valence,
                "arousal": self.indicators.emotional_arousal,
            },
        )


@dataclass
class FusionReadyOutput:
    """Lightweight real-time output. All modality modules should conform to this interface."""

    modality: str = "tone"
    timestamp_ms: int = 0
    trust_score: float = 0.5
    confidence: float = 0.5  # Model confidence, not speaker confidence
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))

    # Key indicators for quick fusion (subset of full indicators)
    indicators: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modality": self.modality,
            "timestamp_ms": self.timestamp_ms,
            "trust_score": float(self.trust_score),
            "confidence": float(self.confidence),
            "embedding": self.embedding.tolist(),
            "indicators": {k: float(v) for k, v in self.indicators.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionReadyOutput":
        """Create from dictionary."""
        return cls(
            modality=data["modality"],
            timestamp_ms=data["timestamp_ms"],
            trust_score=data["trust_score"],
            confidence=data["confidence"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            indicators=data["indicators"],
        )

    def __repr__(self) -> str:
        return (
            f"FusionReadyOutput(modality='{self.modality}', "
            f"trust_score={self.trust_score:.3f}, "
            f"confidence={self.confidence:.3f})"
        )
