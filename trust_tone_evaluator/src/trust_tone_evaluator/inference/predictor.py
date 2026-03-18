"""Inference predictor for trust tone evaluation."""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from trust_tone_evaluator.features.opensmile_extractor import OpenSMILEExtractor
from trust_tone_evaluator.models.mlp_model import MLPTrustModel
from trust_tone_evaluator.output.trust_metrics import (
    TrustIndicators,
    TrustLevel,
    TrustMetrics,
)

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class TrustPredictor:
    """Loads a checkpoint and runs inference. Self-contained — no external config needed."""

    def __init__(
        self,
        model: MLPTrustModel,
        feature_extractor: OpenSMILEExtractor,
        feat_mean: torch.Tensor,
        feat_std: torch.Tensor,
        device: str = "cpu",
        sample_rate: int = 16000,
        max_duration: float = 5.0,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.feature_extractor = feature_extractor
        self.feat_mean = feat_mean.to(device)
        self.feat_std = feat_std.to(device)
        self.device = device
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        feature_set: str = "egemaps",
    ) -> "TrustPredictor":
        """Load a TrustPredictor from a .pt checkpoint file."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Reconstruct model from saved config
        model_config = checkpoint.get("model_config", {})
        input_dim = model_config.pop("input_dim", 88)
        model = MLPTrustModel(input_dim=input_dim, **model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Reconstruct feature extractor
        feature_extractor = OpenSMILEExtractor(
            feature_set=feature_set, feature_level="functionals"
        )

        # Load normalization stats
        norm_stats = checkpoint.get("norm_stats", {})
        feat_mean = norm_stats.get("feat_mean", torch.zeros(input_dim))
        feat_std = norm_stats.get("feat_std", torch.ones(input_dim))

        return cls(
            model=model,
            feature_extractor=feature_extractor,
            feat_mean=feat_mean,
            feat_std=feat_std,
            device=device,
        )

    def predict_file(self, audio_path: str, segment_id: str = "") -> TrustMetrics:
        """Run inference on an audio file. Returns TrustMetrics."""
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required: pip install librosa")

        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration_ms = int(len(audio) / self.sample_rate * 1000)

        result = self.predict_audio(audio)
        result.segment_id = segment_id or Path(audio_path).stem
        result.audio_duration_ms = duration_ms
        return result

    def predict_audio(self, audio: np.ndarray) -> TrustMetrics:
        """Run inference on a mono audio array. Returns TrustMetrics."""
        start = time.perf_counter()

        # Pad or truncate
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # Extract features
        features = self.feature_extractor.extract(audio)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Normalize using training stats
        features = (features - self.feat_mean) / self.feat_std

        # Run model
        with torch.no_grad():
            outputs = self.model(features.unsqueeze(0))

        # Parse outputs
        logits = outputs["trust_logits"][0]
        probs = torch.softmax(logits, dim=-1)
        pred_level_idx = probs.argmax().item()
        confidence = probs[pred_level_idx].item()

        trust_score = outputs["trust_score"][0].item()
        indicators_vec = outputs["indicators"][0].cpu().numpy()
        embedding = outputs["embedding"][0].cpu().numpy()

        elapsed_ms = (time.perf_counter() - start) * 1000

        return TrustMetrics(
            timestamp=datetime.now(),
            trust_level=TrustLevel(pred_level_idx + 1),
            trust_score=trust_score,
            trust_confidence=confidence,
            indicators=TrustIndicators.from_vector(indicators_vec),
            embedding=embedding,
            processing_time_ms=elapsed_ms,
        )

    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract the fusion embedding from audio. Fast path for multimodal pipelines."""
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        features = self.feature_extractor.extract(audio)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        features = (features - self.feat_mean) / self.feat_std

        with torch.no_grad():
            outputs = self.model(features.unsqueeze(0))

        return outputs["embedding"][0].cpu().numpy()
