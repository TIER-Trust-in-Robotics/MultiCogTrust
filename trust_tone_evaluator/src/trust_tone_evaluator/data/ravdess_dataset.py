"""RAVDESS dataset loader with trust label adaptation."""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from trust_tone_evaluator.data.trust_adapter import EmotionToTrustMapper, TrustLabels

logger = logging.getLogger(__name__)


class RAVDESSDataset(Dataset):
    """
    RAVDESS dataset with trust label adaptation. Features precomputed on init.

    Filename: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Emotions: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    """

    # Emotion code to label mapping
    EMOTION_MAP = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    # Emotion name to index mapping for emotion-first classification
    EMOTION_TO_IDX = {
        "neutral": 0,
        "calm": 1,
        "happy": 2,
        "sad": 3,
        "angry": 4,
        "fearful": 5,
        "disgust": 6,
        "surprised": 7,
    }

    # Emotion index to trust level index mapping
    # neutral(0)->3(High), calm(1)->4(VeryHigh), happy(2)->3(High),
    # sad(3)->1(Low), angry(4)->1(Low), fearful(5)->0(VeryLow),
    # disgust(6)->0(VeryLow), surprised(7)->2(Moderate)
    EMOTION_TO_TRUST_IDX = {
        0: 3,  # neutral -> High (index 3)
        1: 4,  # calm -> Very High (index 4)
        2: 3,  # happy -> High (index 3)
        3: 1,  # sad -> Low (index 1)
        4: 1,  # angry -> Low (index 1)
        5: 0,  # fearful -> Very Low (index 0)
        6: 0,  # disgust -> Very Low (index 0)
        7: 2,  # surprised -> Moderate (index 2)
    }

    # Intensity code to value
    INTENSITY_MAP = {
        "01": "normal",
        "02": "strong",
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        emotions: Optional[List[str]] = None,
        sample_rate: int = 16000,
        max_duration: float = 5.0,
        feature_extractor: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        include_song: bool = False,
        test_actors: Optional[List[int]] = None,
        max_feature_frames: int = 500,
        augment: bool = False,
        audio_augment: bool = False,
        num_audio_augments: int = 5,
    ):
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required for RAVDESS dataset")

        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.include_song = include_song
        self.max_feature_frames = max_feature_frames
        self.augment = augment
        self.audio_augment = audio_augment
        self.num_audio_augments = num_audio_augments

        # Filter emotions
        if emotions is None:
            self.emotions = list(self.EMOTION_MAP.values())
        else:
            self.emotions = [e.lower() for e in emotions]

        # Initialize trust mapper
        self.trust_mapper = EmotionToTrustMapper("ravdess")

        # Define train/val/test splits by actor ID
        # Default: actors 1-16 for train, 17-18 for val, 19-24 for test
        if test_actors is None:
            test_actors = [19, 20, 21, 22, 23, 24]

        val_actors = [17, 18]
        train_actors = [i for i in range(1, 25) if i not in test_actors and i not in val_actors]

        if split == "train":
            self.actor_ids = train_actors
        elif split == "val":
            self.actor_ids = val_actors
        elif split == "test":
            self.actor_ids = test_actors
        else:
            # 'all' - use all actors
            self.actor_ids = list(range(1, 25))

        # Load file list
        self.samples = self._load_samples()

        # Precompute all features and labels once
        self._precompute()

    def _augment_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """Create augmented audio copies (time stretch, pitch shift, noise)."""
        augmented = []

        for rate in [0.9, 1.1]:  # Time stretch
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            augmented.append(stretched)

        for n_steps in [-2, 2]:  # Pitch shift +/- 2 semitones
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            augmented.append(shifted)

        augmented.append(audio + np.random.randn(len(audio)) * 0.005)  # Gaussian noise

        return augmented[: self.num_audio_augments]

    def _extract_and_cache(self, audio: np.ndarray, sample: Dict, idx: int):
        """Extract features from audio and add to cache."""
        # Pad or truncate
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # Extract features
        if self.feature_extractor is not None:
            features = self.feature_extractor(audio)
            if features.ndim == 2:
                num_frames, feat_dim = features.shape
                if num_frames > self.max_feature_frames:
                    features = features[: self.max_feature_frames]
                elif num_frames < self.max_feature_frames:
                    pad_width = ((0, self.max_feature_frames - num_frames), (0, 0))
                    features = np.pad(features, pad_width)
        else:
            features = audio

        # Get trust labels
        trust_labels = self.trust_mapper.get_trust_labels(sample["emotion"])
        if sample["intensity"] == "strong":
            trust_labels = self._adjust_for_intensity(trust_labels)

        emotion_idx = self.EMOTION_TO_IDX.get(sample["emotion"], 0)
        self._cached_features.append(torch.tensor(features, dtype=torch.float32))
        self._cached_labels.append({
            "trust_level": torch.tensor(trust_labels.trust_level_index, dtype=torch.long),
            "trust_score": torch.tensor(trust_labels.trust_score, dtype=torch.float32),
            "indicators": torch.tensor(trust_labels.to_indicator_vector(), dtype=torch.float32),
            "emotion_idx": torch.tensor(emotion_idx, dtype=torch.long),
        })

    def _precompute(self):
        """Precompute features and labels for all samples (with optional audio augmentation)."""
        logger.info(f"Precomputing features for {len(self.samples)} samples ({self.split})...")
        self._cached_features = []
        self._cached_labels = []
        self._sample_indices = []  # Maps cached index back to original sample index

        for i, sample in enumerate(self.samples):
            # Load audio
            audio, sr = librosa.load(sample["path"], sr=self.sample_rate)

            # Original sample
            self._extract_and_cache(audio, sample, i)
            self._sample_indices.append(i)

            # Audio-level augmentation (training only)
            if self.audio_augment:
                augmented_audios = self._augment_audio(audio, sr)
                for aug_audio in augmented_audios:
                    self._extract_and_cache(aug_audio, sample, i)
                    self._sample_indices.append(i)

        # Stack features for normalization stats
        all_feats = torch.stack(self._cached_features)
        self._feat_mean = all_feats.mean(dim=0)
        self._feat_std = all_feats.std(dim=0).clamp(min=1e-6)

        logger.info(f"Precomputed {len(self._cached_features)} samples "
                     f"(from {len(self.samples)} originals), "
                     f"feature shape: {self._cached_features[0].shape}")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load and parse all audio file paths."""
        samples = []

        for actor_id in self.actor_ids:
            actor_dir = self.data_dir / f"Actor_{actor_id:02d}"

            if not actor_dir.exists():
                # Try alternative naming
                actor_dir = self.data_dir / f"Actor_{actor_id}"

            if not actor_dir.exists():
                continue

            for audio_file in actor_dir.glob("*.wav"):
                parsed = self._parse_filename(audio_file.name)

                if parsed is None:
                    continue

                # Filter by modality (speech vs song)
                if not self.include_song and parsed["modality"] == "song":
                    continue

                # Filter by emotion
                if parsed["emotion"] not in self.emotions:
                    continue

                samples.append(
                    {
                        "path": str(audio_file),
                        "actor_id": actor_id,
                        "gender": "male" if actor_id % 2 == 1 else "female",
                        **parsed,
                    }
                )

        return samples

    def _parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Parse RAVDESS filename into metadata dict, or None if invalid."""
        try:
            parts = filename.replace(".wav", "").split("-")
            if len(parts) != 7:
                return None

            modality_code, channel_code, emotion_code, intensity_code, statement, repetition, actor = parts

            # Only process audio files
            if modality_code not in ["01", "03"]:  # full-AV or audio-only
                return None

            emotion = self.EMOTION_MAP.get(emotion_code)
            if emotion is None:
                return None

            return {
                "modality": "speech" if channel_code == "01" else "song",
                "emotion": emotion,
                "intensity": self.INTENSITY_MAP.get(intensity_code, "normal"),
                "statement": int(statement),
                "repetition": int(repetition),
            }

        except (ValueError, IndexError):
            return None

    def __len__(self) -> int:
        return len(self._cached_features)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from precomputed cache."""
        features = self._cached_features[idx].clone()
        labels = self._cached_labels[idx]

        # Normalize features using training set statistics
        features = (features - self._feat_mean) / self._feat_std

        # Feature-level augmentation during training
        if self.augment:
            # Gaussian noise (scale relative to normalized features)
            noise = torch.randn_like(features) * 0.05
            features = features + noise

            # Random feature dropout (zero out 5% of features)
            mask = torch.rand_like(features) > 0.05
            features = features * mask

        # Map back to original sample for metadata
        orig_idx = self._sample_indices[idx]

        return {
            "features": features,
            "trust_level": labels["trust_level"],
            "trust_score": labels["trust_score"],
            "indicators": labels["indicators"],
            "emotion_idx": labels["emotion_idx"],
            "emotion": self.samples[orig_idx]["emotion"],
            "metadata": {
                "actor_id": self.samples[orig_idx]["actor_id"],
                "gender": self.samples[orig_idx]["gender"],
                "intensity": self.samples[orig_idx]["intensity"],
                "path": self.samples[orig_idx]["path"],
            },
        }

    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set normalization statistics from training set (for val/test)."""
        self._feat_mean = mean
        self._feat_std = std

    def _adjust_for_intensity(self, labels: TrustLabels) -> TrustLabels:
        """
        Adjust trust labels for strong intensity expressions.

        Strong intensity generally means more pronounced indicators.
        """
        # Increase stress and arousal for strong intensity
        adjusted_stress = min(labels.stress + 0.1, 1.0)
        adjusted_arousal = min(labels.arousal + 0.15, 1.0)

        # More extreme valence
        if labels.valence >= 0:
            adjusted_valence = min(labels.valence + 0.1, 1.0)
        else:
            adjusted_valence = max(labels.valence - 0.1, -1.0)

        return TrustLabels(
            trust_level=labels.trust_level,
            trust_score=labels.trust_score,
            confidence=labels.confidence,
            hesitation=labels.hesitation,
            stress=adjusted_stress,
            engagement=min(labels.engagement + 0.1, 1.0),
            sincerity=labels.sincerity,
            cognitive_load=labels.cognitive_load,
            valence=adjusted_valence,
            arousal=adjusted_arousal,
            dominance=labels.dominance,
        )

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for trust level imbalance."""
        level_counts = np.zeros(5)
        for sample in self.samples:
            labels = self.trust_mapper.get_trust_labels(sample["emotion"])
            level_counts[labels.trust_level_index] += 1

        # Inverse frequency weighting
        total = sum(level_counts)
        weights = total / (5 * level_counts + 1e-6)

        return torch.tensor(weights, dtype=torch.float32)

    def get_emotion_distribution(self) -> Dict[str, int]:
        """Get count of samples per emotion (including augmented)."""
        counts = {}
        for idx in self._sample_indices:
            emotion = self.samples[idx]["emotion"]
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts

    def get_trust_distribution(self) -> Dict[int, int]:
        """Get count of samples per trust level (including augmented)."""
        counts = {i: 0 for i in range(1, 6)}
        for idx in self._sample_indices:
            labels = self.trust_mapper.get_trust_labels(self.samples[idx]["emotion"])
            counts[labels.trust_level] += 1
        return counts

    def __repr__(self) -> str:
        return (
            f"RAVDESSDataset(split='{self.split}', "
            f"samples={len(self._cached_features)} "
            f"(from {len(self.samples)} originals), "
            f"actors={len(self.actor_ids)}, "
            f"emotions={self.emotions})"
        )


def create_ravdess_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    feature_extractor: Optional[Callable] = None,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, val, and test dataloaders for RAVDESS."""
    train_dataset = RAVDESSDataset(
        data_dir,
        split="train",
        feature_extractor=feature_extractor,
        augment=False,
        audio_augment=False,
        **dataset_kwargs,
    )

    val_dataset = RAVDESSDataset(
        data_dir,
        split="val",
        feature_extractor=feature_extractor,
        augment=False,
        **dataset_kwargs,
    )

    test_dataset = RAVDESSDataset(
        data_dir,
        split="test",
        feature_extractor=feature_extractor,
        augment=False,
        **dataset_kwargs,
    )

    # Use training set normalization stats for val/test
    val_dataset.set_normalization_stats(train_dataset._feat_mean, train_dataset._feat_std)
    test_dataset.set_normalization_stats(train_dataset._feat_mean, train_dataset._feat_std)

    # num_workers=0 since features are precomputed in memory (no I/O needed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
