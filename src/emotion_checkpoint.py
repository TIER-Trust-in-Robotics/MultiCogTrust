"""Versioned inference contract shared by training and the live/file predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EMOTION_LABELS = ["anger", "happiness", "sadness", "neutral"]
EMOTION_LABEL_MAP = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    sample_rate: int = 16000
    max_audio_sec: float = 6.0
    min_samples: int = 1024
    max_tokens: int = 297
    waveform_normalization: str = "pcm_float"

    def __post_init__(self):
        if self.sample_rate != 16000 or self.waveform_normalization != "pcm_float":
            raise ValueError("This model requires 16 kHz PCM-scaled float audio")
        if (
            self.min_samples < 400
            or self.max_audio_sec * self.sample_rate < self.min_samples
        ):
            raise ValueError("Invalid minimum/maximum audio length")
        if self.max_tokens < 2:
            raise ValueError("max_tokens must allow special tokens")


def prepare_audio(audio, sample_rate: int, config: PreprocessingConfig) -> torch.Tensor:
    """Mono conversion, resampling, head truncation and short-clip padding."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    array = np.asarray(audio)
    if array.dtype.kind != "f":
        raise ValueError("Audio must be PCM-scaled floating point, not integer PCM")
    if array.ndim == 2:
        array = array.mean(axis=1)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Audio must be a nonempty finite waveform")
    waveform = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    if sample_rate != config.sample_rate:
        from torchaudio.functional import resample

        waveform = resample(waveform, sample_rate, config.sample_rate)
    waveform = waveform[: int(config.max_audio_sec * config.sample_rate)]
    if waveform.numel() < config.min_samples:
        waveform = F.pad(waveform, (0, config.min_samples - waveform.numel()))
    return waveform


def save_checkpoint(
    path,
    model,
    *,
    preprocessing,
    provenance,
    include_encoders=True,
    label_names=None,
    label_map=None,
    task="iemocap_emotion_4",
):
    labels = EMOTION_LABELS if label_names is None else label_names
    mapping = EMOTION_LABEL_MAP if label_map is None else label_map
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if include_encoders or not key.startswith(("text_encoder.", "audio_encoder."))
    }
    if not include_encoders:
        if not (model.freeze_text_encoder and model.freeze_audio_encoder):
            raise ValueError("Cannot omit encoders that were fine-tuned")
        if not all(
            model.model_config.get(k) for k in ("text_revision", "audio_revision")
        ):
            raise ValueError("Head-only checkpoints require pinned encoder revisions")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "label_names": list(labels),
        "label_map": dict(mapping),
        "model_config": dict(model.model_config),
        "preprocessing": asdict(preprocessing),
        "weights_scope": "full" if include_encoders else "fusion_only",
        "model_state_dict": state,
        "provenance": provenance,
    }
    validate_checkpoint(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_checkpoint(checkpoint, expected_task=None):
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        if checkpoint.get("num_labels") == 7 and "trainable_state_dict" in checkpoint:
            raise ValueError(
                "Legacy MOSEI sentiment checkpoint cannot be used for four-class IEMOCAP emotion"
            )
        raise ValueError(
            "Unsupported legacy checkpoint; export with emotion_checkpoint.save_checkpoint"
        )
    required = {
        "task",
        "label_names",
        "label_map",
        "model_config",
        "preprocessing",
        "weights_scope",
        "model_state_dict",
        "provenance",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint missing fields: {sorted(missing)}")
    if expected_task and checkpoint["task"] != expected_task:
        raise ValueError(
            f"Checkpoint task {checkpoint['task']!r} does not match {expected_task!r}"
        )
    labels = checkpoint["label_names"]
    if len(labels) != checkpoint["model_config"]["num_labels"] or len(
        set(labels)
    ) != len(labels):
        raise ValueError("Checkpoint label count/order is invalid")
    if checkpoint["task"] == "iemocap_emotion_4":
        if labels != EMOTION_LABELS or checkpoint["label_map"] != EMOTION_LABEL_MAP:
            raise ValueError("Checkpoint must preserve the four-class IEMOCAP mapping")
    if checkpoint["weights_scope"] not in {"full", "fusion_only"}:
        raise ValueError("Unknown checkpoint weights scope")
    if checkpoint["weights_scope"] == "fusion_only":
        if not all(
            checkpoint["model_config"].get(k)
            for k in ("text_revision", "audio_revision")
        ):
            raise ValueError("Head-only checkpoint is missing encoder revisions")
    PreprocessingConfig(**checkpoint["preprocessing"])


def read_checkpoint(path, expected_task=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, expected_task)
    return checkpoint


def load_checkpoint_weights(model, checkpoint):
    if checkpoint["weights_scope"] == "full":
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    expected_missing = {
        k
        for k in model.state_dict()
        if k.startswith(("text_encoder.", "audio_encoder."))
    }
    if (
        set(incompatible.missing_keys) != expected_missing
        or incompatible.unexpected_keys
    ):
        raise ValueError(
            "Fusion checkpoint does not exactly match the model architecture"
        )
