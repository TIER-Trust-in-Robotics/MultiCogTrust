"""Shared event objects passed between async pipeline workers."""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class AudioChunk:
    """
    A raw microphone chunk.
    Timestamp is the monotonic-clock time, in seconds, at the start of the chunk.
    """

    timestamp: float
    sample_rate: int
    samples: np.ndarray

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def end_time(self) -> float:
        return self.timestamp + self.duration


@dataclass(frozen=True)
class SpeechSegment:
    start_time: float
    end_time: float
    sample_rate: int
    samples: np.ndarray


@dataclass(frozen=True)
class TranscriptEvent:
    start_time: float
    end_time: float
    text: str


@dataclass(frozen=True)
class TextEmbeddingEvent:
    start_time: float
    end_time: float
    text: str
    tokens: list[str]
    words: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_state: torch.Tensor
    word_hidden_states: torch.Tensor
    token_word_indices: list[int | None]
    word_token_indices: list[list[int]]


@dataclass(frozen=True)
class ProsodyFeatureEvent:
    start_time: float
    end_time: float
    utterance_id: str
    words: list[str]
    word_features: np.ndarray
    word_feature_mask: np.ndarray


@dataclass(frozen=True)
class FusionPredictionEvent:
    start_time: float
    end_time: float
    utterance_id: str | None
    text: str
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_label: int
