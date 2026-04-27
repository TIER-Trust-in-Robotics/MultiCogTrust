"""Shared event objects passed between async pipeline workers."""

from dataclasses import dataclass
import numpy as np


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
