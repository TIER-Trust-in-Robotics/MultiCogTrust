"""Compatibility import for the canonical transcription worker."""

from src.text.transcribe import WhisperConfig, run_transcribe_worker

__all__ = ["WhisperConfig", "run_transcribe_worker"]
