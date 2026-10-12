"""One utterance-processing path for both live capture and recorded replay."""

from __future__ import annotations

from dataclasses import asdict
import time

from src.core.events import SpeechSegment
from src.emotion_checkpoint import PreprocessingConfig, prepare_audio


def transcribe_samples(model, samples, *, language="en", beam_size=1):
    # faster-whisper returns a generator: consumption belongs in the worker too.
    segments, _ = model.transcribe(
        samples, language=language, beam_size=beam_size, vad_filter=False
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


class SpeechProcessor:
    def __init__(self, transcriber=None, predictor=None):
        self.transcriber = transcriber
        self.predictor = predictor
        self.preprocessing = (
            predictor.preprocessing if predictor else PreprocessingConfig()
        )

    def process(self, segment: SpeechSegment, transcript: str | None = None):
        started = time.perf_counter()
        audio = prepare_audio(
            segment.samples, segment.sample_rate, self.preprocessing
        ).numpy()
        prepared = time.perf_counter()
        if transcript is None:
            if self.transcriber is None:
                raise ValueError("No ASR model or supplied transcript")
            text = transcribe_samples(self.transcriber, audio)
        else:
            text = transcript.strip()
        transcribed = time.perf_counter()
        prediction = (
            self.predictor.predict(text, audio, self.preprocessing.sample_rate)
            if text and self.predictor
            else None
        )
        finished = time.perf_counter()
        return {
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "input_duration_sec": len(segment.samples) / segment.sample_rate,
            "processed_audio_sec": len(audio) / self.preprocessing.sample_rate,
            "audio_truncated": len(segment.samples) / segment.sample_rate
            > self.preprocessing.max_audio_sec,
            "text": text,
            "transcript_source": "asr" if transcript is None else "provided",
            "status": "ok" if text else "empty_transcript",
            "emotion": asdict(prediction) if prediction else None,
            "model_provenance": {
                key: self.predictor.provenance[key]
                for key in (
                    "kind",
                    "epoch",
                    "train_count",
                    "validation_count",
                    "test_session",
                    "metadata_sha256",
                )
                if key in self.predictor.provenance
            }
            if self.predictor
            else None,
            "timings_ms": {
                "preprocessing": 1000 * (prepared - started),
                "asr": 1000 * (transcribed - prepared),
                "emotion": 1000 * (finished - transcribed),
                "total": 1000 * (finished - started),
            },
        }
