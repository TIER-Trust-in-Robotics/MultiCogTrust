"""One utterance-processing path for both live capture and recorded replay."""

from __future__ import annotations

from dataclasses import asdict
import time

from src.core.events import SpeechSegment
from src.emotion_checkpoint import PreprocessingConfig, prepare_audio


def transcribe_samples(model, samples, *, language="en", beam_size=1, **options):
    # faster-whisper returns a generator: consumption belongs in the worker too.
    # Extra decoding options (e.g. FAST_ASR) are opt-in; the default matches the
    # Session 4 ASR evaluation protocol.
    segments, _ = model.transcribe(
        samples, language=language, beam_size=beam_size, vad_filter=False, **options
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


# No timestamp tokens and no temperature fallback: on Session 4, tiny.en median
# 196 -> 135 ms with 4 threads, worst case 1258 -> 213 ms, WER 0.333 -> 0.300,
# emotion macro F1 within 0.007 for both live models.
FAST_ASR = {"without_timestamps": True, "temperature": 0.0}


class SpeechProcessor:
    def __init__(self, transcriber=None, predictor=None, asr_options=None):
        self.transcriber = transcriber
        self.predictor = predictor
        self.asr_options = dict(asr_options or {})
        self.preprocessing = (
            predictor.preprocessing if predictor else PreprocessingConfig()
        )
        # Checkpoints trained on "previous line [SEP] current line" read the
        # speaker's last transcript as context; one speaker per processor.
        self.text_context = getattr(predictor, "provenance", {}).get("text_context")
        self.previous_text = None

    def model_text(self, text):
        if self.text_context != "previous_same_speaker_line":
            return text
        context = f"{self.previous_text} [SEP] {text}" if self.previous_text else text
        self.previous_text = text
        return context

    def process(self, segment: SpeechSegment, transcript: str | None = None):
        started = time.perf_counter()
        audio = prepare_audio(
            segment.samples, segment.sample_rate, self.preprocessing
        ).numpy()
        prepared = time.perf_counter()
        if transcript is None:
            if self.transcriber is None:
                raise ValueError("No ASR model or supplied transcript")
            text = transcribe_samples(self.transcriber, audio, **self.asr_options)
        else:
            text = transcript.strip()
        transcribed = time.perf_counter()
        previous_text = self.previous_text
        prediction = (
            self.predictor.predict(
                self.model_text(text), audio, self.preprocessing.sample_rate
            )
            if text and self.predictor
            else None
        )
        finished = time.perf_counter()
        result = {
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
                    "text_context",
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
        if self.text_context:
            result["context_text"] = previous_text if prediction else None
        return result
