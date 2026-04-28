from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.core.events import SpeechSegment, TranscriptEvent
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class WhisperConfig:
    model_name: str = "tiny.en"
    language: str = "en"
    device: str = "cpu"  # test with cuda on Jetson!
    compute_type: str = "int8"
    queue_get_timeout_sec: int = 0.5
    queue_put_timeout_sec: int = 0.5
    beam_size: int = 1  # the number of candidate decoding


async def _put_until_stopped(
    output_queue: asyncio.Queue[str],
    transcription: TranscriptEvent,
    stop_event: asyncio.Event,
    timeout_sec: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(output_queue.put(transcription), timeout=timeout_sec)
            return
        except TimeoutError:
            continue


async def run_transcribe_worker(
    input_queue: asyncio.Queue[SpeechSegment],
    output_queue: asyncio.Queue[TranscriptEvent],
    stop_event: asyncio.Event,
    config: WhisperConfig,
) -> None:
    transcriber = WhisperModel(
        config.model_name,
        device=config.device,
        compute_type=config.compute_type,
        beam_size=config.beam_size,
    )
    while not stop_event.is_set():
        try:
            speech = await asyncio.wait_for(
                input_queue.get(),
                timeout=config.queue_get_timeout_sec,
            )
        except TimeoutError:
            continue

        segments, _ = transcriber.transcribe(
            speech.samples,
            language=config.language,
            beam_size=5,
            vad_filter=False,
        )  # look into beam size and what it effects

        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            continue

        event = TranscriptEvent(
            speech.start_time,
            speech.end_time,
            text=text,
        )

        await _put_until_stopped(
            output_queue,
            event,
            stop_event,
            config.queue_put_timeout_sec,
        )
