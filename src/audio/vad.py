"""Async voice detection worker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src import config as audio_config
from src.core.events import AudioChunk, SpeechSegment
from src.segmentAudioTorch import SileroVAD


@dataclass(frozen=True)
class VADConfig:
    sample_rate: int = audio_config.SAMPLE_RATE
    pre_buffer_size: int = audio_config.VAD_PRE_BUFFER_SIZE
    chunk_ms: float = audio_config.VAD_CHUNK_MS
    threshold: float = audio_config.VAD_THRESHOLD
    min_speech_ms: int = audio_config.VAD_MIN_SPEECH_MS
    min_silence_ms: int = audio_config.VAD_MIN_SILENCE_MS
    queue_get_timeout_sec: float = 0.5
    queue_put_timeout_sec: float = 0.25


async def _put_until_stopped(
    output_queue: asyncio.Queue[SpeechSegment],
    stop_event: asyncio.Event,
    segment: SpeechSegment,
    timeout_sec: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(output_queue.put(segment), timeout=timeout_sec)
            return
        except TimeoutError:
            continue


async def run_vad_worker(
    input_queue: asyncio.Queue[AudioChunk],
    output_queue: asyncio.Queue[SpeechSegment],
    stop_event: asyncio.Event,
    config: VADConfig | None = None,
    vad: SileroVAD | None = None,
) -> None:
    config = config or VADConfig()
    if vad is None:
        vad = SileroVAD(
            sample_rate=config.sample_rate,
            pre_buffer_size=config.pre_buffer_size,
            chunk_ms=config.chunk_ms,
            threshold=config.threshold,
            min_speech_ms=config.min_speech_ms,
            min_silence_ms=config.min_silence_ms,
        )

    while not stop_event.is_set():
        try:
            chunk = await asyncio.wait_for(
                input_queue.get(),
                timeout=config.queue_get_timeout_sec,
            )
        except TimeoutError:
            continue

        prob, segment = vad.process_chunk(chunk)

        if segment is not None:
            await _put_until_stopped(
                output_queue,
                stop_event,
                segment,
                config.queue_put_timeout_sec,
            )
