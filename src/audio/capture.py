"""Async microphone capture worker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

import numpy as np
import pyaudio

from src import config as audio_config
from src.core.events import AudioChunk


@dataclass(frozen=True)
class AudioCaptureConfig:
    sample_rate: int = audio_config.SAMPLE_RATE
    channels: int = audio_config.CHANNELS
    frames_per_buffer: int = audio_config.VAD_CHUNK_SAMPLES
    input_device_index: int | None = None
    exception_on_overflow: bool = False
    queue_put_timeout_sec: float = 0.25


def _close_audio(stream: pyaudio.Stream | None, audio: pyaudio.PyAudio | None) -> None:
    if stream is not None:
        try:
            if stream.is_active():
                stream.stop_stream()
        except OSError:
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    if audio is not None:
        try:
            audio.terminate()
        except OSError:
            pass


async def _put_until_stopped(
    output_queue: asyncio.Queue[AudioChunk],
    stop_event: asyncio.Event,
    chunk: AudioChunk,
    timeout_sec: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(output_queue.put(chunk), timeout=timeout_sec)
            return
        except TimeoutError:
            continue


async def run_audio_capture(
    output_queue: asyncio.Queue[AudioChunk],
    stop_event: asyncio.Event,
    config: AudioCaptureConfig | None = None,
) -> None:
    """Read microphone chunks and publish AudioChunk events.


    makes a treats for PyAudio's stream.read() and processes them into AudioChunks for SileroVAD.
    """

    config = config or AudioCaptureConfig()
    audio = None
    stream = None

    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=config.channels,
            rate=config.sample_rate,
            input=True,
            input_device_index=config.input_device_index,
            frames_per_buffer=config.frames_per_buffer,
        )

        while not stop_event.is_set():
            chunk_start = time.monotonic()
            try:
                raw = await asyncio.to_thread(
                    stream.read,
                    config.frames_per_buffer,
                    exception_on_overflow=config.exception_on_overflow,
                )
            except OSError:
                if stop_event.is_set():
                    break
                raise

            samples = np.frombuffer(raw, dtype=np.int16).copy()
            event = AudioChunk(
                timestamp=chunk_start,
                sample_rate=config.sample_rate,
                samples=samples,
            )
            await _put_until_stopped(
                output_queue,
                stop_event,
                event,
                config.queue_put_timeout_sec,
            )
    finally:
        _close_audio(stream, audio)
