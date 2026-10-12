"""Bounded microphone capture with PortAudio ADC times mapped to monotonic time."""

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


def _close_audio(stream, audio):
    if stream is not None:
        try:
            if stream.is_active():
                stream.stop_stream()
        finally:
            stream.close()
    if audio is not None:
        audio.terminate()


def put_latest(queue, item, metrics, counter):
    """Never let slow downstream work grow a capture queue indefinitely."""
    if queue.full():
        queue.get_nowait()
        metrics[counter] = metrics.get(counter, 0) + 1
    queue.put_nowait(item)


async def run_audio_capture(output_queue, stop_event, config=None, *, metrics=None):
    config = config or AudioCaptureConfig()
    metrics = metrics if metrics is not None else {}
    if config.channels != 1:
        raise ValueError("The live emotion pipeline requires mono audio")
    loop = asyncio.get_running_loop()
    audio, stream = None, None
    clock_offset = None

    def deliver(chunk):
        if not stop_event.is_set():
            put_latest(output_queue, chunk, metrics, "dropped_audio_chunks")
            metrics["audio_queue_high_watermark"] = max(
                metrics.get("audio_queue_high_watermark", 0), output_queue.qsize()
            )

    def callback(raw, frame_count, time_info, status_flags):
        nonlocal clock_offset
        now = time.monotonic()
        if clock_offset is None:
            clock_offset = now - time_info["current_time"]
            metrics["audio_clock"] = (
                "PortAudio input_buffer_adc_time plus fixed monotonic offset"
            )
        if stop_event.is_set():
            # One owner stops the stream. Returning paComplete here while the
            # main worker also calls stop_stream can race CoreAudio shutdown.
            return None, pyaudio.paContinue
        timestamp = time_info["input_buffer_adc_time"] + clock_offset
        chunk = AudioChunk(
            timestamp,
            config.sample_rate,
            np.frombuffer(raw, np.float32).copy(),
            bool(status_flags),
        )
        if status_flags:
            metrics["audio_hardware_status_events"] = (
                metrics.get("audio_hardware_status_events", 0) + 1
            )
        loop.call_soon_threadsafe(deliver, chunk)
        return None, pyaudio.paContinue

    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=config.sample_rate,
            input=True,
            input_device_index=config.input_device_index,
            frames_per_buffer=config.frames_per_buffer,
            stream_callback=callback,
            start=False,
        )
        metrics["audio_input_latency_sec"] = stream.get_input_latency()
        stream.start_stream()
        while not stop_event.is_set():
            if not stream.is_active():
                raise RuntimeError("Microphone stream stopped unexpectedly")
            await asyncio.sleep(0.05)
    finally:
        await asyncio.to_thread(_close_audio, stream, audio)
