"""Streaming contracts: causal windows, bounded work, final speech and shutdown."""

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.events import AudioChunk, SpeechSegment
from src.segmentAudioTorch import SileroVAD
from src.stream_runtime import ReplaySource, run_stream
from src.visual_stream import VisualBuffer, VisualFrame


def frame(timestamp, image=True, blendshape=True):
    return VisualFrame(
        timestamp, np.ones(52), np.ones(576), blendshape, image, "test", 1.0
    )


def test_visual_window_excludes_future_stale_and_duplicate_slot_frames():
    buffer = VisualBuffer(50)
    for timestamp in (9.9, 10.02, 10.05, 10.22, 15.99, 16.0, 17.0):
        buffer.append(frame(timestamp, blendshape=timestamp != 10.22))
    row, evidence = buffer.select(10.0, 18.0)
    assert evidence["source_timestamps"] == [10.02, 10.22, 15.99]
    assert evidence["window_end"] == 16.0
    assert evidence["requested_frames"] == 30
    assert evidence["image_frames"] == 3 and evidence["blendshape_frames"] == 2
    assert row["image_mask"].sum() == 3
    assert not row["blendshape_mask"][1] and row["image_mask"][1]
    with pytest.raises(ValueError, match="increase"):
        buffer.append(frame(17.0))
    limited = VisualBuffer(2)
    for timestamp in (1, 2, 3):
        limited.append(frame(timestamp))
    assert len(limited.frames) == 2
    missing, evidence = limited.select(0, 1.5)
    assert not missing["visual_mask"].any()
    assert evidence["observed_frames"] == 0


def test_vad_caps_continuous_speech_flushes_tail_and_discards_gaps(monkeypatch):
    resets = []
    monkeypatch.setattr(
        "src.segmentAudioTorch.load_silero_vad",
        lambda: SimpleNamespace(reset_states=lambda: resets.append(True)),
    )
    vad = SileroVAD(min_speech_ms=32, max_segment_sec=0.096)
    monkeypatch.setattr(vad, "_speach_prob", lambda samples: 1.0)
    chunks = [AudioChunk(i * 0.032, 16000, np.ones(512, np.float32)) for i in range(5)]
    segments = [vad.process_chunk(c)[1] for c in chunks]
    assert segments[2].end_reason == "max_duration"
    assert segments[2].speech_end_time == pytest.approx(0.096)
    final = vad.flush()
    assert final.end_reason == "end_of_stream"
    assert final.start_time == pytest.approx(0.096)
    assert len(final.samples) == 1024 and vad.flush() is None
    vad.process_chunk(chunks[0])
    vad.reset()
    assert resets == [True] and vad.flush() is None


class FinalOnlyVAD:
    def __init__(self):
        self.chunks = []
        self.resets = 0

    def process_chunk(self, chunk):
        self.chunks.append(chunk)
        return 1.0, None

    def flush(self):
        if not self.chunks:
            return None
        chunks, self.chunks = self.chunks, []
        return SpeechSegment(
            chunks[0].timestamp,
            chunks[-1].end_time,
            16000,
            np.concatenate([c.samples for c in chunks]),
            chunks[-1].end_time,
            "end_of_stream",
        )

    def reset(self):
        self.chunks = []
        self.resets += 1


class Processor:
    def __init__(self, delay=0):
        self.delay = delay

    def process(self, segment):
        time.sleep(self.delay)
        return {
            "text": "test speech",
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "emotion": None,
            "timings_ms": {},
        }


def test_paced_replay_drains_final_prediction_after_input_ends():
    events = []
    report = asyncio.run(
        run_stream(
            Processor(0.05),
            events.append,
            replay=ReplaySource(np.ones(1536, np.float32), 16000, None),
            vad=FinalOnlyVAD(),
        )
    )
    assert report["audio_chunks"] == 3 and report["predictions"] == 1
    assert events[-2]["segment_end_reason"] == "end_of_stream"
    assert events[-2]["start_sec"] == pytest.approx(0)
    assert events[-2]["end_sec"] == pytest.approx(0.096)
    assert events[-2]["timings_ms"]["speech_end_to_prediction"] >= 40
    assert events[-1]["event"] == "stream_summary"


def test_slow_consumer_bounds_queue_and_discards_stale_work():
    class EveryChunkVAD(FinalOnlyVAD):
        def process_chunk(self, chunk):
            self.chunks.append(chunk)
            return 1.0, self.flush()

    events = []
    report = asyncio.run(
        run_stream(
            Processor(0.15),
            events.append,
            replay=ReplaySource(np.ones(512 * 20, np.float32), 16000, None),
            vad=EveryChunkVAD(),
            queue_delay_limit=0.1,
        )
    )
    assert report["segment_queue_high_watermark"] <= 4
    assert report["stale_segments"] > 0
    assert (
        report["predictions"] + report["stale_segments"] + report["dropped_segments"]
        == 20
    )
    assert report["dropped_audio_chunks"] == 0


def test_capture_gap_discards_partial_speech_and_stop_ends_deadline(monkeypatch):
    detector, events = FinalOnlyVAD(), []

    async def source(queue, stop, config, *, metrics):
        now = time.monotonic()
        for offset in (0, 0.032, 0.5, 0.532):
            await queue.put(AudioChunk(now + offset, 16000, np.ones(512, np.float32)))
        stop.set()

    monkeypatch.setattr("src.stream_runtime.run_audio_capture", source)
    before = time.monotonic()
    report = asyncio.run(
        run_stream(Processor(), events.append, duration=30, vad=detector)
    )
    assert time.monotonic() - before < 2
    assert report["audio_discontinuities"] == 1 and detector.resets == 1
    prediction = next(e for e in events if e["event"] == "prediction")
    assert prediction["end_time"] - prediction["start_time"] == pytest.approx(0.064)


def test_video_failure_is_reported_without_losing_speech(monkeypatch):
    def broken_video(*args):
        raise RuntimeError("missing camera or source")

    monkeypatch.setattr("src.stream_runtime.video_samples", broken_video)
    events = []
    report = asyncio.run(
        run_stream(
            Processor(),
            events.append,
            extractor=object(),
            vad=FinalOnlyVAD(),
            replay=ReplaySource(np.ones(1024, np.float32), 16000, "missing.avi"),
        )
    )
    assert report["predictions"] == 1 and report["video_frames"] == 0
    assert report["camera_error"] == "missing camera or source"
    assert events[0]["event"] == "vision_unavailable"


def test_portaudio_clock_and_overload_are_observable(monkeypatch):
    from src.audio.capture import run_audio_capture

    closed = []

    class Stream:
        def __init__(self, callback, stop):
            self.callback, self.stop = callback, stop

        def start_stream(self):
            for i in range(40):
                self.callback(
                    np.zeros(512, np.float32).tobytes(),
                    512,
                    {
                        "current_time": 100 + i * 0.032,
                        "input_buffer_adc_time": 99.99 + i * 0.032,
                    },
                    int(i == 39),
                )
            asyncio.get_running_loop().call_later(0.03, self.stop.set)

        def is_active(self):
            return True

        def get_input_latency(self):
            return 0.01

        def stop_stream(self):
            closed.append("stopped")

        def close(self):
            closed.append("closed")

    async def run():
        queue, stop, metrics = asyncio.Queue(32), asyncio.Event(), {}
        audio = SimpleNamespace(
            open=lambda **kw: Stream(kw["stream_callback"], stop),
            terminate=lambda: closed.append("terminated"),
        )
        monkeypatch.setattr("src.audio.capture.pyaudio.PyAudio", lambda: audio)
        await run_audio_capture(queue, stop, metrics=metrics)
        chunks = [queue.get_nowait() for _ in range(queue.qsize())]
        assert len(chunks) == 32 and metrics["dropped_audio_chunks"] == 8
        np.testing.assert_allclose(
            np.diff([c.timestamp for c in chunks]), 0.032, atol=1e-7
        )
        assert chunks[-1].discontinuity and metrics["audio_hardware_status_events"] == 1
        assert closed == ["stopped", "closed", "terminated"]

    asyncio.run(run())
