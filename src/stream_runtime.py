"""Shared bounded audio/video orchestration for device capture and paced replay."""

from __future__ import annotations
import asyncio
from collections import deque
from dataclasses import dataclass
import sys
import time

import cv2
import numpy as np

from src.audio.capture import AudioCaptureConfig, put_latest, run_audio_capture
from src.core.events import AudioChunk
from src.segmentAudioTorch import SileroVAD
from src.visual_stream import VisualBuffer, video_samples


@dataclass
class ReplaySource:
    samples: np.ndarray
    sample_rate: int
    video_path: str | None
    video_start_sec: float = 0.0
    side: str = "full"


def distribution(values):
    return (
        {
            "count": len(values),
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "max_ms": float(max(values)),
        }
        if values
        else {"count": 0}
    )


def peak_rss_mib():
    """Peak resident memory of this process (Windows has no resource module)."""
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset / 1024**2
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)


async def run_stream(
    processor,
    emit,
    *,
    visual_predictor=None,
    extractor=None,
    replay=None,
    camera_index=None,
    microphone_index=None,
    duration=None,
    threshold=0.4,
    dashboard=None,
    queue_delay_limit=3.0,
    vad=None,
):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    detector = vad or SileroVAD(threshold=threshold, max_segment_sec=6.0)
    origin = time.monotonic()
    audio_queue, frame_queue, segments = (
        asyncio.Queue(32),
        asyncio.Queue(2),
        asyncio.Queue(4),
    )
    visual_buffer = VisualBuffer(450)
    if extractor is None:
        visual_buffer.closed = True
    metrics = {
        "clock": "monotonic_seconds",
        "origin": origin,
        "source": "paced_recording" if replay else "live_devices",
        "queue_capacities": {
            "audio": 32,
            "frames": 2,
            "segments": 4,
            "visual_history": 450,
        },
        "dropped_audio_chunks": 0,
        "dropped_video_frames": 0,
        "dropped_segments": 0,
        "audio_discontinuities": 0,
        "stale_segments": 0,
        "audio_chunks": 0,
        "video_frames": 0,
        "predictions": 0,
        "camera_error": None,
        "image_available_frames": 0,
        "blendshape_available_frames": 0,
        "speech_positive_chunks": 0,
        "peak_audio_rms": 0.0,
        "latency_sample_limit": 2000,
        "latency_scope": "most recent 2000 observations per stage",
        "response_ms": deque(maxlen=2000),
        "visual_ms": deque(maxlen=2000),
    }
    if dashboard:
        dashboard.set_stop_callback(lambda: loop.call_soon_threadsafe(stop.set))
        dashboard.update(status="running", source=metrics["source"], origin=origin)

    def publish(event):
        emit(event)
        if dashboard:
            dashboard.record(event)

    async def finish(queue):
        if not asyncio.current_task().cancelling():
            await queue.put(None)

    async def audio_source():
        try:
            if replay is None:
                await run_audio_capture(
                    audio_queue,
                    stop,
                    AudioCaptureConfig(input_device_index=microphone_index),
                    metrics=metrics,
                )
            else:
                if replay.sample_rate != 16000 or replay.samples.ndim != 1:
                    raise ValueError("Paced replay requires mono 16 kHz samples")
                limit = len(replay.samples)
                if duration is not None:
                    limit = min(limit, int(duration * 16000))
                for index in range(0, limit, 512):
                    if stop.is_set():
                        break
                    samples = replay.samples[index : min(index + 512, limit)].astype(
                        np.float32
                    )
                    if len(samples) < 512:
                        metrics["padded_final_audio_samples"] = 512 - len(samples)
                        samples = np.pad(samples, (0, 512 - len(samples)))
                    # Deliver after the complete chunk would have been captured.
                    await asyncio.sleep(
                        max(0, origin + (index + 512) / 16000 - time.monotonic())
                    )
                    put_latest(
                        audio_queue,
                        AudioChunk(origin + index / 16000, 16000, samples),
                        metrics,
                        "dropped_audio_chunks",
                    )
                    metrics["audio_queue_high_watermark"] = max(
                        metrics.get("audio_queue_high_watermark", 0),
                        audio_queue.qsize(),
                    )
        finally:
            await finish(audio_queue)

    def deliver_frame(item):
        if not stop.is_set():
            put_latest(frame_queue, item, metrics, "dropped_video_frames")
            metrics["frame_queue_high_watermark"] = max(
                metrics.get("frame_queue_high_watermark", 0), frame_queue.qsize()
            )

    def camera_pump():
        capture = cv2.VideoCapture(camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open camera {camera_index}")
            metrics["video_clock"] = (
                "monotonic at camera frame receipt; sensor delay not calibrated"
            )
            next_time = 0.0
            while not stop.is_set():
                ok, frame = capture.read()
                timestamp = time.monotonic()
                if not ok:
                    raise RuntimeError("Camera stopped delivering frames")
                if timestamp >= next_time:
                    loop.call_soon_threadsafe(deliver_frame, (timestamp, frame))
                    next_time = timestamp + 0.2
        finally:
            capture.release()

    async def video_source():
        try:
            if replay is not None and replay.video_path:
                seconds = len(replay.samples) / replay.sample_rate
                if duration is not None:
                    seconds = min(seconds, duration)
                frames = video_samples(
                    replay.video_path,
                    replay.side,
                    replay.video_start_sec,
                    replay.video_start_sec + seconds,
                )

                def next_frame():
                    return next(frames, None)

                try:
                    while not stop.is_set():
                        item = await asyncio.to_thread(next_frame)
                        if item is None:
                            break
                        timestamp, frame = item
                        timestamp = origin + timestamp - replay.video_start_sec
                        await asyncio.sleep(max(0, timestamp - time.monotonic()))
                        deliver_frame((timestamp, frame))
                finally:
                    frames.close()
            elif camera_index is not None:
                await asyncio.to_thread(camera_pump)
        except Exception as error:
            metrics["camera_error"] = str(error)
            publish(
                {
                    "event": "vision_unavailable",
                    "message": str(error),
                    "fallback": "audio_text",
                }
            )
        finally:
            await finish(frame_queue)

    async def vision_worker():
        try:
            while True:
                item = await frame_queue.get()
                if item is None:
                    break
                timestamp, frame = item
                event, summary, jpeg = await asyncio.to_thread(
                    extractor.process, timestamp, frame
                )
                visual_buffer.append(event)
                metrics["video_frames"] += 1
                metrics["image_available_frames"] += int(event.image_valid)
                metrics["blendshape_available_frames"] += int(event.blendshape_valid)
                metrics.setdefault("first_video_timestamp", timestamp)
                metrics["last_video_timestamp"] = timestamp
                metrics["visual_ms"].append(event.processing_ms)
                metrics["visual_history_high_watermark"] = max(
                    metrics.get("visual_history_high_watermark", 0),
                    len(visual_buffer.frames),
                )
                if dashboard:
                    dashboard.vision(summary, jpeg)
        finally:
            visual_buffer.closed = True

    def enqueue(segment):
        if segments.full():
            _, old = segments.get_nowait()
            metrics["dropped_segments"] += 1
            publish(
                {
                    "event": "dropped_segment",
                    "reason": "queue_full",
                    "start_time": old.start_time,
                    "end_time": old.end_time,
                }
            )
        segments.put_nowait((time.monotonic(), segment))
        metrics["segment_queue_high_watermark"] = max(
            metrics.get("segment_queue_high_watermark", 0), segments.qsize()
        )

    async def vad_worker():
        previous_end = None
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break
            if chunk.discontinuity or (
                previous_end is not None and abs(chunk.timestamp - previous_end) > 0.01
            ):
                detector.reset()
                metrics["audio_discontinuities"] += 1
                publish(
                    {
                        "event": "audio_discontinuity",
                        "timestamp": chunk.timestamp,
                        "action": "discard_partial_segment",
                    }
                )
            previous_end = chunk.end_time
            probability, segment = await asyncio.to_thread(
                detector.process_chunk, chunk
            )
            metrics["audio_chunks"] += 1
            metrics.setdefault("first_audio_timestamp", chunk.timestamp)
            metrics["last_audio_timestamp"] = chunk.end_time
            metrics["speech_positive_chunks"] += int(probability > threshold)
            rms = float(np.sqrt(np.mean(chunk.samples.astype(np.float32) ** 2)))
            metrics["peak_audio_rms"] = max(metrics["peak_audio_rms"], rms)
            if dashboard:
                dashboard.update(
                    audio={
                        "timestamp": chunk.timestamp,
                        "vad": probability,
                        "rms": rms,
                    }
                )
            if segment is not None:
                enqueue(segment)
        final = detector.flush()
        if final is not None:
            enqueue(final)
        await finish(segments)

    async def consume():
        while True:
            item = await segments.get()
            if item is None:
                break
            queued, segment = item
            queue_ms = 1000 * (time.monotonic() - queued)
            if queue_ms > queue_delay_limit * 1000:
                metrics["stale_segments"] += 1
                publish(
                    {
                        "event": "dropped_segment",
                        "reason": "stale",
                        "start_time": segment.start_time,
                        "end_time": segment.end_time,
                    }
                )
                continue
            result = await asyncio.to_thread(processor.process, segment)
            if visual_predictor is not None:
                deadline = time.monotonic() + 0.4
                required = min(segment.end_time, segment.start_time + 6)
                while (
                    not visual_buffer.closed
                    and visual_buffer.watermark < required
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.01)
                row, evidence = visual_buffer.select(
                    segment.start_time, segment.end_time
                )
                result["visual"] = evidence
                if result["emotion"] is not None:
                    result["audio_text_emotion"] = result["emotion"]
                    before = time.perf_counter()
                    result["emotion"] = await asyncio.to_thread(
                        visual_predictor.predict,
                        result["emotion"]["probabilities"],
                        row,
                    )
                    result["timings_ms"]["visual_fusion"] = 1000 * (
                        time.perf_counter() - before
                    )
            finished = time.monotonic()
            response_ms = 1000 * (
                finished - (segment.speech_end_time or segment.end_time)
            )
            result.update(
                event="prediction",
                source=metrics["source"],
                source_clock="monotonic_seconds",
                origin=origin,
                speech_end_time=segment.speech_end_time,
                segment_end_reason=segment.end_reason,
                emitted_at=finished,
                start_sec=segment.start_time - origin,
                end_sec=segment.end_time - origin,
            )
            result["timings_ms"].update(
                queue_wait=queue_ms, speech_end_to_prediction=response_ms
            )
            metrics["response_ms"].append(response_ms)
            metrics["predictions"] += 1
            publish(result)

    async def deadline():
        try:
            await asyncio.wait_for(stop.wait(), duration)
        except TimeoutError:
            stop.set()

    try:
        async with asyncio.TaskGroup() as group:
            group.create_task(audio_source())
            group.create_task(vad_worker())
            group.create_task(consume())
            if extractor is not None:
                group.create_task(video_source())
                group.create_task(vision_worker())
            if replay is None and duration is not None:
                group.create_task(deadline())
    finally:
        stop.set()
        if dashboard:
            dashboard.set_stop_callback(None)
    report = {
        **metrics,
        "event": "stream_summary",
        "elapsed_sec": time.monotonic() - origin,
        "response_latency": distribution(metrics["response_ms"]),
        "visual_latency": distribution(metrics["visual_ms"]),
        "peak_process_rss_mib": peak_rss_mib(),
    }
    report.pop("response_ms")
    report.pop("visual_ms")
    publish(report)
    if dashboard:
        dashboard.update(status="finished", metrics=report)
    return report
