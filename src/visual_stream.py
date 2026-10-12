"""Timestamped visual features shared by live capture and uncached video replay."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from src.extract_iemocap_visual import decoded_video, sample_times
from src.iemocap_embedding_cache import sha256
from src.visual_features import (
    FaceFeatureExtractor,
    FaceImageEmbedder,
    DETECTOR_PATH,
    STATUS_NAMES,
    BLENDSHAPE_NAMES,
    panel_crops,
)
from src.visual_fusion import visual_row

LANDMARKER = Path(__file__).resolve().parent / "weights/face_landmarker.task"


@dataclass
class VisualFrame:
    timestamp: float
    blendshapes: np.ndarray
    image_embedding: np.ndarray
    blendshape_valid: bool
    image_valid: bool
    status: str
    processing_ms: float


class VisualExtractor:
    def __init__(self, feature_config, device="cpu"):
        if (
            feature_config["schema_version"] != 2
            or feature_config["sample_hz"] != 5
            or feature_config["max_seconds"] != 6
        ):
            raise ValueError(
                "Runtime requires the verified v2 5 Hz / six-second visual contract"
            )
        if (
            sha256(LANDMARKER) != feature_config["landmarker_sha256"]
            or sha256(DETECTOR_PATH) != feature_config["face_detector"]["sha256"]
        ):
            raise ValueError("Runtime face models differ from training")
        self.face = FaceFeatureExtractor(LANDMARKER)
        self.image = FaceImageEmbedder(device)
        if (
            self.image.provenance["weights_sha256"]
            != feature_config["image_encoder"]["weights_sha256"]
        ):
            self.face.close()
            raise ValueError("Runtime image encoder differs from training")

    def close(self):
        self.face.close()

    def process(self, timestamp, bgr):
        start = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face = self.face.extract(rgb, round(timestamp * 1000))
        embedding = (
            self.image.encode([face.crop])[0]
            if face.image_valid
            else np.zeros(576, np.float32)
        )
        result = VisualFrame(
            timestamp,
            face.blendshapes,
            embedding,
            face.blendshape_valid,
            face.image_valid,
            STATUS_NAMES[face.status],
            1000 * (time.perf_counter() - start),
        )
        preview = bgr.copy()
        if face.image_valid:
            x0, y0, x1, y1 = np.asarray(face.box, int)
            cv2.rectangle(
                preview,
                (x0, y0),
                (x1, y1),
                (90, 210, 90) if face.blendshape_valid else (30, 170, 245),
                2,
            )
        scale = min(1, 640 / preview.shape[1])
        if scale < 1:
            preview = cv2.resize(preview, None, fx=scale, fy=scale)
        ok, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 78])
        result.processing_ms = 1000 * (time.perf_counter() - start)
        summary = {
            "timestamp": timestamp,
            "image_available": face.image_valid,
            "blendshapes_available": face.blendshape_valid,
            "status": result.status,
            "processing_ms": result.processing_ms,
            "blendshapes": {
                BLENDSHAPE_NAMES[i]: float(face.blendshapes[i])
                for i in np.argsort(face.blendshapes)[-6:][::-1]
            }
            if face.blendshape_valid
            else {},
        }
        return result, summary, jpeg.tobytes() if ok else None


class VisualBuffer:
    def __init__(self, max_frames=450):
        self.frames = deque(maxlen=max_frames)
        self.watermark = float("-inf")
        self.closed = False

    def append(self, frame):
        if frame.timestamp <= self.watermark:
            raise ValueError("Visual feature timestamps must increase")
        self.frames.append(frame)
        self.watermark = frame.timestamp

    def select(self, start, end, utterance_id="runtime"):
        requested = sample_times(start, end, 5, 6)
        count = len(requested)
        bs, images = (
            np.zeros((count, 52), np.float32),
            np.zeros((count, 576), np.float32),
        )
        bm, im = np.zeros(count, bool), np.zeros(count, bool)
        times = np.zeros(count, np.float32)
        selected = []
        stop = min(end, start + 6)
        # Actual capture timestamps are kept. Each 200 ms slot receives at most
        # one observation; late/future observations cannot fill an earlier slot.
        for frame in self.frames:
            if not start <= frame.timestamp < stop:
                continue
            index = int((frame.timestamp - start) * 5 + 1e-8)
            if index >= count or any(i == index for i, _ in selected):
                continue
            bs[index], images[index] = frame.blendshapes, frame.image_embedding
            bm[index], im[index] = frame.blendshape_valid, frame.image_valid
            times[index] = frame.timestamp - start
            selected.append((index, frame.timestamp))
        row = visual_row(bs, images, times, bm, im, utterance_id)
        evidence = {
            "window_start": start,
            "window_end": stop,
            "requested_frames": count,
            "observed_frames": len(selected),
            "image_frames": int(im.sum()),
            "blendshape_frames": int(bm.sum()),
            "source_timestamps": [t for _, t in selected],
            "image_coverage": float(im.mean()) if count else 0.0,
            "blendshape_coverage": float(bm.mean()) if count else 0.0,
        }
        return row, evidence


def video_samples(video, side, start, end, sample_hz=5):
    """Decode sequentially using the same speaker crop and display times as training."""
    target, crops = start, None
    for _, timestamp, bgr in decoded_video(video):
        if crops is None:
            crops = panel_crops(bgr)
        if timestamp < target - 1e-9:
            continue
        if timestamp >= end:
            break
        if side != "full":
            x0, y0, x1, y1 = crops[side]
            bgr = bgr[y0:y1, x0:x1]
        yield timestamp, bgr
        target += 1 / sample_hz
        # Skip unavailable slots across real timestamp gaps; never duplicate a frame.
        if target <= timestamp:
            target = start + (int((timestamp - start) * sample_hz) + 1) / sample_hz


def extract_interval(extractor, video, side, start, end, notify=None):
    buffer = VisualBuffer()
    for timestamp, frame in video_samples(video, side, start, min(end, start + 6)):
        event, summary, jpeg = extractor.process(timestamp, frame)
        buffer.append(event)
        if notify:
            notify(summary, jpeg)
    buffer.closed = True
    return buffer.select(start, end)
