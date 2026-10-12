"""Causal face blendshapes and compact ImageNet image features on face crops.

Blendshapes are not FACS action units. Image embeddings are not pretrained
emotion predictions. The same frame extractor can be used for file or webcam input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from src.iemocap_embedding_cache import sha256

DETECTOR_PATH = (
    Path(__file__).resolve().parent / "weights/face_detection_yunet_2023mar.onnx"
)

BLENDSHAPE_NAMES = [member.name for member in Blendshapes]
STATUS_NAMES = {
    0: "valid",
    1: "no_face",
    2: "face_too_small",
    3: "ambiguous_faces",
    4: "frame_unavailable",
    5: "invalid_crop",
    6: "landmarks_unavailable",
}


def speaker_side(dialog_id, speaker):
    if (
        len(dialog_id) < 7
        or not dialog_id.startswith("Ses")
        or dialog_id[5] not in "FM"
        or speaker not in ("F", "M")
    ):
        raise ValueError("Invalid IEMOCAP speaker identity")
    return "left" if dialog_id[5] == speaker else "right"


def panel_crops(first_rgb):
    """Trim fixed black borders using only the first available frame."""
    height, width = first_rgb.shape[:2]
    result = {}
    for side, left, right in (("left", 0, width // 2), ("right", width // 2, width)):
        mask = first_rgb[:, left:right].max(axis=2) > 12
        yy, xx = np.where(mask)
        if len(xx):
            x0, y0 = max(left, left + int(xx.min()) - 4), max(0, int(yy.min()) - 4)
            x1, y1 = (
                min(right, left + int(xx.max()) + 5),
                min(height, int(yy.max()) + 5),
            )
        else:
            x0, y0, x1, y1 = left, 0, right, height
        result[side] = (x0, y0, x1, y1)
    return result


def choose_face(boxes, panel_shape, min_area_fraction=0.0125, ambiguity_ratio=0.6):
    """Largest sufficiently large face; mask similar-sized competing faces."""
    if not boxes:
        return None, 1
    height, width = panel_shape[:2]
    areas = np.array([max(0, b[2] - b[0]) * max(0, b[3] - b[1]) for b in boxes])
    order = np.argsort(areas)[::-1]
    if areas[order[0]] < height * width * min_area_fraction:
        return None, 2
    if len(order) > 1 and areas[order[1]] >= areas[order[0]] * ambiguity_ratio:
        return None, 3
    return int(order[0]), 0


def square_face_crop(rgb, box, margin=0.25):
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = box
    size = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = (
        max(0, int(np.floor(cx - size / 2))),
        max(0, int(np.floor(cy - size / 2))),
    )
    right, bottom = (
        min(width, int(np.ceil(cx + size / 2))),
        min(height, int(np.ceil(cy + size / 2))),
    )
    if right - left < 2 or bottom - top < 2:
        return None
    return rgb[top:bottom, left:right].copy()


@dataclass
class FaceFrame:
    blendshapes: np.ndarray
    box: np.ndarray
    crop: np.ndarray | None
    status: int
    face_count: int
    milliseconds: float
    image_valid: bool
    blendshape_valid: bool
    detection_score: float


class FaceFeatureExtractor:
    def __init__(self, model_path, min_area_fraction=0.0125, ambiguity_ratio=0.6):
        self.last_timestamp = -1
        self.min_area_fraction, self.ambiguity_ratio = (
            min_area_fraction,
            ambiguity_ratio,
        )
        # YuNet handles small faces in the wide IEMOCAP speaker panels. Dense
        # landmarking then sees a tight crop rather than the entire body panel.
        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH), "", (320, 320), 0.7, 0.3, 5000
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=mp.tasks.BaseOptions.Delegate.CPU,
                ),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=True,
            )
        )

    def close(self):
        self.landmarker.close()

    def extract(self, rgb, timestamp_ms):
        if timestamp_ms <= self.last_timestamp:
            raise ValueError("Visual frames must have increasing source timestamps")
        self.last_timestamp = timestamp_ms
        started = time.perf_counter()
        height, width = rgb.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        boxes = (
            []
            if detections is None
            else [[d[0], d[1], d[0] + d[2], d[1] + d[3]] for d in detections]
        )
        selected, status = choose_face(
            boxes, rgb.shape, self.min_area_fraction, self.ambiguity_ratio
        )
        features, box, crop = np.zeros(52, np.float32), np.zeros(4, np.float32), None
        image_valid, blendshape_valid, score = False, False, 0.0
        if selected is not None:
            box = np.asarray(boxes[selected], np.float32)
            score = float(detections[selected, -1])
            crop = square_face_crop(rgb, box)
            if crop is None:
                status = 5
            else:
                image_valid = True
                result = self.landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
                )
                if result.face_blendshapes:
                    categories = result.face_blendshapes[0]
                    if len(categories) != 52 or [c.index for c in categories] != list(
                        range(52)
                    ):
                        raise ValueError("Unexpected MediaPipe blendshape schema")
                    features = np.asarray([c.score for c in categories], np.float32)
                    blendshape_valid = True
                else:
                    status = 6
        return FaceFrame(
            features,
            box,
            crop,
            status,
            len(boxes),
            1000 * (time.perf_counter() - started),
            image_valid,
            blendshape_valid,
            score,
        )


class FaceImageEmbedder:
    def __init__(self, device="cpu"):
        self.weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = mobilenet_v3_small(weights=self.weights)
        self.model.classifier = torch.nn.Identity()
        self.model.to(device).eval().requires_grad_(False)
        self.transform, self.device = self.weights.transforms(), device
        weight_path = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / self.weights.url.rsplit("/", 1)[-1]
        )
        self.provenance = {
            "architecture": "mobilenet_v3_small",
            "pretraining": "ImageNet-1K",
            "weights": "IMAGENET1K_V1",
            "weights_url": self.weights.url,
            "weights_sha256": sha256(weight_path),
            "embedding_dim": 576,
            "transform": str(self.transform),
            "device": device,
        }

    @torch.inference_mode()
    def encode(self, crops):
        if not crops:
            return np.empty((0, 576), np.float32)
        batch = torch.stack(
            [self.transform(Image.fromarray(crop)) for crop in crops]
        ).to(self.device)
        return self.model(batch).cpu().numpy().astype(np.float32)
