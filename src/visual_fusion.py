"""Versioned visual head contract and shared inference for replay/live inputs."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from src.emotion_checkpoint import EMOTION_LABELS
from src.iemocap_embedding_cache import sha256
from src.models.visual_temporal import VisualTemporalClassifier, fuse_probabilities
from src.visual_feature_dataset import collate_visual


def save_visual_checkpoint(path, model, alpha, provenance, feature_config):
    payload = {
        "schema_version": 1,
        "task": "iemocap_visual_late_fusion_4",
        "label_names": EMOTION_LABELS,
        "architecture": model.config,
        "alpha": float(alpha),
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "feature_config": feature_config,
        "provenance": provenance,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_visual_checkpoint(path, base_checkpoint=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 1
        or payload.get("task") != "iemocap_visual_late_fusion_4"
        or payload["label_names"] != EMOTION_LABELS
    ):
        raise ValueError("Incompatible visual checkpoint")
    if not 0 <= payload["alpha"] <= 1:
        raise ValueError("Invalid fusion weight")
    if (
        base_checkpoint
        and sha256(base_checkpoint) != payload["provenance"]["base_checkpoint_sha256"]
    ):
        raise ValueError(
            "Visual head requires its original frozen audio/text checkpoint"
        )
    model = VisualTemporalClassifier(**payload["architecture"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def visual_row(
    blendshapes, images, times, blendshape_mask, image_mask, utterance_id="runtime"
):
    count = len(times)
    if not count:
        blendshapes, images, times = (
            np.zeros((1, 52), np.float32),
            np.zeros((1, 576), np.float32),
            [0.0],
        )
        blendshape_mask, image_mask = [False], [False]
    bm, im = (
        torch.as_tensor(blendshape_mask, dtype=torch.bool),
        torch.as_tensor(image_mask, dtype=torch.bool),
    )
    return {
        "utterance_id": utterance_id,
        "label": -1,
        "blendshapes": torch.as_tensor(np.asarray(blendshapes), dtype=torch.float32),
        "image_embeddings": torch.as_tensor(np.asarray(images), dtype=torch.float32),
        "times_sec": torch.as_tensor(times, dtype=torch.float32),
        "blendshape_mask": bm,
        "image_mask": im,
        "visual_mask": bm & im,
        "decoded_mask": torch.ones(len(bm), dtype=torch.bool),
    }


class VisualFusionPredictor:
    def __init__(self, checkpoint, base_checkpoint):
        self.model, self.payload = load_visual_checkpoint(checkpoint, base_checkpoint)
        self.alpha = self.payload["alpha"]
        self.mode = self.payload["architecture"]["mode"]
        self.checkpoint_sha256 = sha256(checkpoint)

    @torch.inference_mode()
    def predict(self, base_probabilities, row):
        base = torch.tensor(
            [[base_probabilities[name] for name in EMOTION_LABELS]], dtype=torch.float32
        )
        logits, available = self.model(collate_visual([row]))
        visual = logits.softmax(-1)
        fused = fuse_probabilities(base, visual, available, self.alpha)[0]
        index = int(fused.argmax())
        return {
            "label": EMOTION_LABELS[index],
            "label_index": index,
            "probabilities": dict(zip(EMOTION_LABELS, fused.tolist())),
            "visual_probabilities": dict(zip(EMOTION_LABELS, visual[0].tolist()))
            if bool(available[0])
            else None,
            "visual_available": bool(available[0]),
            "fallback": not bool(available[0]) or self.alpha == 0,
            "route": "audio_text_missing_vision"
            if not bool(available[0])
            else "audio_text_visual_shadow"
            if self.alpha == 0
            else "visual_probability_fusion",
            "visual_mode": self.mode,
            "visual_weight": self.alpha if bool(available[0]) else 0.0,
            "visual_checkpoint_sha256": self.checkpoint_sha256,
        }
